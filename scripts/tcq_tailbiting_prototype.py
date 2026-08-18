#!/usr/bin/env python3
"""Prototype tail-biting TCQ: seed each block from previous block's final state.

Compares three modes:
1. Free-init: all states equally viable (current implementation)
2. Tail-biting: block N+1 starts with block N's final state (zero cost)
3. Tail-biting soft: block N+1 biases toward block N's final state (low cost)

Tests with sequences of multiple 128-element blocks to simulate KV cache rows.
"""

import numpy as np
import sys
import time

sys.path.insert(0, '.')
from scripts.tcq_train_vectorized import (
	build_predecessor_table, viterbi_batch, lloyd_max_baseline,
	LLOYD_MAX_2BIT, LLOYD_MAX_3BIT
)


def viterbi_single(x, codebook, predecessors, k, L, initial_state=None, bias_penalty=None):
	"""Single-sequence Viterbi with optional initial state bias."""
	T = len(x)
	n_states = 1 << L
	n_out = 1 << k
	INF = 1e30

	if initial_state is not None and bias_penalty is not None:
		# soft tail-biting: penalize states that aren't the initial state
		cost = np.full(n_states, bias_penalty, dtype=np.float32)
		cost[initial_state] = 0.0
	elif initial_state is not None:
		# hard tail-biting: only allow the initial state
		cost = np.full(n_states, INF, dtype=np.float32)
		cost[initial_state] = 0.0
	else:
		# free-init
		cost = np.zeros(n_states, dtype=np.float32)

	bt_prev = np.zeros((T, n_states), dtype=np.int32)

	for t in range(T):
		dist = (x[t] - codebook) ** 2  # [n_states]
		pred_costs = cost[predecessors]  # [n_states, n_out]
		total = pred_costs + dist[:, None]  # [n_states, n_out]

		best_p = total.argmin(axis=1)  # [n_states]
		new_cost = total.min(axis=1)  # [n_states]

		bt_prev[t, :] = predecessors[np.arange(n_states), best_p]
		cost = new_cost

	# backtrace
	state = cost.argmin()
	final_state = state
	states = np.zeros(T, dtype=np.int32)
	for t in range(T - 1, -1, -1):
		states[t] = state
		state = bt_prev[t, state]

	recon = codebook[states]
	mse = np.mean((x - recon) ** 2)
	return mse, states, final_state


def simulate_kv_row(n_blocks, codebook, predecessors, k, L, sigma=1.0, mode='free'):
	"""Simulate encoding a KV cache row of n_blocks × 128 elements."""
	T = 128
	total_mse = 0.0
	prev_final_state = None

	for b in range(n_blocks):
		x = np.random.randn(T).astype(np.float32) * sigma

		if mode == 'free' or b == 0:
			mse, states, final_state = viterbi_single(x, codebook, predecessors, k, L)
		elif mode == 'hard':
			mse, states, final_state = viterbi_single(
				x, codebook, predecessors, k, L, initial_state=prev_final_state)
		elif mode == 'soft':
			mse, states, final_state = viterbi_single(
				x, codebook, predecessors, k, L,
				initial_state=prev_final_state, bias_penalty=0.5)
		elif mode == 'softer':
			mse, states, final_state = viterbi_single(
				x, codebook, predecessors, k, L,
				initial_state=prev_final_state, bias_penalty=0.1)

		total_mse += mse
		prev_final_state = final_state

	return total_mse / n_blocks


def run_experiment(k, L, codebook, n_rows=200, n_blocks_per_row=8):
	"""Compare free-init vs tail-biting across many KV cache rows."""
	predecessors = build_predecessor_table(k, L)
	n_states = 1 << L

	modes = ['free', 'hard', 'soft', 'softer']
	results = {m: [] for m in modes}

	print(f"\nSimulating {n_rows} KV rows × {n_blocks_per_row} blocks × 128 elements")
	print(f"k={k}, L={L}, {n_states} states\n")

	for row in range(n_rows):
		if (row + 1) % 50 == 0:
			print(f"  Row {row+1}/{n_rows}...")
		for mode in modes:
			np.random.seed(row * 1000)  # same data for each mode
			mse = simulate_kv_row(n_blocks_per_row, codebook, predecessors, k, L, mode=mode)
			results[mode].append(mse)

	print(f"\nResults:")
	print(f"{'Mode':>10} {'Mean MSE':>12} {'vs free-init':>14}")
	print(f"{'-'*10:>10} {'-'*12:>12} {'-'*14:>14}")
	free_mse = np.mean(results['free'])
	for mode in modes:
		mean_mse = np.mean(results[mode])
		delta = (mean_mse / free_mse - 1) * 100
		print(f"{mode:>10} {mean_mse:12.6f} {delta:+13.2f}%")

	return results


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--bits', type=int, default=2, choices=[2, 3])
	parser.add_argument('--n-rows', type=int, default=200)
	parser.add_argument('--n-blocks', type=int, default=8)
	parser.add_argument('--seed', type=int, default=42)
	args = parser.parse_args()

	np.random.seed(args.seed)

	if args.bits == 2:
		k, L = 2, 8
		centroids = LLOYD_MAX_2BIT
	else:
		k, L = 3, 9
		centroids = LLOYD_MAX_3BIT

	# Use the current trained codebook by training a quick one
	from scripts.tcq_train_vectorized import train_codebook as tc
	print(f"Training {args.bits}-bit codebook for tail-biting experiment...")
	codebook, _ = tc(k, L, 1.0, centroids, n_train=500, n_iters=15, n_restarts=1, batch_size=500)

	print(f"\n{'='*60}")
	print(f"Tail-biting experiment: {args.bits}-bit TCQ")
	print(f"{'='*60}")

	run_experiment(k, L, codebook, n_rows=args.n_rows, n_blocks_per_row=args.n_blocks)

	print(f"\nNote: 'hard' forces exact previous final state.")
	print(f"'soft' penalizes other states by 0.5, 'softer' by 0.1.")
	print(f"First block in each row always uses free-init.")
	print(f"\nIMPORTANT: Adjacent blocks in a KV cache row have INDEPENDENT data")
	print(f"(different token positions), so the state correlation may be weak.")
	print(f"This tests whether the trellis state carries useful information")
	print(f"even across independent data blocks.")
