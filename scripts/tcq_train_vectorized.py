#!/usr/bin/env python3
"""Vectorized TCQ codebook training — batch Viterbi across all samples.

Supports both 2-bit (k=2, L=8, 256 states) and 3-bit (k=3, L=9, 512 states).
10-50x faster than the pure Python version, enabling n_train=2000+.
"""

import numpy as np
import sys
import time

LLOYD_MAX_2BIT = np.array([-1.510, -0.4528, 0.4528, 1.510])
LLOYD_MAX_3BIT = np.array([-1.748, -1.050, -0.5006, -0.06971, 0.06971, 0.5006, 1.050, 1.748])


def build_predecessor_table(k, L):
	"""Precompute predecessor states for each (next_state, predecessor_index) pair."""
	n_states = 1 << L
	n_out = 1 << k
	mask_lower = (1 << (L - k)) - 1
	# predecessors[ns, p] = the predecessor state that transitions to ns via p-th path
	ns_range = np.arange(n_states)
	p_range = np.arange(n_out)
	predecessors = ((ns_range[:, None] & mask_lower) << k) | p_range[None, :]
	return predecessors  # [n_states, n_out]


def init_coset_codebook(k, L, sigma, centroids):
	"""Initialize codebook with coset-shifted Lloyd-Max centroids."""
	n_states = 1 << L
	n_out = 1 << k
	n_groups = 1 << (L - k)
	c = centroids * sigma
	spacing = c[1] - c[0]
	shifts = np.linspace(-spacing / 2, spacing / 2, n_groups, endpoint=False)
	codebook = np.zeros(n_states)
	for group in range(n_groups):
		for pos in range(n_out):
			state = (group << k) | pos
			codebook[state] = c[pos] + shifts[group]
	return codebook


def viterbi_batch(X, codebook, predecessors, k, L, batch_size=500):
	"""Vectorized Viterbi encode for a batch of samples.

	X: [n_samples, T]
	Returns: states [n_samples, T], mse [n_samples]
	"""
	n_samples, T = X.shape
	n_states = 1 << L
	n_out = 1 << k
	INF = 1e30

	all_states = np.zeros((n_samples, T), dtype=np.int32)
	all_mse = np.zeros(n_samples)

	# process in chunks to limit memory (bt is [batch, T, n_states])
	for start in range(0, n_samples, batch_size):
		end = min(start + batch_size, n_samples)
		B = end - start
		x = X[start:end]  # [B, T]

		cost = np.zeros((B, n_states), dtype=np.float32)  # free init
		bt_prev = np.zeros((B, T, n_states), dtype=np.int16)  # int16 enough for <=32K states

		for t in range(T):
			# dist[b, ns] = (x[b,t] - codebook[ns])^2
			dist = (x[:, t:t+1] - codebook[None, :]) ** 2  # [B, n_states]

			# pred_costs[b, ns, p] = cost[b, predecessors[ns, p]]
			pred_costs = cost[:, predecessors]  # [B, n_states, n_out]

			# total[b, ns, p] = pred_costs[b, ns, p] + dist[b, ns]
			total = pred_costs + dist[:, :, None]  # [B, n_states, n_out]

			# best predecessor for each (b, ns)
			best_p = total.argmin(axis=2)  # [B, n_states]
			cost = total.min(axis=2)  # [B, n_states]

			# store backtrace: the actual predecessor state
			bt_prev[:, t, :] = predecessors[np.arange(n_states)[None, :], best_p]

		# backtrace
		best_final = cost.argmin(axis=1)  # [B]
		state = best_final.copy()
		for t in range(T - 1, -1, -1):
			all_states[start:end, t] = state
			# bt_prev[:, t, state] — need to gather per-sample
			state = bt_prev[np.arange(B), t, state]

		# compute MSE from states
		recon = codebook[all_states[start:end]]  # [B, T]
		all_mse[start:end] = np.mean((x - recon) ** 2, axis=1)

	return all_states, all_mse


def train_codebook(k, L, sigma, centroids, n_train=2000, n_elements=128,
				   n_iters=30, n_restarts=3, batch_size=500):
	"""Train TCQ codebook with GLA (Generalized Lloyd Algorithm)."""
	n_states = 1 << L
	predecessors = build_predecessor_table(k, L)
	best_global_mse = float('inf')
	best_global_codebook = None

	for restart in range(n_restarts):
		if n_restarts > 1:
			print(f"\n=== Restart {restart+1}/{n_restarts} ===")

		codebook = init_coset_codebook(k, L, sigma, centroids)
		# add small random perturbation for restarts > 0
		if restart > 0:
			codebook += np.random.randn(n_states) * sigma * 0.02

		best_mse = float('inf')
		best_codebook = codebook.copy()
		stall_count = 0

		for iteration in range(n_iters):
			t0 = time.time()
			X = np.random.randn(n_train, n_elements).astype(np.float32) * sigma

			states, mses = viterbi_batch(X, codebook, predecessors, k, L, batch_size)
			mse = mses.mean()
			elapsed = time.time() - t0

			improved = ""
			if mse < best_mse:
				best_mse = mse
				best_codebook = codebook.copy()
				improved = " *"
				stall_count = 0
			else:
				stall_count += 1

			# GLA centroid update
			state_sums = np.zeros(n_states, dtype=np.float64)
			state_counts = np.zeros(n_states, dtype=np.int64)
			np.add.at(state_sums, states.ravel(), X.ravel())
			np.add.at(state_counts, states.ravel(), 1)

			used = np.sum(state_counts > 0)
			for s in range(n_states):
				if state_counts[s] > 0:
					codebook[s] = state_sums[s] / state_counts[s]

			# reinitialize dead states from neighbors of most-used states
			dead = state_counts == 0
			n_dead = dead.sum()
			if n_dead > 0 and iteration < n_iters - 1:
				alive_idx = np.where(~dead)[0]
				# pick random alive states and add small perturbation
				donors = np.random.choice(alive_idx, size=n_dead)
				codebook[dead] = codebook[donors] + np.random.randn(n_dead) * sigma * 0.01

			print(f"  iter {iteration+1:2d}: MSE = {mse:.6f}  ({used}/{n_states} used, {elapsed:.1f}s){improved}")

			if stall_count >= 5 and iteration > 8:
				print(f"  Converged (5 iterations without improvement)")
				break

		if best_mse < best_global_mse:
			best_global_mse = best_mse
			best_global_codebook = best_codebook.copy()

	return best_global_codebook, best_global_mse


def evaluate(codebook, k, L, sigma, n_eval=1000, n_elements=128):
	"""Evaluate codebook on fresh data."""
	predecessors = build_predecessor_table(k, L)
	X = np.random.randn(n_eval, n_elements).astype(np.float32) * sigma
	_, mses = viterbi_batch(X, codebook, predecessors, k, L, batch_size=500)
	return mses.mean()


def export_c_codebook(codebook, k, L, scale, mse_reduction, db_gain, name_suffix):
	"""Print codebook as C constant array."""
	n_states = len(codebook)
	scaled = codebook * scale
	print(f"\n// GLA-trained free-init TCQ codebook: k={k}, L={L} ({n_states} states)")
	print(f"// MSE reduction: {mse_reduction:.1f}% vs Lloyd-Max {k}-bit, {db_gain:.2f} dB")
	print(f"// Scaled by 1/sqrt(128) = {scale:.10f}")
	print(f"static __constant__ float d_turbo{k}_tcq_codebook{name_suffix}[{n_states}] = {{")
	for i in range(0, n_states, 8):
		vals = ', '.join(f'{scaled[j]:+.8f}' for j in range(i, min(i + 8, n_states)))
		comma = ',' if i + 8 < n_states else ''
		print(f"    {vals}{comma}")
	print("};")


def lloyd_max_baseline(centroids, sigma, n_eval=2000, n_elements=128):
	"""Compute Lloyd-Max scalar quantization MSE baseline."""
	c = centroids * sigma
	bounds = (c[:-1] + c[1:]) / 2
	X = np.random.randn(n_eval, n_elements) * sigma
	indices = np.searchsorted(bounds, X)
	recon = c[indices]
	return np.mean((X - recon) ** 2)


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Train TCQ codebook')
	parser.add_argument('--bits', type=int, default=2, choices=[2, 3])
	parser.add_argument('--n-train', type=int, default=2000)
	parser.add_argument('--n-iters', type=int, default=30)
	parser.add_argument('--n-restarts', type=int, default=3)
	parser.add_argument('--batch-size', type=int, default=500)
	parser.add_argument('--seed', type=int, default=42)
	args = parser.parse_args()

	np.random.seed(args.seed)
	sigma = 1.0
	scale = 1.0 / np.sqrt(128)

	if args.bits == 2:
		k, L = 2, 8
		centroids = LLOYD_MAX_2BIT
	else:
		k, L = 3, 9
		centroids = LLOYD_MAX_3BIT

	print(f"=== {args.bits}-bit TCQ codebook training ===")
	print(f"k={k}, L={L}, {1<<L} states, n_train={args.n_train}, "
		  f"n_iters={args.n_iters}, n_restarts={args.n_restarts}\n")

	# baseline
	lm_mse = lloyd_max_baseline(centroids, sigma)
	print(f"Lloyd-Max {args.bits}-bit baseline: MSE = {lm_mse:.6f}\n")

	# train
	t0 = time.time()
	codebook, train_mse = train_codebook(
		k, L, sigma, centroids,
		n_train=args.n_train, n_iters=args.n_iters,
		n_restarts=args.n_restarts, batch_size=args.batch_size
	)
	train_time = time.time() - t0
	print(f"\nTraining took {train_time:.1f}s")

	# evaluate on fresh data
	np.random.seed(12345)
	eval_mse = evaluate(codebook, k, L, sigma, n_eval=2000)
	reduction = (1 - eval_mse / lm_mse) * 100
	db_gain = 10 * np.log10(lm_mse / eval_mse)
	print(f"\nEVAL (2000 fresh samples): MSE = {eval_mse:.6f} "
		  f"({reduction:+.1f}% vs LM, {db_gain:+.2f} dB)")

	# export for both compilation units
	export_c_codebook(codebook, k, L, scale, reduction, db_gain, "")
	export_c_codebook(codebook, k, L, scale, reduction, db_gain, "_fattn")

	# export binary (scaled) for runtime loading
	scaled = codebook * scale
	bin_path = f"/tmp/tcq_{k}bit_numpy_s{args.seed}.bin"
	scaled.astype(np.float32).tofile(bin_path)
	print(f"\nBinary codebook written to {bin_path}")
