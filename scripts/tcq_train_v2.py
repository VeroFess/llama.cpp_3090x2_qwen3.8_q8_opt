#!/usr/bin/env python3
"""TCQ codebook training v2 — adds tail-biting and left-shift trellis.

Extends tcq_train_vectorized.py with two QTIP paper improvements:
1. Tail-biting: two-pass Viterbi that constrains start=end state (Algorithm 4)
2. Left-shift trellis: QTIP's shift direction vs our right-shift

The left-shift and right-shift de Bruijn graphs are isomorphic via bit-reversal.
So we train with left-shift, bit-reverse the codebook indices, and use the
existing right-shift CUDA decode kernel unchanged.
"""

import numpy as np
import sys
import time

LLOYD_MAX_2BIT = np.array([-1.510, -0.4528, 0.4528, 1.510])
LLOYD_MAX_3BIT = np.array([-1.748, -1.050, -0.5006, -0.06971, 0.06971, 0.5006, 1.050, 1.748])


def build_predecessor_table(k, L, trellis='right'):
	"""Precompute predecessor states for each (next_state, predecessor_index) pair."""
	n_states = 1 << L
	n_out = 1 << k
	ns_range = np.arange(n_states)
	p_range = np.arange(n_out)

	if trellis == 'right':
		# Right-shift: next = (prev >> k) | (out << (L-k))
		# Predecessor of ns: prev = ((ns & mask_lower) << k) | p
		mask_lower = (1 << (L - k)) - 1
		predecessors = ((ns_range[:, None] & mask_lower) << k) | p_range[None, :]
	else:
		# Left-shift: next = ((prev << k) & mask_L) | c
		# Predecessor of ns: prev = (ns >> k) | (p << (L-k))
		predecessors = (ns_range[:, None] >> k) | (p_range[None, :] << (L - k))

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


def bit_reverse(x, nbits):
	"""Reverse the bit order of integer x within nbits bits."""
	result = 0
	for i in range(nbits):
		result = (result << 1) | ((x >> i) & 1)
	return result


def build_bit_reverse_table(L):
	"""Build a lookup table for L-bit reversal."""
	return np.array([bit_reverse(i, L) for i in range(1 << L)], dtype=np.int32)


def viterbi_batch(X, codebook, predecessors, k, L, batch_size=500,
				  overlap=None, trellis='right'):
	"""Vectorized Viterbi encode for a batch of samples.

	X: [n_samples, T]
	overlap: optional [n_samples] array of (L-k)-bit overlap constraints for tail-biting
	trellis: 'right' or 'left' — determines which bits are constrained by overlap
	Returns: states [n_samples, T], mse [n_samples]
	"""
	n_samples, T = X.shape
	n_states = 1 << L
	n_out = 1 << k
	INF = 1e30
	mask_Lk = (1 << (L - k)) - 1

	# Precompute state bit properties for overlap constraints
	states_arr = np.arange(n_states, dtype=np.int32)
	states_bottom_Lk = states_arr & mask_Lk  # bottom (L-k) bits
	states_top_Lk = states_arr >> k           # top (L-k) bits = shift right by k

	all_states = np.zeros((n_samples, T), dtype=np.int32)
	all_mse = np.zeros(n_samples)

	for start in range(0, n_samples, batch_size):
		end = min(start + batch_size, n_samples)
		B = end - start
		x = X[start:end]  # [B, T]

		if overlap is None:
			cost = np.zeros((B, n_states), dtype=np.float32)  # free init
		else:
			# Constrained init: only allow states whose "incoming context" matches overlap
			# Right-shift: incoming context = bottom (L-k) bits → state & mask_Lk == overlap
			# Left-shift: incoming context = top (L-k) bits → state >> k == overlap
			cost = np.full((B, n_states), INF, dtype=np.float32)
			batch_ov = overlap[start:end]  # [B]
			if trellis == 'right':
				init_match = (states_bottom_Lk[None, :] == batch_ov[:, None])
			else:
				init_match = (states_top_Lk[None, :] == batch_ov[:, None])
			cost[init_match] = 0.0

		bt_prev = np.zeros((B, T, n_states), dtype=np.int16)

		for t in range(T):
			dist = (x[:, t:t+1] - codebook[None, :]) ** 2  # [B, n_states]
			pred_costs = cost[:, predecessors]  # [B, n_states, n_out]
			total = pred_costs + dist[:, :, None]  # [B, n_states, n_out]
			best_p = total.argmin(axis=2)
			cost = total.min(axis=2)
			bt_prev[:, t, :] = predecessors[np.arange(n_states)[None, :], best_p]

		if overlap is not None:
			# Constrained final: the "outgoing context" must match overlap
			# Right-shift: outgoing context = state >> k == overlap
			# Left-shift: outgoing context = state & mask_Lk == overlap
			batch_ov = overlap[start:end]
			if trellis == 'right':
				final_bad = (states_top_Lk[None, :] != batch_ov[:, None])
			else:
				final_bad = (states_bottom_Lk[None, :] != batch_ov[:, None])
			cost[final_bad] = INF

		# backtrace
		best_final = cost.argmin(axis=1)  # [B]
		state = best_final.copy()
		for t in range(T - 1, -1, -1):
			all_states[start:end, t] = state
			state = bt_prev[np.arange(B), t, state]

		recon = codebook[all_states[start:end]]
		all_mse[start:end] = np.mean((x - recon) ** 2, axis=1)

	return all_states, all_mse


def viterbi_batch_tailbiting(X, codebook, predecessors, k, L,
							 batch_size=500, trellis='right'):
	"""Two-pass tail-biting Viterbi (QTIP Algorithm 4).

	Pass 1: rotate by T/2, run free-init Viterbi, extract overlap from midpoint.
	Pass 2: run on original with overlap constraint on start and end states.
	"""
	n_samples, T = X.shape
	half_T = T // 2
	mask_Lk = (1 << (L - k)) - 1

	# Pass 1: rotate sequence by T/2, unconstrained Viterbi
	X_rot = np.roll(X, -half_T, axis=1)
	states_rot, _ = viterbi_batch(X_rot, codebook, predecessors, k, L, batch_size)

	# Extract overlap from midpoint of rotated sequence
	# Position half_T in rotated sequence = original position 0 (the junction point)
	mid_states = states_rot[:, half_T]
	if trellis == 'right':
		# Right-shift: incoming context = bottom (L-k) bits
		overlap = mid_states & mask_Lk
	else:
		# Left-shift: incoming context = top (L-k) bits
		overlap = mid_states >> k

	# Pass 2: constrained Viterbi on original sequence
	states, mses = viterbi_batch(X, codebook, predecessors, k, L, batch_size,
								 overlap=overlap, trellis=trellis)

	return states, mses


def train_codebook(k, L, sigma, centroids, n_train=2000, n_elements=128,
				   n_iters=30, n_restarts=3, batch_size=500,
				   trellis='right', tail_biting=False):
	"""Train TCQ codebook with GLA."""
	n_states = 1 << L
	predecessors = build_predecessor_table(k, L, trellis)
	best_global_mse = float('inf')
	best_global_codebook = None

	encode_fn = viterbi_batch_tailbiting if tail_biting else viterbi_batch

	for restart in range(n_restarts):
		if n_restarts > 1:
			print(f"\n=== Restart {restart+1}/{n_restarts} ===")

		codebook = init_coset_codebook(k, L, sigma, centroids)
		if restart > 0:
			codebook += np.random.randn(n_states) * sigma * 0.02

		best_mse = float('inf')
		best_codebook = codebook.copy()
		stall_count = 0

		for iteration in range(n_iters):
			t0 = time.time()
			X = np.random.randn(n_train, n_elements).astype(np.float32) * sigma

			if tail_biting:
				states, mses = encode_fn(X, codebook, predecessors, k, L,
										 batch_size, trellis=trellis)
			else:
				states, mses = encode_fn(X, codebook, predecessors, k, L, batch_size)
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

			dead = state_counts == 0
			n_dead = dead.sum()
			if n_dead > 0 and iteration < n_iters - 1:
				alive_idx = np.where(~dead)[0]
				donors = np.random.choice(alive_idx, size=n_dead)
				codebook[dead] = codebook[donors] + np.random.randn(n_dead) * sigma * 0.01

			tag = "TB " if tail_biting else ""
			print(f"  {tag}iter {iteration+1:2d}: MSE = {mse:.6f}  ({used}/{n_states} used, {elapsed:.1f}s){improved}")

			if stall_count >= 5 and iteration > 8:
				print(f"  Converged (5 iterations without improvement)")
				break

		if best_mse < best_global_mse:
			best_global_mse = best_mse
			best_global_codebook = best_codebook.copy()

	return best_global_codebook, best_global_mse


def evaluate(codebook, k, L, sigma, n_eval=1000, n_elements=128,
			 trellis='right', tail_biting=False):
	"""Evaluate codebook on fresh data."""
	predecessors = build_predecessor_table(k, L, trellis)
	X = np.random.randn(n_eval, n_elements).astype(np.float32) * sigma
	if tail_biting:
		_, mses = viterbi_batch_tailbiting(X, codebook, predecessors, k, L,
										   batch_size=500, trellis=trellis)
	else:
		_, mses = viterbi_batch(X, codebook, predecessors, k, L, batch_size=500)
	return mses.mean()


def export_c_codebook(codebook, k, L, scale, mse_reduction, db_gain, name_suffix, extra_note=""):
	"""Print codebook as C constant array."""
	n_states = len(codebook)
	scaled = codebook * scale
	print(f"\n// GLA-trained TCQ codebook: k={k}, L={L} ({n_states} states){extra_note}")
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
	parser = argparse.ArgumentParser(description='Train TCQ codebook v2')
	parser.add_argument('--bits', type=int, default=3, choices=[2, 3])
	parser.add_argument('--n-train', type=int, default=4000)
	parser.add_argument('--n-iters', type=int, default=100)
	parser.add_argument('--n-restarts', type=int, default=1)
	parser.add_argument('--batch-size', type=int, default=500)
	parser.add_argument('--seed', type=int, default=99)
	parser.add_argument('--trellis', choices=['left', 'right'], default='right')
	parser.add_argument('--tail-biting', action='store_true')
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

	flags = []
	if args.trellis == 'left':
		flags.append("left-shift")
	if args.tail_biting:
		flags.append("tail-biting")
	flag_str = f" [{', '.join(flags)}]" if flags else ""

	print(f"=== {args.bits}-bit TCQ codebook training v2{flag_str} ===")
	print(f"k={k}, L={L}, {1<<L} states, n_train={args.n_train}, "
		  f"n_iters={args.n_iters}, n_restarts={args.n_restarts}, "
		  f"seed={args.seed}\n")

	# baseline
	lm_mse = lloyd_max_baseline(centroids, sigma)
	print(f"Lloyd-Max {args.bits}-bit baseline: MSE = {lm_mse:.6f}\n")

	# train
	t0 = time.time()
	codebook, train_mse = train_codebook(
		k, L, sigma, centroids,
		n_train=args.n_train, n_iters=args.n_iters,
		n_restarts=args.n_restarts, batch_size=args.batch_size,
		trellis=args.trellis, tail_biting=args.tail_biting
	)
	train_time = time.time() - t0
	print(f"\nTraining took {train_time:.1f}s")

	# If left-shift, convert to right-shift indexing for CUDA compatibility
	if args.trellis == 'left':
		print(f"\nConverting left-shift codebook to right-shift indexing (bit-reversal)...")
		rev_table = build_bit_reverse_table(L)
		codebook_right = np.zeros_like(codebook)
		codebook_right[rev_table] = codebook
		codebook_export = codebook_right
		extra_note = " (left-shift trained, bit-reversed for right-shift decode)"
	else:
		codebook_export = codebook
		extra_note = ""

	if args.tail_biting:
		extra_note += " (tail-biting)"

	# evaluate on fresh data (with same trellis/tail-biting as training)
	np.random.seed(12345)
	eval_mse = evaluate(codebook, k, L, sigma, n_eval=2000,
						trellis=args.trellis, tail_biting=args.tail_biting)
	reduction = (1 - eval_mse / lm_mse) * 100
	db_gain = 10 * np.log10(lm_mse / eval_mse)
	print(f"\nEVAL (2000 fresh, {args.trellis}-shift{', TB' if args.tail_biting else ''}): "
		  f"MSE = {eval_mse:.6f} ({reduction:+.1f}% vs LM, {db_gain:+.2f} dB)")

	# also evaluate with free-init right-shift (what CUDA will actually do)
	if args.trellis == 'left' or args.tail_biting:
		np.random.seed(12345)
		predecessors_right = build_predecessor_table(k, L, 'right')
		X_eval = np.random.randn(2000, 128).astype(np.float32) * sigma
		_, mses_right = viterbi_batch(X_eval, codebook_export, predecessors_right, k, L)
		eval_mse_right = mses_right.mean()
		reduction_right = (1 - eval_mse_right / lm_mse) * 100
		db_gain_right = 10 * np.log10(lm_mse / eval_mse_right)
		print(f"EVAL (right-shift free-init, CUDA-compatible): "
			  f"MSE = {eval_mse_right:.6f} ({reduction_right:+.1f}% vs LM, {db_gain_right:+.2f} dB)")
		# Use right-shift eval for export stats
		reduction = reduction_right
		db_gain = db_gain_right

	# export C codebook (always in right-shift format)
	export_c_codebook(codebook_export, k, L, scale, reduction, db_gain, "", extra_note)
	export_c_codebook(codebook_export, k, L, scale, reduction, db_gain, "_fattn", extra_note)

	# export binary (scaled, right-shift format)
	scaled = codebook_export * scale
	suffix = ""
	if args.trellis == 'left':
		suffix += "_left"
	if args.tail_biting:
		suffix += "_tb"
	bin_path = f"/tmp/tcq_{k}bit{suffix}_s{args.seed}.bin"
	scaled.astype(np.float32).tofile(bin_path)
	print(f"\nBinary codebook written to {bin_path}")
