#!/usr/bin/env python3
"""
TCQ with right-shift trellis for O(1) parallel decode.
Trains codebook for turbo3_tcq CUDA implementation.

Right-shift trellis: next_state = (state >> k) | (output << (L-k))
This allows decode via sliding 9-bit window over output bitstream.
"""

import numpy as np
from scipy.stats import norm as ndist
import sys

LLOYD_MAX_3BIT = np.array([-1.748, -1.050, -0.5006, -0.06971, 0.06971, 0.5006, 1.050, 1.748])

class RightShiftTrellis:
	def __init__(self, k, L, sigma=1.0):
		self.k = k
		self.L = L
		self.n_states = 1 << L
		self.n_outputs = 1 << k
		self.state_mask = self.n_states - 1
		self.sigma = sigma
		self.codebook = self._init_coset_codebook(sigma)

	def _init_coset_codebook(self, sigma):
		n_groups = 1 << (self.L - self.k)
		centroids = LLOYD_MAX_3BIT * sigma
		spacing = centroids[1] - centroids[0]
		shifts = np.linspace(-spacing/2, spacing/2, n_groups, endpoint=False)
		codebook = np.zeros(self.n_states)
		for group in range(n_groups):
			for pos in range(self.n_outputs):
				state = (group << self.k) | pos
				codebook[state] = centroids[pos] + shifts[group]
		return codebook

	def next_state(self, state, output):
		return (state >> self.k) | (output << (self.L - self.k))

	def predecessors(self, ns):
		"""Return (prev_state, output) pairs that lead to ns."""
		# ns = (prev >> k) | (out << (L-k))
		# ns[L-k-1:0] = prev[L-1:k], ns[L-1:L-k] = out
		out = ns >> (self.L - self.k)
		base = (ns & ((1 << (self.L - self.k)) - 1)) << self.k
		preds = []
		for p in range(self.n_outputs):
			prev = base | p
			preds.append((prev, out))
		return preds

	def _encode_single(self, x):
		T = len(x)
		n_states = self.n_states
		n_out = self.n_outputs
		cb = self.codebook
		INF = 1e30

		# Start from state 0 (matches decode convention: 6 prepended zero bits)
		cost = np.full(n_states, INF)
		cost[0] = 0.0

		bt_prev = np.zeros((T, n_states), dtype=np.int32)

		for t in range(T):
			new_cost = np.full(n_states, INF)
			for s in range(n_states):
				if cost[s] >= INF:
					continue
				base_cost = cost[s]
				for out in range(n_out):
					ns = self.next_state(s, out)
					d = (x[t] - cb[ns]) ** 2
					total = base_cost + d
					if total < new_cost[ns]:
						new_cost[ns] = total
						bt_prev[t][ns] = s
			cost = new_cost

		# Backtrack
		state = np.argmin(cost)
		recon = np.zeros(T)
		states = np.zeros(T, dtype=np.int32)
		for t in range(T - 1, -1, -1):
			states[t] = state
			recon[t] = cb[state]
			state = bt_prev[t][state]

		return recon, states

	def encode_viterbi_batch(self, X):
		batch, T = X.shape
		all_recon = np.zeros_like(X)
		all_states = []
		for b in range(batch):
			recon, states = self._encode_single(X[b])
			all_recon[b] = recon
			all_states.append(states)
		mse = np.mean((X - all_recon) ** 2)
		return all_recon, all_states, mse

	def train_codebook(self, n_train=500, n_elements=128, n_iters=20, verbose=True):
		if verbose:
			print(f"  Training right-shift TCQ: k={self.k}, L={self.L}, "
				  f"{self.n_states} states, {n_train} seqs...")
		best_mse = float('inf')
		best_codebook = self.codebook.copy()

		for iteration in range(n_iters):
			X = np.random.randn(n_train, n_elements) * self.sigma
			_, all_states, mse = self.encode_viterbi_batch(X)
			if verbose:
				print(f"    iter {iteration+1:2d}: MSE = {mse:.6f}", end='')
			if mse < best_mse:
				best_mse = mse
				best_codebook = self.codebook.copy()

			state_sums = np.zeros(self.n_states)
			state_counts = np.zeros(self.n_states, dtype=np.int64)
			for b in range(n_train):
				states = all_states[b]
				for t in range(n_elements):
					s = states[t]
					state_sums[s] += X[b, t]
					state_counts[s] += 1

			updated = 0
			for s in range(self.n_states):
				if state_counts[s] > 0:
					self.codebook[s] = state_sums[s] / state_counts[s]
					updated += 1
			if verbose:
				print(f"  ({updated}/{self.n_states} states used)")
			if iteration > 2 and abs(mse - best_mse) / best_mse < 1e-4:
				if verbose:
					print(f"    Converged at iteration {iteration+1}")
				break

		self.codebook = best_codebook
		return best_mse

	def verify_decode(self, x, states):
		"""Verify that sliding-window decode reproduces Viterbi states."""
		T = len(x)
		# Build bitstream: 6 zero bits + 128 outputs of 3 bits
		outputs = np.zeros(T, dtype=np.int32)
		for t in range(T):
			outputs[t] = states[t] >> (self.L - self.k)  # output = top k bits of state

		# Pack into bitstream (6 zero prefix + outputs)
		n_bits = 6 + T * 3
		bitstream = np.zeros((n_bits + 7) // 8, dtype=np.uint8)
		# Skip first 6 bits (zeros = initial state 0)
		for t in range(T):
			out = outputs[t]
			bit_pos = 6 + t * 3
			for b in range(3):
				byte_idx = (bit_pos + b) // 8
				bit_off = (bit_pos + b) % 8
				if out & (1 << b):
					bitstream[byte_idx] |= (1 << bit_off)

		# Decode via sliding window
		for t in range(T):
			bit_pos = t * 3
			byte_idx = bit_pos // 8
			bit_off = bit_pos % 8
			raw = int(bitstream[byte_idx]) | (int(bitstream[byte_idx + 1]) << 8)
			state_decoded = (raw >> bit_off) & 0x1FF
			assert state_decoded == states[t], \
				f"t={t}: decoded state {state_decoded} != viterbi state {states[t]}"
		return True


if __name__ == '__main__':
	np.random.seed(42)
	sigma = 1.0
	quick = '--quick' in sys.argv

	print("Right-Shift TCQ Codebook Training for CUDA Implementation")
	print(f"{'quick' if quick else 'full'} mode\n")

	# Lloyd-Max 3-bit baseline
	centroids = LLOYD_MAX_3BIT * sigma
	bounds = (centroids[:-1] + centroids[1:]) / 2
	n_eval = 200 if quick else 500
	lm_mses = []
	for _ in range(n_eval):
		x = np.random.randn(128) * sigma
		indices = np.searchsorted(bounds, x)
		recon = centroids[indices]
		lm_mses.append(np.mean((x - recon) ** 2))
	lm_mse = np.mean(lm_mses)
	print(f"Lloyd-Max 3-bit baseline: MSE = {lm_mse:.6f}\n")

	# Train right-shift TCQ
	trellis = RightShiftTrellis(k=3, L=9, sigma=sigma)
	n_train = 500 if not quick else 200
	train_mse = trellis.train_codebook(n_train=n_train, n_elements=128, n_iters=15)

	# Verify sliding-window decode
	print("\n  Verifying sliding-window decode...")
	x_test = np.random.randn(128) * sigma
	recon, states = trellis._encode_single(x_test)
	assert trellis.verify_decode(x_test, states), "DECODE VERIFICATION FAILED"
	print("  Decode verification PASSED!")

	# Evaluate
	eval_mses = []
	for _ in range(n_eval):
		x = np.random.randn(128) * sigma
		recon, _ = trellis._encode_single(x)
		eval_mses.append(np.mean((x - recon) ** 2))
	tcq_mse = np.mean(eval_mses)
	reduction = (1 - tcq_mse / lm_mse) * 100
	db_gain = 10 * np.log10(lm_mse / tcq_mse)
	print(f"\n  EVAL: MSE = {tcq_mse:.6f}  ({reduction:+.1f}% vs LM, {db_gain:+.2f} dB)")

	# Export C codebook
	print(f"\n// Right-shift TCQ codebook: k=3, L=9 (512 states)")
	print(f"// MSE reduction: {reduction:.1f}% vs Lloyd-Max, {db_gain:.2f} dB")
	print(f"// Initial state = 0, decode: state_t = read_9_bits(qs, t*3)")
	print(f"static __constant__ float d_turbo3_tcq_codebook[512] = {{")
	for i in range(0, 512, 8):
		vals = ', '.join(f'{trellis.codebook[j]:+.7f}' for j in range(i, min(i+8, 512)))
		print(f"    {vals},")
	print("};")
