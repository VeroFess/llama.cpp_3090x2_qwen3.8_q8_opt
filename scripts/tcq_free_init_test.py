#!/usr/bin/env python3
"""Quick test: free initial state vs state=0 on right-shift trellis."""

import numpy as np
import sys

LLOYD_MAX_3BIT = np.array([-1.748, -1.050, -0.5006, -0.06971, 0.06971, 0.5006, 1.050, 1.748])
LLOYD_MAX_2BIT = np.array([-1.510, -0.4528, 0.4528, 1.510])

class RightShiftTrellis:
	def __init__(self, k, L, sigma=1.0):
		self.k = k
		self.L = L
		self.n_states = 1 << L
		self.n_outputs = 1 << k
		self.state_mask = self.n_states - 1
		self.sigma = sigma
		if k == 3:
			self.base_centroids = LLOYD_MAX_3BIT * sigma
		elif k == 2:
			self.base_centroids = LLOYD_MAX_2BIT * sigma
		else:
			raise ValueError(f"unsupported k={k}")
		self.codebook = self._init_coset_codebook(sigma)

	def _init_coset_codebook(self, sigma):
		n_groups = 1 << (self.L - self.k)
		centroids = self.base_centroids
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

	def _encode_single(self, x, free_init=False):
		T = len(x)
		n_states = self.n_states
		n_out = self.n_outputs
		cb = self.codebook
		INF = 1e30

		cost = np.zeros(n_states) if free_init else np.full(n_states, INF)
		if not free_init:
			cost[0] = 0.0

		bt_prev = np.zeros((T, n_states), dtype=np.int32)

		for t in range(T):
			new_cost = np.full(n_states, INF)
			for s in range(n_states):
				if cost[s] >= INF:
					continue
				for out in range(n_out):
					ns = self.next_state(s, out)
					d = (x[t] - cb[ns]) ** 2
					total = cost[s] + d
					if total < new_cost[ns]:
						new_cost[ns] = total
						bt_prev[t][ns] = s
			cost = new_cost

		state = np.argmin(cost)
		recon = np.zeros(T)
		states = np.zeros(T, dtype=np.int32)
		for t in range(T - 1, -1, -1):
			states[t] = state
			recon[t] = cb[state]
			state = bt_prev[t][state]

		return recon, states, state  # return initial state too

	def train_codebook(self, n_train=500, n_elements=128, n_iters=20, free_init=False, verbose=True):
		tag = "free-init" if free_init else "state=0"
		if verbose:
			print(f"  Training ({tag}): k={self.k}, L={self.L}, {self.n_states} states")
		best_mse = float('inf')
		best_codebook = self.codebook.copy()

		for iteration in range(n_iters):
			X = np.random.randn(n_train, n_elements) * self.sigma
			mses = []
			all_states = []
			for b in range(n_train):
				recon, states, _ = self._encode_single(X[b], free_init=free_init)
				mses.append(np.mean((X[b] - recon) ** 2))
				all_states.append(states)
			mse = np.mean(mses)
			if verbose:
				print(f"    iter {iteration+1:2d}: MSE = {mse:.6f}", end='')
			if mse < best_mse:
				best_mse = mse
				best_codebook = self.codebook.copy()

			state_sums = np.zeros(self.n_states)
			state_counts = np.zeros(self.n_states, dtype=np.int64)
			for b in range(n_train):
				for t in range(n_elements):
					s = all_states[b][t]
					state_sums[s] += X[b, t]
					state_counts[s] += 1

			for s in range(self.n_states):
				if state_counts[s] > 0:
					self.codebook[s] = state_sums[s] / state_counts[s]
			used = np.sum(state_counts > 0)
			if verbose:
				print(f"  ({used}/{self.n_states} states used)")
			if iteration > 2 and abs(mse - best_mse) / best_mse < 1e-4:
				if verbose:
					print(f"    Converged")
				break

		self.codebook = best_codebook
		return best_mse

	def verify_decode_free_init(self, x, states, initial_state):
		"""Verify sliding-window decode with arbitrary initial state."""
		T = len(x)
		outputs = np.array([states[t] >> (self.L - self.k) for t in range(T)], dtype=np.int32)

		# Pack bitstream: prefix = initial_state >> k, then outputs
		prefix_bits = self.L - self.k  # 6 for k=3 L=9
		n_bits = prefix_bits + T * self.k
		bitstream = np.zeros((n_bits + 7) // 8, dtype=np.uint8)

		# Write prefix bits (initial_state >> k)
		prefix_val = initial_state >> self.k
		for b in range(prefix_bits):
			if prefix_val & (1 << b):
				bitstream[b // 8] |= (1 << (b % 8))

		# Write outputs
		for t in range(T):
			out = outputs[t]
			bit_pos = prefix_bits + t * self.k
			for b in range(self.k):
				byte_idx = (bit_pos + b) // 8
				bit_off = (bit_pos + b) % 8
				if out & (1 << b):
					bitstream[byte_idx] |= (1 << bit_off)

		# Decode via sliding window
		state_bits = self.L
		for t in range(T):
			bit_pos = t * self.k
			byte_idx = bit_pos // 8
			bit_off = bit_pos % 8
			raw = int(bitstream[byte_idx]) | (int(bitstream[byte_idx + 1]) << 8)
			state_decoded = (raw >> bit_off) & ((1 << state_bits) - 1)
			if state_decoded != states[t]:
				return False, f"t={t}: decoded {state_decoded} != expected {states[t]}"
		return True, "OK"


if __name__ == '__main__':
	np.random.seed(42)
	sigma = 1.0
	quick = '--quick' in sys.argv
	n_train = 200 if quick else 500
	n_eval = 100 if quick else 300

	print("=" * 60)
	print("TCQ Free Initial State Experiment")
	print("=" * 60)

	configs = [
		(3, 9, "3-bit k=3 L=9 (512 states)"),
		(2, 8, "2-bit k=2 L=8 (256 states)"),
		(2, 6, "2-bit k=2 L=6 (64 states)"),
	]

	for k, L, label in configs:
		print(f"\n{'='*60}")
		print(f"  {label}")
		print(f"{'='*60}")

		# Lloyd-Max baseline
		if k == 3:
			centroids = LLOYD_MAX_3BIT * sigma
		else:
			centroids = LLOYD_MAX_2BIT * sigma
		bounds = (centroids[:-1] + centroids[1:]) / 2
		lm_mses = []
		for _ in range(n_eval):
			x = np.random.randn(128) * sigma
			indices = np.searchsorted(bounds, x)
			recon = centroids[indices]
			lm_mses.append(np.mean((x - recon) ** 2))
		lm_mse = np.mean(lm_mses)
		print(f"\n  Lloyd-Max baseline: MSE = {lm_mse:.6f}")

		# Train + eval with state=0
		t0 = RightShiftTrellis(k=k, L=L, sigma=sigma)
		t0.train_codebook(n_train=n_train, n_elements=128, n_iters=15, free_init=False)
		mses_0 = []
		for _ in range(n_eval):
			x = np.random.randn(128) * sigma
			recon, _, _ = t0._encode_single(x, free_init=False)
			mses_0.append(np.mean((x - recon) ** 2))
		mse_0 = np.mean(mses_0)
		red_0 = (1 - mse_0 / lm_mse) * 100
		db_0 = 10 * np.log10(lm_mse / mse_0)

		# Train + eval with free init
		t1 = RightShiftTrellis(k=k, L=L, sigma=sigma)
		t1.train_codebook(n_train=n_train, n_elements=128, n_iters=15, free_init=True)
		mses_f = []
		for _ in range(n_eval):
			x = np.random.randn(128) * sigma
			recon, _, _ = t1._encode_single(x, free_init=True)
			mses_f.append(np.mean((x - recon) ** 2))
		mse_f = np.mean(mses_f)
		red_f = (1 - mse_f / lm_mse) * 100
		db_f = 10 * np.log10(lm_mse / mse_f)

		# Verify decode for free-init
		x_test = np.random.randn(128) * sigma
		recon_test, states_test, init_state = t1._encode_single(x_test, free_init=True)
		ok, msg = t1.verify_decode_free_init(x_test, states_test, init_state)

		print(f"\n  Results:")
		print(f"    state=0:    MSE = {mse_0:.6f}  ({red_0:+.1f}% vs LM, {db_0:+.2f} dB)")
		print(f"    free-init:  MSE = {mse_f:.6f}  ({red_f:+.1f}% vs LM, {db_f:+.2f} dB)")
		print(f"    free-init improvement over state=0: {(1 - mse_f/mse_0)*100:.1f}%")
		print(f"    Decode verify: {msg}")
		print(f"    Initial state used in verify: {init_state}")
