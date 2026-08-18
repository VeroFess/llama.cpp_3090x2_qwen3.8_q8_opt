#!/usr/bin/env python3
"""Train TCQ codebook with free initial state and export for CUDA."""

import numpy as np
import sys

LLOYD_MAX_3BIT = np.array([-1.748, -1.050, -0.5006, -0.06971, 0.06971, 0.5006, 1.050, 1.748])

class RightShiftTrellis:
	def __init__(self, k, L, sigma=1.0):
		self.k = k
		self.L = L
		self.n_states = 1 << L
		self.n_outputs = 1 << k
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

	def _encode_single(self, x):
		T = len(x)
		n_states = self.n_states
		n_out = self.n_outputs
		cb = self.codebook
		INF = 1e30

		# Free initial state: all states start at cost 0
		cost = np.zeros(n_states)
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
		return recon, states

	def train_codebook(self, n_train=1000, n_elements=128, n_iters=30):
		print(f"Training free-init TCQ: k={self.k}, L={self.L}, {self.n_states} states, {n_train} seqs, {n_iters} iters")
		best_mse = float('inf')
		best_codebook = self.codebook.copy()

		for iteration in range(n_iters):
			X = np.random.randn(n_train, n_elements) * self.sigma
			mses = []
			all_states = []
			for b in range(n_train):
				recon, states = self._encode_single(X[b])
				mses.append(np.mean((X[b] - recon) ** 2))
				all_states.append(states)
			mse = np.mean(mses)
			print(f"  iter {iteration+1:2d}: MSE = {mse:.6f}", end='')
			if mse < best_mse:
				best_mse = mse
				best_codebook = self.codebook.copy()
				print(" *", end='')

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
			print(f"  ({used}/{self.n_states} used)")
			if iteration > 4 and abs(mse - best_mse) / best_mse < 5e-5:
				print(f"  Converged at iteration {iteration+1}")
				break

		self.codebook = best_codebook
		return best_mse


if __name__ == '__main__':
	np.random.seed(42)
	sigma = 1.0
	scale = 1.0 / np.sqrt(128)  # 0.0883883f for post-FWHT data

	# Lloyd-Max baseline
	centroids = LLOYD_MAX_3BIT * sigma
	bounds = (centroids[:-1] + centroids[1:]) / 2
	lm_mses = []
	for _ in range(500):
		x = np.random.randn(128) * sigma
		indices = np.searchsorted(bounds, x)
		recon = centroids[indices]
		lm_mses.append(np.mean((x - recon) ** 2))
	lm_mse = np.mean(lm_mses)
	print(f"Lloyd-Max 3-bit baseline: MSE = {lm_mse:.6f}\n")

	# Train
	trellis = RightShiftTrellis(k=3, L=9, sigma=sigma)
	best_mse = trellis.train_codebook(n_train=500, n_elements=128, n_iters=15)

	# Evaluate
	eval_mses = []
	for _ in range(500):
		x = np.random.randn(128) * sigma
		recon, _ = trellis._encode_single(x)
		eval_mses.append(np.mean((x - recon) ** 2))
	tcq_mse = np.mean(eval_mses)
	reduction = (1 - tcq_mse / lm_mse) * 100
	db_gain = 10 * np.log10(lm_mse / tcq_mse)
	print(f"\nEVAL: MSE = {tcq_mse:.6f} ({reduction:+.1f}% vs LM, {db_gain:+.2f} dB)\n")

	# Export C codebook (scaled by 1/sqrt(128) for post-FWHT data)
	scaled = trellis.codebook * scale
	print(f"// GLA-trained free-init TCQ codebook: k=3, L=9 (512 states)")
	print(f"// MSE reduction: {reduction:.1f}% vs Lloyd-Max, {db_gain:.2f} dB")
	print(f"// Free initial state, decode: state_t = read_9_bits(qs, t*3)")
	print(f"// Scaled by 1/sqrt(128) = {scale:.10f}")
	print(f"static __constant__ float d_turbo3_tcq_codebook[512] = {{")
	for i in range(0, 512, 8):
		vals = ', '.join(f'{scaled[j]:+.8f}' for j in range(i, min(i+8, 512)))
		comma = ',' if i + 8 < 512 else ''
		print(f"    {vals}{comma}")
	print("};")
