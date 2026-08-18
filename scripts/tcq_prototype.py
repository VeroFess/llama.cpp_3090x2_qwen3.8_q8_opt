#!/usr/bin/env python3
"""
TCQ (Trellis-Coded Quantization) prototype for TurboQuant KV cache.
Experiment #61: Compare bitshift-trellis TCQ vs Lloyd-Max scalar quantization
on i.i.d. Gaussian data (post-FWHT KV cache distribution).

Implements iterative codebook training (Generalized Lloyd Algorithm for TCQ):
1. Initialize codebook with shifted Lloyd-Max cosets
2. Encode training data with Viterbi (fix codebook, optimize path)
3. Update codebook centroids (fix paths, optimize reconstruction values)
4. Repeat until convergence
"""

import numpy as np
from scipy.stats import norm as ndist
import sys

# Lloyd-Max centroids for N(0,1)
LLOYD_MAX = {
	2: np.array([-1.510, -0.4528, 0.4528, 1.510]),
	3: np.array([-1.748, -1.050, -0.5006, -0.06971, 0.06971, 0.5006, 1.050, 1.748]),
	4: np.array([
		-2.733, -2.070, -1.618, -1.256, -0.9424, -0.6568, -0.3881, -0.1284,
		 0.1284,  0.3881,  0.6568,  0.9424,  1.256,  1.618,  2.070,  2.733
	]),
}

def lloyd_max_quantize(x, k):
	centroids = LLOYD_MAX[k]
	bounds = (centroids[:-1] + centroids[1:]) / 2
	indices = np.searchsorted(bounds, x)
	recon = centroids[indices]
	mse = np.mean((x - recon) ** 2)
	return indices, recon, mse


class BitshiftTrellis:
	def __init__(self, k, L, sigma=1.0):
		self.k = k
		self.L = L
		self.n_states = 1 << L
		self.n_outputs = 1 << k
		self.state_mask = self.n_states - 1
		self.sigma = sigma
		self.codebook = self._init_coset_codebook(sigma)

	def _init_coset_codebook(self, sigma):
		"""Initialize with shifted Lloyd-Max cosets."""
		n_groups = 1 << (self.L - self.k)
		centroids = LLOYD_MAX[self.k] * sigma
		spacing = centroids[1] - centroids[0]
		shifts = np.linspace(-spacing/2, spacing/2, n_groups, endpoint=False)
		codebook = np.zeros(self.n_states)
		for group in range(n_groups):
			for pos in range(self.n_outputs):
				state = (group << self.k) | pos
				codebook[state] = centroids[pos] + shifts[group]
		return codebook

	def next_state(self, state, output):
		return ((state << self.k) | output) & self.state_mask

	def encode_viterbi_batch(self, X):
		"""
		Vectorized Viterbi encoding for a batch of sequences.
		X: shape (batch, T)
		Returns: recon (batch, T), total_mse
		"""
		batch, T = X.shape
		all_recon = np.zeros_like(X)
		all_states_used = []  # for codebook training

		for b in range(batch):
			recon, states_visited = self._encode_single(X[b])
			all_recon[b] = recon
			all_states_used.append(states_visited)

		mse = np.mean((X - all_recon) ** 2)
		return all_recon, all_states_used, mse

	def _encode_single(self, x):
		"""Viterbi encode single sequence. Returns (recon, states_visited)."""
		T = len(x)
		n_states = self.n_states
		n_out = self.n_outputs
		cb = self.codebook

		INF = 1e30
		cost = np.zeros(n_states)  # open trellis: start anywhere

		bt_out = np.zeros((T, n_states), dtype=np.int32)
		bt_prev = np.zeros((T, n_states), dtype=np.int32)

		for t in range(T):
			new_cost = np.full(n_states, INF)

			for s in range(n_states):
				base_cost = cost[s]
				for out in range(n_out):
					ns = self.next_state(s, out)
					d = (x[t] - cb[ns]) ** 2
					total = base_cost + d
					if total < new_cost[ns]:
						new_cost[ns] = total
						bt_out[t][ns] = out
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

	def train_codebook(self, n_train=5000, n_elements=128, n_iters=20, verbose=True):
		"""
		Generalized Lloyd Algorithm for TCQ codebook training.

		1. Generate training data (i.i.d. Gaussian)
		2. Encode with Viterbi (fix codebook)
		3. Update codebook: each entry = mean of values assigned to that state
		4. Repeat
		"""
		if verbose:
			print(f"  Training TCQ codebook: k={self.k}, L={self.L}, "
				  f"{self.n_states} states, {n_train} training vectors...")

		best_mse = float('inf')
		best_codebook = self.codebook.copy()

		for iteration in range(n_iters):
			# Generate fresh training data each iteration
			X = np.random.randn(n_train, n_elements) * self.sigma

			# Encode with current codebook
			_, all_states, mse = self.encode_viterbi_batch(X)

			if verbose:
				print(f"    iter {iteration+1:2d}: MSE = {mse:.6f}", end='')

			if mse < best_mse:
				best_mse = mse
				best_codebook = self.codebook.copy()

			# Accumulate: for each state, collect all x values mapped to it
			state_sums = np.zeros(self.n_states)
			state_counts = np.zeros(self.n_states, dtype=np.int64)

			for b in range(n_train):
				states = all_states[b]
				for t in range(n_elements):
					s = states[t]
					state_sums[s] += X[b, t]
					state_counts[s] += 1

			# Update codebook: centroid of assigned values
			updated = 0
			for s in range(self.n_states):
				if state_counts[s] > 0:
					self.codebook[s] = state_sums[s] / state_counts[s]
					updated += 1

			if verbose:
				print(f"  ({updated}/{self.n_states} states used)")

			# Convergence check
			if iteration > 2 and abs(mse - best_mse) / best_mse < 1e-4:
				if verbose:
					print(f"    Converged at iteration {iteration+1}")
				break

		self.codebook = best_codebook
		return best_mse


def run_experiment(k, L_values, n_elements=128, n_eval=500, sigma=1.0):
	print(f"\n{'='*60}")
	print(f"Rate: {k} bits/element, {n_elements} elements/block")
	print(f"{'='*60}")

	# Lloyd-Max baseline
	lm_mses = []
	for _ in range(n_eval):
		x = np.random.randn(n_elements) * sigma
		_, _, mse = lloyd_max_quantize(x, k)
		lm_mses.append(mse)
	lm_mse = np.mean(lm_mses)
	print(f"\nLloyd-Max {k}-bit:  MSE = {lm_mse:.6f}")

	for L in L_values:
		print(f"\n  --- L={L} ({1<<L} states) ---")

		# Train codebook
		trellis = BitshiftTrellis(k, L, sigma)
		n_train = 500 if (1 << L) <= 64 else 200
		train_mse = trellis.train_codebook(
			n_train=n_train, n_elements=n_elements,
			n_iters=15, verbose=True)

		# Evaluate on fresh data
		eval_mses = []
		n_eval_tcq = min(n_eval, 200)
		for i in range(n_eval_tcq):
			x = np.random.randn(n_elements) * sigma
			recon, _ = trellis._encode_single(x)
			mse = np.mean((x - recon) ** 2)
			eval_mses.append(mse)

		tcq_mse = np.mean(eval_mses)
		reduction = (1 - tcq_mse / lm_mse) * 100
		db_gain = 10 * np.log10(lm_mse / tcq_mse) if tcq_mse > 0 else float('inf')
		print(f"  EVAL: MSE = {tcq_mse:.6f}  ({reduction:+.1f}% vs LM, {db_gain:+.2f} dB)")

	dr_bound = sigma**2 * 2**(-2*k)
	print(f"\n  D(R) bound: {dr_bound:.6f} ({10*np.log10(lm_mse/dr_bound):+.2f} dB from LM)")


if __name__ == '__main__':
	sigma = 1.0
	quick = '--quick' in sys.argv
	n_eval = 200 if quick else 500

	print("TCQ Prototype with Codebook Training (Experiment #61)")
	print(f"{'quick' if quick else 'full'} mode, {n_eval} eval trials\n")

	# 2-bit: where TCQ matters most
	run_experiment(k=2, L_values=[4, 6, 8],
				   n_elements=128, n_eval=n_eval, sigma=sigma)

	# 3-bit: user's preferred rate
	run_experiment(k=3, L_values=[6, 9],
				   n_elements=128, n_eval=n_eval, sigma=sigma)
