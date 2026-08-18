#!/usr/bin/env python3
"""Analyze TCQ error covariance structure.

Theory: coset-structured codebooks produce more diagonal Σ_e (error covariance),
which is robust to any Q structure. GLA-optimized codebooks produce non-diagonal
Σ_e whose off-diagonal structure interacts with model's Q covariance to hurt PPL.

Tests prediction #2: old codebook Σ_e is more diagonal than trained codebooks.
"""

import numpy as np
import sys
import time
import re
import struct

T = 128
K_BITS = 3
L = 9
N_STATES = 1 << L   # 512
N_OUT = 1 << K_BITS  # 8
SIGMA = 1.0 / np.sqrt(128.0)

LM_CENTROIDS = np.array([-1.748, -1.050, -0.5006, -0.06971,
	0.06971, 0.5006, 1.050, 1.748])

def init_coset():
	cb = np.zeros(N_STATES)
	spacing = (LM_CENTROIDS[1] - LM_CENTROIDS[0]) * SIGMA
	n_groups = 1 << (L - K_BITS)
	for g in range(n_groups):
		shift = spacing * (g / n_groups - 0.5)
		for p in range(N_OUT):
			cb[(g << K_BITS) | p] = LM_CENTROIDS[p] * SIGMA + shift
	return cb

def precompute_predecessors():
	mask = (1 << (L - K_BITS)) - 1
	preds = np.zeros((N_STATES, N_OUT), dtype=np.int32)
	for s in range(N_STATES):
		for p in range(N_OUT):
			preds[s, p] = ((s & mask) << K_BITS) | p
	return preds

def viterbi_batch(data, codebook, preds):
	"""Vectorized Viterbi for batch of samples."""
	n_samples = data.shape[0]
	bt = np.zeros((n_samples, T, N_STATES), dtype=np.int8)
	cost = np.zeros((n_samples, N_STATES))

	for t in range(T):
		x_t = data[:, t:t+1]
		dist = (x_t - codebook[None, :]) ** 2
		pred_costs = cost[:, preds]
		best_p = np.argmin(pred_costs, axis=2)
		best_pred_cost = np.take_along_axis(pred_costs, best_p[:, :, None], axis=2).squeeze(2)
		cost = best_pred_cost + dist
		bt[:, t, :] = best_p

	states = np.zeros((n_samples, T), dtype=np.int32)
	states[:, T-1] = np.argmin(cost, axis=1)
	mask_lower = (1 << (L - K_BITS)) - 1

	for t in range(T-2, -1, -1):
		ns = states[:, t+1]
		bp = bt[np.arange(n_samples), t+1, ns]
		states[:, t] = ((ns & mask_lower) << K_BITS) | bp

	return states

def parse_codebook_from_file(filepath):
	"""Extract codebook values from CUDA trainer output file."""
	with open(filepath) as f:
		text = f.read()
	m = re.search(r'static __constant__ float d_turbo3_tcq_codebook\[512\] = \{\n(.*?)\};', text, re.DOTALL)
	if not m:
		return None
	vals = []
	for line in m.group(1).strip().split('\n'):
		for v in line.strip().rstrip(',').split(','):
			v = v.strip().rstrip('f')
			if v:
				vals.append(float(v))
	return np.array(vals)

def analyze_cov(errors, name):
	"""Full covariance analysis of error vectors."""
	n = errors.shape[0]
	cov = np.cov(errors.T)  # 128x128
	eigs = np.sort(np.linalg.eigvalsh(cov))[::-1]

	tr = np.trace(cov)
	mse = tr / T
	diag = np.diag(np.diag(cov))
	off = cov - diag
	off_frob = np.linalg.norm(off, 'fro')
	diag_frob = np.linalg.norm(diag, 'fro')
	off_ratio = off_frob / diag_frob
	mean_d = np.mean(np.diag(cov))
	max_off = np.max(np.abs(off))

	# effective rank via entropy
	e_norm = eigs / np.sum(eigs)
	ent = -np.sum(e_norm * np.log(e_norm + 1e-30))
	eff_rank = np.exp(ent)

	print(f"\n{'='*60}")
	print(f"  {name}")
	print(f"{'='*60}")
	print(f"  MSE:                     {mse:.8f}")
	print(f"  Off-diag/diag Frobenius: {off_ratio:.6f}")
	print(f"  Max |off-diag| / mean(diag): {max_off/mean_d:.6f}")
	print(f"  Eigenvalue max/min:      {eigs[0]/eigs[-1]:.2f}")
	print(f"  Eigenvalue CV:           {np.std(eigs)/np.mean(eigs):.4f}")
	print(f"  Effective rank:          {eff_rank:.1f} / 128")
	print(f"  Top 5 eigenvalues:       {eigs[:5]}")
	print(f"  Bottom 5 eigenvalues:    {eigs[-5:]}")

	# lag autocorrelations from the covariance matrix
	print(f"  Autocorrelations:")
	for lag in [1, 2, 3, 5, 10, 20, 64]:
		if lag < T:
			pairs = [(i, i+lag) for i in range(T-lag)]
			ac = np.mean([cov[i,j] for i,j in pairs]) / mean_d
			print(f"    lag-{lag:2d}: {ac:+.6f}")

	# sum of absolute off-diagonal (total correlation energy)
	total_offdiag = np.sum(np.abs(off))
	total_diag = np.sum(np.abs(diag))
	print(f"  Total |off-diag| / total |diag|: {total_offdiag/total_diag:.6f}")

	# monotonicity of groups
	cb_name = name  # for labeling
	return cov, eigs

def count_monotonic_groups(cb):
	n_groups = 1 << (L - K_BITS)
	mono = 0
	for g in range(n_groups):
		vals = [cb[(g << K_BITS) | p] for p in range(N_OUT)]
		if all(vals[i] <= vals[i+1] for i in range(N_OUT-1)):
			mono += 1
	return mono

def main():
	# Load post-FWHT K data
	print("Loading post-FWHT K data from /tmp/turbo_postrot.bin...")
	n_vec = 10000
	raw = np.fromfile("/tmp/turbo_postrot.bin", dtype=np.float32, count=n_vec * T)
	data = raw.reshape(n_vec, T)
	print(f"  {n_vec} vectors, mean={np.mean(data):.6f}, std={np.std(data):.6f} (expect {SIGMA:.6f})")

	# Load codebooks
	print("\nLoading codebooks...")
	old_cb = np.fromfile("/tmp/old_codebook_3bit.bin", dtype=np.float32)
	assert len(old_cb) == N_STATES, f"old codebook has {len(old_cb)} entries, expected {N_STATES}"
	print(f"  Old numpy: {count_monotonic_groups(old_cb)}/64 monotonic groups")

	coset_cb = init_coset()
	print(f"  Coset init: {count_monotonic_groups(coset_cb)}/64 monotonic groups")

	# Parse trained codebooks from sweep outputs
	codebooks = [
		("Coset init (0 iter, PPL 5.9194)", coset_cb),
		("Old numpy (100 iter, PPL 5.8236)", old_cb),
	]

	for iters, ppl in [(3, "5.8450"), (10, "5.9386"), (30, "5.8733")]:
		path = f"/tmp/tcq_sweep_3bit_{iters}.txt"
		try:
			cb = parse_codebook_from_file(path)
			if cb is not None and len(cb) == N_STATES:
				mg = count_monotonic_groups(cb)
				codebooks.append((f"CUDA {iters}-iter (PPL {ppl}, {mg}/64 mono)", cb))
				print(f"  CUDA {iters}-iter: {mg}/64 monotonic groups")
		except FileNotFoundError:
			print(f"  {path} not found, skipping")

	# Precompute trellis
	preds = precompute_predecessors()
	batch_size = 500

	# Scalar quantization baseline (no trellis)
	print("\n--- Scalar quantization (no trellis) ---")
	lm_vals = LM_CENTROIDS * SIGMA
	scalar_q = np.zeros_like(data)
	for t in range(T):
		dists = (data[:, t:t+1] - lm_vals[None, :]) ** 2
		idx = np.argmin(dists, axis=1)
		scalar_q[:, t] = lm_vals[idx]
	scalar_err = data - scalar_q
	scalar_cov, scalar_eig = analyze_cov(scalar_err, "Scalar Lloyd-Max (no trellis)")

	# TCQ quantization for each codebook
	results = [("Scalar LM", np.mean(scalar_err**2), scalar_cov, scalar_eig)]

	for name, cb in codebooks:
		print(f"\n--- Quantizing with {name} ---")
		t0 = time.time()
		all_states = []
		for b in range(0, n_vec, batch_size):
			s = viterbi_batch(data[b:b+batch_size], cb, preds)
			all_states.append(s)
			if b % 2000 == 0:
				print(f"  {b}/{n_vec}...")
		states = np.concatenate(all_states)
		quantized = cb[states]
		errors = data - quantized
		elapsed = time.time() - t0
		print(f"  Done in {elapsed:.1f}s")

		cov, eigs = analyze_cov(errors, name)
		results.append((name, np.mean(errors**2), cov, eigs))

	# Comparison table
	print("\n" + "="*80)
	print("COMPARISON SUMMARY")
	print("="*80)
	print(f"{'Codebook':<45} {'MSE':>10} {'Off/Diag':>10} {'EffRank':>8} {'EigMax/Min':>10}")
	print("-" * 83)
	for name, mse, cov, eigs in results:
		diag = np.diag(np.diag(cov))
		off = cov - diag
		off_ratio = np.linalg.norm(off, 'fro') / np.linalg.norm(diag, 'fro')
		e_norm = eigs / np.sum(eigs)
		ent = -np.sum(e_norm * np.log(e_norm + 1e-30))
		eff_rank = np.exp(ent)
		eig_ratio = eigs[0] / eigs[-1] if eigs[-1] > 0 else float('inf')
		short = name[:44]
		print(f"{short:<45} {mse:>10.8f} {off_ratio:>10.6f} {eff_rank:>8.1f} {eig_ratio:>10.2f}")

	# Key test: if Q had identity covariance, error = MSE * ||q||^2
	# With non-identity Q: error = tr(Σ_q · Σ_e)
	# Diagonal Σ_e is worst-case optimal (maximizes effective rank)
	# Off-diagonal structure is a liability that can interact with Q structure
	print("\nKEY FINDING:")
	if len(results) >= 3:
		# Compare old vs a trained codebook
		old_off = None
		for name, mse, cov, eigs in results:
			diag = np.diag(np.diag(cov))
			off = cov - diag
			ratio = np.linalg.norm(off, 'fro') / np.linalg.norm(diag, 'fro')
			if "Old numpy" in name:
				old_off = ratio
		if old_off is not None:
			print(f"  Old codebook off-diagonal ratio: {old_off:.6f}")
			for name, mse, cov, eigs in results:
				if "Old numpy" not in name and "Scalar" not in name:
					diag = np.diag(np.diag(cov))
					off = cov - diag
					ratio = np.linalg.norm(off, 'fro') / np.linalg.norm(diag, 'fro')
					print(f"  {name[:40]}: {ratio:.6f} ({ratio/old_off:.2f}x)")

if __name__ == "__main__":
	main()
