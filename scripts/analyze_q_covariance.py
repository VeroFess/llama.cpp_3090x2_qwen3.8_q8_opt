#!/usr/bin/env python3
"""Combined Q covariance + error covariance analysis.

Tests:
  #1: Σ_q (rotated Q covariance) is NOT proportional to identity
  #2: Old codebook Σ_e is more diagonal (already confirmed by prior script)
  #3: tr(Σ_q · Σ_e) correlates with PPL better than tr(Σ_e) (MSE) alone
"""

import numpy as np
import sys
import time
import re

T = 128
K_BITS = 3
L = 9
N_STATES = 512
N_OUT = 8
SIGMA = 1.0 / np.sqrt(128.0)

LM_CENTROIDS = np.array([-1.748, -1.050, -0.5006, -0.06971,
	0.06971, 0.5006, 1.050, 1.748])

SIGNS1 = np.array([-1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1], dtype=np.float64)

SIGNS2 = np.array([1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1], dtype=np.float64)

def fwht_forward(x):
	"""Forward FWHT with signs: signs1 -> butterfly -> signs2 -> normalize."""
	n = len(x)
	assert n == 128
	out = x.copy() * SIGNS1
	h = 1
	while h < n:
		for i in range(0, n, h * 2):
			for j in range(i, i + h):
				a = out[j]
				b = out[j + h]
				out[j] = a + b
				out[j + h] = a - b
		h *= 2
	out *= SIGNS2 / np.sqrt(n)
	return out

def fwht_forward_batch(data):
	"""Apply FWHT to each row of data [n_samples, 128]."""
	n = data.shape[0]
	out = data.copy() * SIGNS1[None, :]
	h = 1
	while h < 128:
		for i in range(0, 128, h * 2):
			a = out[:, i:i+h].copy()
			b = out[:, i+h:i+2*h].copy()
			out[:, i:i+h] = a + b
			out[:, i+h:i+2*h] = a - b
		h *= 2
	out *= SIGNS2[None, :] / np.sqrt(128)
	return out

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

def main():
	# ========================================
	# PART 1: Load and analyze Q vectors
	# ========================================
	print("="*70)
	print("PART 1: Q COVARIANCE ANALYSIS (Prediction #1)")
	print("="*70)

	print("\nLoading Q vectors from /tmp/q_vectors_raw.bin...")
	with open("/tmp/q_vectors_raw.bin", "rb") as f:
		hdr = np.frombuffer(f.read(16), dtype=np.int32)
		head_dim, n_tokens, n_heads_total, n_layers = hdr
		print(f"  Header: head_dim={head_dim}, n_tokens={n_tokens}, n_heads={n_heads_total}, n_layers={n_layers}")

		# We dumped 4 heads per layer (heads 0,6,12,18)
		n_heads_dumped = 4
		floats_per_head = head_dim * n_tokens
		q_data = {}

		for layer in range(n_layers):
			for hi, head in enumerate([0, 6, 12, 18]):
				raw = np.frombuffer(f.read(floats_per_head * 4), dtype=np.float32)
				q_data[(layer, head)] = raw.reshape(n_tokens, head_dim)

	print(f"  Loaded {len(q_data)} (layer, head) pairs")
	print(f"  Shape per pair: ({n_tokens}, {head_dim})")

	# Analyze raw Q covariance (before FWHT)
	# Split head_dim=256 into two 128-element blocks
	print("\n--- Raw Q covariance (before FWHT) ---")
	all_q_blocks_raw = []
	for layer in range(n_layers):
		for head in [0, 6, 12, 18]:
			q = q_data[(layer, head)]  # [512, 256]
			# Split into 128-element blocks
			block0 = q[:, :128]     # [512, 128]
			block1 = q[:, 128:256]  # [512, 128]
			all_q_blocks_raw.append(block0)
			all_q_blocks_raw.append(block1)

	all_q_raw = np.concatenate(all_q_blocks_raw, axis=0)  # [N, 128]
	print(f"  Total Q samples (raw): {all_q_raw.shape[0]}")
	print(f"  Mean: {np.mean(all_q_raw):.6f}, Std: {np.std(all_q_raw):.6f}")

	cov_q_raw = np.cov(all_q_raw.T)
	eigs_q_raw = np.sort(np.linalg.eigvalsh(cov_q_raw))[::-1]
	print(f"  Eigenvalue max/min: {eigs_q_raw[0]/eigs_q_raw[-1]:.2f}")
	print(f"  Eigenvalue CV: {np.std(eigs_q_raw)/np.mean(eigs_q_raw):.4f}")
	e_norm = eigs_q_raw / np.sum(eigs_q_raw)
	eff_rank = np.exp(-np.sum(e_norm * np.log(e_norm + 1e-30)))
	print(f"  Effective rank: {eff_rank:.1f} / 128")
	print(f"  Top 5 eigenvalues: {eigs_q_raw[:5]}")
	print(f"  Bottom 5 eigenvalues: {eigs_q_raw[-5:]}")

	# Analyze FWHT-rotated Q covariance
	print("\n--- Rotated Q covariance (after FWHT) = Σ_q ---")
	all_q_blocks_rot = []
	for layer in range(n_layers):
		for head in [0, 6, 12, 18]:
			q = q_data[(layer, head)]  # [512, 256]
			block0 = q[:, :128]
			block1 = q[:, 128:256]
			rot0 = fwht_forward_batch(block0)
			rot1 = fwht_forward_batch(block1)
			all_q_blocks_rot.append(rot0)
			all_q_blocks_rot.append(rot1)

	all_q_rot = np.concatenate(all_q_blocks_rot, axis=0)  # [N, 128]
	print(f"  Total Q samples (rotated): {all_q_rot.shape[0]}")
	print(f"  Mean: {np.mean(all_q_rot):.6f}, Std: {np.std(all_q_rot):.6f}")

	cov_q_rot = np.cov(all_q_rot.T)
	eigs_q_rot = np.sort(np.linalg.eigvalsh(cov_q_rot))[::-1]
	print(f"  Eigenvalue max/min: {eigs_q_rot[0]/eigs_q_rot[-1]:.2f}")
	print(f"  Eigenvalue CV: {np.std(eigs_q_rot)/np.mean(eigs_q_rot):.4f}")
	e_norm = eigs_q_rot / np.sum(eigs_q_rot)
	eff_rank_rot = np.exp(-np.sum(e_norm * np.log(e_norm + 1e-30)))
	print(f"  Effective rank: {eff_rank_rot:.1f} / 128")
	print(f"  Top 5 eigenvalues: {eigs_q_rot[:5]}")
	print(f"  Bottom 5 eigenvalues: {eigs_q_rot[-5:]}")

	# Per-layer analysis
	print("\n--- Per-layer Σ_q anisotropy ---")
	print(f"{'Layer':>5} {'Head':>4} {'EigMax/Min':>10} {'EffRank':>8} {'CV':>8}")
	print("-" * 40)
	layer_cov_q = {}
	for layer in [0, 10, 19, 20, 30, 39]:
		for head in [0, 6, 12, 18]:
			if (layer, head) not in q_data:
				continue
			q = q_data[(layer, head)]
			blocks = np.concatenate([
				fwht_forward_batch(q[:, :128]),
				fwht_forward_batch(q[:, 128:256])
			], axis=0)
			cov = np.cov(blocks.T)
			eigs = np.sort(np.linalg.eigvalsh(cov))[::-1]
			cv = np.std(eigs) / np.mean(eigs)
			e_norm = eigs / np.sum(eigs)
			er = np.exp(-np.sum(e_norm * np.log(e_norm + 1e-30)))
			ratio = eigs[0] / eigs[-1] if eigs[-1] > 0 else float('inf')
			print(f"{layer:>5} {head:>4} {ratio:>10.2f} {er:>8.1f} {cv:>8.4f}")
			layer_cov_q[(layer, head)] = cov

	# ========================================
	# PART 2: Error covariance + combined metric
	# ========================================
	print("\n" + "="*70)
	print("PART 2: COMBINED METRIC tr(Σ_q · Σ_e) (Prediction #3)")
	print("="*70)

	# Load K data
	print("\nLoading post-FWHT K data...")
	n_vec = 10000
	k_data = np.fromfile("/tmp/turbo_postrot.bin", dtype=np.float32, count=n_vec*T).reshape(n_vec, T)

	# Load codebooks
	old_cb = np.fromfile("/tmp/old_codebook_3bit.bin", dtype=np.float32)
	coset_cb = init_coset()

	codebooks = [
		("Coset 0-iter", coset_cb, 5.9194),
		("Old numpy 100-iter", old_cb, 5.8236),
	]
	for iters, ppl in [(3, 5.8450), (10, 5.9386), (30, 5.8733)]:
		path = f"/tmp/tcq_sweep_3bit_{iters}.txt"
		try:
			cb = parse_codebook_from_file(path)
			if cb is not None and len(cb) == N_STATES:
				codebooks.append((f"CUDA {iters}-iter", cb, ppl))
		except FileNotFoundError:
			pass

	preds = precompute_predecessors()

	# Use the aggregate rotated Q covariance
	Sigma_q = cov_q_rot

	print(f"\n{'Codebook':<25} {'MSE':>10} {'tr(Σ_e)':>10} {'tr(Σ_q·Σ_e)':>12} {'PPL':>8} {'Ratio':>8}")
	print("-" * 78)

	results = []
	for name, cb, ppl in codebooks:
		print(f"  Quantizing with {name}...", end=" ", flush=True)
		t0 = time.time()
		all_states = []
		for b in range(0, n_vec, 500):
			s = viterbi_batch(k_data[b:b+500], cb, preds)
			all_states.append(s)
		states = np.concatenate(all_states)
		quantized = cb[states]
		errors = k_data - quantized
		elapsed = time.time() - t0

		cov_e = np.cov(errors.T)
		mse = np.trace(cov_e) / T
		tr_e = np.trace(cov_e)
		tr_qe = np.trace(Sigma_q @ cov_e)

		# Ratio: how much worse is the weighted metric relative to MSE?
		# If Σ_q = I, ratio = 1 always. Higher ratio means more harm from Q structure.
		ratio = (tr_qe / tr_e) / (np.trace(Sigma_q) / 128)

		results.append((name, mse, tr_e, tr_qe, ppl, ratio, cov_e))
		print(f"{elapsed:.0f}s")
		print(f"{name:<25} {mse:>10.8f} {tr_e:>10.6f} {tr_qe:>12.6f} {ppl:>8.4f} {ratio:>8.4f}")

	# Correlation analysis
	print("\n--- Correlation with PPL ---")
	ppls = np.array([r[4] for r in results])
	mses = np.array([r[1] for r in results])
	tr_qes = np.array([r[3] for r in results])
	ratios = np.array([r[5] for r in results])

	if len(results) >= 3:
		corr_mse = np.corrcoef(mses, ppls)[0, 1]
		corr_trqe = np.corrcoef(tr_qes, ppls)[0, 1]
		corr_ratio = np.corrcoef(ratios, ppls)[0, 1]
		print(f"  Pearson correlation(MSE, PPL):        {corr_mse:+.4f}")
		print(f"  Pearson correlation(tr(Σ_q·Σ_e), PPL): {corr_trqe:+.4f}")
		print(f"  Pearson correlation(ratio, PPL):      {corr_ratio:+.4f}")

		# Rank correlation (more robust)
		from scipy.stats import spearmanr
		r_mse, _ = spearmanr(mses, ppls)
		r_trqe, _ = spearmanr(tr_qes, ppls)
		r_ratio, _ = spearmanr(ratios, ppls)
		print(f"  Spearman correlation(MSE, PPL):        {r_mse:+.4f}")
		print(f"  Spearman correlation(tr(Σ_q·Σ_e), PPL): {r_trqe:+.4f}")
		print(f"  Spearman correlation(ratio, PPL):      {r_ratio:+.4f}")

	# Per-layer tr(Σ_q · Σ_e) for a few representative layers
	print("\n--- Per-layer sensitivity (layers with per-layer Σ_q) ---")
	for name, cb, ppl in codebooks:
		cov_e = [r[6] for r in results if r[0] == name][0]
		per_layer_vals = []
		for layer in [0, 10, 19, 20, 30, 39]:
			head = 0
			if (layer, head) in layer_cov_q:
				tr_val = np.trace(layer_cov_q[(layer, head)] @ cov_e)
				per_layer_vals.append((layer, tr_val))
		if per_layer_vals:
			vals_str = ", ".join([f"L{l}:{v:.6f}" for l, v in per_layer_vals])
			print(f"  {name}: {vals_str}")

	# ========================================
	# PART 3: Eigenvector alignment analysis
	# ========================================
	print("\n" + "="*70)
	print("PART 3: EIGENVECTOR ALIGNMENT ANALYSIS")
	print("="*70)

	# Get Σ_q eigenvectors (directions Q cares most about)
	eigs_q, vecs_q = np.linalg.eigh(Sigma_q)
	# Sort descending
	idx = np.argsort(eigs_q)[::-1]
	eigs_q = eigs_q[idx]
	vecs_q = vecs_q[:, idx]

	print(f"\nΣ_q top-5 eigenvalues: {eigs_q[:5]}")
	print(f"Σ_q bottom-5 eigenvalues: {eigs_q[-5:]}")
	print(f"\nFor each codebook: error variance along Q's top/bottom eigenvectors")

	for name, cb, ppl in codebooks:
		cov_e = [r[6] for r in results if r[0] == name][0]
		# Project error covariance onto Q eigenvectors
		proj = np.diag(vecs_q.T @ cov_e @ vecs_q)
		# Error variance along Q's top eigenvectors (most sensitive directions)
		top5_err = np.mean(proj[:5])
		bot5_err = np.mean(proj[-5:])
		mid_err = np.mean(proj[60:68])
		total_err = np.mean(proj)
		print(f"  {name:<25}: top5={top5_err:.8f} mid={mid_err:.8f} bot5={bot5_err:.8f} avg={total_err:.8f}  (top/avg={top5_err/total_err:.3f})")

	print("\nDone.")

if __name__ == "__main__":
	main()
