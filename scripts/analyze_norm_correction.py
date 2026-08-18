#!/usr/bin/env python3
"""Analyze TCQ norm correction differences between codebooks.

The SET_ROWS kernel:
1. Normalizes input to unit norm: x /= ||x||, stores saved_norm = ||x||
2. Applies FWHT rotation
3. Viterbi encodes with codebook → reconstruction r
4. Computes recon_norm = ||r||
5. Stores corrected_norm = saved_norm / recon_norm as fp16

If recon_norm differs systematically between codebooks, the norm correction
introduces different magnitude errors. +0.002 PPL/layer requires only ~0.1%
systematic norm bias.
"""
import numpy as np
import struct, sys

K_BITS = 3
N_STATES = 512
BLOCK_SIZE = 128
L_BITS = 9  # log2(512) = 9 state bits
MASK = (1 << (L_BITS - K_BITS)) - 1  # 0x3F = 63

# FWHT sign arrays (must match CUDA)
SIGNS1 = np.array([
	+1,+1,+1,-1,-1,-1,-1,-1,+1,+1,-1,+1,-1,-1,+1,+1,
	+1,-1,+1,+1,-1,-1,-1,+1,-1,-1,-1,-1,+1,-1,+1,-1,
	+1,+1,+1,-1,-1,+1,-1,+1,+1,-1,+1,-1,-1,+1,+1,-1,
	+1,-1,-1,-1,-1,+1,-1,+1,+1,-1,+1,-1,-1,-1,-1,+1,
	+1,+1,-1,-1,-1,+1,+1,-1,-1,-1,+1,+1,+1,+1,+1,+1,
	-1,-1,-1,-1,+1,-1,-1,+1,+1,+1,-1,+1,+1,+1,-1,+1,
	+1,+1,-1,-1,+1,+1,+1,+1,-1,-1,-1,-1,-1,+1,+1,+1,
	+1,-1,+1,-1,+1,-1,-1,+1,+1,-1,-1,+1,+1,-1,+1,-1,
], dtype=np.float32)

SIGNS2 = np.array([
	-1,-1,+1,+1,+1,+1,-1,-1,-1,-1,-1,-1,-1,-1,+1,+1,
	-1,-1,-1,+1,-1,-1,+1,-1,+1,-1,-1,+1,-1,+1,+1,+1,
	+1,+1,+1,+1,+1,-1,-1,+1,-1,-1,-1,-1,-1,-1,-1,-1,
	+1,-1,-1,+1,+1,+1,+1,-1,+1,+1,-1,-1,-1,+1,-1,+1,
	+1,+1,-1,+1,-1,+1,-1,-1,+1,-1,+1,+1,+1,+1,+1,-1,
	+1,-1,-1,+1,-1,+1,+1,-1,-1,-1,-1,+1,+1,-1,-1,-1,
	-1,+1,+1,-1,+1,-1,+1,-1,-1,-1,+1,-1,+1,+1,+1,-1,
	+1,-1,-1,-1,+1,+1,-1,+1,-1,+1,-1,-1,-1,-1,-1,+1,
], dtype=np.float32)

def fwht_forward(data):
	"""Apply forward FWHT with sign arrays (matching CUDA)."""
	d = data.copy()
	d *= SIGNS1
	n = len(d)
	h = 1
	while h < n:
		for i in range(0, n, h * 2):
			for j in range(i, i + h):
				a, b = d[j], d[j + h]
				d[j], d[j + h] = a + b, a - b
		h *= 2
	d *= (1.0 / np.sqrt(n)) * SIGNS2
	return d

def precompute_predecessors():
	"""Precompute trellis predecessors for right-shift bitshift trellis."""
	preds = np.zeros((N_STATES, 8), dtype=np.int32)
	for sid in range(N_STATES):
		base = (sid & MASK) << K_BITS
		for p in range(8):
			preds[sid, p] = base | p
	return preds

def viterbi_encode(data, codebook):
	"""Viterbi encode a single 128-element block. Returns states and recon_norm."""
	preds = precompute_predecessors()
	cost = np.zeros(N_STATES, dtype=np.float64)
	bt = np.zeros((BLOCK_SIZE, N_STATES), dtype=np.int32)

	for t in range(BLOCK_SIZE):
		xt = float(data[t])
		new_cost = np.full(N_STATES, 1e30, dtype=np.float64)
		new_bt = np.zeros(N_STATES, dtype=np.int32)
		for sid in range(N_STATES):
			dist = (xt - codebook[sid]) ** 2
			prev_costs = cost[preds[sid]]
			best_p = np.argmin(prev_costs)
			new_cost[sid] = prev_costs[best_p] + dist
			new_bt[sid] = best_p
		cost = new_cost
		bt[t] = new_bt

	# Backtrack
	final_state = np.argmin(cost)
	states = np.zeros(BLOCK_SIZE, dtype=np.int32)
	state = final_state
	for t in range(BLOCK_SIZE - 1, -1, -1):
		states[t] = state
		p = bt[t, state]
		state = ((state & MASK) << K_BITS) | p

	# Reconstruction
	recon = codebook[states]
	recon_norm = np.sqrt(np.sum(recon ** 2))

	return states, recon, recon_norm

def viterbi_batch(data_blocks, codebook):
	"""Batch Viterbi for multiple blocks. Returns per-block recon_norms and reconstructions."""
	preds = precompute_predecessors()
	n_blocks = len(data_blocks)
	all_recon_norms = np.zeros(n_blocks)
	all_recons = np.zeros_like(data_blocks)
	all_mse = np.zeros(n_blocks)

	for b in range(n_blocks):
		if b % 500 == 0 and b > 0:
			print(f"  block {b}/{n_blocks}", file=sys.stderr)
		data = data_blocks[b]
		cost = np.zeros(N_STATES, dtype=np.float64)
		bt = np.zeros((BLOCK_SIZE, N_STATES), dtype=np.int32)

		for t in range(BLOCK_SIZE):
			xt = float(data[t])
			new_cost = np.full(N_STATES, 1e30, dtype=np.float64)
			new_bt = np.zeros(N_STATES, dtype=np.int32)
			for sid in range(N_STATES):
				dist = (xt - codebook[sid]) ** 2
				prev_costs = cost[preds[sid]]
				best_p = np.argmin(prev_costs)
				new_cost[sid] = prev_costs[best_p] + dist
				new_bt[sid] = best_p
			cost = new_cost
			bt[t] = new_bt

		final_state = np.argmin(cost)
		states = np.zeros(BLOCK_SIZE, dtype=np.int32)
		state = final_state
		for t in range(BLOCK_SIZE - 1, -1, -1):
			states[t] = state
			p = bt[t, state]
			state = ((state & MASK) << K_BITS) | p

		recon = codebook[states]
		recon_norm = np.sqrt(np.sum(recon ** 2))
		all_recon_norms[b] = recon_norm
		all_recons[b] = recon
		all_mse[b] = np.mean((data - recon) ** 2)

	return all_recon_norms, all_recons, all_mse

def load_codebook_from_file(path):
	"""Load codebook from CUDA trainer text output."""
	import re
	with open(path) as f:
		text = f.read()
	m = re.search(r'static __constant__ float d_turbo3_tcq_codebook\[512\] = \{\n(.*?)\};', text, re.DOTALL)
	if not m:
		raise ValueError(f"Codebook not found in {path}")
	vals = [float(x.strip().rstrip('f')) for x in m.group(1).replace('\n', ',').split(',') if x.strip()]
	return np.array(vals[:512], dtype=np.float32)

def fp16_roundtrip(x):
	"""Simulate fp16 storage of norm value."""
	return np.float16(x).astype(np.float32)

def main():
	# Load K data
	print("Loading post-FWHT K data...")
	raw = np.fromfile('/tmp/turbo_postrot.bin', dtype=np.float32)
	n_vec = len(raw) // BLOCK_SIZE
	k_data = raw[:n_vec * BLOCK_SIZE].reshape(n_vec, BLOCK_SIZE)
	print(f"  {n_vec} vectors loaded")

	# Use 2000 blocks for tractable Viterbi (still takes a few minutes)
	n_blocks = min(2000, n_vec)
	k_data = k_data[:n_blocks]

	# Normalize each block to unit norm (as SET_ROWS does before FWHT)
	# Note: turbo_postrot.bin is ALREADY post-FWHT, so we're analyzing the
	# same data the Viterbi sees. We just need to normalize.
	norms = np.sqrt(np.sum(k_data ** 2, axis=1, keepdims=True))
	norms = np.maximum(norms, 1e-10)
	saved_norms = norms.ravel()  # these would be stored
	k_unit = k_data / norms  # unit-norm data fed to Viterbi

	# Load codebooks
	print("Loading codebooks...")
	cb_old = np.fromfile('/tmp/old_codebook_3bit.bin', dtype=np.float32)[:512]

	codebooks = {}
	codebooks['old_numpy'] = cb_old
	for iters in [3, 10, 30]:
		path = f'/tmp/tcq_sweep_3bit_{iters}.txt'
		try:
			codebooks[f'cuda_{iters}'] = load_codebook_from_file(path)
		except:
			print(f"  SKIP {path}")

	print(f"\n{'='*80}")
	print("NORM CORRECTION ANALYSIS")
	print(f"{'='*80}")
	print(f"Blocks: {n_blocks}, each normalized to unit norm before Viterbi\n")

	# Analyze each codebook
	results = {}
	for name, cb in codebooks.items():
		print(f"\n--- {name} ---")
		recon_norms, recons, block_mse = viterbi_batch(k_unit, cb)

		# Corrected norm = saved_norm / recon_norm
		corrected_norms = saved_norms[:n_blocks] / recon_norms

		# fp16 roundtrip
		corrected_fp16 = np.array([fp16_roundtrip(x) for x in corrected_norms])
		fp16_err = corrected_fp16 - corrected_norms  # fp16 quantization error
		rel_fp16_err = fp16_err / corrected_norms

		# Reconstruction with norm correction
		recon_scaled = recons * corrected_norms[:, None]
		data_scaled = k_unit * saved_norms[:n_blocks, None]

		# Full reconstruction error (after norm correction)
		full_mse = np.mean((data_scaled - recon_scaled) ** 2)

		# Reconstruction with fp16 norm
		recon_fp16 = recons * corrected_fp16[:, None]
		full_mse_fp16 = np.mean((data_scaled - recon_fp16) ** 2)

		# Norm of reconstruction vs norm of original
		recon_output_norms = np.sqrt(np.sum(recon_scaled ** 2, axis=1))
		norm_ratio = recon_output_norms / saved_norms[:n_blocks]

		# Per-block direction error (cosine distance)
		dot_products = np.sum(k_unit * (recons / recon_norms[:, None]), axis=1)
		cos_dist = 1 - dot_products

		print(f"  recon_norm: mean={np.mean(recon_norms):.6f}, std={np.std(recon_norms):.6f}, "
			  f"min={np.min(recon_norms):.6f}, max={np.max(recon_norms):.6f}")
		print(f"  |recon_norm - 1|: mean={np.mean(np.abs(recon_norms - 1)):.6f}, "
			  f"median={np.median(np.abs(recon_norms - 1)):.6f}")
		print(f"  recon_norm bias (mean-1): {np.mean(recon_norms) - 1:.6f}")
		print(f"  corrected_norm: mean={np.mean(corrected_norms):.6f}")
		print(f"  fp16 norm rel err: mean={np.mean(rel_fp16_err):.8f}, "
			  f"std={np.std(rel_fp16_err):.8f}, max={np.max(np.abs(rel_fp16_err)):.6f}")
		print(f"  output norm ratio (should be 1.0): mean={np.mean(norm_ratio):.8f}, "
			  f"std={np.std(norm_ratio):.8f}")
		print(f"  block MSE (unit-norm): mean={np.mean(block_mse):.8f}")
		print(f"  full MSE (with norm): {full_mse:.8f}")
		print(f"  full MSE (fp16 norm): {full_mse_fp16:.8f}")
		print(f"  MSE increase from fp16 norm: {(full_mse_fp16/full_mse - 1)*100:.4f}%")
		print(f"  cosine distance: mean={np.mean(cos_dist):.8f}, std={np.std(cos_dist):.8f}")

		results[name] = {
			'recon_norm_mean': np.mean(recon_norms),
			'recon_norm_std': np.std(recon_norms),
			'recon_norm_bias': np.mean(recon_norms) - 1,
			'abs_norm_dev': np.mean(np.abs(recon_norms - 1)),
			'block_mse': np.mean(block_mse),
			'full_mse': full_mse,
			'full_mse_fp16': full_mse_fp16,
			'fp16_rel_err_mean': np.mean(rel_fp16_err),
			'fp16_rel_err_std': np.std(rel_fp16_err),
			'cos_dist_mean': np.mean(cos_dist),
			'norm_ratio_std': np.std(norm_ratio),
			'recon_norms': recon_norms,
		}

	# Comparative summary
	print(f"\n{'='*80}")
	print("COMPARATIVE SUMMARY")
	print(f"{'='*80}")
	print(f"{'Codebook':<15} {'norm_bias':>12} {'|norm-1|':>10} {'norm_std':>10} {'unit_MSE':>12} {'cos_dist':>12}")
	for name, r in results.items():
		print(f"{name:<15} {r['recon_norm_bias']:>+12.6f} {r['abs_norm_dev']:>10.6f} "
			  f"{r['recon_norm_std']:>10.6f} {r['block_mse']:>12.8f} {r['cos_dist_mean']:>12.8f}")

	# Direct comparison: old vs 10-iter
	if 'old_numpy' in results and 'cuda_10' in results:
		print(f"\n{'='*80}")
		print("OLD NUMPY vs CUDA 10-ITER")
		print(f"{'='*80}")
		rn_old = results['old_numpy']['recon_norms']
		rn_10 = results['cuda_10']['recon_norms']
		diff = rn_10 - rn_old
		print(f"  recon_norm difference (10-iter minus old):")
		print(f"    mean: {np.mean(diff):+.8f}")
		print(f"    std:  {np.std(diff):.8f}")
		print(f"    corr: {np.corrcoef(rn_old, rn_10)[0,1]:.4f}")
		print(f"  Are norm corrections systematically different? "
			  f"{'YES' if abs(np.mean(diff)) > 2*np.std(diff)/np.sqrt(len(diff)) else 'NO'} "
			  f"(|mean|/SE = {abs(np.mean(diff)) / (np.std(diff)/np.sqrt(len(diff))):.1f})")

if __name__ == '__main__':
	main()
