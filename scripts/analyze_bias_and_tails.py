#!/usr/bin/env python3
"""Test remaining hypotheses for MSE-PPL divergence:
1. Quantization bias E[e] ≠ 0 — mean error per channel
2. Higher-order moments — kurtosis/skewness of per-channel errors
3. Tail behavior — how softmax-weighted errors differ from MSE
4. Input-dependent error structure — do errors correlate with input magnitude?
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

def fwht_forward_batch(data):
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

def simulate_attention_error(q_rot, k_data, errors):
	"""Simulate actual attention error from quantization.
	q_rot: [n_q, 128] — rotated Q vectors
	k_data: [n_k, 128] — rotated K vectors (clean)
	errors: [n_k, 128] — quantization errors
	Returns: statistics about attention weight perturbation.
	"""
	n_q = min(200, q_rot.shape[0])
	n_k = min(500, k_data.shape[0])
	q = q_rot[:n_q]     # [n_q, 128]
	k = k_data[:n_k]    # [n_k, 128]
	e = errors[:n_k]    # [n_k, 128]

	sqrt_d = np.sqrt(128.0)

	# Clean dot products: q · k
	dots_clean = (q @ k.T) / sqrt_d  # [n_q, n_k]
	# Noisy dot products: q · (k + e)
	dots_noisy = (q @ (k + e).T) / sqrt_d  # [n_q, n_k]
	# Error in dot products
	dot_errors = dots_noisy - dots_clean  # = q · e / sqrt_d

	# Softmax of clean
	dots_clean_shifted = dots_clean - dots_clean.max(axis=1, keepdims=True)
	attn_clean = np.exp(dots_clean_shifted)
	attn_clean /= attn_clean.sum(axis=1, keepdims=True)

	# Softmax of noisy
	dots_noisy_shifted = dots_noisy - dots_noisy.max(axis=1, keepdims=True)
	attn_noisy = np.exp(dots_noisy_shifted)
	attn_noisy /= attn_noisy.sum(axis=1, keepdims=True)

	# Attention weight error
	attn_err = attn_noisy - attn_clean
	attn_l1 = np.mean(np.sum(np.abs(attn_err), axis=1))
	attn_kl = np.mean(np.sum(attn_clean * np.log((attn_clean + 1e-30) / (attn_noisy + 1e-30)), axis=1))

	# Dot product error statistics
	dot_err_var = np.var(dot_errors)
	dot_err_mean = np.mean(dot_errors)
	dot_err_kurtosis = np.mean((dot_errors - dot_err_mean)**4) / (np.var(dot_errors)**2) - 3

	# Attention-weighted dot error: errors at high-attention positions matter more
	weighted_err = np.sum(attn_clean * np.abs(dot_errors - np.mean(dot_errors, axis=1, keepdims=True)), axis=1)
	attn_weighted_err = np.mean(weighted_err)

	return {
		'attn_l1': attn_l1,
		'attn_kl': attn_kl,
		'dot_err_var': dot_err_var,
		'dot_err_mean': dot_err_mean,
		'dot_err_kurtosis': dot_err_kurtosis,
		'attn_weighted_err': attn_weighted_err,
	}

def main():
	print("Loading data...")
	n_vec = 10000
	k_data = np.fromfile("/tmp/turbo_postrot.bin", dtype=np.float32, count=n_vec*T).reshape(n_vec, T)
	old_cb = np.fromfile("/tmp/old_codebook_3bit.bin", dtype=np.float32)
	coset_cb = init_coset()

	# Load Q vectors for attention simulation
	print("Loading Q vectors...")
	with open("/tmp/q_vectors_raw.bin", "rb") as f:
		hdr = np.frombuffer(f.read(16), dtype=np.int32)
		head_dim, n_tokens, n_heads_total, n_layers = hdr
		n_heads_dumped = 4
		floats_per_head = head_dim * n_tokens
		# Load all Q data from all layers, just head 0
		q_blocks = []
		for layer in range(n_layers):
			raw = np.frombuffer(f.read(floats_per_head * 4), dtype=np.float32).reshape(n_tokens, head_dim)
			q_blocks.append(fwht_forward_batch(raw[:, :128]))
			q_blocks.append(fwht_forward_batch(raw[:, 128:256]))
			# Skip heads 6, 12, 18
			for _ in range(3):
				f.read(floats_per_head * 4)
	q_rot = np.concatenate(q_blocks, axis=0)  # rotated Q vectors
	print(f"  Q rotated: {q_rot.shape}")

	codebooks = [
		("Coset 0-iter", coset_cb, 5.9194),
		("Old numpy 100-iter", old_cb, 5.8236),
	]
	for iters, ppl in [(3, 5.8450), (5, 5.8576), (10, 5.9386), (20, 5.9712), (30, 5.8733)]:
		path = f"/tmp/tcq_sweep_3bit_{iters}.txt"
		try:
			cb = parse_codebook_from_file(path)
			if cb is not None and len(cb) == N_STATES:
				codebooks.append((f"CUDA {iters}-iter", cb, ppl))
		except FileNotFoundError:
			pass

	preds = precompute_predecessors()

	# ========================================
	# ANALYSIS 1: Quantization bias E[e]
	# ========================================
	print("\n" + "="*70)
	print("ANALYSIS 1: QUANTIZATION BIAS E[e]")
	print("="*70)

	all_results = []
	for name, cb, ppl in codebooks:
		print(f"\n  Quantizing with {name}...", end=" ", flush=True)
		t0 = time.time()
		all_states = []
		for b in range(0, n_vec, 500):
			s = viterbi_batch(k_data[b:b+500], cb, preds)
			all_states.append(s)
		states = np.concatenate(all_states)
		quantized = cb[states]
		errors = k_data - quantized
		print(f"{time.time()-t0:.0f}s")

		# Per-channel mean error (bias)
		mean_err = np.mean(errors, axis=0)  # [128]
		global_bias = np.mean(errors)
		bias_norm = np.linalg.norm(mean_err)
		max_bias = np.max(np.abs(mean_err))
		rms_bias = np.sqrt(np.mean(mean_err**2))
		mse = np.mean(errors**2)

		# Bias as fraction of RMS error
		bias_fraction = rms_bias / np.sqrt(mse)

		# Per-channel error kurtosis
		ch_kurtosis = []
		for c in range(T):
			e = errors[:, c]
			mu = np.mean(e)
			var = np.var(e)
			if var > 0:
				k4 = np.mean((e - mu)**4) / var**2 - 3
				ch_kurtosis.append(k4)
		mean_kurtosis = np.mean(ch_kurtosis)
		max_kurtosis = np.max(ch_kurtosis)

		# Per-channel skewness
		ch_skew = []
		for c in range(T):
			e = errors[:, c]
			mu = np.mean(e)
			var = np.var(e)
			if var > 0:
				s3 = np.mean((e - mu)**3) / var**1.5
				ch_skew.append(s3)
		mean_skew = np.mean(ch_skew)
		max_abs_skew = np.max(np.abs(ch_skew))

		# Error magnitude vs input magnitude correlation
		# Does the error get larger when the input is larger?
		input_mag = np.abs(k_data)
		error_mag = np.abs(errors)
		# Per-channel correlation
		mag_corrs = []
		for c in range(T):
			corr = np.corrcoef(input_mag[:, c], error_mag[:, c])[0, 1]
			mag_corrs.append(corr)
		mean_mag_corr = np.mean(mag_corrs)

		# Position-dependent MSE (beginning vs end of block)
		mse_first16 = np.mean(errors[:, :16]**2)
		mse_mid = np.mean(errors[:, 56:72]**2)
		mse_last16 = np.mean(errors[:, 112:]**2)

		# 99th percentile absolute error
		p99_err = np.percentile(np.abs(errors), 99)
		p999_err = np.percentile(np.abs(errors), 99.9)

		print(f"    MSE:             {mse:.8f}")
		print(f"    Global bias:     {global_bias:+.8f}")
		print(f"    Bias norm:       {bias_norm:.8f}")
		print(f"    RMS bias:        {rms_bias:.8f} ({bias_fraction*100:.2f}% of RMS error)")
		print(f"    Max |bias|:      {max_bias:.8f}")
		print(f"    Mean kurtosis:   {mean_kurtosis:+.4f}")
		print(f"    Max kurtosis:    {max_kurtosis:+.4f}")
		print(f"    Mean skewness:   {mean_skew:+.4f}")
		print(f"    Max |skewness|:  {max_abs_skew:.4f}")
		print(f"    Mag correlation: {mean_mag_corr:+.4f}")
		print(f"    MSE by position: first16={mse_first16:.8f}  mid={mse_mid:.8f}  last16={mse_last16:.8f}")
		print(f"    P99 |error|:     {p99_err:.6f}")
		print(f"    P99.9 |error|:   {p999_err:.6f}")

		all_results.append({
			'name': name,
			'ppl': ppl,
			'mse': mse,
			'global_bias': global_bias,
			'rms_bias': rms_bias,
			'bias_fraction': bias_fraction,
			'mean_kurtosis': mean_kurtosis,
			'max_kurtosis': max_kurtosis,
			'mean_skew': mean_skew,
			'max_abs_skew': max_abs_skew,
			'mag_corr': mean_mag_corr,
			'mse_first16': mse_first16,
			'mse_mid': mse_mid,
			'mse_last16': mse_last16,
			'p99': p99_err,
			'p999': p999_err,
			'errors': errors,
		})

	# ========================================
	# ANALYSIS 2: Simulated attention error
	# ========================================
	print("\n" + "="*70)
	print("ANALYSIS 2: SIMULATED ATTENTION ERROR (with real Q)")
	print("="*70)

	for r in all_results:
		print(f"\n  Simulating {r['name']}...", end=" ", flush=True)
		t0 = time.time()
		attn = simulate_attention_error(q_rot, k_data, r['errors'])
		print(f"{time.time()-t0:.1f}s")
		r.update(attn)
		print(f"    Attention L1:         {attn['attn_l1']:.8f}")
		print(f"    Attention KL div:     {attn['attn_kl']:.8f}")
		print(f"    Dot error variance:   {attn['dot_err_var']:.8f}")
		print(f"    Dot error mean:       {attn['dot_err_mean']:+.8f}")
		print(f"    Dot error kurtosis:   {attn['dot_err_kurtosis']:+.4f}")
		print(f"    Attn-weighted error:  {attn['attn_weighted_err']:.8f}")

	# ========================================
	# CORRELATION ANALYSIS
	# ========================================
	print("\n" + "="*70)
	print("CORRELATION WITH PPL")
	print("="*70)

	ppls = np.array([r['ppl'] for r in all_results])
	metrics = [
		('MSE', [r['mse'] for r in all_results]),
		('RMS bias', [r['rms_bias'] for r in all_results]),
		('Bias fraction', [r['bias_fraction'] for r in all_results]),
		('Mean kurtosis', [r['mean_kurtosis'] for r in all_results]),
		('Max |skewness|', [r['max_abs_skew'] for r in all_results]),
		('Mag correlation', [r['mag_corr'] for r in all_results]),
		('MSE first16', [r['mse_first16'] for r in all_results]),
		('MSE last16', [r['mse_last16'] for r in all_results]),
		('First/Last MSE ratio', [r['mse_first16']/r['mse_last16'] for r in all_results]),
		('P99.9 error', [r['p999'] for r in all_results]),
		('Attention L1', [r['attn_l1'] for r in all_results]),
		('Attention KL', [r['attn_kl'] for r in all_results]),
		('Dot error variance', [r['dot_err_var'] for r in all_results]),
		('Dot error mean', [r['dot_err_mean'] for r in all_results]),
		('Dot error kurtosis', [r['dot_err_kurtosis'] for r in all_results]),
		('Attn-weighted error', [r['attn_weighted_err'] for r in all_results]),
	]

	try:
		from scipy.stats import spearmanr
		has_scipy = True
	except ImportError:
		has_scipy = False

	print(f"\n{'Metric':<25} {'Pearson':>8} {'Spearman':>8}  Values")
	print("-" * 90)
	for metric_name, values in metrics:
		vals = np.array(values)
		if np.std(vals) == 0:
			continue
		pearson = np.corrcoef(vals, ppls)[0, 1]
		if has_scipy:
			spearman, _ = spearmanr(vals, ppls)
		else:
			spearman = float('nan')
		vals_str = " ".join([f"{v:.6f}" for v in vals[:5]])
		print(f"{metric_name:<25} {pearson:>+8.4f} {spearman:>+8.4f}  {vals_str}")

	# ========================================
	# SUMMARY TABLE
	# ========================================
	print("\n" + "="*70)
	print("FULL COMPARISON TABLE")
	print("="*70)
	print(f"{'Codebook':<22} {'PPL':>7} {'MSE':>10} {'Bias%':>6} {'Kurt':>6} {'MagCor':>6} {'AttnL1':>10} {'AttnKL':>10} {'1st/Last':>8}")
	print("-" * 96)
	for r in all_results:
		fl_ratio = r['mse_first16'] / r['mse_last16']
		print(f"{r['name']:<22} {r['ppl']:>7.4f} {r['mse']:>10.8f} {r['bias_fraction']*100:>5.2f}% {r['mean_kurtosis']:>+6.3f} {r['mag_corr']:>+6.4f} {r['attn_l1']:>10.6f} {r['attn_kl']:>10.8f} {fl_ratio:>8.4f}")

	print("\nDone.")

if __name__ == "__main__":
	main()
