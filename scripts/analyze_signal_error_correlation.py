#!/usr/bin/env python3
"""Test: Signal-dependent quantization noise in TCQ.

Theory: As GLA training increases, the TCQ quantizer evolves from memoryless
(coset structure) to predictive (state-dependent). This makes errors correlated
with the input signal. Through Q-K correlation in attention, signal-dependent
error creates a systematic temperature perturbation in softmax that degrades PPL.

Key measurements:
1. Per-element signal-error correlation: corr(k[i], e[i])
2. Cross-position signal-error: corr(k[i-1], e[i]) — trellis memory effect
3. Attention logit-error correlation: corr(q·k, q·e) with real Q vectors
4. Temperature perturbation: regression slope of q·e on q·k
5. Coset state diversity: within-coset variance (memorylessness metric)
"""

import numpy as np
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

def scalar_quantize(data, centroids):
	"""Scalar (memoryless) quantization — no trellis."""
	quantized = np.zeros_like(data)
	for t in range(data.shape[1]):
		dists = (data[:, t:t+1] - centroids[None, :]) ** 2
		idx = np.argmin(dists, axis=1)
		quantized[:, t] = centroids[idx]
	return quantized

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

def coset_state_diversity(cb):
	"""Measure within-coset variance — how much does the trellis state affect output?
	0 = perfectly memoryless (all states in a coset produce same value)
	Higher = more predictive (state-dependent reconstruction)
	"""
	n_cosets = N_STATES >> K_BITS  # 64 cosets
	coset_stds = []
	for coset in range(n_cosets):
		vals = []
		for p in range(N_OUT):
			# All states that produce output p in this coset context
			# State = (coset << K_BITS) | p
			state = (coset << K_BITS) | p
			vals.append(cb[state])
		# But actually, the "coset" in a bitshift trellis groups states by their
		# upper bits. States sharing the same 3-bit output (same p) can have
		# 64 different upper-bit contexts. Let's measure per-output-symbol.
		pass

	# Better: for each of the 8 output symbols, measure variance of codebook
	# entries across all 64 states that map to that symbol
	per_symbol_stds = []
	for p in range(N_OUT):
		vals = [cb[s] for s in range(N_STATES) if (s & ((1 << K_BITS) - 1)) == p]
		per_symbol_stds.append(np.std(vals))

	# Also: for each group of 8 consecutive states (same upper bits),
	# measure how much they differ from the coset-init pattern
	group_devs = []
	for g in range(N_STATES >> K_BITS):
		group_vals = [cb[(g << K_BITS) | p] for p in range(N_OUT)]
		# Compare spacing to Lloyd-Max
		lm_scaled = LM_CENTROIDS * SIGMA
		# Best-fit: find offset that minimizes deviation
		offsets = [group_vals[p] - lm_scaled[p] for p in range(N_OUT)]
		mean_offset = np.mean(offsets)
		dev = np.std([offsets[p] - mean_offset for p in range(N_OUT)])
		group_devs.append(dev)

	return {
		'per_symbol_std': np.mean(per_symbol_stds),
		'per_symbol_stds': per_symbol_stds,
		'group_deviation': np.mean(group_devs),
		'group_deviation_max': np.max(group_devs),
	}


def main():
	print("=" * 70)
	print("SIGNAL-DEPENDENT QUANTIZATION NOISE ANALYSIS")
	print("=" * 70)

	# ================================================================
	# Load data
	# ================================================================
	print("\nLoading post-FWHT K data...")
	n_vec = 10000
	k_data = np.fromfile("/tmp/turbo_postrot.bin", dtype=np.float32, count=n_vec * T).reshape(n_vec, T).astype(np.float64)
	print(f"  K: {k_data.shape}, mean={np.mean(k_data):.6f}, std={np.std(k_data):.6f}")

	print("Loading Q vectors...")
	with open("/tmp/q_vectors_raw.bin", "rb") as f:
		hdr = np.frombuffer(f.read(16), dtype=np.int32)
		head_dim, n_tokens, n_heads_total, n_layers = hdr
		print(f"  Header: head_dim={head_dim}, n_tokens={n_tokens}, n_layers={n_layers}")
		n_heads_dumped = 4
		floats_per_head = head_dim * n_tokens
		q_blocks = []
		for layer in range(n_layers):
			for hi in range(n_heads_dumped):
				raw = np.frombuffer(f.read(floats_per_head * 4), dtype=np.float32).reshape(n_tokens, head_dim)
				# FWHT rotate both 128-element halves
				q_blocks.append(fwht_forward_batch(raw[:, :128].astype(np.float64)))
				q_blocks.append(fwht_forward_batch(raw[:, 128:256].astype(np.float64)))
	q_rot = np.concatenate(q_blocks, axis=0)
	print(f"  Q rotated: {q_rot.shape}")
	# Subsample Q for attention analysis
	n_q_use = min(2000, q_rot.shape[0])
	q_sub = q_rot[np.random.choice(q_rot.shape[0], n_q_use, replace=False)]
	n_k_use = min(5000, n_vec)

	# ================================================================
	# Codebooks
	# ================================================================
	old_cb = np.fromfile("/tmp/old_codebook_3bit.bin", dtype=np.float32).astype(np.float64)
	coset_cb = init_coset()

	codebooks = [
		("Scalar LM", None, None),  # special: no trellis
		("Coset 0-iter", coset_cb, 5.9194),
		("Old numpy", old_cb, 5.8236),
	]
	for iters, ppl in [(3, 5.8450), (5, 5.8576), (10, 5.9386), (20, 5.9712), (30, 5.8733)]:
		path = f"/tmp/tcq_sweep_3bit_{iters}.txt"
		try:
			cb = parse_codebook_from_file(path)
			if cb is not None and len(cb) == N_STATES:
				codebooks.append((f"CUDA {iters}-iter", cb.astype(np.float64), ppl))
		except FileNotFoundError:
			pass

	preds = precompute_predecessors()
	lm_vals = LM_CENTROIDS * SIGMA

	# ================================================================
	# ANALYSIS 1: Per-element signal-error correlation
	# ================================================================
	print("\n" + "=" * 70)
	print("ANALYSIS 1: PER-ELEMENT SIGNAL-ERROR CORRELATION  corr(k[i], e[i])")
	print("=" * 70)
	print("  If quantizer is memoryless, this depends only on centroid spacing.")
	print("  If predictive (trellis memory), this should increase with training.")
	print()

	all_results = []
	for name, cb, ppl in codebooks:
		t0 = time.time()
		if cb is None:  # scalar
			quantized = scalar_quantize(k_data, lm_vals)
			errors = k_data - quantized
		else:
			all_states = []
			for b in range(0, n_vec, 500):
				s = viterbi_batch(k_data[b:b+500], cb, preds)
				all_states.append(s)
			states = np.concatenate(all_states)
			quantized = cb[states]
			errors = k_data - quantized
		elapsed = time.time() - t0

		mse = np.mean(errors ** 2)

		# Per-position correlation corr(k[i], e[i])
		pos_corrs = []
		for i in range(T):
			r = np.corrcoef(k_data[:, i], errors[:, i])[0, 1]
			pos_corrs.append(r)
		mean_corr = np.mean(pos_corrs)
		std_corr = np.std(pos_corrs)

		# Signed: E[k[i] * e[i]] / (σ_k * σ_e) — but also just raw E[k*e]
		eke = np.mean(k_data * errors)  # global average of k*e

		# Cross-position: corr(k[i-1], e[i]) — does previous value predict current error?
		cross_corrs = []
		for i in range(1, T):
			r = np.corrcoef(k_data[:, i-1], errors[:, i])[0, 1]
			cross_corrs.append(r)
		mean_cross_corr = np.mean(cross_corrs)

		# Cross-position lag-2: corr(k[i-2], e[i])
		cross_corrs_2 = []
		for i in range(2, T):
			r = np.corrcoef(k_data[:, i-2], errors[:, i])[0, 1]
			cross_corrs_2.append(r)
		mean_cross_corr_2 = np.mean(cross_corrs_2)

		result = {
			'name': name, 'ppl': ppl, 'mse': mse,
			'mean_corr': mean_corr, 'std_corr': std_corr,
			'eke': eke,
			'cross_corr_lag1': mean_cross_corr,
			'cross_corr_lag2': mean_cross_corr_2,
			'errors': errors,
			'cb': cb,
		}
		all_results.append(result)

		print(f"  {name:<20} ({elapsed:.0f}s)")
		print(f"    MSE:                    {mse:.8f}")
		print(f"    corr(k[i], e[i]):       {mean_corr:+.6f}  (std {std_corr:.6f})")
		print(f"    E[k*e]:                 {eke:+.8f}")
		print(f"    corr(k[i-1], e[i]):     {mean_cross_corr:+.6f}  (trellis lag-1)")
		print(f"    corr(k[i-2], e[i]):     {mean_cross_corr_2:+.6f}  (trellis lag-2)")
		print()

	# ================================================================
	# ANALYSIS 2: Attention logit-error correlation
	# ================================================================
	print("=" * 70)
	print("ANALYSIS 2: ATTENTION LOGIT-ERROR CORRELATION  corr(q·k, q·e)")
	print("=" * 70)
	print("  If > 0: high-attention keys get logits inflated → sharpening")
	print("  If < 0: high-attention keys get logits decreased → broadening")
	print("  If ≈ 0: no systematic temperature bias → benign")
	print()

	sqrt_d = np.sqrt(128.0)
	k_sub = k_data[:n_k_use]

	for r in all_results:
		name = r['name']
		errors = r['errors'][:n_k_use]
		t0 = time.time()

		# Compute logits and logit errors for each Q vector
		# logits[i,j] = q_sub[i] · k_sub[j] / √d
		# lerrs[i,j]  = q_sub[i] · errors[j] / √d
		logits = (q_sub @ k_sub.T) / sqrt_d      # [n_q, n_k]
		lerrs  = (q_sub @ errors.T) / sqrt_d      # [n_q, n_k]

		# Per-query correlation: for each q, corr across keys
		per_q_corr = []
		for i in range(n_q_use):
			c = np.corrcoef(logits[i], lerrs[i])[0, 1]
			if not np.isnan(c):
				per_q_corr.append(c)
		mean_logit_corr = np.mean(per_q_corr)
		std_logit_corr = np.std(per_q_corr)
		median_logit_corr = np.median(per_q_corr)

		# Temperature perturbation: fit q·e = ε · q·k + intercept
		# ε > 0 means systematic sharpening
		# Use all (q,k) pairs for a global fit
		logits_flat = logits.ravel()
		lerrs_flat = lerrs.ravel()
		# Linear regression: ε = cov(z, δ) / var(z)
		epsilon = np.cov(logits_flat, lerrs_flat)[0, 1] / np.var(logits_flat)
		# Also: mean relative error for large logits (|z| > 1)
		large_mask = np.abs(logits_flat) > 1.0
		if np.any(large_mask):
			mean_rel = np.mean(lerrs_flat[large_mask] / logits_flat[large_mask])
		else:
			large_mask = np.abs(logits_flat) > 0.5
			mean_rel = np.mean(lerrs_flat[large_mask] / logits_flat[large_mask]) if np.any(large_mask) else 0

		# Softmax analysis: does the error systematically change attention entropy?
		# For each q, compute entropy of clean vs noisy attention
		entropy_changes = []
		n_entropy = min(500, n_q_use)
		for i in range(n_entropy):
			z_clean = logits[i] - np.max(logits[i])
			p_clean = np.exp(z_clean); p_clean /= p_clean.sum()
			z_noisy = (logits[i] + lerrs[i])
			z_noisy = z_noisy - np.max(z_noisy)
			p_noisy = np.exp(z_noisy); p_noisy /= p_noisy.sum()
			h_clean = -np.sum(p_clean * np.log(p_clean + 1e-30))
			h_noisy = -np.sum(p_noisy * np.log(p_noisy + 1e-30))
			entropy_changes.append(h_noisy - h_clean)
		mean_entropy_change = np.mean(entropy_changes)
		std_entropy_change = np.std(entropy_changes)

		elapsed = time.time() - t0
		r['logit_corr'] = mean_logit_corr
		r['logit_corr_std'] = std_logit_corr
		r['logit_corr_median'] = median_logit_corr
		r['epsilon'] = epsilon
		r['mean_rel'] = mean_rel
		r['entropy_change'] = mean_entropy_change
		r['entropy_change_std'] = std_entropy_change

		print(f"  {name:<20} ({elapsed:.1f}s)")
		print(f"    corr(q·k, q·e):         {mean_logit_corr:+.6f}  (median {median_logit_corr:+.6f}, std {std_logit_corr:.6f})")
		print(f"    Temperature ε:           {epsilon:+.8f}")
		print(f"    Mean relative error:     {mean_rel:+.8f}")
		print(f"    Entropy change:          {mean_entropy_change:+.6f}  (std {std_entropy_change:.6f})")
		print()

	# ================================================================
	# ANALYSIS 3: Coset state diversity
	# ================================================================
	print("=" * 70)
	print("ANALYSIS 3: COSET STATE DIVERSITY (memorylessness)")
	print("=" * 70)
	print("  per_symbol_std: variance of codebook values for same output symbol")
	print("  across 64 different trellis states. 0 = perfectly memoryless.")
	print()

	for r in all_results:
		if r['cb'] is None:
			print(f"  {r['name']:<20} — scalar (no trellis)")
			continue
		div = coset_state_diversity(r['cb'])
		r['state_diversity'] = div['per_symbol_std']
		r['group_deviation'] = div['group_deviation']
		print(f"  {r['name']:<20}")
		print(f"    Per-symbol std:         {div['per_symbol_std']:.8f}")
		print(f"    Per-symbol breakdown:   {['%.6f' % s for s in div['per_symbol_stds']]}")
		print(f"    Group deviation (mean): {div['group_deviation']:.8f}")
		print(f"    Group deviation (max):  {div['group_deviation_max']:.8f}")
		print()

	# ================================================================
	# SUMMARY TABLE
	# ================================================================
	print("=" * 70)
	print("SUMMARY TABLE")
	print("=" * 70)
	header = f"{'Codebook':<20} {'PPL':>7} {'MSE':>10} {'corr(k,e)':>10} {'E[k*e]':>10} {'cross_lag1':>10} {'corr(qk,qe)':>12} {'epsilon':>10} {'ΔH':>8}"
	print(header)
	print("-" * len(header))
	for r in all_results:
		ppl_str = f"{r['ppl']:.4f}" if r['ppl'] else "  N/A "
		print(f"{r['name']:<20} {ppl_str:>7} {r['mse']:>10.8f} {r['mean_corr']:>+10.6f} {r['eke']:>+10.8f} {r['cross_corr_lag1']:>+10.6f} {r['logit_corr']:>+12.6f} {r['epsilon']:>+10.8f} {r['entropy_change']:>+8.5f}")

	# ================================================================
	# THEORY PREDICTIONS — PASS/FAIL
	# ================================================================
	print("\n" + "=" * 70)
	print("THEORY PREDICTIONS")
	print("=" * 70)

	# Prediction 1: corr(k,e) increases with GLA iterations
	cuda_results = [r for r in all_results if r['name'].startswith('CUDA')]
	if len(cuda_results) >= 2:
		corrs = [(r['name'], r['mean_corr']) for r in cuda_results]
		corrs_sorted = sorted(corrs, key=lambda x: int(x[0].split()[1].split('-')[0]))
		increasing = all(corrs_sorted[i][1] <= corrs_sorted[i+1][1] for i in range(len(corrs_sorted)-1))
		# More lenient: is 30-iter > 3-iter?
		first = [r for r in cuda_results if '3-iter' in r['name']]
		last = [r for r in cuda_results if '30-iter' in r['name'] or '20-iter' in r['name']]
		if first and last:
			trend = last[-1]['mean_corr'] > first[0]['mean_corr']
		else:
			trend = increasing
		status = "CONFIRMED" if trend else "REFUTED"
		print(f"\n  P1: |corr(k,e)| increases with GLA iterations: {status}")
		for n, c in corrs_sorted:
			print(f"      {n}: {c:+.6f}")

	# Prediction 2: corr(q·k, q·e) increases with iterations
	if len(cuda_results) >= 2:
		lcorrs = [(r['name'], r['logit_corr']) for r in cuda_results]
		lcorrs_sorted = sorted(lcorrs, key=lambda x: int(x[0].split()[1].split('-')[0]))
		if first and last:
			trend2 = last[-1]['logit_corr'] > first[0]['logit_corr']
		else:
			trend2 = False
		status2 = "CONFIRMED" if trend2 else "REFUTED"
		print(f"\n  P2: corr(q·k, q·e) increases with iterations: {status2}")
		for n, c in lcorrs_sorted:
			print(f"      {n}: {c:+.6f}")

	# Prediction 3: coset (0-iter) matches scalar baseline
	scalar_r = [r for r in all_results if r['name'] == 'Scalar LM']
	coset_r = [r for r in all_results if r['name'] == 'Coset 0-iter']
	if scalar_r and coset_r:
		diff = abs(coset_r[0]['mean_corr'] - scalar_r[0]['mean_corr'])
		status3 = "CONFIRMED" if diff < 0.01 else "REFUTED"
		print(f"\n  P3: Coset TCQ ≈ scalar (memoryless): {status3}")
		print(f"      Scalar: {scalar_r[0]['mean_corr']:+.6f}")
		print(f"      Coset:  {coset_r[0]['mean_corr']:+.6f}")
		print(f"      Diff:   {diff:.6f}")

	# Prediction 4: temperature epsilon is non-zero and increases
	if len(cuda_results) >= 2:
		eps = [(r['name'], r['epsilon']) for r in cuda_results]
		eps_sorted = sorted(eps, key=lambda x: int(x[0].split()[1].split('-')[0]))
		print(f"\n  P4: Temperature perturbation trend:")
		for n, e in eps_sorted:
			print(f"      {n}: ε = {e:+.8f}")

	# Correlation with PPL
	print("\n  Correlation of metrics with PPL (excluding scalar):")
	ppls = []
	metrics_dict = {}
	for r in all_results:
		if r['ppl'] is None:
			continue
		ppls.append(r['ppl'])
		for key in ['mean_corr', 'eke', 'cross_corr_lag1', 'logit_corr', 'epsilon', 'mse', 'entropy_change']:
			if key not in metrics_dict:
				metrics_dict[key] = []
			metrics_dict[key].append(r[key])

	if len(ppls) >= 3:
		ppls_arr = np.array(ppls)
		for key, vals in metrics_dict.items():
			vals_arr = np.array(vals)
			if np.std(vals_arr) > 0:
				pearson = np.corrcoef(vals_arr, ppls_arr)[0, 1]
				print(f"      {key:<20} Pearson with PPL: {pearson:+.4f}")

	print("\nDone.")

if __name__ == "__main__":
	main()
