#!/usr/bin/env python3
"""Temporal Decay Prototype — validate 3->2 bit requantization quality.

Adapted from TheTom/turboquant_plus/benchmarks/temporal_decay_prototype.py
Self-contained: no external dependencies beyond numpy.

Tests whether progressive requantization (turbo3 -> effective 2-bit) preserves
enough quality for old KV cache tokens.
"""

import numpy as np

# 3-bit centroids (Lloyd-Max, d=128 Gaussian)
CENTROIDS_3BIT = np.array([
	-0.190685, -0.117832, -0.065717, -0.021460,
	 0.021460,  0.065717,  0.117832,  0.190685
])

# 2-bit centroids (Lloyd-Max, d=128 Gaussian)
CENTROIDS_2BIT = np.array([
	-0.133462, -0.039994, 0.039994, 0.133462
])

MIDPOINTS_3BIT = np.array([
	-0.154259, -0.091775, -0.043589, 0.0, 0.043589, 0.091775, 0.154259
])

MIDPOINTS_2BIT = np.array([
	-0.086728, 0.0, 0.086728
])


def fwht_128(x):
	"""In-place Fast Walsh-Hadamard Transform, length 128, with 1/sqrt(128) normalization."""
	n = len(x)
	h = 1
	while h < n:
		for i in range(0, n, h * 2):
			for j in range(i, i + h):
				a = x[j]
				b = x[j + h]
				x[j] = a + b
				x[j + h] = a - b
		h *= 2
	x *= 1.0 / np.sqrt(n)
	return x


def rotate_forward(x, signs1, signs2):
	"""signs1 -> FWHT -> signs2 (matching TurboQuant rotation)."""
	out = x.copy()
	out *= signs1
	fwht_128(out)
	out *= signs2
	return out


def rotate_inverse(x, signs1, signs2):
	"""signs2 -> FWHT -> signs1 (inverse rotation, FWHT is self-inverse)."""
	out = x.copy()
	out *= signs2
	fwht_128(out)
	out *= signs1
	return out


def quantize_3bit(x_normalized):
	return np.digitize(x_normalized, MIDPOINTS_3BIT).astype(np.uint8)


def quantize_2bit(x_normalized):
	return np.digitize(x_normalized, MIDPOINTS_2BIT).astype(np.uint8)


def dequantize_3bit(indices, norm):
	centroids = CENTROIDS_3BIT[indices]
	recon_norm = np.sqrt(np.sum(centroids ** 2))
	corrected_norm = norm / recon_norm if recon_norm > 1e-10 else norm
	return centroids * corrected_norm


def dequantize_2bit(indices, norm):
	centroids = CENTROIDS_2BIT[indices]
	recon_norm = np.sqrt(np.sum(centroids ** 2))
	corrected_norm = norm / recon_norm if recon_norm > 1e-10 else norm
	return centroids * corrected_norm


def requantize_3to2(indices_3bit, norm_3bit):
	"""Core of temporal decay: dequant 3-bit -> requant to 2-bit."""
	values = dequantize_3bit(indices_3bit, norm_3bit)
	recon_norm = np.linalg.norm(values)
	if recon_norm > 1e-10:
		normalized = values / recon_norm
	else:
		normalized = values
		recon_norm = 0.0

	indices_2bit = quantize_2bit(normalized)

	centroids_2bit = CENTROIDS_2BIT[indices_2bit]
	centroid_norm = np.sqrt(np.sum(centroids_2bit ** 2))
	corrected_norm = recon_norm / centroid_norm if centroid_norm > 1e-10 else recon_norm

	return indices_2bit, corrected_norm


def cosine_similarity(a, b):
	dot = np.dot(a, b)
	na = np.linalg.norm(a)
	nb = np.linalg.norm(b)
	if na < 1e-10 or nb < 1e-10:
		return 0.0
	return dot / (na * nb)


def run_synthetic_test(d=128, n_vectors=1000, seed=42):
	print(f"\n{'='*60}")
	print(f"SYNTHETIC TEST: d={d}, n_vectors={n_vectors}")
	print(f"{'='*60}\n")

	rng = np.random.default_rng(seed)

	# Generate random sign arrays (matching TurboQuant's approach)
	signs1 = rng.choice([-1.0, 1.0], size=d).astype(np.float32)
	signs2 = rng.choice([-1.0, 1.0], size=d).astype(np.float32)

	cos_sims_3bit = []
	cos_sims_2bit_direct = []
	cos_sims_decay = []
	mse_3bit = []
	mse_2bit = []
	mse_decay = []

	for i in range(n_vectors):
		x = rng.standard_normal(d).astype(np.float32)
		norm = np.linalg.norm(x)
		if norm < 1e-10:
			continue
		x_normalized = x / norm

		x_rotated = rotate_forward(x_normalized, signs1, signs2)

		# Path A: 3-bit (current turbo3)
		indices_3bit = quantize_3bit(x_rotated)
		recon_3bit_rotated = dequantize_3bit(indices_3bit, norm)
		recon_3bit = rotate_inverse(recon_3bit_rotated / norm, signs1, signs2) * norm

		# Path B: Direct 2-bit (theoretical best)
		indices_2bit_direct = quantize_2bit(x_rotated)
		recon_2bit_rotated = dequantize_2bit(indices_2bit_direct, norm)
		recon_2bit = rotate_inverse(recon_2bit_rotated / norm, signs1, signs2) * norm

		# Path C: Temporal decay (3-bit -> requant to 2-bit)
		indices_decay, norm_decay = requantize_3to2(indices_3bit, norm)
		recon_decay_rotated = dequantize_2bit(indices_decay, norm_decay)
		recon_decay = rotate_inverse(recon_decay_rotated / norm_decay, signs1, signs2) * norm_decay

		cos_sims_3bit.append(cosine_similarity(x, recon_3bit))
		cos_sims_2bit_direct.append(cosine_similarity(x, recon_2bit))
		cos_sims_decay.append(cosine_similarity(x, recon_decay))

		mse_3bit.append(np.mean((x - recon_3bit) ** 2))
		mse_2bit.append(np.mean((x - recon_2bit) ** 2))
		mse_decay.append(np.mean((x - recon_decay) ** 2))

	cs3 = np.mean(cos_sims_3bit)
	cs2 = np.mean(cos_sims_2bit_direct)
	csd = np.mean(cos_sims_decay)
	m3 = np.mean(mse_3bit)
	m2 = np.mean(mse_2bit)
	md = np.mean(mse_decay)

	print(f"{'Method':<25} {'Cosine Sim':>12} {'MSE':>12} {'vs 3-bit':>10}")
	print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*10}")
	print(f"{'turbo3 (3-bit)':<25} {cs3:>12.6f} {m3:>12.6f} {'baseline':>10}")
	print(f"{'Direct 2-bit':<25} {cs2:>12.6f} {m2:>12.6f} {m2/m3:>10.2f}x")
	print(f"{'Decay 3->2 (requant)':<25} {csd:>12.6f} {md:>12.6f} {md/m3:>10.2f}x")
	print()

	print("Quality Assessment:")
	if csd > 0.80:
		print(f"  OK  Decay cosine sim {csd:.4f} > 0.80 threshold -- VIABLE")
	else:
		print(f"  BAD Decay cosine sim {csd:.4f} < 0.80 threshold -- TOO LOSSY")

	if csd > cs2 * 0.95:
		print(f"  OK  Decay within 5% of direct 2-bit -- requant doesn't add much error")
	else:
		gap = (1 - csd/cs2) * 100
		print(f"  WARN Decay {gap:.1f}% worse than direct 2-bit -- double-quant error")

	# Inner product preservation
	print(f"\nInner Product Preservation (attention score proxy):")
	ip_errors_3bit = []
	ip_errors_decay = []
	for i in range(min(100, n_vectors)):
		x = rng.standard_normal(d).astype(np.float32)
		q = rng.standard_normal(d).astype(np.float32)
		norm = np.linalg.norm(x)
		if norm < 1e-10:
			continue
		x_rot = rotate_forward(x / norm, signs1, signs2)

		ip_true = np.dot(x, q)

		idx3 = quantize_3bit(x_rot)
		r3 = dequantize_3bit(idx3, norm)
		r3_unrot = rotate_inverse(r3 / norm, signs1, signs2) * norm
		ip_3bit = np.dot(r3_unrot, q)

		idx_d, norm_d = requantize_3to2(idx3, norm)
		rd = dequantize_2bit(idx_d, norm_d)
		rd_unrot = rotate_inverse(rd / norm_d, signs1, signs2) * norm_d
		ip_decay = np.dot(rd_unrot, q)

		ip_errors_3bit.append(abs(ip_3bit - ip_true) / (abs(ip_true) + 1e-10))
		ip_errors_decay.append(abs(ip_decay - ip_true) / (abs(ip_true) + 1e-10))

	print(f"  3-bit mean relative error: {np.mean(ip_errors_3bit):.4f}")
	print(f"  Decay mean relative error: {np.mean(ip_errors_decay):.4f}")
	print(f"  Decay/3-bit error ratio:   {np.mean(ip_errors_decay)/np.mean(ip_errors_3bit):.2f}x")

	return {
		"cosine_3bit": cs3, "cosine_2bit": cs2, "cosine_decay": csd,
		"mse_3bit": m3, "mse_2bit": m2, "mse_decay": md,
		"ip_error_3bit": np.mean(ip_errors_3bit),
		"ip_error_decay": np.mean(ip_errors_decay),
	}


def run_memory_savings_estimate():
	print(f"\n{'='*60}")
	print(f"MEMORY SAVINGS ESTIMATE (Qwen3.5-27B: 40 layers, 4 KV heads, d=256)")
	print(f"{'='*60}\n")

	n_layers = 40
	n_kv_heads = 4
	d_head = 256

	turbo3_bpe = 3.5 / 8
	turbo2_bpe = 2.0 / 8

	for context in [32768, 65536, 131072, 262144]:
		kv_no_decay = context * n_layers * n_kv_heads * d_head * 2 * turbo3_bpe

		recent = min(4096, context)
		old = context - recent

		# Decay 32 of 40 layers, keep first4+last4 at turbo3
		decay_layers = 32
		no_decay_layers = 8

		kv_decay = (
			recent * n_layers * n_kv_heads * d_head * 2 * turbo3_bpe +
			old * decay_layers * n_kv_heads * d_head * 2 * turbo2_bpe +
			old * no_decay_layers * n_kv_heads * d_head * 2 * turbo3_bpe
		)

		savings_pct = (1 - kv_decay / kv_no_decay) * 100
		print(f"  {context//1024:>4}K context: "
			f"no decay={kv_no_decay/1024/1024:.1f} MB, "
			f"with decay={kv_decay/1024/1024:.1f} MB, "
			f"savings={savings_pct:.0f}%")

	print()
	print("  Savings increase with context because the 'old' token fraction grows.")


if __name__ == "__main__":
	print("TurboQuant Temporal Decay Prototype")
	print("=" * 60)

	synthetic = run_synthetic_test(d=128, n_vectors=1000)
	run_memory_savings_estimate()

	print(f"\n{'='*60}")
	print("SUMMARY")
	print(f"{'='*60}")
	print(f"  Synthetic cosine sim: 3-bit={synthetic['cosine_3bit']:.4f}, "
		f"decay={synthetic['cosine_decay']:.4f}")
	print(f"  Inner product error:  3-bit={synthetic['ip_error_3bit']:.4f}, "
		f"decay={synthetic['ip_error_decay']:.4f} "
		f"({synthetic['ip_error_decay']/synthetic['ip_error_3bit']:.1f}x)")
	print()

	if synthetic['cosine_decay'] > 0.80:
		print("  VIABLE -- proceed to CUDA implementation")
	elif synthetic['cosine_decay'] > 0.70:
		print("  MARGINAL -- may work for non-critical layers, test with PPL")
	else:
		print("  TOO LOSSY -- need a different approach")
