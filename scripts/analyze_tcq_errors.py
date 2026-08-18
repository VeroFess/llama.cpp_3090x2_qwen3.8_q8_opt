#!/usr/bin/env python3
"""Analyze TCQ quantization error autocorrelation from kernel dump.

Usage:
	1. Run llama-perplexity with TURBO_TCQ_DUMP_ERRORS=N (e.g. 1000)
	2. Copy /tmp/tcq_errors.bin from server
	3. python3 scripts/analyze_tcq_errors.py [tcq_errors.bin] [codebook.bin] [--bits 3]

The codebook.bin should be the same codebook used during quantization
(512 floats for 3-bit, 256 floats for 2-bit).
"""

import sys
import struct
import numpy as np

def load_dump(path):
	with open(path, 'rb') as f:
		n_groups = struct.unpack('i', f.read(4))[0]
		x = np.frombuffer(f.read(n_groups * 128 * 4), dtype=np.float32).reshape(n_groups, 128)
		out = np.frombuffer(f.read(n_groups * 128), dtype=np.uint8).reshape(n_groups, 128)
	return n_groups, x, out

def reconstruct_states_3bit(out, n_states=512):
	"""Reconstruct trellis states from 3-bit output symbols."""
	n_groups, seq_len = out.shape
	states = np.zeros((n_groups, seq_len), dtype=np.int32)
	# We don't know the initial state, but for error analysis we can reconstruct
	# from 3 consecutive outputs (for t >= 2)
	for t in range(seq_len):
		if t < 2:
			# Can't reconstruct without initial state; use output directly
			# state = out[t] << 6 (partial, but codebook lookup is what matters)
			if t == 0:
				states[:, t] = out[:, 0].astype(np.int32) << 6
			else:
				states[:, t] = (out[:, 0].astype(np.int32) & 0x7) | (out[:, 1].astype(np.int32) << 3) | (out[:, 1].astype(np.int32) << 6)
		else:
			states[:, t] = (
				(out[:, t-2].astype(np.int32) & 0x7) |
				((out[:, t-1].astype(np.int32) & 0x7) << 3) |
				((out[:, t].astype(np.int32) & 0x7) << 6)
			)
	return states

def reconstruct_states_2bit(out, n_states=256):
	"""Reconstruct trellis states from 2-bit output symbols."""
	n_groups, seq_len = out.shape
	states = np.zeros((n_groups, seq_len), dtype=np.int32)
	for t in range(seq_len):
		if t < 3:
			# Partial reconstruction
			states[:, t] = 0
			for i in range(min(t+1, 4)):
				if t-3+1+i >= 0:
					states[:, t] |= (out[:, t-3+1+i].astype(np.int32) & 0x3) << (i*2)
		else:
			states[:, t] = (
				(out[:, t-3].astype(np.int32) & 0x3) |
				((out[:, t-2].astype(np.int32) & 0x3) << 2) |
				((out[:, t-1].astype(np.int32) & 0x3) << 4) |
				((out[:, t].astype(np.int32) & 0x3) << 6)
			)
	return states

def compute_errors(x, states, codebook, skip_initial=3):
	"""Compute quantization errors: original - reconstruction."""
	n_groups, seq_len = x.shape
	recon = codebook[states]
	errors = x - recon
	# Skip initial positions where state reconstruction is unreliable
	return errors[:, skip_initial:]

def autocorrelation(errors, max_lag=10):
	"""Compute autocorrelation at lags 1..max_lag, averaged over groups."""
	n_groups, seq_len = errors.shape
	# Normalize each group's errors to zero mean
	errors = errors - errors.mean(axis=1, keepdims=True)
	var = (errors ** 2).mean(axis=1)

	acf = np.zeros(max_lag + 1)
	for lag in range(max_lag + 1):
		if lag == 0:
			acf[0] = 1.0
			continue
		products = (errors[:, :-lag] * errors[:, lag:]).mean(axis=1)
		# Avoid division by zero
		valid = var > 1e-20
		acf[lag] = (products[valid] / var[valid]).mean()
	return acf

def main():
	dump_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/tcq_errors.bin'
	cb_path = sys.argv[2] if len(sys.argv) > 2 else None
	bits = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 3

	if '--bits' in sys.argv:
		idx = sys.argv.index('--bits')
		bits = int(sys.argv[idx + 1])

	print(f"Loading dump from {dump_path} ({bits}-bit TCQ)")
	n_groups, x, out = load_dump(dump_path)
	print(f"Loaded {n_groups} groups, x shape: {x.shape}, out shape: {out.shape}")

	# Load codebook
	if cb_path:
		codebook = np.fromfile(cb_path, dtype=np.float32)
	else:
		# Use compiled-in codebook (hardcoded from turbo-quant-cuda.cuh)
		print("WARNING: No codebook provided, using default 3-bit compiled-in codebook")
		print("For accurate results, provide the codebook used during quantization")
		sys.exit(1)

	n_states = 512 if bits == 3 else 256
	assert len(codebook) == n_states, f"Codebook size {len(codebook)} != expected {n_states}"

	# Reconstruct states
	if bits == 3:
		states = reconstruct_states_3bit(out)
		skip = 2
	else:
		states = reconstruct_states_2bit(out)
		skip = 3

	# Compute errors
	errors = compute_errors(x, states, codebook, skip_initial=skip)
	print(f"\nError statistics (positions {skip}-127):")
	print(f"  Mean:   {errors.mean():.6f}")
	print(f"  Std:    {errors.std():.6f}")
	print(f"  Min:    {errors.min():.6f}")
	print(f"  Max:    {errors.max():.6f}")

	# Autocorrelation
	max_lag = 10
	acf = autocorrelation(errors, max_lag)
	print(f"\nAutocorrelation (theoretical prediction: lag-1 ≈ 0.15-0.30):")
	for lag in range(max_lag + 1):
		bar = '#' * int(abs(acf[lag]) * 50)
		sign = '+' if acf[lag] >= 0 else '-'
		print(f"  lag {lag:2d}: {acf[lag]:+.4f}  {sign}{bar}")

	# Cross-group statistics
	per_group_acf1 = np.zeros(n_groups)
	for g in range(n_groups):
		e = errors[g] - errors[g].mean()
		v = (e ** 2).mean()
		if v > 1e-20:
			per_group_acf1[g] = (e[:-1] * e[1:]).mean() / v
	print(f"\nPer-group lag-1 autocorrelation distribution:")
	print(f"  Mean:   {per_group_acf1.mean():.4f}")
	print(f"  Std:    {per_group_acf1.std():.4f}")
	print(f"  Median: {np.median(per_group_acf1):.4f}")
	print(f"  P5:     {np.percentile(per_group_acf1, 5):.4f}")
	print(f"  P95:    {np.percentile(per_group_acf1, 95):.4f}")

	# Compare with iid baseline: for iid errors, lag-1 acf ≈ -1/(N-1) ≈ -0.008
	iid_expected = -1.0 / (128 - skip - 1)
	print(f"\n  iid baseline: {iid_expected:.4f}")
	print(f"  Observed excess: {per_group_acf1.mean() - iid_expected:.4f}")

	if abs(per_group_acf1.mean()) > 0.05:
		print(f"\n  SIGNIFICANT autocorrelation detected!")
	else:
		print(f"\n  No significant autocorrelation (< 0.05)")

if __name__ == '__main__':
	main()
