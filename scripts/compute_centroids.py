#!/usr/bin/env python3
"""Compute empirical Lloyd-Max centroids from post-FWHT KV cache data.

Usage:
	python3 scripts/compute_centroids.py /tmp/turbo_postrot.bin

Reads float32 samples dumped by TURBO_EXTRACT mode and computes
optimal 4-bit (16-level), 3-bit (8-level), and 2-bit (4-level)
Lloyd-Max centroids via iterative conditional-expectation algorithm.
"""
import sys
import struct
import numpy as np
from scipy.stats import norm

def lloyd_max_empirical(samples, n_centroids, n_iter=200):
	"""Iterative Lloyd-Max on empirical data."""
	# Init from quantiles of the sample distribution
	quantiles = np.linspace(0.5/n_centroids, 1 - 0.5/n_centroids, n_centroids)
	centroids = np.quantile(samples, quantiles)

	for it in range(n_iter):
		# Decision boundaries = midpoints
		boundaries = np.concatenate([
			[-np.inf],
			(centroids[:-1] + centroids[1:]) / 2,
			[np.inf]
		])
		# Conditional expectations
		new_centroids = np.zeros(n_centroids)
		for i in range(n_centroids):
			mask = (samples >= boundaries[i]) & (samples < boundaries[i+1])
			if mask.sum() > 0:
				new_centroids[i] = samples[mask].mean()
			else:
				new_centroids[i] = centroids[i]
		if np.allclose(centroids, new_centroids, atol=1e-10):
			break
		centroids = new_centroids

	return centroids

def lloyd_max_gaussian(n_centroids, sigma, n_iter=200):
	"""Analytical Lloyd-Max for N(0, sigma^2)."""
	quantiles = np.linspace(0.5/n_centroids, 1 - 0.5/n_centroids, n_centroids)
	centroids = np.array([sigma * norm.ppf(q) for q in quantiles])

	for it in range(n_iter):
		boundaries = np.concatenate([
			[-np.inf],
			(centroids[:-1] + centroids[1:]) / 2,
			[np.inf]
		])
		new_centroids = np.zeros(n_centroids)
		for i in range(n_centroids):
			a, b = boundaries[i], boundaries[i+1]
			num = sigma * (norm.pdf(a/sigma) - norm.pdf(b/sigma))
			den = norm.cdf(b/sigma) - norm.cdf(a/sigma)
			new_centroids[i] = num / den if den > 1e-15 else centroids[i]
		if np.allclose(centroids, new_centroids, atol=1e-10):
			break
		centroids = new_centroids

	return centroids

def format_c_array(name, values):
	"""Format as C constant array."""
	lines = []
	half = len(values) // 2
	neg = values[:half]
	pos = values[half:]
	neg_str = ", ".join(f"{v:11.6f}f" for v in neg)
	pos_str = ", ".join(f"{v:11.6f}f" for v in pos)
	lines.append(f"// {name}")
	lines.append(f"    {neg_str},")
	lines.append(f"     {pos_str},")
	return "\n".join(lines)

def compute_midpoints(centroids):
	return (centroids[:-1] + centroids[1:]) / 2

def mse_for_centroids(samples, centroids):
	"""Compute MSE of quantizing samples with given centroids."""
	midpoints = compute_midpoints(centroids)
	indices = np.searchsorted(midpoints, samples)
	recon = centroids[indices]
	return np.mean((samples - recon) ** 2)

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(f"Usage: {sys.argv[0]} <postrot.bin>")
		sys.exit(1)

	path = sys.argv[1]
	with open(path, "rb") as f:
		data = f.read()

	n_samples = len(data) // 4
	samples = np.frombuffer(data, dtype=np.float32)
	print(f"Loaded {n_samples:,} samples from {path}")
	print(f"  mean:     {samples.mean():.6f}")
	print(f"  std:      {samples.std():.6f}")
	print(f"  expected: {1/np.sqrt(128):.6f}")
	print(f"  kurtosis: {float(np.mean((samples/samples.std())**4)):.3f}  (Gaussian=3.0)")
	print(f"  min/max:  {samples.min():.6f} / {samples.max():.6f}")
	print()

	sigma = 1.0 / np.sqrt(128)

	for n_bits, n_cent in [(4, 16), (3, 8), (2, 4)]:
		print(f"=== {n_bits}-bit ({n_cent} centroids) ===")

		# Empirical Lloyd-Max
		c_emp = lloyd_max_empirical(samples, n_cent)
		mse_emp = mse_for_centroids(samples, c_emp)

		# Gaussian Lloyd-Max (current)
		c_gauss = lloyd_max_gaussian(n_cent, sigma)
		mse_gauss = mse_for_centroids(samples, c_gauss)

		improvement = (mse_gauss - mse_emp) / mse_gauss * 100

		print(f"  Gaussian Lloyd-Max MSE: {mse_gauss:.8f}")
		print(f"  Empirical Lloyd-Max MSE: {mse_emp:.8f}")
		print(f"  Improvement: {improvement:+.2f}%")
		print()

		print("  Gaussian centroids:")
		print(f"    {format_c_array('gaussian', c_gauss)}")
		print()
		print("  Empirical centroids:")
		print(f"    {format_c_array('empirical', c_emp)}")
		print()

		# Show per-centroid comparison
		print(f"  {'Idx':>3} {'Gaussian':>11} {'Empirical':>11} {'Diff%':>8}")
		for i in range(n_cent):
			diff_pct = (c_emp[i] - c_gauss[i]) / abs(c_gauss[i]) * 100 if abs(c_gauss[i]) > 1e-10 else 0
			print(f"  {i+1:3d} {c_gauss[i]:11.6f} {c_emp[i]:11.6f} {diff_pct:+8.2f}%")
		print()

		# Midpoints for the empirical centroids (needed for quantization)
		mid_emp = compute_midpoints(c_emp)
		mid_str = ", ".join(f"{v:11.6f}f" for v in mid_emp)
		print(f"  Empirical midpoints ({n_cent-1} values):")
		print(f"    {mid_str}")
		print()
