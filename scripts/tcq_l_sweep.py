#!/usr/bin/env python3
"""Sweep trellis depth L for 2-bit TCQ to find the quality/speed sweet spot.

Tests L=6,7,8,9,10 for k=2 (2-bit) and L=7,8,9,10,11 for k=3 (3-bit).
Reports MSE reduction vs Lloyd-Max baseline for each.
"""

import numpy as np
import sys
import time

# Import from vectorized trainer
sys.path.insert(0, '.')
from scripts.tcq_train_vectorized import (
	build_predecessor_table, init_coset_codebook, viterbi_batch,
	train_codebook, evaluate, lloyd_max_baseline,
	LLOYD_MAX_2BIT, LLOYD_MAX_3BIT
)


def sweep(k, L_values, sigma=1.0, n_train=1000, n_iters=20, n_restarts=1):
	if k == 2:
		centroids = LLOYD_MAX_2BIT
	else:
		centroids = LLOYD_MAX_3BIT

	lm_mse = lloyd_max_baseline(centroids, sigma, n_eval=2000)
	print(f"\n{'='*60}")
	print(f"{k}-bit TCQ L sweep (Lloyd-Max baseline MSE = {lm_mse:.6f})")
	print(f"n_train={n_train}, n_iters={n_iters}, n_restarts={n_restarts}")
	print(f"{'='*60}\n")

	results = []
	for L in L_values:
		n_states = 1 << L
		prefix_bits = L - k
		total_bits = prefix_bits + 128 * k
		total_bytes = (total_bits + 7) // 8
		bpv = (2 + total_bytes) / 128 * 8  # 2 bytes norm + qs bytes

		print(f"\n--- L={L}: {n_states} states, prefix={prefix_bits}b, "
			  f"block={total_bytes}+2 bytes, {bpv:.3f} bpv ---")

		t0 = time.time()
		codebook, train_mse = train_codebook(
			k, L, sigma, centroids,
			n_train=n_train, n_iters=n_iters,
			n_restarts=n_restarts, batch_size=500
		)
		elapsed = time.time() - t0

		np.random.seed(54321)
		eval_mse = evaluate(codebook, k, L, sigma, n_eval=2000)
		reduction = (1 - eval_mse / lm_mse) * 100
		db_gain = 10 * np.log10(lm_mse / eval_mse)

		results.append({
			'L': L, 'states': n_states, 'bpv': bpv,
			'mse': eval_mse, 'reduction': reduction, 'db': db_gain,
			'time': elapsed
		})

		print(f"  EVAL: MSE = {eval_mse:.6f} ({reduction:+.1f}% vs LM, {db_gain:+.2f} dB) [{elapsed:.0f}s]")

	print(f"\n{'='*60}")
	print(f"Summary ({k}-bit TCQ L sweep)")
	print(f"{'='*60}")
	print(f"{'L':>3} {'States':>7} {'bpv':>6} {'MSE':>10} {'Reduction':>10} {'dB':>7} {'Time':>7}")
	print(f"{'-'*3:>3} {'-'*7:>7} {'-'*6:>6} {'-'*10:>10} {'-'*10:>10} {'-'*7:>7} {'-'*7:>7}")
	for r in results:
		print(f"{r['L']:3d} {r['states']:7d} {r['bpv']:6.3f} {r['mse']:10.6f} {r['reduction']:+9.1f}% {r['db']:+6.2f} {r['time']:6.0f}s")


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--bits', type=int, default=2, choices=[2, 3])
	parser.add_argument('--n-train', type=int, default=1000)
	parser.add_argument('--n-iters', type=int, default=20)
	parser.add_argument('--n-restarts', type=int, default=1)
	parser.add_argument('--seed', type=int, default=42)
	args = parser.parse_args()

	np.random.seed(args.seed)

	if args.bits == 2:
		sweep(k=2, L_values=[6, 7, 8, 9, 10],
			  n_train=args.n_train, n_iters=args.n_iters, n_restarts=args.n_restarts)
	else:
		sweep(k=3, L_values=[7, 8, 9, 10, 11],
			  n_train=args.n_train, n_iters=args.n_iters, n_restarts=args.n_restarts)
