#!/usr/bin/env python3
"""Test: Rare state fragility hypothesis.

With more GLA iterations, some trellis states become rare. Their centroids
are trained on few samples → noisy. When these rare states fire on critical
tokens during inference, the reconstruction is poor in ways that aggregate
statistics miss but PPL catches.

Measurements:
1. State frequency distribution (entropy, min, max, Gini coefficient)
2. Number of rare states (< N threshold)
3. Per-state MSE — do rare states have larger errors?
4. Error contribution: what fraction of total error comes from rare states?
5. Worst-case per-block analysis: max error in any block, correlated with rare state usage
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

def gini_coefficient(freq):
	"""Gini coefficient of state frequencies. 0 = perfectly uniform, 1 = maximally skewed."""
	sorted_f = np.sort(freq)
	n = len(sorted_f)
	cumulative = np.cumsum(sorted_f)
	return (2.0 * np.sum((np.arange(1, n+1) * sorted_f)) / (n * np.sum(sorted_f))) - (n + 1) / n

def main():
	print("=" * 70)
	print("RARE STATE FRAGILITY ANALYSIS")
	print("=" * 70)

	print("\nLoading post-FWHT K data...")
	n_vec = 10000
	k_data = np.fromfile("/tmp/turbo_postrot.bin", dtype=np.float32, count=n_vec * T).reshape(n_vec, T).astype(np.float64)
	total_assignments = n_vec * T  # 1,280,000

	old_cb = np.fromfile("/tmp/old_codebook_3bit.bin", dtype=np.float32).astype(np.float64)
	coset_cb = init_coset()

	codebooks = [
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

	# ================================================================
	# ANALYSIS 1: State frequency distribution
	# ================================================================
	print("\n" + "=" * 70)
	print("ANALYSIS 1: STATE FREQUENCY DISTRIBUTION")
	print("=" * 70)
	print(f"  Total state assignments per codebook: {total_assignments:,}")
	print(f"  Uniform average: {total_assignments / N_STATES:.0f} per state")
	print()

	all_results = []
	for name, cb, ppl in codebooks:
		t0 = time.time()
		all_states = []
		for b in range(0, n_vec, 500):
			s = viterbi_batch(k_data[b:b+500], cb, preds)
			all_states.append(s)
		states = np.concatenate(all_states)  # [n_vec, T]
		quantized = cb[states]
		errors = k_data - quantized
		elapsed = time.time() - t0

		# State frequencies
		freq = np.bincount(states.ravel(), minlength=N_STATES).astype(np.float64)
		freq_norm = freq / freq.sum()

		# Distribution statistics
		entropy = -np.sum(freq_norm * np.log2(freq_norm + 1e-30))
		max_entropy = np.log2(N_STATES)  # 9.0 for 512 states
		min_freq = int(np.min(freq))
		max_freq = int(np.max(freq))
		p5_freq = int(np.percentile(freq, 5))
		p95_freq = int(np.percentile(freq, 95))
		gini = gini_coefficient(freq)
		n_zero = int(np.sum(freq == 0))
		n_rare_10 = int(np.sum(freq < 10))
		n_rare_100 = int(np.sum(freq < 100))
		n_rare_500 = int(np.sum(freq < 500))
		cv = np.std(freq) / np.mean(freq)

		mse = np.mean(errors ** 2)

		# Per-state MSE
		state_mse = np.zeros(N_STATES)
		state_count = np.zeros(N_STATES)
		for s in range(N_STATES):
			mask = states.ravel() == s
			if np.any(mask):
				state_mse[s] = np.mean(errors.ravel()[mask] ** 2)
				state_count[s] = np.sum(mask)

		# Rare vs common state error analysis
		# Define "rare" as bottom 10% by frequency
		freq_threshold = np.percentile(freq[freq > 0], 10)
		rare_mask = (freq > 0) & (freq <= freq_threshold)
		common_mask = freq > np.percentile(freq, 90)

		rare_states_idx = np.where(rare_mask)[0]
		common_states_idx = np.where(common_mask)[0]

		rare_mse = np.mean(state_mse[rare_mask]) if np.any(rare_mask) else 0
		common_mse = np.mean(state_mse[common_mask]) if np.any(common_mask) else 0
		rare_to_common = rare_mse / common_mse if common_mse > 0 else 0

		# Error contribution from rare states
		rare_total_err = 0
		rare_total_count = 0
		for s in rare_states_idx:
			mask = states.ravel() == s
			rare_total_err += np.sum(errors.ravel()[mask] ** 2)
			rare_total_count += np.sum(mask)
		rare_fraction = rare_total_count / total_assignments if total_assignments > 0 else 0
		rare_err_fraction = rare_total_err / np.sum(errors ** 2) if np.sum(errors ** 2) > 0 else 0

		# Per-block worst-case: blocks with most rare-state usage
		rare_state_set = set(rare_states_idx)
		rare_per_block = np.zeros(n_vec)
		block_mse = np.zeros(n_vec)
		for i in range(n_vec):
			rare_per_block[i] = sum(1 for s in states[i] if s in rare_state_set)
			block_mse[i] = np.mean(errors[i] ** 2)

		# Correlation between rare-state usage and block MSE
		if np.std(rare_per_block) > 0:
			corr_rare_mse = np.corrcoef(rare_per_block, block_mse)[0, 1]
		else:
			corr_rare_mse = 0.0

		# Top-10% worst blocks: how many rare states do they have?
		worst_blocks = np.argsort(block_mse)[-n_vec // 10:]
		best_blocks = np.argsort(block_mse)[:n_vec // 10]
		rare_in_worst = np.mean(rare_per_block[worst_blocks])
		rare_in_best = np.mean(rare_per_block[best_blocks])

		result = {
			'name': name, 'ppl': ppl, 'mse': mse,
			'entropy': entropy, 'max_entropy': max_entropy,
			'min_freq': min_freq, 'max_freq': max_freq,
			'p5_freq': p5_freq, 'p95_freq': p95_freq,
			'gini': gini, 'cv': cv,
			'n_zero': n_zero, 'n_rare_10': n_rare_10,
			'n_rare_100': n_rare_100, 'n_rare_500': n_rare_500,
			'rare_mse': rare_mse, 'common_mse': common_mse,
			'rare_to_common': rare_to_common,
			'rare_fraction': rare_fraction, 'rare_err_fraction': rare_err_fraction,
			'corr_rare_mse': corr_rare_mse,
			'rare_in_worst': rare_in_worst, 'rare_in_best': rare_in_best,
			'state_mse': state_mse, 'freq': freq,
		}
		all_results.append(result)

		print(f"  {name:<20} ({elapsed:.0f}s)")
		print(f"    MSE:                  {mse:.8f}")
		print(f"    State entropy:        {entropy:.4f} / {max_entropy:.1f} bits  ({entropy/max_entropy*100:.2f}%)")
		print(f"    Freq range:           [{min_freq}, {max_freq}]  (p5={p5_freq}, p95={p95_freq})")
		print(f"    Gini coefficient:     {gini:.6f}")
		print(f"    Freq CV:              {cv:.6f}")
		print(f"    Dead states (0):      {n_zero}")
		print(f"    Rare states (<10):    {n_rare_10}")
		print(f"    Rare states (<100):   {n_rare_100}")
		print(f"    Rare states (<500):   {n_rare_500}")
		print(f"    Rare-state MSE:       {rare_mse:.8f}")
		print(f"    Common-state MSE:     {common_mse:.8f}")
		print(f"    Rare/Common MSE:      {rare_to_common:.4f}x")
		print(f"    Rare % of samples:    {rare_fraction*100:.2f}%")
		print(f"    Rare % of total err:  {rare_err_fraction*100:.2f}%")
		print(f"    corr(rare_count, block_MSE): {corr_rare_mse:+.4f}")
		print(f"    Rare states in worst 10% blocks: {rare_in_worst:.2f}")
		print(f"    Rare states in best 10% blocks:  {rare_in_best:.2f}")
		print()

	# ================================================================
	# ANALYSIS 2: State MSE distribution shape
	# ================================================================
	print("=" * 70)
	print("ANALYSIS 2: PER-STATE MSE DISTRIBUTION")
	print("=" * 70)
	print("  Does the spread of per-state MSE increase with training?")
	print("  If rare states have disproportionately large MSE, this should show up")
	print("  as higher variance/skewness of per-state MSE distribution.")
	print()

	for r in all_results:
		sm = r['state_mse']
		freq = r['freq']
		active = freq > 0
		sm_active = sm[active]

		mean_sm = np.mean(sm_active)
		std_sm = np.std(sm_active)
		cv_sm = std_sm / mean_sm if mean_sm > 0 else 0
		max_sm = np.max(sm_active)
		p99_sm = np.percentile(sm_active, 99)
		skew_sm = np.mean(((sm_active - mean_sm) / std_sm) ** 3) if std_sm > 0 else 0

		# Weighted by frequency: do high-frequency states have lower/higher MSE?
		freq_active = freq[active]
		weighted_corr = np.corrcoef(freq_active, sm_active)[0, 1] if len(freq_active) > 2 else 0

		print(f"  {r['name']:<20}")
		print(f"    Active states:           {np.sum(active)}/512")
		print(f"    Per-state MSE: mean={mean_sm:.8f}  std={std_sm:.8f}  CV={cv_sm:.4f}")
		print(f"    Per-state MSE: max={max_sm:.8f}  p99={p99_sm:.8f}  max/mean={max_sm/mean_sm:.2f}x")
		print(f"    Per-state MSE skewness:  {skew_sm:+.4f}")
		print(f"    corr(freq, state_MSE):   {weighted_corr:+.4f}")
		print()

	# ================================================================
	# ANALYSIS 3: Transition structure
	# ================================================================
	print("=" * 70)
	print("ANALYSIS 3: STATE TRANSITION PATTERNS")
	print("=" * 70)
	print("  Do trained codebooks create more 'specialized' trellis paths?")
	print()

	for r_idx, (name, cb, ppl) in enumerate(codebooks):
		# Requantize to get states (we didn't save them above, recompute on subset)
		n_trans = 2000
		all_states = []
		for b in range(0, n_trans, 500):
			s = viterbi_batch(k_data[b:b+500], cb, preds)
			all_states.append(s)
		states = np.concatenate(all_states)

		# Count transitions
		trans_count = np.zeros((N_STATES, N_STATES), dtype=np.int64)
		for i in range(n_trans):
			for t in range(T - 1):
				trans_count[states[i, t], states[i, t+1]] += 1

		total_trans = trans_count.sum()
		n_possible = 0
		n_used = 0
		for s in range(N_STATES):
			# Possible successors from state s (bitshift trellis)
			mask_lower = (1 << (L - K_BITS)) - 1
			for p in range(N_OUT):
				next_s = ((s & mask_lower) << K_BITS) | p
				n_possible += 1
				if trans_count[s, next_s] > 0:
					n_used += 1

		# Transition entropy (conditional)
		trans_entropy = 0
		for s in range(N_STATES):
			row_sum = trans_count[s].sum()
			if row_sum == 0:
				continue
			for ns in range(N_STATES):
				if trans_count[s, ns] > 0:
					p = trans_count[s, ns] / row_sum
					trans_entropy -= (row_sum / total_trans) * p * np.log2(p)

		# Path diversity: for each block, compute the number of unique states
		unique_states_per_block = np.array([len(set(states[i])) for i in range(n_trans)])
		mean_unique = np.mean(unique_states_per_block)
		min_unique = np.min(unique_states_per_block)

		print(f"  {name:<20}")
		print(f"    Transitions used:       {n_used}/{n_possible} ({n_used/n_possible*100:.1f}%)")
		print(f"    Conditional entropy:     {trans_entropy:.4f} bits  (max {np.log2(N_OUT):.1f})")
		print(f"    Unique states/block:     mean={mean_unique:.1f}  min={min_unique}")
		print()

	# ================================================================
	# SUMMARY
	# ================================================================
	print("=" * 70)
	print("SUMMARY TABLE")
	print("=" * 70)
	header = f"{'Codebook':<20} {'PPL':>7} {'MSE':>10} {'Entropy':>8} {'Gini':>8} {'Dead':>5} {'<100':>5} {'R/C MSE':>8} {'corr':>8} {'MaxStateMSE':>12}"
	print(header)
	print("-" * len(header))
	for r in all_results:
		ppl_str = f"{r['ppl']:.4f}" if r['ppl'] else "  N/A "
		sm_max = np.max(r['state_mse'][r['freq'] > 0])
		print(f"{r['name']:<20} {ppl_str:>7} {r['mse']:>10.8f} {r['entropy']:>8.4f} {r['gini']:>8.6f} {r['n_zero']:>5} {r['n_rare_100']:>5} {r['rare_to_common']:>8.4f} {r['corr_rare_mse']:>+8.4f} {sm_max:>12.8f}")

	# Correlation with PPL
	print("\n  Correlation of state metrics with PPL:")
	ppls = []
	metrics = {}
	for r in all_results:
		if r['ppl'] is None:
			continue
		ppls.append(r['ppl'])
		for key in ['entropy', 'gini', 'cv', 'n_rare_100', 'n_rare_500',
					 'rare_to_common', 'rare_err_fraction', 'corr_rare_mse',
					 'rare_in_worst', 'mse']:
			if key not in metrics:
				metrics[key] = []
			metrics[key].append(r[key])

	if len(ppls) >= 3:
		ppls_arr = np.array(ppls)
		for key, vals in metrics.items():
			vals_arr = np.array(vals)
			if np.std(vals_arr) > 0:
				pearson = np.corrcoef(vals_arr, ppls_arr)[0, 1]
				print(f"    {key:<25} Pearson: {pearson:+.4f}")

	# ================================================================
	# THEORY VERDICT
	# ================================================================
	print("\n" + "=" * 70)
	print("THEORY VERDICT: RARE STATE FRAGILITY")
	print("=" * 70)

	# Check: does state entropy decrease with iterations?
	cuda_results = [r for r in all_results if r['name'].startswith('CUDA')]
	if len(cuda_results) >= 2:
		ents = [(r['name'], r['entropy']) for r in cuda_results]
		ents_sorted = sorted(ents, key=lambda x: int(x[0].split()[1].split('-')[0]))
		print(f"\n  State entropy trend (decreasing = more skewed):")
		for n, e in ents_sorted:
			print(f"    {n}: {e:.4f} bits")

		ginis = [(r['name'], r['gini']) for r in cuda_results]
		ginis_sorted = sorted(ginis, key=lambda x: int(x[0].split()[1].split('-')[0]))
		print(f"\n  Gini coefficient trend (increasing = more skewed):")
		for n, g in ginis_sorted:
			print(f"    {n}: {g:.6f}")

		rares = [(r['name'], r['n_rare_100']) for r in cuda_results]
		rares_sorted = sorted(rares, key=lambda x: int(x[0].split()[1].split('-')[0]))
		print(f"\n  Rare states (<100) trend:")
		for n, r in rares_sorted:
			print(f"    {n}: {r}")

		ratios = [(r['name'], r['rare_to_common']) for r in cuda_results]
		ratios_sorted = sorted(ratios, key=lambda x: int(x[0].split()[1].split('-')[0]))
		print(f"\n  Rare/Common MSE ratio trend:")
		for n, r in ratios_sorted:
			print(f"    {n}: {r:.4f}x")

	print("\nDone.")

if __name__ == "__main__":
	main()
