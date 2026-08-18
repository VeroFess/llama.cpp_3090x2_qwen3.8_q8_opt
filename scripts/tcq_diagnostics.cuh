// Shared diagnostic utilities for TCQ codebook trainers
// Used by tcq_train_cuda.cu and tcq_train_product.cu
#pragma once

#include <cstdio>
#include <climits>
#include <sys/stat.h>

// Count groups with monotonically increasing codebook entries
template<int K>
int compute_monotonicity(const float* codebook, int n_groups) {
	constexpr int N_OUT = 1 << K;
	int mono = 0;
	for (int g = 0; g < n_groups; g++) {
		bool ok = true;
		for (int p = 1; p < N_OUT; p++) {
			if (codebook[(g << K) | p] <= codebook[(g << K) | (p - 1)]) {
				ok = false;
				break;
			}
		}
		if (ok) mono++;
	}
	return mono;
}

// Largest out-of-order distance within any group
template<int K>
float compute_max_crossover(const float* codebook, int n_groups) {
	constexpr int N_OUT = 1 << K;
	float max_cross = 0.0f;
	for (int g = 0; g < n_groups; g++) {
		for (int p = 1; p < N_OUT; p++) {
			float diff = codebook[(g << K) | (p - 1)] - codebook[(g << K) | p];
			if (diff > max_cross) max_cross = diff;
		}
	}
	return max_cross;
}

// State balance statistics
inline void compute_state_balance(const int* counts, int n_states, int* out_min, int* out_max, double* out_mean) {
	*out_min = INT_MAX;
	*out_max = 0;
	double sum = 0.0;
	for (int s = 0; s < n_states; s++) {
		if (counts[s] < *out_min) *out_min = counts[s];
		if (counts[s] > *out_max) *out_max = counts[s];
		sum += counts[s];
	}
	*out_mean = sum / n_states;
}

// Print all diagnostic metrics
template<int K>
void print_diagnostics(const float* codebook, const int* counts, int n_states) {
	int n_groups = n_states >> K;

	int mono = compute_monotonicity<K>(codebook, n_groups);
	float max_cross = compute_max_crossover<K>(codebook, n_groups);

	int bmin, bmax;
	double bmean;
	compute_state_balance(counts, n_states, &bmin, &bmax, &bmean);

	printf("  diag: mono=%d/%d  crossover=%.6f  balance=%d/%d/%.0f(%.1fx)",
		   mono, n_groups, max_cross, bmin, bmax, bmean,
		   (bmin > 0) ? (double)bmax / bmin : 0.0);
}

// PAVA isotonic regression — enforce monotonic non-decreasing within each group
template<int K>
void apply_monotonicity_constraint(float* codebook, int n_groups) {
	constexpr int N_OUT = 1 << K;
	float vals[16]; // max N_OUT we'd ever use

	for (int g = 0; g < n_groups; g++) {
		for (int p = 0; p < N_OUT; p++)
			vals[p] = codebook[(g << K) | p];

		// PAVA: pool adjacent violators using block-based approach
		int block_start[16], block_end[16];
		float block_val[16];
		int n_blocks = 0;

		for (int i = 0; i < N_OUT; i++) {
			block_start[n_blocks] = i;
			block_end[n_blocks] = i;
			block_val[n_blocks] = vals[i];
			n_blocks++;

			while (n_blocks >= 2 && block_val[n_blocks - 2] > block_val[n_blocks - 1]) {
				float sum = block_val[n_blocks - 2] * (block_end[n_blocks - 2] - block_start[n_blocks - 2] + 1)
				          + block_val[n_blocks - 1] * (block_end[n_blocks - 1] - block_start[n_blocks - 1] + 1);
				int count = block_end[n_blocks - 1] - block_start[n_blocks - 2] + 1;
				block_end[n_blocks - 2] = block_end[n_blocks - 1];
				block_val[n_blocks - 2] = sum / count;
				n_blocks--;
			}
		}

		for (int b = 0; b < n_blocks; b++) {
			for (int i = block_start[b]; i <= block_end[b]; i++) {
				vals[i] = block_val[b];
			}
		}

		for (int p = 0; p < N_OUT; p++)
			codebook[(g << K) | p] = vals[p];
	}
}

// Save codebook to binary file
inline void save_codebook(const float* codebook, int n_states, const char* path) {
	FILE* fp = fopen(path, "wb");
	if (!fp) {
		fprintf(stderr, "WARNING: Cannot write codebook to %s\n", path);
		return;
	}
	fwrite(codebook, sizeof(float), n_states, fp);
	fclose(fp);
}
