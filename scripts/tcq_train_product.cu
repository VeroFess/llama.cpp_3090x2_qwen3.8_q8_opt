// Product-aware CUDA TCQ codebook training
// Theory: Ordentlich & Polyanskiy "Optimal Quantization for Matrix Multiplication"
// (arXiv:2410.13780) proves MSE-optimal quantizers are NOT optimal for approximating
// A^T B. The product-aware distortion metric weights errors by the query covariance
// structure: D_product = e^T Σ_q e, not just ||e||².
//
// Three training modes:
//   mse       — standard MSE (baseline, same as tcq_train_cuda.cu)
//   isotropy  — adaptive per-position weights to equalize error across dimensions
//              (ensures Σ_e ≈ σ²I, which the theory says minimizes product distortion)
//   qweights  — explicit per-position weights from Q covariance diagonal file
//
// Evaluation always reports both MSE and product distortion (using anisotropic Q).
//
// Compile: nvcc -O3 -arch=sm_86 -o tcq_train_product tcq_train_product.cu -lcurand
// Usage:   ./tcq_train_product --bits 3 --n-train 100000 --n-iters 200 --mode isotropy

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <cstdint>
#include <curand.h>
#include <cuda_runtime.h>
#include "tcq_diagnostics.cuh"

#define CHECK_CUDA(call) do { \
	cudaError_t err = (call); \
	if (err != cudaSuccess) { \
		fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		exit(1); \
	} \
} while(0)

#define CHECK_CURAND(call) do { \
	curandStatus_t err = (call); \
	if (err != CURAND_STATUS_SUCCESS) { \
		fprintf(stderr, "cuRAND error at %s:%d: %d\n", __FILE__, __LINE__, err); \
		exit(1); \
	} \
} while(0)

#define T 128
#define MAX_STATES 1024

__constant__ float d_codebook[MAX_STATES];
__constant__ float d_weights[T];    // per-position weights for product-aware training
__constant__ float d_q_scale[T];    // Q anisotropy profile for product evaluation

// ============================================================================
// Weighted Viterbi: one threadblock per sample, one thread per state
// Cost at position t = w[t] * (x_t - codebook[state])²
// ============================================================================
template<int K, int L>
__global__ void k_viterbi_encode(
	const float* __restrict__ data,
	int16_t*     __restrict__ states_out,
	float*       __restrict__ mse_out,
	float*       __restrict__ wmse_out,   // weighted MSE (NULL if not needed)
	int n_train
) {
	const int sample = blockIdx.x;
	if (sample >= n_train) return;

	constexpr int N_STATES = 1 << L;
	constexpr int N_OUT = 1 << K;
	constexpr int MASK_LOWER = (1 << (L - K)) - 1;

	const int sid = threadIdx.x;
	if (sid >= N_STATES) return;

	const float* x = data + sample * T;

	extern __shared__ char smem_raw[];
	float* cost     = (float*)smem_raw;
	float* new_cost = cost + N_STATES;
	uint8_t* bt = (uint8_t*)(new_cost + N_STATES);

	cost[sid] = 0.0f;
	__syncthreads();

	for (int t = 0; t < T; t++) {
		float x_t = x[t];
		float cb_val = d_codebook[sid];
		float raw_dist = (x_t - cb_val) * (x_t - cb_val);
		float dist = d_weights[t] * raw_dist;

		float best_cost = FLT_MAX;
		int best_p = 0;

		#pragma unroll
		for (int p = 0; p < N_OUT; p++) {
			int prev_s = ((sid & MASK_LOWER) << K) | p;
			float c = cost[prev_s] + dist;
			if (c < best_cost) {
				best_cost = c;
				best_p = p;
			}
		}

		new_cost[sid] = best_cost;

		int bt_idx = t * (N_STATES / 2) + sid / 2;
		if (sid % 2 == 0) {
			bt[bt_idx] = (uint8_t)(best_p & 0xF);
		}
		__syncthreads();
		if (sid % 2 == 1) {
			bt[bt_idx] |= ((uint8_t)(best_p & 0xF)) << 4;
		}
		__syncthreads();

		float tmp = new_cost[sid];
		cost[sid] = tmp;
		__syncthreads();
	}

	if (sid == 0) {
		float best = FLT_MAX;
		int best_state = 0;
		for (int s = 0; s < N_STATES; s++) {
			if (cost[s] < best) {
				best = cost[s];
				best_state = s;
			}
		}

		int16_t* out_states = states_out + sample * T;
		float mse = 0.0f;
		float wmse = 0.0f;
		int state = best_state;
		for (int t = T - 1; t >= 0; t--) {
			out_states[t] = (int16_t)state;
			float recon = d_codebook[state];
			float diff = x[t] - recon;
			float sq = diff * diff;
			mse += sq;
			wmse += d_weights[t] * sq;

			int bt_idx = t * (N_STATES / 2) + state / 2;
			int p;
			if (state % 2 == 0) {
				p = bt[bt_idx] & 0xF;
			} else {
				p = (bt[bt_idx] >> 4) & 0xF;
			}
			state = ((state & MASK_LOWER) << K) | p;
		}

		mse_out[sample] = mse / T;
		if (wmse_out) wmse_out[sample] = wmse / T;
	}
}

// ============================================================================
// Weighted centroid collection: accumulates w[t]*x and w[t] per state
// ============================================================================
__global__ void k_collect_centroids_weighted(
	const float*   __restrict__ data,
	const int16_t* __restrict__ states,
	double*        __restrict__ state_wsums,   // Σ w*x per state
	double*        __restrict__ state_waccum,  // Σ w per state
	int*           __restrict__ state_counts,  // raw count (for dead state detection)
	int n_train
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = n_train * T;
	if (idx >= total) return;

	int t = idx % T;
	int state = (int)(unsigned short)states[idx];
	float val = data[idx];
	float w = d_weights[t];

	atomicAdd(&state_wsums[state], (double)(w * val));
	atomicAdd(&state_waccum[state], (double)w);
	atomicAdd(&state_counts[state], 1);
}

// ============================================================================
// Per-position MSE: measures error isotropy across the T=128 positions
// ============================================================================
__global__ void k_position_mse(
	const float*   __restrict__ data,
	const int16_t* __restrict__ states,
	double*        __restrict__ pos_sum_sq,
	int n_train
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = n_train * T;
	if (idx >= total) return;

	int t = idx % T;
	int state = (int)(unsigned short)states[idx];
	float err = data[idx] - d_codebook[state];

	atomicAdd(&pos_sum_sq[t], (double)(err * err));
}

// ============================================================================
// Product distortion evaluation: E[(q·e)²] where q ~ N(0, Σ_q), e = x - x̂
// One block per sample, parallel reduction for dot product
// ============================================================================
__global__ void k_product_eval(
	const float*   __restrict__ data,
	const float*   __restrict__ queries,
	const int16_t* __restrict__ states,
	float*         __restrict__ product_err,
	int n_eval
) {
	const int sample = blockIdx.x;
	if (sample >= n_eval) return;

	extern __shared__ float s_reduce[];

	const float* k = data + sample * T;
	const float* q = queries + sample * T;
	const int16_t* s = states + sample * T;

	// each thread handles T/blockDim.x positions
	float local_qdot = 0.0f;
	for (int t = threadIdx.x; t < T; t += blockDim.x) {
		int state = (int)(unsigned short)s[t];
		float err = k[t] - d_codebook[state];
		local_qdot += q[t] * err;
	}

	s_reduce[threadIdx.x] = local_qdot;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride)
			s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		product_err[sample] = s_reduce[0] * s_reduce[0];
	}
}

// ============================================================================
// Scale query vectors by anisotropy profile: q[t] *= scale[t]
// ============================================================================
__global__ void k_scale_queries(float* queries, int n_samples) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n_samples * T) return;
	int t = idx % T;
	queries[idx] *= d_q_scale[t];
}

// ============================================================================
// Host code
// ============================================================================

static const float LM_2BIT[] = {-1.510f, -0.4528f, 0.4528f, 1.510f};
static const float LM_3BIT[] = {-1.748f, -1.050f, -0.5006f, -0.06971f, 0.06971f, 0.5006f, 1.050f, 1.748f};

void init_coset_codebook(float* codebook, int k, int L, float sigma, const float* centroids) {
	int n_out = 1 << k;
	int n_groups = 1 << (L - k);

	float spacing = (centroids[1] - centroids[0]) * sigma;
	for (int group = 0; group < n_groups; group++) {
		float shift = spacing * ((float)group / n_groups - 0.5f);
		for (int pos = 0; pos < n_out; pos++) {
			int state = (group << k) | pos;
			codebook[state] = centroids[pos] * sigma + shift;
		}
	}
}

// generate synthetic Q anisotropy profile: exponential decay with given effective rank
void make_q_profile(float* scale, int effective_rank) {
	// eigenvalues λ_i = exp(-i / effective_rank), so scale = sqrt(λ)
	float sum = 0.0f;
	for (int i = 0; i < T; i++) {
		scale[i] = expf(-0.5f * (float)i / effective_rank);
		sum += scale[i] * scale[i];
	}
	// normalize so total variance = T * sigma²
	float norm = sqrtf((float)T / sum);
	for (int i = 0; i < T; i++) scale[i] *= norm;
}

enum TrainMode { MODE_MSE, MODE_ISOTROPY, MODE_QWEIGHTS };

template<int K, int L>
void train(int n_train, int n_iters, int n_restarts, TrainMode mode,
           const char* init_file, const char* qweight_file, const char* out_file,
           const char* data_file, int q_rank, int base_seed,
           const char* output_dir, bool constrain_mono) {
	constexpr int N_STATES = 1 << L;
	constexpr int N_GROUPS = 1 << (L - K);
	const float sigma = 1.0f / sqrtf(128.0f);
	const float* centroids = (K == 2) ? LM_2BIT : LM_3BIT;

	if (output_dir) { mkdir(output_dir, 0755); printf("Output dir: %s\n", output_dir); }
	if (constrain_mono) printf("Monotonicity constraint: ENABLED\n");

	// load real data from model dump if provided
	float* h_real_data = nullptr;
	int real_data_n = 0;
	if (data_file) {
		FILE* fp = fopen(data_file, "rb");
		if (!fp) { fprintf(stderr, "Cannot open data file: %s\n", data_file); exit(1); }

		fseek(fp, 0, SEEK_END);
		long file_size = ftell(fp);
		fseek(fp, 0, SEEK_SET);

		// auto-detect: header format (int32 count + data) or raw format (just floats)
		int32_t maybe_count;
		fread(&maybe_count, sizeof(int32_t), 1, fp);

		int count;
		if (maybe_count > 0 && maybe_count < 100000000 &&
		    (long)maybe_count * T * sizeof(float) + sizeof(int32_t) == file_size) {
			count = maybe_count;
			printf("Data file has header: %d blocks\n", count);
		} else {
			fseek(fp, 0, SEEK_SET);
			count = file_size / (T * sizeof(float));
			printf("Data file (raw): %d blocks (%.1f MB)\n", count, file_size / 1e6);
		}

		if (count > n_train) count = n_train;
		real_data_n = count;
		n_train = count;
		h_real_data = (float*)malloc((size_t)count * T * sizeof(float));
		size_t read = fread(h_real_data, sizeof(float), (size_t)count * T, fp);
		fclose(fp);
		if ((int)(read / T) < count) {
			fprintf(stderr, "Data file truncated: got %d blocks, expected %d\n", (int)(read/T), count);
			count = (int)(read / T);
			real_data_n = count;
			n_train = count;
		}
		printf("Loaded %d real K blocks from %s\n", real_data_n, data_file);
	}

	printf("Mode: %s, n_train=%d%s\n", mode == MODE_MSE ? "mse" :
	       mode == MODE_ISOTROPY ? "isotropy" : "qweights",
	       n_train, data_file ? " (real data)" : " (synthetic)");

	// initialize per-position weights
	float h_weights[T];
	for (int i = 0; i < T; i++) h_weights[i] = 1.0f;

	if (mode == MODE_QWEIGHTS && qweight_file) {
		FILE* fp = fopen(qweight_file, "rb");
		if (!fp) { fprintf(stderr, "Cannot open Q-weight file: %s\n", qweight_file); exit(1); }
		if (fread(h_weights, sizeof(float), T, fp) != T) {
			fprintf(stderr, "Q-weight file must contain %d floats\n", T); exit(1);
		}
		fclose(fp);
		// normalize so mean weight = 1
		float wsum = 0;
		for (int i = 0; i < T; i++) wsum += h_weights[i];
		for (int i = 0; i < T; i++) h_weights[i] *= T / wsum;
		printf("Loaded Q-weights from %s (range: %.3f - %.3f)\n",
		       qweight_file, h_weights[0], h_weights[T-1]);
	}

	CHECK_CUDA(cudaMemcpyToSymbol(d_weights, h_weights, T * sizeof(float)));

	// Q anisotropy profile for product evaluation
	float h_q_scale[T];
	make_q_profile(h_q_scale, q_rank);
	CHECK_CUDA(cudaMemcpyToSymbol(d_q_scale, h_q_scale, T * sizeof(float)));
	printf("Q anisotropy: effective rank %d (scale range: %.3f - %.3f)\n",
	       q_rank, h_q_scale[0], h_q_scale[T-1]);

	// allocate GPU memory
	float* d_data;
	int16_t* d_states;
	float *d_mse, *d_wmse;
	double *d_state_wsums, *d_pos_sum_sq;
	double* d_state_waccum;
	int* d_state_counts;
	float *d_queries, *d_product_err;

	CHECK_CUDA(cudaMalloc(&d_data, (size_t)n_train * T * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&d_states, (size_t)n_train * T * sizeof(int16_t)));
	CHECK_CUDA(cudaMalloc(&d_mse, (size_t)n_train * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&d_wmse, (size_t)n_train * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&d_state_wsums, N_STATES * sizeof(double)));
	CHECK_CUDA(cudaMalloc(&d_state_waccum, N_STATES * sizeof(double)));
	CHECK_CUDA(cudaMalloc(&d_state_counts, N_STATES * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&d_pos_sum_sq, T * sizeof(double)));
	CHECK_CUDA(cudaMalloc(&d_queries, (size_t)n_train * T * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&d_product_err, (size_t)n_train * sizeof(float)));

	float* h_codebook = (float*)malloc(N_STATES * sizeof(float));
	float* h_mse = (float*)malloc(n_train * sizeof(float));
	double* h_wsums = (double*)malloc(N_STATES * sizeof(double));
	double* h_waccum = (double*)malloc(N_STATES * sizeof(double));
	int* h_counts = (int*)malloc(N_STATES * sizeof(int));
	double* h_pos_mse = (double*)malloc(T * sizeof(double));
	float* h_product_err = (float*)malloc(n_train * sizeof(float));

	curandGenerator_t gen;
	CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

	size_t smem_viterbi = 2 * N_STATES * sizeof(float) + T * (N_STATES / 2) * sizeof(uint8_t);
	printf("Shared memory per block: %zu bytes\n", smem_viterbi);
	if (smem_viterbi > 48 * 1024) {
		CHECK_CUDA(cudaFuncSetAttribute(
			k_viterbi_encode<K, L>,
			cudaFuncAttributeMaxDynamicSharedMemorySize,
			smem_viterbi));
	}

	float best_global_mse = FLT_MAX;
	float best_global_product = FLT_MAX;
	float best_global_codebook[MAX_STATES];

	for (int restart = 0; restart < n_restarts; restart++) {
		if (n_restarts > 1) printf("\n=== Restart %d/%d ===\n", restart+1, n_restarts);

		// reset weights for each restart
		for (int i = 0; i < T; i++) h_weights[i] = 1.0f;
		if (mode == MODE_QWEIGHTS && qweight_file) {
			FILE* fp = fopen(qweight_file, "rb");
			if (fp) {
				fread(h_weights, sizeof(float), T, fp);
				fclose(fp);
				float wsum = 0;
				for (int i = 0; i < T; i++) wsum += h_weights[i];
				for (int i = 0; i < T; i++) h_weights[i] *= T / wsum;
			}
		}
		CHECK_CUDA(cudaMemcpyToSymbol(d_weights, h_weights, T * sizeof(float)));

		// initialize codebook
		if (init_file && restart == 0) {
			FILE* fp = fopen(init_file, "rb");
			if (!fp) { fprintf(stderr, "Cannot open init file: %s\n", init_file); exit(1); }
			if (fread(h_codebook, sizeof(float), N_STATES, fp) != (size_t)N_STATES) {
				fprintf(stderr, "Init file too small\n"); exit(1);
			}
			fclose(fp);
			printf("Loaded initial codebook from %s\n", init_file);
		} else {
			init_coset_codebook(h_codebook, K, L, sigma, centroids);
		}
		memcpy(best_global_codebook, h_codebook, N_STATES * sizeof(float));
		if (restart > 0) {
			srand(restart * 7919);
			for (int s = 0; s < N_STATES; s++) {
				h_codebook[s] += ((float)rand() / RAND_MAX - 0.5f) * sigma * 0.04f;
			}
		}

		float best_mse = FLT_MAX;
		float best_product = FLT_MAX;
		float best_codebook[MAX_STATES];
		memcpy(best_codebook, h_codebook, N_STATES * sizeof(float));
		int stall = 0;

		// upload real data once before iteration loop
		if (h_real_data) {
			CHECK_CUDA(cudaMemcpy(d_data, h_real_data, (size_t)n_train * T * sizeof(float), cudaMemcpyHostToDevice));
		}

		for (int iter = 0; iter < n_iters; iter++) {
			CHECK_CUDA(cudaMemcpyToSymbol(d_codebook, h_codebook, N_STATES * sizeof(float)));

			if (!h_real_data) {
				// generate fresh synthetic K data
				CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, base_seed + restart * 10000 + iter));
				CHECK_CURAND(curandGenerateNormal(gen, d_data, (size_t)n_train * T, 0.0f, sigma));
			}

			// generate Q vectors for product eval (always synthetic)
			CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, base_seed + 50000 + restart * 10000 + iter));
			CHECK_CURAND(curandGenerateNormal(gen, d_queries, (size_t)n_train * T, 0.0f, sigma));
			{
				int threads = 256;
				int blocks = ((int)(n_train * T) + threads - 1) / threads;
				k_scale_queries<<<blocks, threads>>>(d_queries, n_train);
			}

			// Viterbi encode
			k_viterbi_encode<K, L><<<n_train, N_STATES, smem_viterbi>>>(
				d_data, d_states, d_mse, d_wmse, n_train);
			CHECK_CUDA(cudaGetLastError());

			// compute mean MSE
			CHECK_CUDA(cudaMemcpy(h_mse, d_mse, n_train * sizeof(float), cudaMemcpyDeviceToHost));
			double mean_mse = 0.0;
			for (int i = 0; i < n_train; i++) mean_mse += h_mse[i];
			mean_mse /= n_train;

			// compute product distortion
			{
				size_t smem_prod = 128 * sizeof(float);
				k_product_eval<<<n_train, 128, smem_prod>>>(
					d_data, d_queries, d_states, d_product_err, n_train);
				CHECK_CUDA(cudaGetLastError());
			}
			CHECK_CUDA(cudaMemcpy(h_product_err, d_product_err, n_train * sizeof(float), cudaMemcpyDeviceToHost));
			double mean_product = 0.0;
			for (int i = 0; i < n_train; i++) mean_product += h_product_err[i];
			mean_product /= n_train;

			// track best by product distortion if product-aware, else by MSE
			bool improved = false;
			if (mode == MODE_MSE) {
				if (mean_mse < best_mse) { improved = true; stall = 0; }
				else stall++;
			} else {
				if (mean_product < best_product) { improved = true; stall = 0; }
				else stall++;
			}
			if (improved) {
				best_mse = mean_mse;
				best_product = mean_product;
				memcpy(best_codebook, h_codebook, N_STATES * sizeof(float));
			}

			// per-position MSE for isotropy measurement + adaptive weights
			CHECK_CUDA(cudaMemset(d_pos_sum_sq, 0, T * sizeof(double)));
			{
				int threads = 256;
				int blocks = ((int)(n_train * T) + threads - 1) / threads;
				k_position_mse<<<blocks, threads>>>(d_data, d_states, d_pos_sum_sq, n_train);
			}
			CHECK_CUDA(cudaMemcpy(h_pos_mse, d_pos_sum_sq, T * sizeof(double), cudaMemcpyDeviceToHost));

			double pos_mean = 0, pos_max = 0, pos_min = 1e30;
			for (int t = 0; t < T; t++) {
				double pm = h_pos_mse[t] / n_train;
				pos_mean += pm;
				if (pm > pos_max) pos_max = pm;
				if (pm < pos_min) pos_min = pm;
			}
			pos_mean /= T;
			double pos_cv = 0;
			for (int t = 0; t < T; t++) {
				double pm = h_pos_mse[t] / n_train;
				pos_cv += (pm - pos_mean) * (pm - pos_mean);
			}
			pos_cv = sqrt(pos_cv / T) / pos_mean; // coefficient of variation

			// collect weighted centroids
			CHECK_CUDA(cudaMemset(d_state_wsums, 0, N_STATES * sizeof(double)));
			CHECK_CUDA(cudaMemset(d_state_waccum, 0, N_STATES * sizeof(double)));
			CHECK_CUDA(cudaMemset(d_state_counts, 0, N_STATES * sizeof(int)));
			{
				int threads = 256;
				int blocks = ((int)(n_train * T) + threads - 1) / threads;
				k_collect_centroids_weighted<<<blocks, threads>>>(
					d_data, d_states, d_state_wsums, d_state_waccum, d_state_counts, n_train);
			}

			CHECK_CUDA(cudaMemcpy(h_wsums, d_state_wsums, N_STATES * sizeof(double), cudaMemcpyDeviceToHost));
			CHECK_CUDA(cudaMemcpy(h_waccum, d_state_waccum, N_STATES * sizeof(double), cudaMemcpyDeviceToHost));
			CHECK_CUDA(cudaMemcpy(h_counts, d_state_counts, N_STATES * sizeof(int), cudaMemcpyDeviceToHost));

			int used = 0;
			for (int s = 0; s < N_STATES; s++) {
				if (h_counts[s] > 0) {
					h_codebook[s] = (float)(h_wsums[s] / h_waccum[s]);
					used++;
				}
			}

			// reinitialize dead states
			if (used < N_STATES && iter < n_iters - 1) {
				for (int s = 0; s < N_STATES; s++) {
					if (h_counts[s] == 0) {
						int donor = 0;
						for (int d = 1; d < N_STATES; d++) {
							if (h_counts[d] > h_counts[donor]) donor = d;
						}
						h_codebook[s] = h_codebook[donor] + ((float)rand() / RAND_MAX - 0.5f) * sigma * 0.02f;
					}
				}
			}

			// apply monotonicity constraint if enabled
			if (constrain_mono) {
				apply_monotonicity_constraint<K>(h_codebook, N_GROUPS);
			}

			// adaptive isotropy: update weights for next iteration
			if (mode == MODE_ISOTROPY && iter < n_iters - 1) {
				for (int t = 0; t < T; t++) {
					float pm = (float)(h_pos_mse[t] / n_train);
					float ratio = pm / (float)pos_mean;
					// blend: 70% adaptive + 30% uniform to prevent oscillation
					h_weights[t] = 0.7f * ratio + 0.3f;
				}
				CHECK_CUDA(cudaMemcpyToSymbol(d_weights, h_weights, T * sizeof(float)));
			}

			printf("  iter %3d: MSE=%.6f  prod=%.3e  isotropy=%.3f  (%d/%d used)",
			       iter+1, (float)mean_mse, mean_product, pos_cv, used, N_STATES);
			print_diagnostics<K>(h_codebook, h_counts, N_STATES);
			printf("%s\n", improved ? " *" : "");
			fflush(stdout);

			// save codebook for this iteration
			if (output_dir) {
				char fname[512];
				snprintf(fname, sizeof(fname), "%s/cb_iter%03d.bin", output_dir, iter + 1);
				save_codebook(h_codebook, N_STATES, fname);
			}

			if (stall >= 10 && iter > 20) {
				printf("  Converged (10 iters without improvement)\n");
				break;
			}
		}

		if (mode == MODE_MSE) {
			if (best_mse < best_global_mse) {
				best_global_mse = best_mse;
				best_global_product = best_product;
				memcpy(best_global_codebook, best_codebook, N_STATES * sizeof(float));
			}
		} else {
			if (best_product < best_global_product) {
				best_global_mse = best_mse;
				best_global_product = best_product;
				memcpy(best_global_codebook, best_codebook, N_STATES * sizeof(float));
			}
		}
	}

	// ========================================================================
	// Final evaluation on fresh data
	// ========================================================================
	printf("\n=== Final evaluation (10000 fresh samples) ===\n");

	CHECK_CUDA(cudaMemcpyToSymbol(d_codebook, best_global_codebook, N_STATES * sizeof(float)));
	// reset weights to uniform for fair MSE comparison
	float uniform_w[T];
	for (int i = 0; i < T; i++) uniform_w[i] = 1.0f;
	CHECK_CUDA(cudaMemcpyToSymbol(d_weights, uniform_w, T * sizeof(float)));

	int n_eval = 10000;
	float *d_eval_data, *d_eval_queries, *d_eval_mse, *d_eval_product;
	int16_t* d_eval_states;
	CHECK_CUDA(cudaMalloc(&d_eval_data, (size_t)n_eval * T * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&d_eval_queries, (size_t)n_eval * T * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&d_eval_states, (size_t)n_eval * T * sizeof(int16_t)));
	CHECK_CUDA(cudaMalloc(&d_eval_mse, (size_t)n_eval * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&d_eval_product, (size_t)n_eval * sizeof(float)));

	CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 99999));
	CHECK_CURAND(curandGenerateNormal(gen, d_eval_data, (size_t)n_eval * T, 0.0f, sigma));

	// Viterbi with uniform weights (pure MSE)
	k_viterbi_encode<K, L><<<n_eval, N_STATES, smem_viterbi>>>(
		d_eval_data, d_eval_states, d_eval_mse, NULL, n_eval);
	CHECK_CUDA(cudaGetLastError());

	float* h_eval_mse = (float*)malloc(n_eval * sizeof(float));
	CHECK_CUDA(cudaMemcpy(h_eval_mse, d_eval_mse, n_eval * sizeof(float), cudaMemcpyDeviceToHost));
	double eval_mse = 0.0;
	for (int i = 0; i < n_eval; i++) eval_mse += h_eval_mse[i];
	eval_mse /= n_eval;

	// product eval with isotropic Q
	CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 77777));
	CHECK_CURAND(curandGenerateNormal(gen, d_eval_queries, (size_t)n_eval * T, 0.0f, sigma));
	{
		size_t smem_prod = 128 * sizeof(float);
		k_product_eval<<<n_eval, 128, smem_prod>>>(
			d_eval_data, d_eval_queries, d_eval_states, d_eval_product, n_eval);
	}
	float* h_eval_prod = (float*)malloc(n_eval * sizeof(float));
	CHECK_CUDA(cudaMemcpy(h_eval_prod, d_eval_product, n_eval * sizeof(float), cudaMemcpyDeviceToHost));
	double eval_prod_iso = 0.0;
	for (int i = 0; i < n_eval; i++) eval_prod_iso += h_eval_prod[i];
	eval_prod_iso /= n_eval;

	// product eval with anisotropic Q (using configured rank)
	CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 88888));
	CHECK_CURAND(curandGenerateNormal(gen, d_eval_queries, (size_t)n_eval * T, 0.0f, sigma));
	{
		int threads = 256;
		int blocks = ((int)(n_eval * T) + threads - 1) / threads;
		k_scale_queries<<<blocks, threads>>>(d_eval_queries, n_eval);
	}
	{
		size_t smem_prod = 128 * sizeof(float);
		k_product_eval<<<n_eval, 128, smem_prod>>>(
			d_eval_data, d_eval_queries, d_eval_states, d_eval_product, n_eval);
	}
	CHECK_CUDA(cudaMemcpy(h_eval_prod, d_eval_product, n_eval * sizeof(float), cudaMemcpyDeviceToHost));
	double eval_prod_aniso = 0.0;
	for (int i = 0; i < n_eval; i++) eval_prod_aniso += h_eval_prod[i];
	eval_prod_aniso /= n_eval;

	// per-position MSE for final isotropy check
	double* d_eval_pos;
	CHECK_CUDA(cudaMalloc(&d_eval_pos, T * sizeof(double)));
	CHECK_CUDA(cudaMemset(d_eval_pos, 0, T * sizeof(double)));
	{
		int threads = 256;
		int blocks = ((int)(n_eval * T) + threads - 1) / threads;
		k_position_mse<<<blocks, threads>>>(d_eval_data, d_eval_states, d_eval_pos, n_eval);
	}
	double h_eval_pos[T];
	CHECK_CUDA(cudaMemcpy(h_eval_pos, d_eval_pos, T * sizeof(double), cudaMemcpyDeviceToHost));
	double epos_mean = 0, epos_max = 0, epos_min = 1e30;
	for (int t = 0; t < T; t++) {
		double pm = h_eval_pos[t] / n_eval;
		epos_mean += pm;
		if (pm > epos_max) epos_max = pm;
		if (pm < epos_min) epos_min = pm;
	}
	epos_mean /= T;
	double epos_cv = 0;
	for (int t = 0; t < T; t++) {
		double pm = h_eval_pos[t] / n_eval;
		epos_cv += (pm - epos_mean) * (pm - epos_mean);
	}
	epos_cv = sqrt(epos_cv / T) / epos_mean;

	printf("MSE:               %.6f\n", (float)eval_mse);
	printf("Product (iso Q):   %.3e\n", eval_prod_iso);
	printf("Product (aniso Q): %.3e  (rank=%d)\n", eval_prod_aniso, q_rank);
	printf("Error isotropy CV: %.4f  (max/min=%.2f)\n", epos_cv, epos_max/epos_min);

	// position MSE profile (first and last few positions)
	printf("\nPer-position MSE profile:\n  pos  0-7:  ");
	for (int t = 0; t < 8; t++) printf("%.5f ", (float)(h_eval_pos[t]/n_eval));
	printf("\n  pos 60-67: ");
	for (int t = 60; t < 68; t++) printf("%.5f ", (float)(h_eval_pos[t]/n_eval));
	printf("\n  pos120-127:");
	for (int t = 120; t < 128; t++) printf("%.5f ", (float)(h_eval_pos[t]/n_eval));
	printf("\n");

	// save binary codebook
	if (out_file) {
		FILE* fp = fopen(out_file, "wb");
		if (fp) {
			fwrite(best_global_codebook, sizeof(float), N_STATES, fp);
			fclose(fp);
			printf("\nSaved codebook to %s (%d floats)\n", out_file, N_STATES);
		}
	}

	// print C arrays
	printf("\n// Product-aware TCQ codebook: k=%d, L=%d (%d states)\n", K, L, N_STATES);
	printf("// Mode: %s, MSE=%.6f, Product(aniso,rank=%d)=%.3e, isotropy CV=%.4f\n",
	       mode == MODE_MSE ? "mse" : mode == MODE_ISOTROPY ? "isotropy" : "qweights",
	       (float)eval_mse, q_rank, eval_prod_aniso, epos_cv);

	const char* suffixes[] = {"", "_fattn"};
	for (int v = 0; v < 2; v++) {
		printf("static __constant__ float d_turbo%d_tcq_codebook%s[%d] = {\n", K, suffixes[v], N_STATES);
		for (int i = 0; i < N_STATES; i += 8) {
			printf("    ");
			for (int j = i; j < i + 8 && j < N_STATES; j++) {
				printf("%+.8ff", best_global_codebook[j]);
				if (j < N_STATES - 1) {
					printf(",");
					if ((j + 1) % 8 != 0) printf(" ");
				}
			}
			printf("\n");
		}
		printf("};\n\n");
	}

	// cleanup
	if (h_real_data) free(h_real_data);
	free(h_codebook); free(h_mse); free(h_wsums); free(h_waccum);
	free(h_counts); free(h_pos_mse); free(h_product_err);
	free(h_eval_mse); free(h_eval_prod);
	cudaFree(d_data); cudaFree(d_states); cudaFree(d_mse); cudaFree(d_wmse);
	cudaFree(d_state_wsums); cudaFree(d_state_waccum); cudaFree(d_state_counts);
	cudaFree(d_pos_sum_sq); cudaFree(d_queries); cudaFree(d_product_err);
	cudaFree(d_eval_data); cudaFree(d_eval_queries); cudaFree(d_eval_states);
	cudaFree(d_eval_mse); cudaFree(d_eval_product); cudaFree(d_eval_pos);
	curandDestroyGenerator(gen);
}

int main(int argc, char** argv) {
	int bits = 3;
	int n_train = 100000;
	int n_iters = 200;
	int n_restarts = 3;
	int seed = 42;
	int q_rank = 5;  // default: effective rank 5 (matching our aggregate Q measurement)
	TrainMode mode = MODE_MSE;
	const char* init_file = nullptr;
	const char* qweight_file = nullptr;
	const char* out_file = nullptr;
	const char* data_file = nullptr;
	const char* output_dir = nullptr;
	bool constrain_mono = false;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--bits") == 0 && i+1 < argc) bits = atoi(argv[++i]);
		else if (strcmp(argv[i], "--n-train") == 0 && i+1 < argc) n_train = atoi(argv[++i]);
		else if (strcmp(argv[i], "--n-iters") == 0 && i+1 < argc) n_iters = atoi(argv[++i]);
		else if (strcmp(argv[i], "--n-restarts") == 0 && i+1 < argc) n_restarts = atoi(argv[++i]);
		else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) seed = atoi(argv[++i]);
		else if (strcmp(argv[i], "--init") == 0 && i+1 < argc) init_file = argv[++i];
		else if (strcmp(argv[i], "--out") == 0 && i+1 < argc) out_file = argv[++i];
		else if (strcmp(argv[i], "--data-file") == 0 && i+1 < argc) data_file = argv[++i];
		else if (strcmp(argv[i], "--q-weights") == 0 && i+1 < argc) qweight_file = argv[++i];
		else if (strcmp(argv[i], "--q-rank") == 0 && i+1 < argc) q_rank = atoi(argv[++i]);
		else if (strcmp(argv[i], "--output-dir") == 0 && i+1 < argc) output_dir = argv[++i];
		else if (strcmp(argv[i], "--constrain-monotonicity") == 0) constrain_mono = true;
		else if (strcmp(argv[i], "--mode") == 0 && i+1 < argc) {
			i++;
			if (strcmp(argv[i], "mse") == 0) mode = MODE_MSE;
			else if (strcmp(argv[i], "isotropy") == 0) mode = MODE_ISOTROPY;
			else if (strcmp(argv[i], "qweights") == 0) mode = MODE_QWEIGHTS;
			else { fprintf(stderr, "Unknown mode: %s (mse|isotropy|qweights)\n", argv[i]); return 1; }
		}
	}

	printf("=== Product-aware CUDA TCQ codebook training ===\n");
	printf("bits=%d, n_train=%d, n_iters=%d, n_restarts=%d, seed=%d\n",
	       bits, n_train, n_iters, n_restarts, seed);
	if (init_file) printf("init=%s\n", init_file);
	if (out_file) printf("out=%s\n", out_file);
	if (data_file) printf("data=%s\n", data_file);
	if (output_dir) printf("output_dir=%s\n", output_dir);
	if (constrain_mono) printf("monotonicity constraint: ON\n");
	printf("\n");

	if (bits == 2) {
		train<2, 8>(n_train, n_iters, n_restarts, mode, init_file, qweight_file, out_file, data_file, q_rank, seed, output_dir, constrain_mono);
	} else if (bits == 3) {
		train<3, 9>(n_train, n_iters, n_restarts, mode, init_file, qweight_file, out_file, data_file, q_rank, seed, output_dir, constrain_mono);
	} else {
		fprintf(stderr, "Unsupported bits: %d\n", bits);
		return 1;
	}

	return 0;
}
