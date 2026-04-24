#pragma once
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

#define CUDA_CHECK(condition)                                          \
  do {                                                                 \
    cudaError_t error = condition;                                     \
    if (error != cudaSuccess) {                                        \
      printf(                                                          \
          "CUDA_CHECK error in line %d of file %s \
              : %s \n",                                                \
          __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError())); \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

// #define DEBUG

#ifdef DEBUG
#define DEBUG_BLOCK(expr) \
  do {                    \
    expr                  \
  } while (0)
#else
#define DEBUG_BLOCK(...) \
  do {                   \
  } while (0)
#endif

__global__ void naive_nrow_gemm(float *A, float *B, float *C, float a, float b, int M, int N, int K, int mBlock);
__global__ void row_softmax(float *input, float *output, int n);
__global__ void naive_pv(float *P, float *V, float *O, int M, int N, int mBlock);

bool all_close(float *A, float *B, int m, int n) {
  for (int i = 0; i < m * n; i++) {
    if (fabs(A[i] - B[i]) > 1e-3f) {
      printf("A[%d] = %f, B[%d] = %f\n", i, A[i], i, B[i]);
      return false;
    }
  }
  return true;
}

// print matrix
void print_host_matrix(float *matrix, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f, ", matrix[i * n + j]);
    }
    printf("\n");
  }
}

void print_device_matrix(float *dev_ptr, int m, int n) {
  float *host_ptr = new float[m * n];
  cudaMemcpy(host_ptr, dev_ptr, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f, ", host_ptr[i * n + j]);
    }
    printf("\n");
  }
  free(host_ptr);
}

void self_attention_cuda(float *Q, float *K, float *V, float *O, int m, int n) {
    int mBlock = 2;
    assert(m % mBlock == 0 && "mBlock should align");

    float sm_scale = 1.f / sqrtf(static_cast<float>(n));
    float *sm_o;
    cudaMalloc((void **)&sm_o, sizeof(float) * m * m);

    dim3 qk_block(m / mBlock, 1, 1);
    naive_nrow_gemm<<<1, qk_block>>>(Q, K, sm_o, sm_scale, 0, m, m, n, mBlock);
    cudaDeviceSynchronize();
    DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError()); printf("== naive QK ==\n");
                print_device_matrix(sm_o, m, m););

    // QK[M, M]
    dim3 sm_block(m, 1, 1);
    row_softmax<<<1, sm_block>>>(sm_o, sm_o, m);
    cudaDeviceSynchronize();
    DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError());
                printf("== naive softmax(QK) ==\n");
                print_device_matrix(sm_o, m, m););

    // QK[M, M] @ V[M, N]
    dim3 qkv_block(m / mBlock, 1, 1);
    naive_pv<<<1, qkv_block>>>(sm_o, V, O, m, n, mBlock);
    cudaDeviceSynchronize();
    DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError());
                printf("== naive softmax(QK)V ==\n");
                print_device_matrix(O, m, n););

    cudaFree(sm_o);
}

// naive gemm implement with slice-k
// perform C = aA@B + bC
// A[M, K] x B[K, N] = C[M, N]
// each thread process mblock rows of A
__global__ void naive_nrow_gemm(float *A, float *B, float *C, float a, float b, int M, int N, int K, int mBlock) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // each thread process a range of rows
    idx *= mBlock;

    // A[mBlock, K] x B[N, K].T = C[mBlock, N]
    for (int i = idx; i < idx + mBlock; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            // C[M, N]
            // C = aA@B + bC
            C[i * N + j] = a * sum + b * C[i * N + j];
        }
    }
}

// perform QK[M, M] @ V[M, N]
__global__ void naive_pv(float *P, float *V, float *O, int M, int N, int mBlock) {
    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    // each thread process a range of rows
    idx *= mBlock;
    int K = M;
    // P[mBlock, M] x V[M, N] = O[mBlock, N]
    for (int i = idx; i < idx + mBlock; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                sum += P[i * K + k] * V[k * N + j];
            }
            // C[M, N]
            O[i * N + j] = sum;
        }
    }
}

// each thread process one row of softmax
__global__ void row_softmax(float *input, float *output, int n) {
    // assume id will not exceed row number of input
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    float max = -INFINITY;
    float sum = 0.f;

    // Find max
    for (int i = 0; i < n; i++) {
        if (input[idx * n + i] > max) {
            max = input[idx * n + i];
        }
    }

    // Compute numerator and denominator
    for (int i = 0; i < n; i++) {
        output[idx * n + i] = exp(input[idx * n + i] - max);
        sum += output[idx * n + i];
    }

    // Compute softmax
    for (int i = 0; i < n; i++) {
        output[idx * n + i] /= sum;
    }
}