#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>    // for fabsf
#include <fstream>  // for CSV output
#include <iostream>
#include <vector>

#define BLOCK_SIZE 128
#define TOL 1e-5f

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (*(reinterpret_cast<const float4 *>(&(pointer))))     // 返回一个 float4
#define FETCH_FLOAT4_VAR(pointer) (*(reinterpret_cast<float4 *>(&(pointer))))

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}

void checkCublasError(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << msg << " CUBLAS ERROR: " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}

/*
    该版本没有引入缓存机制。
    V6 将线程块处理逻辑进一步细化到 Warp 级别
*/

template <const int BM, const int BN, const int BK, const int a_move,
          const int b_move>
__device__ void load_from_gmem(int N, int K, const float *A, const float *B,
                               float *As, float *Bs, int a_px,
                               int a_py, int b_px,
                               int b_py) {
  for(int i = 0;i < BM;i += a_move) {
    float4 tmp = FETCH_FLOAT4(A[OFFSET(a_px + i, a_py, K)]);
    As[OFFSET(a_py, a_px + i, BM)] = tmp.x;
    As[OFFSET(a_py + 1, a_px + i, BM)] = tmp.y;
    As[OFFSET(a_py + 2, a_px + i, BM)] = tmp.z;
    As[OFFSET(a_py + 3, a_px + i, BM)] = tmp.w;
  }
  for(int i = 0;i < BK;i += b_move) {
    float4 tmp = FETCH_FLOAT4(B[OFFSET(b_px + i, b_py, N)]); 
    FETCH_FLOAT4_VAR(Bs[OFFSET(b_px + i, b_py, BN)]) = tmp;
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void process_from_smem(float *reg_a, float *reg_b,
                                  float *thread_results, const float *As,
                                  const float *Bs, const int warp_idx_x,
                                  const int warp_idx_y,
                                  const int thread_x_in_warp,
                                  const int thread_y_in_warp) {
  //  每一个线程需要计算 WMITER * WNITER 个 TM * TN 数据
  for(int bk_idx = 0;bk_idx < BK;bk_idx++) {
    for(int w_r = 0;w_r < WMITER;w_r++) {
      for(int m = 0;m < TM;m++) {
        reg_a[w_r * TM + m] = 
        As[OFFSET(bk_idx, warp_idx_x * WM + w_r * WSUBM +thread_x_in_warp * TM + m, BM)];
      }
    }
    for(int w_c = 0;w_c < WNITER;w_c++) {
      for(int n = 0;n < TN;n++) {
        reg_b[w_c * TN + n] = 
        Bs[OFFSET(bk_idx, warp_idx_y * WN + w_c * WSUBN + thread_y_in_warp * TN + n, BN)];
      }
    }
    for(int w_r = 0;w_r < WMITER;w_r++) {
      for(int w_c = 0;w_c < WNITER;w_c++) {
        for(int m = 0;m < TM;m++) {
          for(int n = 0;n < TN;n++) {
            thread_results[OFFSET((w_r * TM + m), (w_c * TN + n), WNITER * TN)] += 
            reg_a[w_r * TM + m] * reg_b[w_c * TN + n];
          }
        }
      }
    }
  }
}

constexpr int WARP_SIZE = 32;

/*
  将线程块划分为若干个 warp 块，每一个 warp 块内划分出多个 Warp Tile，
  每一个 Warp Tile 内划分出 warpSize 个 thread Tile，每一个线程处理一个 thread Tile...
*/

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) mysgemm_warptiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
  
  
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  //  每一个warp 处理一个形状为 WM * WN 的子矩阵 
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int warp_idx_y = warp_idx % (BN / WN);
  const int warp_idx_x = warp_idx / (BN / WN);

  //  每一个warp子矩阵进一步划分为 Warp Tile (形状为 WSUBM * WSUBN)
  constexpr int WMITER = (WM * WN) / (WARP_SIZE * TM * TN * WNITER);
  constexpr int WSUBM = WM / WMITER;
  constexpr int WSUBN = WN / WNITER;

  //  每一个线程在 Warp Tile 内部的位置
  const int thread_idx_in_warp = threadIdx.x % WARP_SIZE;
  const int thread_y_in_warp = thread_idx_in_warp % (WSUBN / TN);
  const int thread_x_in_warp = thread_idx_in_warp / (WSUBN / TN);

  A = &A[bx * BM * K];
  B = &B[by * BN];
  C = &C[(bx * BM + warp_idx_x * WM) * N + (by * BN + warp_idx_y * WN)];

  __shared__ float As[BK * BM];
  __shared__ float Bs[BK * BN];

  //  针对每一个线程，存储它所有的计算元素，因此存储的数据量为 WMITER * WNITER * TM * TN
  float thread_results[WMITER * WNITER * TM * TN] = { 0.0 };
  float reg_a[WMITER * TM];
  float reg_b[WNITER * TN];

  //  加载 As 和 Bs，加载是线程块级别的
  int a_py = (threadIdx.x % (BK / 4)) * 4;
  int a_px = threadIdx.x / (BK / 4);
  const int a_move = NUM_THREADS / (BK / 4);

  int b_py = (threadIdx.x % (BN / 4)) * 4;
  int b_px = threadIdx.x / (BN / 4);
  const int b_move = NUM_THREADS / (BN / 4);

  for(int bk_idx = 0;bk_idx < K;bk_idx += BK) {
    load_from_gmem<BM, BN, BK, a_move, b_move> (
      N, K, A, B, As, Bs, a_px, a_py, b_px, b_py
    );
    __syncthreads();
    process_from_smem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
      reg_a, reg_b, thread_results, As, Bs, warp_idx_x, warp_idx_y, thread_x_in_warp, thread_y_in_warp
    );
    A += BK;
    B += BK * N;
    __syncthreads();
  }

  for(int w_r = 0;w_r < WMITER;w_r++) {
    for(int w_c = 0;w_c < WNITER;w_c++) {
      float* tmpC = C + w_r * WSUBM * N + w_c * WSUBN;    //  定位到具体的 Warp Tile
      for(int m = 0;m < TM;m++) {
        for(int n = 0;n < TN;n += 4) {
          //  thread_results 的读取位置为：(w_r * TM + m) * (TN * WNITER) + (w_c * TN + n)
          const int strat = OFFSET(w_r * TM + m, w_c * TN + n, WNITER * TN);
          //  tmpC 的读取位置为：(thread_x_in_warp * TM + m, thread_y_in_warp * TN + n)
          float4 tmp = FETCH_FLOAT4_VAR(tmpC[OFFSET(thread_x_in_warp * TM + m, thread_y_in_warp * TN + n, N)]);
          tmp.x = beta * tmp.x;
          tmp.y = beta * tmp.y;
          tmp.z = beta * tmp.z;
          tmp.w = beta * tmp.w;
          tmp.x += alpha * thread_results[strat + 0];
          tmp.y += alpha * thread_results[strat + 1];
          tmp.z += alpha * thread_results[strat + 2];
          tmp.w += alpha * thread_results[strat + 3];
          FETCH_FLOAT4_VAR(tmpC[OFFSET(thread_x_in_warp * TM + m, thread_y_in_warp * TN + n, N)]) = tmp;
        }
      }
    }
  }
}

std::vector<int> generateSizes() {
  std::vector<int> sizes;
  //  8192
  for (int i = 256; i <= 8192; i += 256) {
    sizes.push_back(i);
  }
  return sizes;
}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)
int main() {
  std::vector<int> sizes = generateSizes();

  // 打开CSV文件
  std::ofstream csv_file("sgemm_benchmark_v7.csv");
  csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched,Ratio" << std::endl;

  for (int N : sizes) {
    std::cout << "Testing size: " << N << std::endl;

    size_t size = N * N * sizeof(float);
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C_cublas = (float *)malloc(size);
    float *C_v1 = (float *)malloc(size);

    float *d_A, *d_B, *d_C_v1;
    checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
    checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
    checkCudaError(cudaMalloc(&d_C_v1, size), "cudaMalloc d_C_v1 failed");

    bool out_of_memory = false;

    try {
      // 初始化矩阵 A 和 B
      for (int i = 0; i < N * N; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
      }

      // 拷贝到设备
      checkCudaError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice),
                     "cudaMemcpy A to device failed");
      checkCudaError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice),
                     "cudaMemcpy B to device failed");

      cublasHandle_t handle;
      checkCublasError(cublasCreate(&handle), "cublasCreate failed");

      float alpha = 1.0f;
      float beta = 0.0f;

      cudaEvent_t start, stop;
      checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
      checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

      // warmup
      int warpup_time = 0;  // 热身次数
      for (int i = 0; i < warpup_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                         "cublasSgemm failed");
      }
      cudaDeviceSynchronize();

      // cuBLAS SGEMM
      int repeat_time = 1;
      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start cublas) failed");
      for (int i = 0; i < repeat_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                         "cublasSgemm failed");
      }

      checkCudaError(cudaEventRecord(stop),
                     "cudaEventRecord(stop cublas) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize cublas failed");

      float cublas_time = 0;
      checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                     "cudaEventElapsedTime cublas failed");

      // 拷贝 cuBLAS 结果
      checkCudaError(cudaMemcpy(C_cublas, d_C_v1, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_cublas failed");

      // mysgemm_v1
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

      const uint K10_NUM_THREADS = 128;
      const uint K10_BN = 128;
      const uint K10_BM = 128;
      const uint K10_BK = 16;
      const uint K10_WN = 64;
      const uint K10_WM = 64;
      const uint K10_WNITER = 4;
      const uint K10_TN = 4;
      const uint K10_TM = 8;
      dim3 blockDim(K10_NUM_THREADS);

      constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

      // warptile in threadblocktile
      static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
      static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);
      // threads in warpsubtile
      static_assert(
          (K10_WM * K10_WN) % (WARP_SIZE * K10_TM * K10_TN * K10_WNITER) == 0);
      constexpr uint K10_WMITER =
          (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
      // warpsubtile in warptile
      static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));

      static_assert(
          (K10_NUM_THREADS * 4) % K10_BK == 0,
          "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
          "issues during GMEM->SMEM tiling (loading only parts of the "
          "final row of Bs during each iteraion)");
      static_assert(
          (K10_NUM_THREADS * 4) % K10_BN == 0,
          "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
          "issues during GMEM->SMEM tiling (loading only parts of the "
          "final row of As during each iteration)");
      static_assert(
          K10_BN % (16 * K10_TN) == 0,
          "BN must be a multiple of 16*TN to avoid quantization effects");
      static_assert(
          K10_BM % (16 * K10_TM) == 0,
          "BM must be a multiple of 16*TM to avoid quantization effects");
      static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                    "BM*BK must be a multiple of 4*256 to vectorize loads");
      static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                    "BN*BK must be a multiple of 4*256 to vectorize loads");

      dim3 gridDim(CEIL_DIV(N, K10_BM), CEIL_DIV(N, K10_BN));

      for (int i = 0; i < warpup_time; ++i) {
        mysgemm_warptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER,
                           K10_TM, K10_TN, K10_NUM_THREADS>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }
      cudaDeviceSynchronize();

      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v1) failed");
      for (int i = 0; i < repeat_time; ++i) {
        mysgemm_warptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER,
                           K10_TM, K10_TN, K10_NUM_THREADS>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }
      checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize v1 failed");
      checkCudaError(cudaGetLastError(), "cuda get last error failed");
      float v1_time = 0;
      checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                     "cudaEventElapsedTime v1 failed");

      // 拷贝手写 kernel 结果
      checkCudaError(cudaMemcpy(C_v1, d_C_v1, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_v1 failed");

      int error_count = 0;
      for (int i = 0; i < N * N && error_count < 10; ++i) {
        if (fabsf(C_cublas[i] - C_v1[i]) > TOL) {
          error_count++;
        }
      }

      float cublas_gflops =
          repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f);  // GFlops
      float v1_gflops =
          repeat_time * 2.0f * N * N * N / (v1_time * 1e6f);  // GFlops

      float ratio = v1_gflops / cublas_gflops;
      // 写入CSV
      csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
               << (error_count == 0 ? "1" : "0") << "," << ratio << std::endl;

      // 释放资源
      cublasDestroy(handle);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C_v1);

      free(A);
      free(B);
      free(C_cublas);
      free(C_v1);
      cudaDeviceSynchronize();
    } catch (...) {
      std::cerr << "Out of memory or error during testing size: " << N
                << std::endl;
      out_of_memory = true;
    }

    if (!out_of_memory) {
      std::cout << "Finished size: " << N << std::endl;
    } else {
      csv_file << N << ",OOM,OOM,0" << std::endl;
    }
  }

  csv_file.close();

  std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark.csv'"
            << std::endl;
  return 0;
}
