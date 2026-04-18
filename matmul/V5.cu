#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>    // for fabsf
#include <fstream>  // for CSV output
#include <iostream>
#include <vector>

#define BLOCK_SIZE 128
#define TOL 1e-5f

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
 
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

/*
    在 V1 ~ V4 版本中，数据的读写 和 计算是并行的。
    V5引入了缓存机制，数据的读写不再需要等待计算的完成，在当前轮次计算时便提前缓存下一轮计算所需的数据.
*/

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(256) mysgemm_v7(int M, int N, int K, float alpha, float *A, float *B, float beta,float *C) {

    int bk = blockIdx.x;
    int by = blockIdx.y;

    A = &A[bk * BM * K];
    B = &B[by * BN];
    C = &C[bk * BM * N + by * BN];

    //  Thread Tile
    const int block_col_thread = BN / TN;
    const int block_row_thread = BM / TM;
    const int thread_num = block_row_thread * block_col_thread;

    int ty = (threadIdx.x % block_col_thread) * TN;
    int tx = (threadIdx.x / block_col_thread) * TM;
    
    //  加载 As 和 Bs
    __shared__ float As[2][BK * BM];  //  采取转置的方式
    __shared__ float Bs[2][BK * BN];

    const int ldg_a_num = BM * BK / thread_num / 4;
    const int ldg_b_num = BN * BK / thread_num / 4;

    int a_py = threadIdx.x % (BK / 4) * 4;
    int a_px = threadIdx.x / (BK / 4);
    int a_move = BM / ldg_a_num;

    int b_py = threadIdx.x % (BN / 4) * 4;
    int b_px = threadIdx.x / (BN / 4);
    int b_move = BK / ldg_b_num;

    float accum[TM][TN] = {0.};
    float ldg_a_reg[4 * ldg_a_num]; //  缓冲全局显存  A，加载到共享内存
    float ldg_b_reg[4 * ldg_b_num]; //  缓冲全局显存  B，加载到共享内存
    
    float a_frag[2][TM];  //  缓冲 共享内存，用于计算
    float b_frag[2][TN];  //  缓冲 共享内存，用于计算

    /*
        缓冲流程：
        A -> ldg_a_reg -> As  (局部加载，单线程加载)
        As -> a_frag -> accum (Thread Tile 的全局加载，需要全局 As)
    */

    //  注意，本例中 A 和 B 后续不再执行 BK 方向的偏移

    int load_index;
    #pragma unroll
    for(int i = 0;i < BM;i += a_move) {
        //  加载 A 到 ldg_a_reg，加载的起始位置为 (a_px + i, a_py)
        load_index = i / a_move * 4;
        FETCH_FLOAT4(ldg_a_reg[load_index]) = FETCH_FLOAT4(A[OFFSET(a_px + i, a_py, K)]);
        As[0][OFFSET(a_py, a_px + i, BM)] = ldg_a_reg[load_index];
        As[0][OFFSET(a_py + 1, a_px + i, BM)] = ldg_a_reg[load_index + 1];
        As[0][OFFSET(a_py + 2, a_px + i, BM)] = ldg_a_reg[load_index + 2];
        As[0][OFFSET(a_py + 3, a_px + i, BM)] = ldg_a_reg[load_index + 3];
    }
    #pragma unroll
    for(int i = 0;i < BK;i += b_move) {
        FETCH_FLOAT4(Bs[0][OFFSET(b_px + i, b_py, BN)]) =  FETCH_FLOAT4(B[OFFSET(b_px + i, b_py, N)]);
    }

    __syncthreads();

    //  加载 a_frag[0] 和 b_frag[0]
    //  固定 k 值，加载 A[0 ~ TM - 1][k] 和 B[k][0 ~ TN - 1]
    #pragma unroll
    for(int m = 0;m < TM;m += 4) {
        //  倘若 As 没有转置，那么 加载元素为 As[0][m + tx][0]
        FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(As[0][OFFSET(0, m + tx, BM)]);
    }

    #pragma unroll
    for(int n = 0;n < TN;n += 4) {
        //  加载元素为 Bs[0][0][n + ty]
        FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(Bs[0][OFFSET(0, n + ty, BN)]);
    }

    //  As 和 Bs 的下一个读取的块索引
    int write_index = 1;
    int k = 0;

    /*
        大缓冲：ldg_a_reg -> As & ldg_b_reg -> Bs
        小缓冲：As -> a_frag & Bs -> b_frag
        缓冲原则：在数据消耗之前，提前准备好下一轮数据，例如 在使用 a_frag 和 b_frag 求算 accum 前，提前缓冲好下一轮的 a_fraq 和 b_frag...
                 在循环开始前，由于 ldg_a_reg 和 ldg_b_reg 已经被用于加载 As 和 Bs，因此优先缓冲好 ldg_a_reg 和 ldg_b_reg...
                 在下一轮循环开始前，As 和 Bs 需要重新计算...
    */

    do{
        k += BK;

        if(k < K) {
            //  缓存 ldg_a_reg 和 ldg_b_reg
            #pragma unrollA
            for(int i = 0;i < BM;i += a_move) {
                load_index = i / a_move * 4;
                //  线程加载的起始位置为 (a_px + i, a_py)
                FETCH_FLOAT4(ldg_a_reg[load_index]) = FETCH_FLOAT4(A[OFFSET(a_px + i, a_py + k, K)]);
            }
            #pragma unroll
            for(int i = 0;i < BK;i += b_move) {
                load_index = i / b_move * 4;
                //  线程加载的起始位置为 (b_px + i, b_py) 
                FETCH_FLOAT4(ldg_b_reg[load_index]) = FETCH_FLOAT4(B[OFFSET(b_px + i + k, b_py, N)]);
            }
        }

        int loading_index = write_index ^ 1;

        //  求算 accum[][]，为了方便缓冲a_frag[TM] 和 b_frag[TN]，遍历 k 的范围是 [0, BK - 2]，特判 A[][BK - 1] * B[BK - 1][]
        #pragma unroll
        for(int ck = 0;ck < BK - 1;ck++) {
          #pragma unroll
            for(int m = 0;m < TM;m += 4) {
                //  当前加载A的坐标为 (tx + m, bk + 1)
                FETCH_FLOAT4(a_frag[(ck + 1) % 2][m]) = FETCH_FLOAT4(As[loading_index][OFFSET(ck + 1, tx + m, BM)]);
            }
            #pragma unroll
            for(int n = 0;n < TN;n += 4) {
                //  当前加加载B的坐标为(bk + 1, ty + n)
                FETCH_FLOAT4(b_frag[(ck + 1) % 2][n]) = FETCH_FLOAT4(Bs[loading_index][OFFSET(ck + 1, ty + n, BN)]);
            }
            //  求算 A[i][bk] * B[bk][j] -> accum[i][j]
            #pragma unroll
            for(int m = 0;m < TM;m++) {
                #pragma unroll
                for(int n = 0;n < TN;n++) {
                    accum[m][n] += a_frag[ck % 2][m] * b_frag[ck % 2][n];
                }
            }
        }

        //  缓冲下一轮的 As 和 Bs
        if(k < K) {
            #pragma unroll
            for(int i = 0;i < BM;i += a_move) {
                //  加载坐标为 (a_px + i, a_py)
                load_index = i / a_move * 4;
                As[write_index][OFFSET(a_py, a_px + i, BM)] = ldg_a_reg[load_index];
                As[write_index][OFFSET(a_py + 1, a_px + i, BM)] = ldg_a_reg[load_index + 1];
                As[write_index][OFFSET(a_py + 2, a_px + i, BM)] = ldg_a_reg[load_index + 2];
                As[write_index][OFFSET(a_py + 3, a_px + i, BM)] = ldg_a_reg[load_index + 3];
            }
            #pragma unroll
            for(int i = 0;i < BK;i += b_move) {
                //  加载坐标为 (b_px + i, b_py)
                load_index = i / b_move * 4;
                FETCH_FLOAT4(Bs[write_index][OFFSET(b_px + i, b_py, BN)]) = FETCH_FLOAT4(ldg_b_reg[load_index]);
            }
            __syncthreads();
            //  固定 k = 0
            #pragma unroll
            for(int m = 0;m < TM;m += 4) {
                //  加载As的起始位置为 (tx + m, 0)，一次性加载四个元素 
                FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(As[write_index][OFFSET(0, tx + m, BM)]);
            }
            #pragma unroll
            for(int n = 0;n < TN;n += 4) {
                //  加载Bs的起始位置为 (0, ty + n)，一次性加载四个元素
                FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(Bs[write_index][OFFSET(0, ty + n, BN)]);
            }
            write_index ^= 1;
        }
        
        #pragma unroll
        for(int m = 0;m < TM;m++) {
          #pragma unroll
            for(int n = 0;n < TN;n++) {
                accum[m][n] += a_frag[(BK - 1) % 2][m] * b_frag[(BK - 1) % 2][n];
            }
        }
    }while(k < K);

    #pragma unroll
    for(int m = 0;m < TM;m++) {
      #pragma unroll
        for(int n = 0;n < TN;n += 4) {
            float4 ctmp = FETCH_FLOAT4(C[OFFSET(tx + m, ty + n, N)]);
            ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
            ctmp.y = alpha * accum[m][n + 1] + beta * ctmp.y;
            ctmp.z = alpha * accum[m][n + 2] + beta * ctmp.z;
            ctmp.w = alpha * accum[m][n + 3] + beta * ctmp.w;
            FETCH_FLOAT4(C[OFFSET(tx + m, ty + n, N)]) = ctmp;
        }
    }

}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

std::vector<int> generateSizes() {
  std::vector<int> sizes;
  for (int i = 256; i <= 8192; i += 256) {
    sizes.push_back(i);
  }
  return sizes;
}
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
      int warpup_time = 10;  // 热身次数
      for (int i = 0; i < warpup_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                         "cublasSgemm failed");
      }
      cudaDeviceSynchronize();

      // cuBLAS SGEMM
      int repeat_time = 5;
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

      dim3 blockDim(256);
      dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

      for (int i = 0; i < warpup_time; ++i) {
        mysgemm_v7<128, 128, 8, 8, 8>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }
      cudaDeviceSynchronize();

      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v1) failed");
      for (int i = 0; i < repeat_time; ++i) {
        mysgemm_v7<128, 128, 8, 8, 8>
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
      // 结果比较
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
