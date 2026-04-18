#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>    // for fabsf
#include <fstream>  // for CSV output
#include <iostream>
#include <vector>

#define TOL 1e-5f
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

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
    相比 V2，V3的 线程模型的坐标布局遵循 内存模型的坐标布局（以 x 轴为行，以 y 轴为列）
    引入了 Thread Tile（大小为 TM * TN），每一个线程处理一个 Thread Tile，即 计算 TM * TN 个元素，提高了单线程的访存密度。
    按照我的理解，V2的每一个线程所承担的计算量较少，线程计算负荷小。在 V3 版本中，每一个线程的计算负荷有所提升。
    就全局而言，所需的线程块数量大幅度减少，寄存器 和 共享内存等资源更加充裕，active warp 数量增多，调度速度更快.

    笔者初学，浅见陋识，尚祈读者明鉴。
*/

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm_v4(int M, int N, int K, float alpha, float *A, float *B,float beta, float *C) {

    int bx = blockIdx.x;
    int by = blockIdx.y;

    A = &A[bx * BM * K];
    B = &B[by * BN];
    C = &C[bx * BM * N + by * BN];

    //  将子块C 划分为若干个 Thread Tile (TM * TN)
    const int thread_row_num = BM / TM;
    const int thread_col_num = BN / TN;
    const int thread_num = thread_col_num * thread_row_num;

    //  当前线程的 Thread Tile 的起始位置
    int tx = (threadIdx.x / thread_col_num) * TM;
    int ty = (threadIdx.x % thread_col_num) * TN;

    //  加载 As
    int a_py = threadIdx.x % BK;
    int a_px = threadIdx.x / BK;
    int a_move = thread_num / BK;

    //  加载 Bs
    int b_py = threadIdx.x % BN;
    int b_px = threadIdx.x / BN;
    int b_move = thread_num / BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BM];

    float tmp[TM][TN] = {0.0};
    #pragma unroll
    for(int k = 0;k < K;k += BK) {
        #pragma unroll
        for(int i = 0;i < BM;i += a_move) {
            As[OFFSET(i + a_px, a_py, BK)] = A[OFFSET(i + a_px, a_py, K)];
        }
        #pragma unroll
        for(int i = 0;i < BK;i += b_move) {
            Bs[OFFSET(i + b_px, b_py, BN)] = B[OFFSET(i + b_px, b_py, N)];
        }
        __syncthreads();
        A += BK;
        B += BK * N;
        #pragma unroll
        for(int l = 0;l < BK;l++) {
            for(int i = 0;i < TM;i++) {
                for(int j = 0;j < TN;j++) {
                    tmp[i][j] += As[OFFSET(tx + i, l, BK)] * Bs[OFFSET(l, ty + j, BN)];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int i = 0;i < TM;i++) {
        #pragma unroll
        for(int j = 0;j < TN;j++) {
            //  C_sub[tx + i][ty + j]
            C[OFFSET(tx + i, ty + j, N)] = alpha * tmp[i][j] + beta * C[OFFSET(tx + i, ty + j, N)];
        }
    }
}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

// std::vector<int> generateSizes() { return {128, 256, 512, 1024, 2048, 4096, 8192}; }
std::vector<int> generateSizes() { return {4096}; }
int main() {
    int device_id = 0;
    checkCudaError(cudaSetDevice(device_id), "cudaSetDevice failed");
    std::vector<int> sizes = generateSizes();

    // 打开CSV文件
    std::ofstream csv_file("sgemm_benchmark_v3.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

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

        // mysgemm_v4
        checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

   


        dim3 blockDim(256);     //  每一个线程处理 64 个元素，因此布局可以抽象为 128 * 128，故网格的形状为 (N / 128, N / 128)
        dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

        for (int i = 0; i < warpup_time; ++i) {
            mysgemm_v4<128, 128, 8, 8, 8>
                <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
        }

        cudaDeviceSynchronize();
        checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

        checkCudaError(cudaEventRecord(start),
                        "cudaEventRecord(start v1) failed");

        for (int i = 0; i < repeat_time; ++i) {
            mysgemm_v4<128, 128, 8, 8, 8>
                <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
        }
        checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
        checkCudaError(cudaEventSynchronize(stop),
                        "cudaEventSynchronize v1 failed");
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
        // 写入CSV
        csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
                << (error_count == 0 ? "1" : "0") << std::endl;

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