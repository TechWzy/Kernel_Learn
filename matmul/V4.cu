#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>    // for fabsf
#include <fstream>  // for CSV output
#include <iostream>
#include <vector>

#define TOL 1e-5f
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

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
    在 V3 的基础上，采取 float4 的方式读写。float4 一次性能够读取 4 个 float4，进一步提高了 带宽的利用率。
    此外，访存指令的数量理论上降低到 V3版本的 1/4，减轻了指令发射压力。

    笔者初学，浅见陋识，尚祈读者明鉴。
*/

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm_v6(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    int bx = blockIdx.x;
    int by = blockIdx.y;

    //  每一个线程块逻辑上处理的区域二维形状为 128 * 128，即使实际线程块的一维形状是 128
    //  求算 C 的子块区域的起始位置为 (bx * BM, by * BN)

    C = &C[bx * BM * N + by * BN];
    A = &A[bx * BM * K];
    B = &B[by * BN];

    //  每一个线程块处理一个C子块，而每一个线程处理 64 个元素
    //  采取 Thread Tile 技术，将C子块进一步划分为 多个 TM * TN 小块.
    //  每一个小块由 一个线程负责处理

    const int sblock_row_num = BM / TM;     //  行方向需要多少个线程处理?
    const int sblock_col_num = BN / TN;     //  列方向需要多少个线程处理?

    //  确保 thread_num == blockSize
    const int thread_num = sblock_col_num * sblock_row_num;    

    //  每一个线程负责一个小块，根据上述信息可以抽象出小块的起始位置(tx, ty)
    int ty = (threadIdx.x % sblock_col_num) * TN;
    int tx = (threadIdx.x / sblock_col_num) * TM;

    //  每一个线程还需要加载 A 和 B 到 A_sub 和 B_sub 
    //  因此需要进一步抽象出处理 A_sub 和 B_sub 的指针，此时视角为整个线程块

    //  每一个线程一次性取一个 float4，在形状为 BN * BK 的A_sub 里面加载
    int a_py = threadIdx.x % (BK / 4) * 4;
    int a_px = threadIdx.x / (BK / 4);
    //  每一个线程需要处理的元素数量，也可以理解为该线程需要上移的次数
    const int ldg_a_num = BM * BK / thread_num / 4;     
    int a_move = BM / ldg_a_num;

    //  同理
    int b_py = threadIdx.x % (BN / 4) * 4;
    int b_px = threadIdx.x / (BN / 4);
    const int ldg_b_num = BK * BN / thread_num / 4;
    int b_move = BK / ldg_b_num;

    __shared__ float As[BK * BM];   //  实际处理 As 的转置
    __shared__ float Bs[BN * BK];

    //  每一个线程处理 4 * ldg_a_num 个元素
    float ldg_a_reg[4];
    float accum[TM][TN] = {0.0};

    //  求算 A[i][k] * B[k][j] 时，固定 k 值
    //  要求加载 A[i][] 和 B[][j] 到 a_frag 和 b_frag
    float a_frag[TM];
    float b_frag[TN];

    #pragma unroll
    for(int k = 0;k < K;k += BK) {
        #pragma unroll
        for(int i = 0;i < BM;i += a_move) {
            //  当前位置为: (a_px + i, a_py)，沿着行方向加载
            FETCH_FLOAT4(ldg_a_reg[0]) = FETCH_FLOAT4(A[OFFSET(a_px + i, a_py, K)]);
            //  存储在 As 中的实际位置为 (a_py, a_px + i)
            As[OFFSET(a_py, a_px + i, BM)] = ldg_a_reg[0];
            As[OFFSET(a_py + 1, a_px + i, BM)] = ldg_a_reg[1];
            As[OFFSET(a_py + 2, a_px + i, BM)] = ldg_a_reg[2];
            As[OFFSET(a_py + 3, a_px + i, BM)] = ldg_a_reg[3];
        }
        #pragma unroll
        for(int i = 0;i < BK;i += b_move) {
            //  当前位置为：(b_px + i, b_py)，沿着行方向加载
            FETCH_FLOAT4(Bs[OFFSET(b_px + i, b_py, BN)]) = FETCH_FLOAT4(B[OFFSET(b_px + i, b_py, N)]);
        }
        __syncthreads();
        A += BK;
        B += BK * N;
        #pragma unroll
        for(int l = 0;l < BK;l++) { //  固定 k 值
            #pragma unroll
            for(int m = 0;m < TM;m += 4) {
                //  加载的起始位置为 (tx + m, l) -> 转置位置 (l, tx + m)
                FETCH_FLOAT4(a_frag[m]) = FETCH_FLOAT4(As[OFFSET(l, tx + m, BM)]);
            }
            #pragma unroll
            for(int n = 0;n < TN;n += 4) {
                //  加载的起始位置为 (l, ty + n)
                FETCH_FLOAT4(b_frag[n]) = FETCH_FLOAT4(Bs[OFFSET(l, ty + n, BN)]);
            }
            #pragma unroll
            for(int m = 0;m < TM;m++) {
                for(int n = 0;n < TN;n++) {
                    accum[m][n] += a_frag[m] * b_frag[n];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int m = 0;m < TM;m++) {
        #pragma unroll
        //  一次性取出 1 个 float4
        for(int n = 0;n < TN;n += 4) {
            //  处理的起始位置为 (tx + m, ty + n)
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
std::vector<int> generateSizes() { return {4096}; }
// std::vector<int> generateSizes() { return {128, 256, 512, 1024, 2048, 4096, 8192}; }
int main() {
    int device_id = 0;
    checkCudaError(cudaSetDevice(device_id), "cudaSetDevice failed");
    std::vector<int> sizes = generateSizes();
    // 打开CSV文件
    std::ofstream csv_file("sgemm_benchmark_v4.csv");
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
        int warpup_time = 2;  // 热身次数
        for (int i = 0; i < warpup_time; ++i) {
            checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                        &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                            "cublasSgemm failed");
        }
        cudaDeviceSynchronize();

        // cuBLAS SGEMM
        int repeat_time = 2;
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
            mysgemm_v6<128, 128, 8, 8, 8>
                <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
        }

        cudaDeviceSynchronize();
        checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

        checkCudaError(cudaEventRecord(start),
                        "cudaEventRecord(start v1) failed");

        for (int i = 0; i < repeat_time; ++i) {
            mysgemm_v6<128, 128, 8, 8, 8>
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
