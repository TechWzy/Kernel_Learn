#include <chrono>   // for timing
#include <cmath>    // for INFINITY
#include <cstdlib>  // for malloc/free
#include <iostream> 

// Function to compare results
bool compare_results(const float *cpu, const float *gpu, int N, int C,
                     float epsilon = 1e-3f) {
  for (int i = 0; i < N * C; ++i) {
    if (fabs(cpu[i] - gpu[i]) > epsilon) {
      std::cout << "Difference at index " << i << ": CPU=" << cpu[i]
                << ", GPU=" << gpu[i] << ", diff=" << fabs(cpu[i] - gpu[i])
                << std::endl;
      return false;
    }
  }
  return true;
}

void softmax_forward_cpu(float *out, const float *input, int N, int C) {
    for(int i = 0;i < N;i++) {
        const float* inp_row = input + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        for(int j = 0;j < C;j++) {
            if(inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }

        float sum = 0.f;
        for(int j = 0;j < C;j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }

        for(int j = 0;j < C;j++) {
            out_row[j] /= sum;
        }
    }
}

/*
    线程块大小为 256，每一个线程处理 4096 / 256 = 16 个元素
*/


__global__ void softmax_forward_kernel2(float* out, const float* input, int N, int C) {

    extern __shared__ float shared[]; 
    
    const float* inp_row = input + blockIdx.x * C;
    float* out_row = out + blockIdx.x * C;

    //  第一个任务：先求算 inp_row[0 : C - 1] 的最大值，依旧采用归约计算的方式
    float maxval = -INFINITY;
    for(int i = threadIdx.x;i < C;i += blockDim.x) {
        maxval = fmaxf(maxval, inp_row[i]);
    }

    shared[threadIdx.x] = maxval;
    __syncthreads();

    for(int stride = blockDim.x / 2;stride >= 1;stride /= 2) {
        __syncthreads();
        if(threadIdx.x < stride) {        
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
    }
    __syncthreads();
    maxval = shared[0];

    for(int i = threadIdx.x;i < C;i += blockDim.x) {
        out_row[i] = exp(inp_row[i] - maxval);
    }
    __syncthreads();

    float sum = 0.f;
    for(int i = threadIdx.x;i < C;i += blockDim.x) {
        sum += out_row[i];
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    
    //  归约计算
    for(int stride = blockDim.x / 2;stride >= 1;stride /= 2) {
        __syncthreads();
        if(threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
    }

    __syncthreads();
    sum = shared[0];

    for(int i = threadIdx.x;i < C;i += blockDim.x) {
        out_row[i] /= sum;
    }
}

int main() {

    int N = 128;
    int C = 4096;

    size_t num_element = N * C;
    size_t nbytes = num_element * sizeof(float);

    float* input = (float*)malloc(nbytes);
    float* out_cpu = (float*)malloc(nbytes);
    float* out_gpu = (float*)malloc(nbytes);

    for(int i = 0;i < N;i++) {
        for(int j = 0;j < C;j++) {
            input[i * C + j] = float(j);
        }
    }

    // Run CPU version and measure time
    auto start_cpu = std::chrono::high_resolution_clock::now();
    softmax_forward_cpu(out_cpu, input, N, C);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

    // Run GPU version and measure time using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float* d_out, *d_input;
    cudaMalloc((void**) &d_out, nbytes);
    cudaMalloc((void**) &d_input, nbytes);
    cudaMemcpy(d_input, input, nbytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlock = N;
    cudaEventRecord(start);
    softmax_forward_kernel2<<<numBlock, blockSize>>>(d_out, d_input, N, C);
    cudaEventRecord(stop);

    // 重点：务必等待 stop 记录结束
    cudaEventSynchronize(stop);

    // Calculate milliseconds
    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    // Copy result back to host
    cudaMemcpy(out_gpu, d_out, nbytes, cudaMemcpyDeviceToHost);

    // Compare results
    bool success = compare_results(out_cpu, out_gpu, N, C);
    std::cout << "Results match: " << (success ? "YES" : "NO") << std::endl;

    // Print performance comparison
    std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_time_ms << " ms" << std::endl;
    std::cout << "Speedup: " << (cpu_time.count() / (gpu_time_ms)) << "x"<< std::endl;

    // Cleanup
    cudaFree(d_out);
    cudaFree(d_input);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(input);
    free(out_cpu);
    free(out_gpu);

    return 0;
}