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
    //  串行计算
    for(int i = 0;i < N;i++) {
        //  求算第 i 个向量输入和输出的起始地址
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

__inline__ __device__
float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__inline__ __device__
float warpReduceMax(float val) {
    for(int offset = warpSize / 2;offset > 0;offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

/*

    Warp 级别的调度在代码中具象化.
    每一个线程负责 16 个元素，优先求算 16 个元素的局部最大值。针对同一个 Warp的所有线程的局部最大值，借助 warpReduceMax 能够
    高效求算出一个 Warp 的局部最大值，最后在对所有 Warp 的局部最大值，依旧借助 warpReduceMax 求算出全局最大值.
    同理可以求算出全局和...
*/

__global__ void softmax_forward_kernel3(float* out, const float* input, int N ,int C) {

    static __shared__ float shared[32];

    const float* input_row = input + blockIdx.x * C;
    float* out_row = out + blockIdx.x * C;

    int lane = threadIdx.x % warpSize;    //  warp 级编号
    int wid = threadIdx.x / warpSize;     // warp 编号，含义是 当前线程位于第几个 warp 里面
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    //  任务1：求算向量的最大值
    float maxval = -INFINITY;
    for(int i = tid;i < C;i += blockSize) {
        maxval = fmaxf(maxval, input_row[i]);
    }

    __syncthreads();
    maxval = warpReduceMax(maxval);
    if(lane == 0) {
        shared[wid] = maxval;
    }
    __syncthreads();

    if(wid == 0) {
        float val = (tid < blockDim.x / warpSize)?shared[tid] : -INFINITY;
        shared[lane] = warpReduceMax(val);
    }

    __syncthreads();
    maxval = shared[0];

    //  任务2：求算全局和
    for(int i = tid;i < C;i += blockSize) {
        out_row[i] = exp(input_row[i] - maxval);
    }

    __syncthreads();

    float partialSum = 0.f;
    for(int i = tid;i < C;i += blockSize) {
        partialSum += out_row[i];
    }

    __syncthreads();
    partialSum = warpReduceSum(partialSum);
    if(lane == 0) {
        shared[wid] = partialSum;
    }

    __syncthreads();
    float sum = 0.f;
    if(wid == 0) {
        //  注意此处是 lane 而不是 tid，后者会出现数组越界
        sum = (lane < blockDim.x / warpSize)?shared[lane] : 0;
        sum = warpReduceSum(sum);
        shared[lane] = sum;
    }

    __syncthreads();
    sum = shared[0];

    for(int i = tid;i < C;i += blockDim.x) {
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

    //  数据初始化
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
    softmax_forward_kernel3<<<numBlock, blockSize>>>(d_out, d_input, N, C);
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