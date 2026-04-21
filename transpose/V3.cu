#include <cuda_runtime.h>
#include <iostream>

/*
    在 V2 版本中，共享内存的读取(tile[dx][dy]) 存在 bank conflict，会减低读取效率.
    在 V3 版本中，共享内存的读取消除了 bank confict，读取效率有所提升.
*/

#define BDIMX 32
#define BDIMY 16
__global__ void transposeSmemUnpad(float *out, float *in, const int nx,
                                   const int ny) {
  const int pad = 1;
  __shared__ float tile[BDIMY][BDIMX + pad];
  // original
  unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
  // linear global memory index for original
  unsigned int ti = iy * nx + ix;
  // thread index in transposed block
  unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;

  unsigned int dy = bidx / blockDim.y;
  unsigned int dx = bidx % blockDim.y;
  // coordinate in transposed matrix
  ix = blockIdx.y * blockDim.y + dx;
  iy = blockIdx.x * blockDim.x + dy;

  // linear global memory index for transposed matrix
  unsigned int to = iy * ny + ix;

  if (ix < nx && iy < ny) {
    tile[threadIdx.y][threadIdx.x] = in[ti];
    __syncthreads();
    out[to] = tile[dx][dy];
  }
}

void call_transposeSmemUnpad(float *d_out, float *d_in, const int nx,
                             const int ny) {
  // Assuming BDIMX and BDIMY are defined as the block dimensions
  dim3 blockSize(BDIMX, BDIMY);
  // Number of blocks in each dimension for original matrix traversal
  dim3 gridSize((nx + BDIMX - 1) / BDIMX, (ny + BDIMY - 1) / BDIMY);

  // Launch the kernel
  transposeSmemUnpad<<<gridSize, blockSize>>>(d_out, d_in, nx, ny);
}

void naiveSmemWrapperUnpad() {
  int nx = 4096;
  int ny = 4096;
  size_t size = nx * ny * sizeof(float);

  // 主机内存分配
  float *h_in = (float *)malloc(size);
  float *h_out = (float *)malloc(size);

  // 初始化输入矩阵
  for (int i = 0; i < nx * ny; i++) {
    h_in[i] = float(int(i) % 11);
  }

  // 设备内存分配
  float *d_in, *d_out;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);

  // 将数据从主机复制到设备
  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int warp_up_iter = 5;
  for (int i = 0; i < warp_up_iter; ++i) {
    call_transposeSmemUnpad(d_out, d_in, nx, ny);
  }

  int bench_iter = 5;
  // 开始计时
  cudaEventRecord(start);

  for (int i = 0; i < bench_iter; ++i) {
    // 调用核函数
    call_transposeSmemUnpad(d_out, d_in, nx, ny);
  }

  // 结束计时
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    return;
  }

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Smem transpose unpad kernel execution time: "
            << milliseconds / float(bench_iter) << " ms" << std::endl;

  // 将结果从设备复制回主机
  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
  // 释放内存
  free(h_in);
  free(h_out);
  cudaFree(d_in);
  cudaFree(d_out);

  std::cout << "Matrix transposition completed successfully." << std::endl;
}

int main() {
  naiveSmemWrapperUnpad();
  return 0;
}