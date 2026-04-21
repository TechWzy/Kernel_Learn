#include <cuda_runtime.h>
#include <iostream>

#define BDIMX 32
#define BDIMY 16

/*
  全局内存的内存控制器会以 128 字节 为最小单位进行内存事务。
  当同一个 Warp 内的32个线程，读写 连续且对齐 的全局内存地址时，硬件（SM 的内存控制器）会把这 32 个独立的线程请求，
  合并成 1 个内存事务，减少了实际发送到 DRAM 芯片的内存事务数量。

  合并的两个必要条件：
  1. 32个线程的地址必须是连续的；
  2. 起始地址必须是 128 字节的整数倍.

  案例分析：
  V1版本：
  对于 out[ix * ny + iy] = in[iy * nx + ix]， 我们发现在写入 out[] 时，相邻的线程写入全局内存的地址相差 ny * 4 个字节。
  （同一个 Warp 内 iy 是相同的） 显然，写入并非合并的，因此性能效率很差。

  V2版本：
  tile[threadIdx.y][threadIdx.x] = in[ti];
  __syncthreads();
  out[to] = tile[dx][dy];

  观察发现，ti = iy * nx + ix 在同一个 Warp 内是连续的，故读取合并；
  经过块内数据重排，to = iy * ny + ix 是 Warp 内连续的，故写入合并.
*/

__global__ void transposeSmem(float *out, float *in, const int nx, const int ny) {
  
  __shared__ float tile[BDIMY][BDIMX];
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  
  int ti = iy * nx + ix; //  根据 (ix, iy) 求算出元素转置前的全局索引
  int bidx = threadIdx.y * blockDim.x + threadIdx.x;
  
  //  块内数据重排，当前坐标 (dx, dy) 用于获取转置前的元素(dy, dx)，跟当前元素 (threadIdx.x, threadIdx.y) 无关...
  int dx = bidx % blockDim.y;  // dx in [0, 16)
  int dy = bidx / blockDim.y;  // dy in [0, 32)

  ix = blockIdx.y * blockDim.y + dx;
  iy = blockIdx.x * blockDim.x + dy;

  int to = iy * ny + ix;

  if(ix < ny && iy < nx) {
    //  在转置前，元素 (x, y) 存储在共享内存 tile[y][x] ... 
    tile[threadIdx.y][threadIdx.x] = in[ti];
    __syncthreads();
    //  在转置后，元素 (dx, dy) 在转置前的坐标为 (dy, dx)，它存储在共享内存 tile[dx][dy]...
    out[to] = tile[dx][dy];
  }
}

void call_transposeSmem(float *d_out, float *d_in, const int nx, const int ny) {
  // Assuming BDIMX and BDIMY are defined as the block dimensions
  dim3 blockSize(BDIMX, BDIMY);
  // Number of blocks in each dimension for original matrix traversal
  dim3 gridSize((nx + BDIMX - 1) / BDIMX, (ny + BDIMY - 1) / BDIMY);
  // Launch the kernel
  transposeSmem<<<gridSize, blockSize>>>(d_out, d_in, nx, ny);
}

void naiveSmemWrapper() {
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
    call_transposeSmem(d_out, d_in, nx, ny);
  }

  int bench_iter = 5;
  // 开始计时
  cudaEventRecord(start);

  for (int i = 0; i < bench_iter; ++i) {
    // 调用核函数
    call_transposeSmem(d_out, d_in, nx, ny);
  }

  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

  for(int i = 0;i < nx * ny;i++) {
    int x = i % nx;
    int y = i / nx;
    if(h_in[i] != h_out[x * ny + y]) {
      printf("Wrong, i = %d, h_in = %f, h_out = %f\n", i, h_in[i], h_out[x * ny + y]);
      break;
    }
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
  std::cout << "Smem transpose kernel execution time: "
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
  naiveSmemWrapper();
  return 0;
}