#include <cuda_runtime.h>
#include <iostream>

#define BDIMX 32
#define BDIMY 16

__global__ void transposeSmemUnrollPad(float *out, float *in, int nx, int ny) {

  //  物理模型的形状为：(girdDim.x * blockDim.x) * (gridDim.y * blockDim.y)
  //  抽象模型的形状为：(2 * girdDim.x * blockDim.x) * (gridDim.y * blockDim.y)
  const int IPAD = 1;
  __shared__ float tile[BDIMY * (BDIMX * 2 + IPAD)];

  //  没有转置前，当前元素的位置
  int ix = 2 * blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int ti = iy * nx + ix;

  const int bidx = blockDim.x * threadIdx.y + threadIdx.x;
  const int bx = bidx % blockDim.y;
  const int by = bidx / blockDim.y;

  //  转置后，索引在 bidx 的元素的全局坐标
  ix = blockDim.y * blockIdx.y + bx;
  iy = (2 * blockDim.x) * blockIdx.x + by;

  if(ix < ny && (iy + BDIMX) < nx) {
    //  加载 转置前的元素 (ix, iy) 和 (ix + BDIMX, iy) 到全局内存
    int write_in_tile = threadIdx.y * (blockDim.x * 2 + IPAD) + threadIdx.x;
    tile[write_in_tile] = in[ti];
    tile[write_in_tile + BDIMX] = in[ti + BDIMX];

    __syncthreads();
    
    //  转置后，加载索引在 bid 的元素 (ix, iy) 和 (ix, iy + BDMIX) 到全局内存
    write_in_tile = bx * (blockDim.x * 2 + IPAD) + by;
    out[iy * ny + ix] = tile[write_in_tile];
    out[(iy + BDIMX) * ny + ix] = tile[write_in_tile + BDIMX];
  }

}

void call_transposeSmemUnrollUnpad(float *d_out, float *d_in, const int nx,
                                   const int ny) {
  // Assuming BDIMX and BDIMY are defined as the block dimensions
  dim3 blockSize(BDIMX, BDIMY);
  // Number of blocks in each dimension for original matrix traversal
  auto grid = (nx + BDIMX - 1) / BDIMX;
  dim3 gridSize(int(grid / 2), (ny + BDIMY - 1) / BDIMY);

  // Launch the kernel
  transposeSmemUnrollPad<<<gridSize, blockSize>>>(d_out, d_in, nx, ny);
}

void naiveSmemWrapperUnrollUnpad() {
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
    call_transposeSmemUnrollUnpad(d_out, d_in, nx, ny);
  }

  int bench_iter = 5;
  // 开始计时
  cudaEventRecord(start);

  for (int i = 0; i < bench_iter; ++i) {
    // 调用核函数
    call_transposeSmemUnrollUnpad(d_out, d_in, nx, ny);
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
  std::cout << "Smem transpose unroll unpad kernel execution time: "
            << milliseconds / float(bench_iter) << " ms" << std::endl;

  // 将结果从设备复制回主机
  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

  // //  判断转置的正确性
  // const float eps = 1e-5;
  // for(int i = 0;i < nx * ny;i++) {
  //   //  (i % nx, i / nx) -> (i / nx, i % nx)
  //   if(fabs(h_out[(i % nx) * ny + i / nx] - h_in[i]) >= eps) {
  //       printf("Wrong\n");
  //       break;
  //   }
  // }

  // 释放内存
  free(h_in);
  free(h_out);
  cudaFree(d_in);
  cudaFree(d_out);

  std::cout << "Matrix transposition completed successfully." << std::endl;
}

int main() {
  naiveSmemWrapperUnrollUnpad();
  return 0;
}