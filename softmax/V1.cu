#include <chrono>  
#include <cmath>   
#include <cstdlib> 
#include <iostream> 

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