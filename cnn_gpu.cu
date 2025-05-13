// Header inclusions, if any...
#include "lib/cnn.cuh"
#include "cnn_gpu.cuh"

// Using declarations, if any...

__global__ void cnn_gpu(
    float* input,
    float* weight,
    float* bias,
    float* output)
{

    int i = blockIdx.z * blockDim.z + threadIdx.z; // output channel
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    int h_base = h * 2;
    int w_base = w * 2;

    float c00 = bias[i], c01 = bias[i], c10 = bias[i], c11 = bias[i];

    for (int j = 0; j < kNum; ++j) {
        for (int p = 0; p < kKernel; ++p) {
            for (int q = 0; q < kKernel; ++q) {
                int ih0 = h_base + p;
                int ih1 = h_base + 1 + p;
                int iw0 = w_base + q;
                int iw1 = w_base + 1 + q;

                float w_val = weight(i,j,p,q);

                c00 += w_val * input(j,ih0,iw0);
                c01 += w_val * input(j,ih0,iw1);
                c10 += w_val * input(j,ih1,iw0);
                c11 += w_val * input(j,ih1,iw1);
            }
        }
    }

    // Apply ReLU
    c00 = fmaxf(0.f, c00);
    c01 = fmaxf(0.f, c01);
    c10 = fmaxf(0.f, c10);
    c11 = fmaxf(0.f, c11);

    // Maxpool
    float pooled = fmaxf(fmaxf(c00, c01), fmaxf(c10, c11));
    output(i,h,w) = pooled;
}
