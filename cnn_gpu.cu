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

    int i = blockIdx.x * blockDim.x + threadIdx.x; // output channel
    int h = blockIdx.y * blockDim.y + threadIdx.y; // height
    int w = blockIdx.z * blockDim.z + threadIdx.z; // width

    if (i >= kNum || h >= kImSize || w >= kImSize) return;

    // Convolution + bias
    float acc = bias[i];

    for (int j = 0; j < kNum; ++j) {
        for (int p = 0; p < kKernel; ++p) {
            for (int q = 0; q < kKernel; ++q) {
                acc += weight(i,j,p,q) * input(j,h+p,w+q);
            }
        }
    }

    // ReLU
    acc = fmaxf(0.f, acc);

    // Maxpool (2x2)
    if (h % 2 == 0 && w % 2 == 0 && h / 2 < kOutImSize && w / 2 < kOutImSize) {
        float v00 = acc;
        float v01 = (w + 1 < kImSize) ? fmaxf(0.f, bias[i]) : 0.f;
        float v10 = (h + 1 < kImSize) ? fmaxf(0.f, bias[i]) : 0.f;
        float v11 = ((h + 1 < kImSize) && (w + 1 < kImSize)) ? fmaxf(0.f, bias[i]) : 0.f;

        output(i,h/2,w/2) = fmaxf(fmaxf(v00, v01), fmaxf(v10, v11));
    }
}
