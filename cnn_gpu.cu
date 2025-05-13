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
    int h = blockIdx.y * blockDim.y + threadIdx.y; // pooled row
    int w = blockIdx.x * blockDim.x + threadIdx.x; // pooled col

    if (i >= kNum || h >= kOutImSize || w >= kOutImSize) return;

    int h_base = h * 2;
    int w_base = w * 2;

    float v00 = bias[i], v01 = bias[i], v10 = bias[i], v11 = bias[i];

    for (int j = 0; j < kNum; ++j) {
        for (int p = 0; p < kKernel; ++p) {
            for (int q = 0; q < kKernel; ++q) {
                int hin0 = h_base + p;
                int win0 = w_base + q;
                int hin1 = h_base + 1 + p;
                int win1 = w_base + 1 + q;

                float w_val = weight[(((i * kNum + j) * kKernel + p) * kKernel + q)];

                v00 += w_val * input[((j * kInImSize + hin0) * kInImSize + win0)];
                v01 += w_val * input[((j * kInImSize + hin0) * kInImSize + win1)];
                v10 += w_val * input[((j * kInImSize + hin1) * kInImSize + win0)];
                v11 += w_val * input[((j * kInImSize + hin1) * kInImSize + win1)];
            }
        }
    }

    // Apply ReLU
    v00 = fmaxf(0.f, v00);
    v01 = fmaxf(0.f, v01);
    v10 = fmaxf(0.f, v10);
    v11 = fmaxf(0.f, v11);

    // Maxpool
    float pooled = fmaxf(fmaxf(v00, v01), fmaxf(v10, v11));
    output[(i * kOutImSize + h) * kOutImSize + w] = pooled;
}
