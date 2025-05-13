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

  // Get thread indices
  int i = blockIdx.x;  // output channel
  int h = blockIdx.y;  // output height
  int w = blockIdx.z;  // output width
  
  // Initialize output with bias
  output(i,h,w) = bias[i];
  
  // Convolution
  float sum = 0.0f;
  for (int j = 0; j < kNum; ++j) {         // input channel
      for (int p = 0; p < kKernel; ++p) {  // kernel height
          for (int q = 0; q < kKernel; ++q) { // kernel width
              sum += weight(i,j,p,q) * input(j,h+p,w+q);
          }
      }
  }
  
  // Store result
  output(i,h,w) = sum;
  
  // ReLU
  output(i,h,w) = max(0.0f, output(i,h,w));
}
