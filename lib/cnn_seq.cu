#include "cnn.cuh"
#include "cnn_seq.cuh"

// Sequential CNN implementation
void cnn_seq(
    const float *input,
    const float *weight, // kNum (output filters) x kNum (input filters) x kKernel x kKernel
    const float *bias, // one bias for each filter (256 total)
    float *output
  ) {

  // Allocate memory on heap to avoid stack overflow.
  auto c_size = kNum * kImSize * kImSize * sizeof(float); // size of image after convolution
  float *C = static_cast<float*>(malloc(c_size));
  // i x h x w

  // Bias
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w) {
        C(i,h,w) = bias[i];
      }
    }
  }

  // Convolution
  for (int i = 0; i < kNum; ++i) { // output channel
    for (int j = 0; j < kNum; ++j) { // input channel
      for (int h = 0; h < kImSize; ++h) { // height
        for (int w = 0; w < kImSize; ++w) { // width
          for (int p = 0; p < kKernel; ++p) { // kernel height
            for (int q = 0; q < kKernel; ++q) { // kernel width
              C(i,h,w) += weight(i,j,p,q) * input(j,h+p,w+q);
            }
          }
        }
      }
    }
  }

  // ReLU
  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w) {
        C(i,h,w) = max(0.f, C(i,h,w));
      }
    }
  }

  // Max pooling
  for (int i = 0; i < kNum; ++i) { // i refers to output filter channel
    for (int h = 0; h < kOutImSize; ++h) {
      for (int w = 0; w < kOutImSize; ++w) {
        output(i,h,w) = max(
            max(C(i, h*2, w*2  ), C(i, h*2+1, w*2  )),
            max(C(i, h*2, w*2+1), C(i, h*2+1, w*2+1)));
      }
    }
  }

  delete C;
}