#ifndef GPU_H
#define GPU_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#include <cuda.h>
#include "../cuda_util.h"

namespace GPU {

// Idea from: https://developer.nvidia.com/blog/unified-memory-in-cuda-6/
class GPUManaged {
public:
  void *operator new(size_t len) {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    return ptr;
  }

  void operator delete(void *ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
};

}

#endif