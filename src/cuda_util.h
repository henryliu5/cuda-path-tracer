#ifndef UTIL_H
#define UTIL_H
#include <iostream>
#include <cmath>

// https://stackoverflow.com/a/14038590/15471686
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ double d_distance(double* a, double* b, int a_idx, int b_idx, int dims)
{
    double sum = 0;
    for (int i = 0; i < dims; i++) {
        double val = a[a_idx * dims + i] - b[b_idx * dims + i];
        sum += val * val;
    }
    return sqrt(sum);
}

void do_cudaFree(double* a){
    cudaFree(a);
}

// CUDA Kernel function to print
__global__ void print(double* data, double* cent, int dims, int num_pts, int k)
{
    printf("device dims: %d, num_pts: %d, k: %d\n", dims, num_pts, k);
    printf("device data last: %lf\n", data[num_pts * dims - 1]);
    printf("device cent last: %lf\n", cent[k * dims - 1]);
}

// Copy a host array to device and return a pointer to device memory
template <typename T>
T* copy_host_to_device(T* host_a, int size){
    T* d_a;
    gpuErrchk(cudaMalloc((void **) &d_a, size));
    gpuErrchk(cudaMemcpy(d_a, host_a, size, cudaMemcpyHostToDevice));
    return d_a;
}

// Copy a device array to host and return a pointer to host memory
template <typename T>
T* copy_device_to_host(T* device_a, int size){
    T* host_a = new T[size / sizeof(T)];
    gpuErrchk(cudaMemcpy(host_a, device_a, size, cudaMemcpyDeviceToHost));
    return host_a;
}

#endif