#include <cuda.h>
#ifndef COMMAND_LINE_ONLY
#include "ui/GraphicalUI.h"
#endif

#include "RayTracer.h"
#include "ui/CommandLineUI.h"

using namespace std;

RayTracer* theRayTracer;
TraceUI* traceUI;
int TraceUI::m_threads = max(std::thread::hardware_concurrency(), (unsigned)1);
int TraceUI::rayCount[MAX_THREADS];

// usage : ray [option] in.ray out.bmp
// Simply keying in ray will invoke a graphics mode version.
// Use "ray --help" to see the detailed usage.
//
// Graphics mode will be substantially slower than text mode because of
// event handling overhead.
int main(int argc, char** argv)
{
	if (argc != 1) {
		// text mode
		traceUI = new CommandLineUI(argc, argv);
	} else {
#ifdef COMMAND_LINE_ONLY
		// still text mode
		traceUI = new CommandLineUI(argc, argv);
#else
		// graphics mode
		traceUI = new GraphicalUI();
#endif
	}

	theRayTracer = new RayTracer();

	traceUI->setRayTracer(theRayTracer);
	return traceUI->run();
}
// #include <cuda.h>
// #include <glm/glm.hpp>
// #include <iostream>

// __global__
// void add(int n, float *x, float *y)
// {
//   int index = threadIdx.x;
//   int stride = blockDim.x;
//   for (int i = index; i < n; i += stride)
//       y[i] = x[i] + y[i];
// }

// int main(void)
// {
//   int N = 1<<20;
//   float *x, *y;

//   // Allocate Unified Memory – accessible from CPU or GPU
//   cudaMallocManaged(&x, N*sizeof(float));
//   cudaMallocManaged(&y, N*sizeof(float));

//   // initialize x and y arrays on the host
//   for (int i = 0; i < N; i++) {
//     x[i] = 1.0f;
//     y[i] = 2.0f;
//   }

//   // Run kernel on 1M elements on the GPU
//   add<<<1, 256>>>(N, x, y);

//   // Wait for GPU to finish before accessing on host
//   cudaDeviceSynchronize();

//   // Check for errors (all values should be 3.0f)
//   float maxError = 0.0f;
//   for (int i = 0; i < N; i++)
//     maxError = fmax(maxError, fabs(y[i]-3.0f));
//   std::cout << "Max error: " << maxError << std::endl;

//   // Free memory
//   cudaFree(x);
//   cudaFree(y);
  
//   return 0;
// }