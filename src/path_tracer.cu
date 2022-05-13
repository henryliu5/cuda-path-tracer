#include <glm/glm.hpp>
#include "RayTracer.h"
#include "gpu/GPUScene.cuh"
#include "ui/TraceUI.h"
#include <curand.h>
#include <curand_kernel.h>

extern TraceUI* traceUI;

// Do recursive ray tracing!  You'll want to insert a lot of code here
// (or places called from here) to handle reflection, refraction, etc etc.
__host__
glm::dvec3 pathTraceRay(ray& r, int depth, BVHTree* bvhTree) {
	static uniform_real_distribution<double> unif(0, 1);
	static default_random_engine re;
	isect i;
	glm::dvec3 colorC;
	#if VERBOSE
		std::cerr << "== current depth: " << depth << std::endl;
	#endif

	if(bvhTree->traverse(r, i)) {
		const Material& m = i.getMaterial();	
		colorC += m.ke(i);
		// colorC += m.shade(scene.get(), r, i);	
		if (depth > 0) {
			glm::dvec3 normal = glm::normalize(i.getN());
			// glm::dvec3 rand_dir = glm::normalize(sampleHemisphere(normal));
			// double p = 1 / (2 * PI); 
			// double theta = glm::dot(ray2.getDirection(), normal); 

			// Cosine weighted sample
			double r1 = 2 * PI * unif(re);
			double r2 = unif(re);
			double r2s = sqrt(r2);

			auto w = normal;
			auto u = glm::normalize(glm::cross((abs(w.x) > .1 ? glm::dvec3(0, 1, 0) : glm::dvec3(1, 0, 0)), w));
			auto v = glm::cross(w,u);
			auto rand_dir = glm::normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1 - r2));
			double p = 1 / PI;
			double theta = 1;

			// // cout << rand_dir << " " << glm::dot(normal, rand_dir) << endl;

            ray ray2(r.at(i) + normal * 1e-12, rand_dir, r.getAtten(), ray::REFLECTION);

			
			glm::dvec3 brdf = m.kd(i) / PI;

			glm::dvec3 res = pathTraceRay(ray2, depth - 1, bvhTree);
			colorC += (brdf * res * theta / p);
		} 
	}
	return colorC;
}


glm::dvec3 tracePixelPath(unsigned char* buffer, Scene* scene, BVHTree* bvhTree, int pixelI, int pixelJ, int bufferWidth, int bufferHeight, int maxDepth, int samplesPerPixel) {
	glm::dvec3 colSum(0, 0, 0);

	static uniform_real_distribution<double> unif(-0.5, 0.5);
	static default_random_engine re;

	// update pixel buffer w color
	for(int sample = 0; sample < samplesPerPixel; ++sample) {
		// Shoot ray through camera
		ray r(glm::dvec3(0,0,0), glm::dvec3(0,0,0), glm::dvec3(1,1,1), ray::VISIBILITY);

		double iShift = unif(re);
		double jShift = unif(re);
		double newI = pixelI + iShift;
		double newJ = pixelJ + jShift;
		if(newI < 0) newI = 0; if(newI >= bufferWidth) newI = bufferWidth - 1;
		if(newJ < 0) newJ = 0; if(newJ >= bufferHeight) newJ = bufferHeight - 1;
		double x = newI/double(bufferWidth);
		double y = newJ/double(bufferHeight);
		// double x = double(pixelI)/double(bufferWidth);
		// double y = double(pixelJ)/double(bufferHeight);

		scene->getCamera().rayThrough(x,y,r);
		auto color = pathTraceRay(r, maxDepth, bvhTree);
		// color = glm::clamp(color, 0.0, 1.0);
		colSum += color;
	}

	// colSum = glm::clamp(colSum, 0.0, (double) samplesPerPixel);
	colSum = glm::dvec3(colSum.x / samplesPerPixel, colSum.y / samplesPerPixel, colSum.z / samplesPerPixel);
	colSum = glm::clamp(colSum, 0.0, 1.0);

	unsigned char *pixel = buffer + ( pixelI + pixelJ * bufferWidth ) * 3;
	pixel[0] = (int)( 255.0 * colSum.x);
	pixel[1] = (int)( 255.0 * colSum.y);
	pixel[2] = (int)( 255.0 * colSum.z);
	return colSum;
}


// Generate random unit direction in cosine weighted hemisphere of normal
__device__
glm::dvec3 sampleCosineWeightedHemisphere(glm::dvec3& normal, curandState* local_rand_state){
	// Cosine weighted sample
	double r1 = 2 * PI * curand_uniform_double(local_rand_state);
	double r2 = curand_uniform_double(local_rand_state);
	double r2s = sqrt(r2);

	auto& w = normal;
	auto u = glm::normalize(glm::cross((abs(w.x) > .1 ? glm::dvec3(0, 1, 0) : glm::dvec3(1, 0, 0)), w));
	auto v = glm::cross(w,u);
	return glm::normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1 - r2));
}

__device__
glm::dvec3 pathTraceRayGPU(GPU::Ray& r, GPU::Scene* scene, int depth, curandState local_rand_state, bool print ) {
	GPU::Isect i;
	glm::dvec3 colorC;

	GPU::Ray curRay(r);
	glm::dvec3 curAtten(1.0,1.0,1.0);
	for(int bounce = 0; bounce <= depth + 1; bounce++){
		if(scene->intersect(curRay, i)) {
			GPU::Material* m = i.getMaterial();	
			colorC += m->ke() * curAtten;

			glm::dvec3 normal = glm::normalize(i.getN());
			glm::dvec3 rand_dir = sampleCosineWeightedHemisphere(normal, &local_rand_state);

			constexpr double p = 1 / PI;
			constexpr double theta = 1;

            curRay = GPU::Ray(curRay.at(i) + normal * 1e-12, rand_dir);
			glm::dvec3 brdf = m->kd() / PI;
			curAtten *= brdf * theta / p;
			// curAtten *= m->kd(); // happens to work out like this given
		} 
	}
	return colorC;
}


__global__
void pathTraceKernel(unsigned char* buf, int w, int h, GPU::Scene* scene, int maxDepth, int samplesPerPixel, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= w) || (j >= h)) return;

	int pixelIndex = i + j * w;

	//Each thread gets same seed, a different sequence number, no offset
   	curand_init(1984, pixelIndex, 0, &rand_state[pixelIndex]);
	// Local random state for this pixel
	curandState local_rand_state = rand_state[pixelIndex];

	// Average samples
	glm::dvec3 colSum(0, 0, 0);

	for(int sample = 0; sample < samplesPerPixel; sample++){
		GPU::Ray r(glm::dvec3(0,0,0), glm::dvec3(0,0,0));

		double iShift = curand_uniform_double(&local_rand_state);
		double jShift = curand_uniform_double(&local_rand_state);

		double newI = i + iShift;
		double newJ = j + jShift;

		double x = newI/double(w);
		double y = newJ/double(h);

		scene->camera.rayThrough(x, y, r);

		auto color = pathTraceRayGPU(r, scene, maxDepth, local_rand_state, pixelIndex == 100);
		colSum += color;
	}
	colSum = glm::dvec3(colSum.x / samplesPerPixel, colSum.y / samplesPerPixel, colSum.z / samplesPerPixel);
	colSum = glm::clamp(colSum, 0.0, 1.0);

	// Set values in buffer
	int pixel = pixelIndex * 3;
	buf[pixel] = (int)( 255.0 * colSum.x);
	buf[pixel + 1] = (int)( 255.0 * colSum.y);
	buf[pixel + 2] = (int)( 255.0 * colSum.z);
}

__global__
void testKernel(unsigned char* buf, int w, int h, curandState *rand_state){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= w) || (j >= h)) return;

	int pixel_index = j*w*3 + i*3;
	buf[pixel_index + 0] = 128;
    buf[pixel_index + 1] = (float(j) / float(h)) * 255.0;
    buf[pixel_index + 2] = 128;
}

void RayTracer::traceImageGPU(int w, int h){
	int N = w * h;

	// Load scene on GPU
    GPU::Scene* d_gpuScene = new GPU::Scene(scene.get());

	// Copy over buffer for writing
	unsigned char* buf = copy_host_to_device(buffer.data(), buffer.size() * sizeof(unsigned char));


	// Create random state for each pixel
   	curandState *d_rand_state;
   	gpuErrchk(cudaMalloc((void **)&d_rand_state, N*sizeof(curandState)));

	// Launch kernel
	int threadsX = 8;
    int threadsY = 8;
    // Render our buffer
    dim3 blocks(w/threadsX+1, h/threadsY+1);
    dim3 threads(threadsY, threadsY);

	// int blockSize = 256;
	// int numBlocks = (N + blockSize - 1) / blockSize;
	// pathTraceKernel<<<numBlocks, blockSize>>>(buf, w, h, d_gpuScene, traceUI->getDepth(), SAMPLES_PER_PIXEL);
	// testKernel<<<numBlocks, blockSize>>>(buf, w, h, rand_state);
	// testKernel<<<blocks, threads>>>(buf, w, h, d_rand_state);
	pathTraceKernel<<<blocks, threads>>>(buf, w, h, d_gpuScene, traceUI->getDepth(), SAMPLES_PER_PIXEL, d_rand_state);
	gpuErrchk(cudaDeviceSynchronize());
	// Copy buffer back
	cudaMemcpy(buffer.data(), buf, buffer.size() * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
}