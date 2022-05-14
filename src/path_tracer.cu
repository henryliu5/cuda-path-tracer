#include <glm/glm.hpp>
#include "RayTracer.h"
#include "gpu/GPUScene.cuh"
#include "ui/TraceUI.h"
#include <curand.h>
#include <curand_kernel.h>

extern TraceUI* traceUI;
#define APETURE 0.2

// Test light attenuation for perfectly specular surfaces
__host__ __device__ double specAtten(GPU::Isect& i){
	double d = i.getT();
	// constexpr double a = 0.000045492;
	// constexpr double b = 0.003372407;
	// constexpr double c = 0.25;
	// constexpr double a = 0.1;
	// constexpr double b = 0.003372407;
	// constexpr double c = 0.25;
	return 1;
	// return min(1.0, 1/(a * d*d + b * d + c));
}

// Do recursive ray tracing!  You'll want to insert a lot of code here
// (or places called from here) to handle reflection, refraction, etc etc.
__host__
glm::dvec3 pathTraceRay(ray& r, int depth, BVHTree* bvhTree) {
	static uniform_real_distribution<double> unboundedUnif;
	static uniform_real_distribution<double> unif(0, 1);
	static default_random_engine re;
	isect i;
	glm::dvec3 colorC;
	glm::dvec3 myContrib;
	#if VERBOSE
		std::cerr << "== current depth: " << depth << std::endl;
	#endif

	double coeffRR = 1.0;
	// if(traceUI->getDepth() - depth >= min(3, traceUI->getDepth())) {
	// 	double probRR = 0.30;
	// 	double sampleRR = unif(re);
	// 	if (sampleRR >= probRR) {
	// 		return colorC;
	// 	}
	// 	coeffRR = 1 / probRR;
	// }

	if(bvhTree->traverse(r, i)) {
		// cout << "Emissive" << endl;
		const Material& m = i.getMaterial();
		myContrib += (m.ke(i) * coeffRR);		
		colorC += (m.ke(i) * coeffRR);
		// colorC += m.shade(scene.get(), r, i);	
		if (depth > 0) {
			// cout << "Kd of Material at i is : " << m.kd(i).x << " " << m.kd(i).y << " " << m.kd(i).z << " " << m.Diff() << " \n" << endl;
				if (m.Diff()) {
					//DIFFUSE BRDF
					glm::dvec3 normal = glm::normalize(i.getN());

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

					ray ray2(r.at(i) + normal * 1e-12, rand_dir, r.getAtten(), ray::REFLECTION);

					
					glm::dvec3 brdf = m.kd(i) / PI;

					glm::dvec3 res = pathTraceRay(ray2, depth - 1, bvhTree);
					colorC += ((brdf * res * theta / p) * coeffRR);
				}
				if (m.Spec()) {
					// cout << "Specular" << endl;
					//SPECULAR BRDF (assuming glossy surface with no roughness)
					glm::dvec3 w_in = r.getDirection();
					glm::dvec3 normal = i.getN();
					if(r.currentIndex != 1.0){
						normal *= -1.0;
					}
					glm::dvec3 w_normal = glm::dot(w_in, normal) * normal;
					glm::dvec3 w_tan = w_in - w_normal;
					glm::dvec3 w_ref = -w_normal + w_tan;
					w_ref = glm::normalize(w_ref);

					ray reflect(r.at(i) + normal * 1e-12, w_ref, r.getAtten(), ray::REFLECTION);
					// ray reflect(r.at(i) + normal * 1e-12, w_ref + (rand_dir*.1), r.getAtten(), ray::REFLECTION);
					reflect.currentIndex = r.currentIndex;

					glm::dvec3 temp = pathTraceRay(reflect, depth - 1, bvhTree);
					colorC += ((m.ks(i) * temp) * coeffRR);

				}
				if (m.Trans()) {
					// cout << "Transmissive" << endl;
					glm::dvec3 w_in = r.getDirection();
					glm::dvec3 normal = i.getN();

					double n1;
					double n2;
					glm::dvec3 trans = {1,1,1};
					//Based on entering/exiting a material, set n1, n2, and trans
					if(r.currentIndex == 1.0){
						n1 = r.currentIndex;
						n2 = m.index(i);
					} else{
						n1 = m.index(i);
						n2 = 1.0;
						normal *= -1;
						trans = glm::pow(m.kt(i), {i.getT(),i.getT(),i.getT()});
					}

					double r0 = (n1 - n2)/(n1 + n2);
					r0 *= r0;

					double n = n1/n2;

					w_in = -glm::normalize(w_in);
					
					double cosI = glm::dot(normal, w_in); //cosine of theta_1 (incident angle)

					double cosR = 1 - n*n * (1-cosI*cosI); //cosine of theta_2 (refraction angle)
					if(cosR >= 0){
						//Potentially Refract/Reflect
						double reflCoeff = r0 + (1 - r0)*pow((1 - cosI), 5.0); //Schlick's Approximation of Fresnel's Constant

						double sampleFresnel = unboundedUnif(re);

						if (sampleFresnel > reflCoeff) {
							//refraction
							double cosT = glm::sqrt(cosR);
							glm::dvec3 refrac = (n*cosI - cosT) * normal - n*w_in; //direction of the refracted ray

							ray r2(r.at(i) - normal * 1e-12, refrac, r.getAtten(), ray::REFRACTION);
							r2.currentIndex = n2;

							glm::dvec3 temp = pathTraceRay(r2, depth - 1, bvhTree);

							colorC += ((trans * temp) * coeffRR);
						} else {
							//reflection
							glm::dvec3 w_in = r.getDirection();
							glm::dvec3 normal = i.getN();
							if(r.currentIndex != 1.0){
								normal *= -1.0;
							}
							glm::dvec3 w_normal = glm::dot(w_in, normal) * normal;
							glm::dvec3 w_tan = w_in - w_normal;
							glm::dvec3 w_ref = -w_normal + w_tan;
							w_ref = glm::normalize(w_ref);

							ray reflect(r.at(i) + normal * 1e-12, w_ref, r.getAtten(), ray::REFLECTION);
							// ray reflect(r.at(i) + normal * 1e-12, w_ref + (rand_dir*.1), r.getAtten(), ray::REFLECTION);
							reflect.currentIndex = r.currentIndex;

							glm::dvec3 temp = pathTraceRay(reflect, depth - 1, bvhTree);
							colorC += ((m.ks(i) * temp) * coeffRR);
						}	
					} else {
						//Total Internal Reflection 
						glm::dvec3 w_in = r.getDirection();
						glm::dvec3 w_normal = glm::dot(w_in, normal) * normal;
						glm::dvec3 w_tan = w_in - w_normal;
						glm::dvec3 w_ref = -w_normal + w_tan;

						w_ref = glm::normalize(w_ref);
						ray reflect(r.at(i) + normal * 1e-12, w_ref, r.getAtten(), ray::REFLECTION);
						reflect.currentIndex = r.currentIndex;

						glm::dvec3 temp = pathTraceRay(reflect, depth - 1, bvhTree);
						colorC += ((m.kr(i) * trans * temp) * coeffRR);
					}
				}	
				
		} 
	} else {
			// No intersection.  This ray travels to infinity, so we color
			// it according to the background color, which in this (simple) case
			// is just black.
			colorC = glm::dvec3(0.0, 0.0, 0.0);
			if(traceUI->cubeMap()){
				CubeMap* cubeMap = traceUI->getCubeMap();
				glm::dvec3 kd = cubeMap->getColor(r);
				colorC += kd;
			}
	}
	// if(depth >= 1){
	// 	return colorC - myContrib;
	// }
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
glm::dvec3 pathTraceRayGPU(GPU::Ray& in, GPU::Scene* scene, int depth, curandState local_rand_state, bool print ) {
	GPU::Isect i;
	glm::dvec3 colorC;

	GPU::Ray curRay(in);
	glm::dvec3 curAtten(1.0,1.0,1.0);
	for(int bounce = 0; bounce <= depth; bounce++){
		if(scene->intersect(curRay, i)) {
			GPU::Material* m = i.getMaterial();	
			colorC += m->ke() * curAtten;

			if(!(m->Diff() || m->Spec())){
				return colorC;
			}

			// Use probabilities for sampling
			double spec = (m->ks().x + m->ks().y + m->ks().z) / 3.0;
			double diff = (m->kd().x + m->kd().y + m->kd().z) / 3.0;
			double trans = (m->kt().x + m->kt().y + m->kt().z) / 3.0;
			double total = spec + diff + trans;

			double diffProb = diff / total;
			double specProb = spec / total;
			double transProb = trans / total;


			double prob = curand_uniform_double(&local_rand_state);

			if(prob < diffProb) {
				glm::dvec3 normal = glm::normalize(i.getN());
				glm::dvec3 rand_dir = sampleCosineWeightedHemisphere(normal, &local_rand_state);

				constexpr double p = 1 / PI;
				constexpr double theta = 1;

				curRay = GPU::Ray(curRay.at(i) + normal * 1e-12, rand_dir);
				glm::dvec3 brdf = m->kd() / PI;
				curAtten *= brdf * theta / (p * diffProb);
			} else if (prob < diffProb + specProb){
				glm::dvec3 w_in = glm::normalize(curRay.getDirection());
				glm::dvec3 normal = glm::normalize(i.getN());

				if(curRay.currentIndex != 1.0){
					normal *= -1.0;
				}
				glm::dvec3 w_normal = glm::dot(w_in, normal) * normal;
				glm::dvec3 w_tan = w_in - w_normal;
				glm::dvec3 w_ref = -w_normal + w_tan;
				w_ref = glm::normalize(w_ref);

				curRay = GPU::Ray(curRay.at(i) + normal * 1e-12, w_ref);
				 
				curAtten *= m->ks() * specAtten(i)/ specProb;

			} else {
				// cout << "Transmissive" << endl;
				glm::dvec3 w_in = curRay.getDirection();
				glm::dvec3 normal = i.getN();

				double n1;
				double n2;
				glm::dvec3 trans = {1,1,1};
				//Based on entering/exiting a material, set n1, n2, and trans
				if(curRay.currentIndex == 1.0){
					n1 = curRay.currentIndex;
					n2 = m->index();
				} else{
					n1 = m->index();
					n2 = 1.0;
					normal *= -1;
					trans = glm::pow(m->kt(), {i.getT(),i.getT(),i.getT()});
				}

				double r0 = (n1 - n2)/(n1 + n2);
				r0 *= r0;

				double n = n1/n2;

				w_in = -glm::normalize(w_in);
				
				double cosI = glm::dot(normal, w_in); //cosine of theta_1 (incident angle)

				double cosR = 1 - n*n * (1-cosI*cosI); //cosine of theta_2 (refraction angle)
				if(cosR >= 0){
					//Potentially Refract/Reflect
					double reflCoeff = r0 + (1 - r0)*pow((1 - cosI), 5.0); //Schlick's Approximation of Fresnel's Constant

					double sampleFresnel = curand_uniform_double(&local_rand_state);

					if (sampleFresnel > reflCoeff) {
						//refraction
						double cosT = glm::sqrt(cosR);
						glm::dvec3 refrac = (n*cosI - cosT) * normal - n*w_in; //direction of the refracted ray

						curRay = GPU::Ray(curRay.at(i) - normal * 1e-12, refrac);
						curRay.currentIndex = n2;

						// glm::dvec3 temp = pathTraceRay(r2, depth - 1, bvhTree);

						// colorC += trans * temp;
						curAtten *= trans;
					} else {
						//reflection
						glm::dvec3 w_in = curRay.getDirection();
						glm::dvec3 normal = i.getN();
						if(curRay.currentIndex != 1.0){
							normal *= -1.0;
						}
						glm::dvec3 w_normal = glm::dot(w_in, normal) * normal;
						glm::dvec3 w_tan = w_in - w_normal;
						glm::dvec3 w_ref = -w_normal + w_tan;
						w_ref = glm::normalize(w_ref);

						double oldIndex = curRay.currentIndex;
						curRay = GPU::Ray(curRay.at(i) + normal * 1e-12, w_ref);
						curRay.currentIndex = oldIndex;
						// reflect.currentIndex = r.currentIndex;

						// glm::dvec3 temp = pathTraceRay(reflect, depth - 1, bvhTree);
						// colorC += m->ks(i) * temp;
						curAtten *= m->ks();
					}	
				} else {
					//Total Internal Reflection 
					glm::dvec3 w_in = curRay.getDirection();
					glm::dvec3 w_normal = glm::dot(w_in, normal) * normal;
					glm::dvec3 w_tan = w_in - w_normal;
					glm::dvec3 w_ref = -w_normal + w_tan;

					w_ref = glm::normalize(w_ref);

					double oldIndex = curRay.currentIndex;
					curRay = GPU::Ray(curRay.at(i) + normal * 1e-12, w_ref);
					curRay.currentIndex = oldIndex;

					// glm::dvec3 temp = pathTraceRay(reflect, depth - 1, bvhTree);
					// colorC += m->kr() * trans * temp;
					curAtten *= m->kr() * trans;
				}
				curAtten *= 1 / transProb;
			}
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
void pathTraceDOFKernel(unsigned char* buf, int w, int h, GPU::Scene* scene, int maxDepth, int samplesPerPixel, curandState *rand_state, double focal_distance, double aperture) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= w) || (j >= h)) return;

	int pixelIndex = i + j * w;

	//Each thread gets same seed, a different sequence number, no offset
   	curand_init(1984, pixelIndex, 0, &rand_state[pixelIndex]);
	// Local random state for this pixel
	curandState local_rand_state = rand_state[pixelIndex];

	//Average samples
	glm::dvec3 colSum(0, 0, 0);

	//Get Focal Point
	glm::dvec3 eye = scene->camera.getEye();
	double x = double(i)/double(w);
	double y = double(j)/double(h);
    GPU::Ray temp(glm::dvec3(0, 0, 0), glm::dvec3(0, 0, 0));
    scene->camera.rayThrough(x, y, temp);
	glm::dvec3 focalPoint = temp.at(focal_distance);

	for(int sample = 0; sample < samplesPerPixel; sample++){
		double iShift = curand_uniform_double(&local_rand_state);
		double jShift = curand_uniform_double(&local_rand_state);

		iShift *= (2 * aperture) - aperture;
		jShift *= (2 * aperture) - aperture;

		glm::dvec3 jitteredEye = eye + scene->camera.getU() * iShift + scene->camera.getV() * jShift;
        // cout << "jitteredEye: " << jitteredEye << endl;
        GPU::Ray r(jitteredEye, glm::normalize(focalPoint - jitteredEye));

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
	// Enable/disable depth of field with below config
	const bool USE_DOF = false;
    // Good config for easy3: fd = 8.5, a = 0.2
	// Good config for reflection2: fd = 6, a = 0.3
    const double FOCAL_DISTANCE = 8.5;
    // Side length of camera aperture (square)
    const double APERTURE = 0.6;
	
	int N = w * h;
	
	// Load scene on GPU
    GPU::Scene* d_gpuScene = new GPU::Scene(scene.get());
	d_gpuScene->copyBVH(&bvhTree);

	// Copy over buffer for writing
	unsigned char* d_buf = copy_host_to_device(buffer.data(), buffer.size() * sizeof(unsigned char));


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
	if(USE_DOF) {
		pathTraceDOFKernel<<<blocks, threads>>>(d_buf, w, h, d_gpuScene, traceUI->getDepth(), SAMPLES_PER_PIXEL, d_rand_state, FOCAL_DISTANCE, APERTURE);
	} else {
		pathTraceKernel<<<blocks, threads>>>(d_buf, w, h, d_gpuScene, traceUI->getDepth(), SAMPLES_PER_PIXEL, d_rand_state);
	}
	gpuErrchk(cudaDeviceSynchronize());
	// Copy buffer back
	cudaMemcpy(buffer.data(), d_buf, buffer.size() * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
}