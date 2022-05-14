#include <glm/glm.hpp>
#include "RayTracer.h"
#include "gpu/GPUScene.cuh"
#include "ui/TraceUI.h"
#include <curand.h>
#include <curand_kernel.h>
#include <glm/gtx/io.hpp>
#include "gpu/GPUManaged.cuh"
#include "path_tracing.cuh"
#include <thrust/partition.h>

extern TraceUI* traceUI;


class PathSegment : public GPU::GPUManaged {
public:
    CUDA_CALLABLE_MEMBER
    PathSegment() :
        ray({1,1,1}, {1,1,1}),
        color(0.0,0.0,0.0),
        previousColorSum(0.0,0.0,0.0),
        atten(1.0,1.0,1.0) {}

    __device__
    void initIndex(int index) {
        myIndex = index;
    }

    __device__
    void reset(GPU::Ray& r, int bounces) {
        ray = r;
        // Accumulate
        previousColorSum += color;
        color = glm::dvec3(0.0,0.0,0.0);
        atten = glm::dvec3(1.0,1.0,1.0);
        remainingBounces = bounces;
    }

    GPU::Ray ray;
    glm::dvec3 previousColorSum;
    glm::dvec3 color;
    glm::dvec3 atten;
    int myIndex;
    int remainingBounces;
};

// Compaction inspired by https://github.com/hanbollar/CUDA-Stream-Compaction#additional-features
struct split_by_completed {
  __host__ __device__
  bool operator() (const PathSegment &segment) {
    return segment.remainingBounces > 0;
  }
};


__global__ void pathTrace(PathSegment* pathSegments, int numPaths, GPU::Scene* scene, int maxDepth, curandState *rand_state) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Update all path segments this thread is responsible for
    for (int pathIndex = index; pathIndex < numPaths; pathIndex += stride){
        
        PathSegment& pathSegment = pathSegments[pathIndex];

        // Path segment already completed
        if(pathSegment.remainingBounces <= 0) continue;
        
        // Set references to adjust this path segment
        GPU::Ray& curRay = pathSegment.ray;
        glm::dvec3& curAtten = pathSegment.atten;
        glm::dvec3& colorC = pathSegment.color;
        pathSegment.remainingBounces--;

        // Try intersecting with scene, will update pathSegment
        GPU::Isect i;
        if(scene->intersect(curRay, i)) {
            GPU::Material* m = i.getMaterial();	
            colorC += m->ke() * curAtten;

            if(!(m->Diff() || m->Spec() || m->Trans())){
                // Stop this path segment
                pathSegment.remainingBounces = 0;
                return;
            }

            // Use probabilities for sampling
            double spec = (m->ks().x + m->ks().y + m->ks().z) / 3.0;
            double diff = (m->kd().x + m->kd().y + m->kd().z) / 3.0;
            double trans = (m->kt().x + m->kt().y + m->kt().z) / 3.0;
            double total = spec + diff + trans;

            double diffProb = diff / total;
            double specProb = spec / total;
            double transProb = trans / total;


            double prob = curand_uniform_double(&rand_state[pathIndex]);

            if(prob < diffProb) {
                glm::dvec3 normal = glm::normalize(i.getN());
                glm::dvec3 rand_dir = sampleCosineWeightedHemisphere(normal, &rand_state[pathIndex]);

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

                    double sampleFresnel = curand_uniform_double(&rand_state[pathIndex]);

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
        } else {
            // Missed scene
            pathSegment.remainingBounces = 0;
        }

    }
}

// Shoot initial rays through camera to populate initial pathSegments
__global__ void initRays(PathSegment* pathSegments, int n, int w, int h, GPU::Scene* scene, int maxDepth, curandState *rand_state, int sampleNum) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i >= w) || (j >= h)) return;

	int pathIndex = i + j * w;

    PathSegment& path = pathSegments[pathIndex];
    if(sampleNum == 0){
        path.initIndex(pathIndex);
    }

    // Generate initial ray for this path
    GPU::Ray r(glm::dvec3(0,0,0), glm::dvec3(0,0,0));
    double iShift = curand_uniform_double(&rand_state[pathIndex]);
    double jShift = curand_uniform_double(&rand_state[pathIndex]);

    int pixelI = path.myIndex % w;
    int pixelJ = path.myIndex / w;
    double newI = pixelI + iShift;
    double newJ = pixelJ + jShift;

    double x = newI/double(w);
    double y = newJ/double(h);

    scene->camera.rayThrough(x, y, r);

    // Bind new ray to path
    path.reset(r, maxDepth);
}

// Takes resulting colors from completed path segments and writes to color sum buffer
__global__ void fillColors(double* colorBuffer, int w, int h, PathSegment* pathSegments){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i >= w) || (j >= h)) return;
    int pixelIndex = i + j * w;

    colorBuffer[pixelIndex*3] = pathSegments[pixelIndex].color.x;
	colorBuffer[pixelIndex*3 + 1] = pathSegments[pixelIndex].color.y;
	colorBuffer[pixelIndex*3 + 2] = pathSegments[pixelIndex].color.z;
}

__global__ void randInit(int w, int h, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i >= w) || (j >= h)) return;

	int pixelIndex = i + j * w;
    //Each thread gets same seed, a different sequence number, no offset
   	curand_init(1984, pixelIndex, 0, &rand_state[pixelIndex]);
}

void RayTracer::betterPathTracing(int w, int h){
	// Load scene on GPU
    GPU::Scene* d_gpuScene = new GPU::Scene(scene.get());
	d_gpuScene->copyBVH(&bvhTree);

    PathSegment* d_pathSegments;
    // Allocate path segment for every pixel
    const long long n = w * h;
    gpuErrchk(cudaMalloc(&d_pathSegments, n * sizeof(PathSegment)));

    cout << "finished allocation" << endl;

    // Create random state for each path segment
   	curandState *d_rand_state;
   	gpuErrchk(cudaMalloc((void **)&d_rand_state, n * sizeof(curandState)));

    int threadsX = 8;
    int threadsY = 8;
    dim3 blocks(w/threadsX+1, h/threadsY+1); dim3 threads(threadsY, threadsY);
    randInit<<<blocks, threads>>>(w, h, d_rand_state);

    gpuErrchk(cudaDeviceSynchronize());

    for(int sample = 0; sample < SAMPLES_PER_PIXEL; sample++){
        // if(sample % 10 == 0)
            // cout << "sample: " << sample << "\n";
        /*************************
         * Generate initial rays *
         *************************/
        {
            int threadsX = 8;
            int threadsY = 8;
            dim3 blocks(w/threadsX+1, h/threadsY+1); dim3 threads(threadsY, threadsY);
            initRays<<<blocks, threads>>>(d_pathSegments, n, w, h, d_gpuScene, traceUI->getDepth() + 1, d_rand_state, sample);

            gpuErrchk(cudaDeviceSynchronize());
        }

        /*****************************
         * Path trace to next bounce *
         *****************************/
        int numPaths = n;
        for(int depth = 0; depth < traceUI->getDepth() + 1; depth++){
            // Partition array so nonzero bounces are in front
            if(depth != 0 && depth % 7 == 0){
                PathSegment* pivotIndex = thrust::partition(thrust::device, d_pathSegments, d_pathSegments + numPaths, split_by_completed());
                numPaths = pivotIndex - d_pathSegments;
            }

            // Only launch kernels on paths with nonzero bounces remaining
            int blockSize = 256;
            int numBlocks = (numPaths + blockSize - 1) / blockSize;
            pathTrace<<<numBlocks, blockSize>>>(d_pathSegments, numPaths, d_gpuScene, traceUI->getDepth(),  d_rand_state);

            gpuErrchk(cudaDeviceSynchronize());
        }
    }

    // /****************************************************
    //  * Write path segment results into color sum buffer *
    //  ****************************************************/
    PathSegment* c_pathSegments = copy_device_to_host(d_pathSegments,  n * sizeof(PathSegment));
    for(int i = 0; i < w * h; i++){
        glm::dvec3 colSum = c_pathSegments[i].previousColorSum + c_pathSegments[i].color;
        colSum /= SAMPLES_PER_PIXEL;
        colSum = glm::clamp(colSum, 0.0, 1.0);
        int pixelIndex = c_pathSegments[i].myIndex;
        setPixel(pixelIndex % w, pixelIndex / w, colSum);
    }

}