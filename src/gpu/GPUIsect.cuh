#ifndef GPU_ISECT_H
#define GPU_ISECT_H

#include "GPUManaged.cuh"
#include "GPUMaterial.cuh"
#include <cuda.h>

namespace GPU {

class Isect : public GPUManaged {
public:
    CUDA_CALLABLE_MEMBER Isect() : t(0.0), N(), material(nullptr) {}

	// Get/Set Time of flight
	CUDA_CALLABLE_MEMBER void setT(double tt) { t = tt; }
	CUDA_CALLABLE_MEMBER double getT() const { return t; }
	// Get/Set surface normal at this intersection.
	CUDA_CALLABLE_MEMBER void setN(const glm::dvec3& n) { N = n; }
	CUDA_CALLABLE_MEMBER glm::dvec3 getN() const { return N; }

	CUDA_CALLABLE_MEMBER void setMaterial(Material* m)
	{
		// if (material)
			// *material = m;
		// else
			// material.reset(new Material(m));
        material = m;
	}

	CUDA_CALLABLE_MEMBER GPU::Material* getMaterial(){ return material; }

    double t;
	glm::dvec3 N;
    Material* material;
};

}

#endif