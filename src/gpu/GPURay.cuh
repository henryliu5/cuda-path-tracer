#ifndef GPU_RAY_H
#define GPU_RAY_H

#include "GPUManaged.cuh"
#include "GPUIsect.cuh"
#include <cuda.h>

namespace GPU {

class Ray : public GPUManaged {
public:
    CUDA_CALLABLE_MEMBER Ray(const glm::dvec3& pp, const glm::dvec3& dd) : p(pp), d(dd) {}
    
	CUDA_CALLABLE_MEMBER glm::dvec3 at(double t) const { return p + (t * d); }
	CUDA_CALLABLE_MEMBER glm::dvec3 at(const GPU::Isect& i) const;

	CUDA_CALLABLE_MEMBER glm::dvec3 getPosition() const { return p; }
	CUDA_CALLABLE_MEMBER glm::dvec3 getDirection() const { return d; }

	CUDA_CALLABLE_MEMBER void setPosition(const glm::dvec3& pp) { p = pp; }
	CUDA_CALLABLE_MEMBER void setDirection(const glm::dvec3& dd) { d = dd; }


	glm::dvec3 p;
	glm::dvec3 d;
};

}

#endif