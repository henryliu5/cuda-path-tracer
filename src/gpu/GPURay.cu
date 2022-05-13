#include "GPURay.cuh"
#include "GPUIsect.cuh"

glm::dvec3 GPU::Ray::at(const GPU::Isect& i) const
{
	return at(i.getT());
}
