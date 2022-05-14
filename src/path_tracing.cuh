#pragma once
#include "glm/vec3.hpp"

__device__
glm::dvec3 sampleCosineWeightedHemisphere(glm::dvec3& normal, curandState* local_rand_state);
__host__ __device__ double specAtten(GPU::Isect& i);