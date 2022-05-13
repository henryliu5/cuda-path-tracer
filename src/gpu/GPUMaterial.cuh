#ifndef GPU_MATERIAL_H
#define GPU_MATERIAL_H

#include "GPUManaged.cuh"
#include "../scene/material.h"
#include <cuda.h>

namespace GPU {

class Material : public GPUManaged {
public:
    CUDA_CALLABLE_MEMBER Material(glm::dvec3 kd, glm::dvec3 ke) : _kd(kd), _ke(ke) {}
    
    // TODO handle specular stuff
    CUDA_CALLABLE_MEMBER bool Recur() { return false; }

    CUDA_CALLABLE_MEMBER glm::dvec3 ke() { return _ke; }
    CUDA_CALLABLE_MEMBER glm::dvec3 kd() { return _kd; }

protected:
    glm::dvec3 _ke;
    glm::dvec3 _kd;
};

}

#endif