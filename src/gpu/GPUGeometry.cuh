#ifndef GPU_GEOMETRY_H
#define GPU_GEOMETRY_H

#include "GPUManaged.cuh"
#include "GPUIsect.cuh"
#include "GPURay.cuh"
#include "GPUGeometry.cuh"
#include <cuda.h>

namespace GPU {

class Geometry : public GPUManaged {
public:
    enum GeometryType { NONE, TRIMESH, TRIMESH_FACE, SPHERE }; // Avoid virtual function calls
    GeometryType myType;

    Geometry() : myType(NONE) {}

    Geometry(GeometryType g) : myType(g) {}

    CUDA_CALLABLE_MEMBER bool intersect(GPU::Ray& r, GPU::Isect& i);
};

}

#endif