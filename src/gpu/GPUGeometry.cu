#include "GPUGeometry.cuh"
#include "GPUTrimesh.cuh"

bool GPU::Geometry::intersect(GPU::Ray& r, GPU::Isect& i){
    if(myType == TRIMESH){
        return ((GPU::Trimesh *) this)->intersect(r, i);
    }
    if(myType == TRIMESH_FACE){
        return ((GPU::TrimeshFace *) this)->intersect(r, i);
    }
    return false;
}