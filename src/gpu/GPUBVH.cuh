#ifndef GPU_BVH_H
#define GPU_BVH_H

#include "GPUManaged.cuh"
#include "GPUIsect.cuh"
#include "GPUbb.cuh"
#include "GPUGeometry.cuh"
#include "../scene/bvh.h"
#include <cuda.h>
#include <unordered_map>

namespace GPU {

class Geometry;
class Scene;

class BVHNode : public GPUManaged {
public:
    BVHNode(GPU::BVHNode* l, GPU::BVHNode* r, GPU::Geometry* g) : left(l), right(r), geometry(g) {}
    void setBB(GPU::BoundingBox b){ bb = b; }
    BVHNode* left;
    BVHNode* right;
    GPU::Geometry* geometry;
    GPU::BoundingBox bb;

    CUDA_CALLABLE_MEMBER bool isLeaf() { return left == nullptr && right == nullptr; }
};


class BVH : public GPUManaged {
public:
    BVH(GPU::Scene* scene){}
    // Copy from CPU scene
    BVH(BVHTree* b, std::unordered_map<::Geometry*, GPU::Geometry*>& cpuToGpuGeo);
    
	// void build(GPU::Scene* scene);
    CUDA_CALLABLE_MEMBER bool traverse(GPU::Ray& r, GPU::Isect& i);
    CUDA_CALLABLE_MEMBER bool traverseIterative(GPU::Ray& r, GPU::Isect& i);
    GPU::BVHNode* root;

protected:
    GPU::BVHNode* copyHelper(::BVHNode* cpuNode, std::unordered_map<::Geometry*, GPU::Geometry*>& cpuToGpuGeo);
	CUDA_CALLABLE_MEMBER GPU::Isect traverseHelper(GPU::Ray& r, GPU::BVHNode* n);
    int numNodes;
};

}

#endif