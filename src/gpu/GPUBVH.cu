#include "GPUBVH.cuh"
#include "../scene/bvh.h"
#include <unordered_map>
#include "GPUScene.cuh"
#include <glm/gtx/io.hpp>

namespace GPU {

BVH::BVH(BVHTree* cpuTree, std::unordered_map<::Geometry*, GPU::Geometry*>& cpuToGpuGeo) : numNodes(0) {
    root = copyHelper(cpuTree->root, cpuToGpuGeo);
    cout << "BVH Copied: " << numNodes << " nodes" << endl;
}

GPU::BVHNode* BVH::copyHelper(::BVHNode* cpuNode, std::unordered_map<::Geometry*, GPU::Geometry*>& cpuToGpuGeo){
    if(!cpuNode){
        return nullptr;
    }
    GPU::BVHNode* gpuLeft = copyHelper(cpuNode->left, cpuToGpuGeo);
    GPU::BVHNode* gpuRight = copyHelper(cpuNode->right, cpuToGpuGeo);

    // Copy this node
    GPU::BoundingBox bb(cpuNode->bb.getMin(), cpuNode->bb.getMax());
    // cout << "min: " << bb.bmin << " max: " << bb.bmax << endl;
    // cout << "cpu address: " << cpuNode->geometry << endl;
    // cout << "address: " << cpuToGpuGeo[cpuNode->geometry] << endl;
    GPU::BVHNode* gpuNode = new GPU::BVHNode(gpuLeft, gpuRight, cpuToGpuGeo[cpuNode->geometry]);
    gpuNode->setBB(bb);

    numNodes++;
    return gpuNode;
}

// void BVH::build(GPU::Scene* scene){

// }

// Inspired by https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/
bool BVH::traverseIterative(GPU::Ray& r, GPU::Isect& i){
    // Allocate traversal stack from thread-local memory,
    // and push NULL to indicate that there are no postponed nodes.
    GPU::BVHNode* stack[64];
    GPU::BVHNode** stackPtr = stack;
    *stackPtr++ = NULL; // push
    i.setT(-1);
    // Traverse nodes starting from the root.
    GPU::BVHNode* node = root;
    int count = 0;
    do
    {
        count++;
        // Check each child node for overlap.
        GPU::BVHNode* childL = node->left;
        GPU::BVHNode* childR = node->right;

        // Check bounding box of left
        double tMinL;
        double tMaxL;
        bool overlapL = childL->bb.intersect(r, tMinL, tMaxL);

        double tMinR;
        double tMaxR;
        bool overlapR = childR->bb.intersect(r, tMinR, tMaxR);

        // Query overlaps a leaf node => test intersection and take min
        if (overlapL && childL->isLeaf()){
            GPU::Isect test;
            if(childL->geometry->intersect(r, test)){
                if(i.getT() < 0 || test.getT() < i.getT()){
                    i = test;
                }
            }
        }
            
        if (overlapR && childR->isLeaf()){
            GPU::Isect test;
            if(childR->geometry->intersect(r, test)){
                if(i.getT() < 0 || test.getT() < i.getT()){
                    i = test;
                }
            }
        }
            
        // Query overlaps an internal node => traverse.
        bool traverseL = (overlapL && !childL->isLeaf());
        bool traverseR = (overlapR && !childR->isLeaf());

        if (!traverseL && !traverseR)
            node = *--stackPtr; // pop
        else
        {
            node = (traverseL) ? childL : childR;
            if (traverseL && traverseR)
                *stackPtr++ = childR; // push
        }
    }
    while (node != NULL);

    return i.getT() >= 0;
}

bool BVH::traverse(GPU::Ray& r, GPU::Isect& i){
    i = traverseHelper(r, root);
    return i.getT() >= 0;
}

GPU::Isect BVH::traverseHelper(GPU::Ray& r, GPU::BVHNode* n){
    GPU::Isect res;
    res.setT(-1);

    if (!n) {
        return res;
    }

    // Check bounding box of current
    double tMin;
    double tMax;

    if(!n->bb.intersect(r, tMin, tMax)){
        // if(debugMode) cout << "skipping bb" << endl;
        // misses++;
        res.setT(-2);
        return res;
    }

    // If leaf node
    if (n->geometry) {
        if(n->geometry->intersect(r, res)){
            return res;
        }
        res.setT(-1);
        return res;
    }

    GPU::Isect li = traverseHelper(r, n->left);
    GPU::Isect ri = traverseHelper(r, n->right);

    if(li.getT() < 0){
        return ri;
    } else if(ri.getT() < 0){
        return li;
    } else {
        return li.getT() < ri.getT() ? li : ri;
    }
}

}