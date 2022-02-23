#pragma once

#include <glm/vec3.hpp>
#include "ray.h"
#include "scene.h"
#include <vector>

struct BVHNode{
    BVHNode(BVHNode* l, BVHNode* r, Geometry* g) : left(l), right(r), geometry(g) {}
    BVHNode* left;
    BVHNode* right;
    Geometry* geometry;
    BoundingBox bb;
};

class BVHTree {

public:
    BVHTree();
    void build(std::unique_ptr<Scene>& scene);
    bool traverse(ray& r, isect& i);
    BVHNode* root;

protected:
    BVHNode* buildHelper(std::vector<Geometry*>& geo, int st, int end);
    isect traverseHelper(ray& r, BVHNode* n);

    bool built;
    int size;
    int leaves;
    int misses;
    int traverseCount;
    Scene* scene;
};
