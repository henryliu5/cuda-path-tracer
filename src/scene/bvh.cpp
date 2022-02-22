#include "bvh.h"
#include <glm/gtx/io.hpp>
#include <iostream>
#include <limits>

using namespace std;

BVHTree::BVHTree(){
    built = false;
}

void BVHTree::build(unique_ptr<Scene>& scene){
    if(built) {
        cout << "built" << endl; 
        return;
    }
    vector<Geometry*> geo;
    cout << "build" << endl;
    for(Geometry* g: scene->objects){
        auto x = g->getAll();
        geo.insert(geo.end(), x.begin(), x.end());
    }
    cout << "build" << endl;
    root = buildHelper(geo, 0, geo.size());
    built = true;
}

BoundingBox getBox(vector<Geometry*> geo){
    double minX, minY, minZ = numeric_limits<double>::max();
    double maxX, maxY, maxZ = numeric_limits<double>::min();

    for(Geometry* g: geo){
        if(!g->hasBoundingBoxCapability()){
            cout << "rip" << endl;
            exit(0);
        }

        glm::dvec3 mins = g->getBoundingBox().getMin();
        glm::dvec3 maxs = g->getBoundingBox().getMax();

        minX = min(mins.x, minX);
        minY = min(mins.y, minY);
        minZ = min(mins.z, minZ);

        maxX = max(maxs.x, maxX);
        maxY = max(maxs.y, maxY);
        maxZ = max(maxs.z, maxZ);
    }

    return BoundingBox(glm::dvec3(minX, minY, minZ), glm::dvec3(maxX,maxY,maxZ));
}

BVHNode* BVHTree::buildHelper(vector<Geometry*>& geo, int st, int end){
    auto node = new BVHNode(0, 0, nullptr);
    if(st >= end){
        cout << "geo 0 " << endl;
        exit(0);
    }
    cout << "x" << endl;
    cout << end - st << endl;
    cout << "size" << endl;
    // Leaf node
    if(end - st <= 1){
        node->geometry = geo[st];
        return node;
    }
    cout << "2" << endl;
    // Compute bounding boxes
    node->bb = getBox(geo);

    glm::dvec3 mins = node->bb.getMin();
    glm::dvec3 maxs = node->bb.getMax();

    auto diff = maxs - mins;

    cout << "mins: " << mins << endl;
    cout << "maxs: " << maxs << endl;

    // Find longest axis
    double center;
    int centerIndex;
    if(diff[0] > diff[1] && diff[0] > diff[2]){
        // x
        center = diff[0] / 2 + mins[0];
        centerIndex = 0;
    } else if(diff[1] > diff[0] && diff[1] > diff[2]){
        // y
        center = diff[1] / 2 + mins[1];
        centerIndex = 1;
    } else {
        // z
        center = diff[2] / 2 + mins[2];
        centerIndex = 2;
    }

    vector<Geometry*> l, r;
    sort(geo.begin() + st, geo.begin() + end, [=](Geometry* a, Geometry* b) {
        glm::dvec3 a_c = (a->getBoundingBox().getMin() + a->getBoundingBox().getMax()) / glm::dvec3 { 2.0, 2.0, 2.0 };
        glm::dvec3 b_c = (b->getBoundingBox().getMin() + b->getBoundingBox().getMax()) / glm::dvec3 { 2.0, 2.0, 2.0 };
        return a_c[centerIndex] < b_c[centerIndex];
    });

    // for(int i = 0; i < geo.size(); i++){
    //     if (i < geo.size() / 2) {
    //         l.push_back(geo[i]);
    //     } else{
    //         r.push_back(geo[i]);
    //     }
    // }

    int mid = st + (end - st) / 2;

    cout << "starting children" << endl;
    cout << "l: " << mid - st << endl;
    cout << "r: " << end - mid << endl;
    cout << "entering l" << endl;
    node->left = buildHelper(geo, st, mid);
    cout << "entering r" << endl;
    node->right = buildHelper(geo, mid, end);
    return node;
}


bool BVHTree::traverse(ray& r, isect& i)
{
    i = traverseHelper(r, root);
    return i.getT() != -1;
}

isect BVHTree::traverseHelper(ray& r, BVHNode* n){
    isect res;
    if(!n){
        return res;
    }
    res.setT(-1);

    // If leaf node
    if (n->geometry) {
        if(n->geometry->intersect(r, res)){
            return res;
        }
        res.setT(-1);
        return res;
    }

    // Check bounding box of current
    double tMin = -1;
    double tMax = -1;
    if(!n->bb.intersect(r, tMin, tMax)){
        return res;
    }
    
    isect li = traverseHelper(r, n->left);
    isect ri = traverseHelper(r, n->right);

    if(li.getT() == -1){
        return ri;
    } else if(ri.getT() == -1){
        return li;
    } else {
        return li.getT() < ri.getT() ? li : ri;
    }
}