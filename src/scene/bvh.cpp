#include "bvh.h"
#include "../SceneObjects/trimesh.h"
#include "../ui/TraceUI.h"
#include <glm/gtx/io.hpp>
#include <iostream>
#include <limits>

extern bool debugMode;

using namespace std;

BVHTree::BVHTree(){
    built = false;
    size = 0;
    leaves = 0;
    misses = 0;
}

void BVHTree::build(unique_ptr<Scene>& scene){
    this->scene = scene.get();

    if(built) {
        cout << "built" << endl; 
        return;
    }
    vector<Geometry*> geo;
    for(Geometry* g: scene->objects){
        auto x = g->getAll();
        geo.insert(geo.end(), x.begin(), x.end());
    }

    root = buildHelper(geo, 0, geo.size());
    built = true;

    cout << leaves << " " << geo.size() << endl;
    if(leaves != geo.size()){
        cout << "missing elements" << endl;
        exit(0);
    }
}

BoundingBox getBox(vector<Geometry*>& geo, int st, int end){
    double minX, minY, minZ = numeric_limits<double>::max();
    double maxX, maxY, maxZ = numeric_limits<double>::min();

    for(int i = st; i < end; i++){
        if(!geo[i]->hasBoundingBoxCapability()){
            cout << "rip" << endl;
            exit(0);
        }

        glm::dvec3 mins = geo[i]->getBoundingBox().getMin();
        glm::dvec3 maxs = geo[i]->getBoundingBox().getMax();

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
    size++;

    if(st >= end){
        cout << "geo 0 " << endl;
        return nullptr;
    }
    // Leaf node
    if(end - st <= 1){
        node->geometry = geo[st];
        leaves++;
        return node;
    }

    // Compute bounding boxes
    node->bb = getBox(geo, st, end);

    glm::dvec3 mins = node->bb.getMin();
    glm::dvec3 maxs = node->bb.getMax();

    auto diff = maxs - mins;

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

    // cout << "ci: " << centerIndex << endl;
    // for(int i = 0; i < 10; i++){
    //     TrimeshFace* tri = ((TrimeshFace* )geo[i]);

    //     cout << "v: " << tri->parent->vertices[tri->ids[0]] << " " << tri->parent->vertices[tri->ids[1]] << " " << tri->parent->vertices[tri->ids[2]] << endl;
    //     // auto x  = (geo[i]->getBoundingBox().getMin() + geo[i]->getBoundingBox().getMax()) / glm::dvec3 { 2.0, 2.0, 2.0 };
    //     // cout << x << endl;
    //     cout << "min: " << geo[i]->getBoundingBox().getMin() << endl;
    //     cout << "max: " << geo[i]->getBoundingBox().getMax() << endl;

    // }
    // exit(0);

    sort(geo.begin() + st, geo.begin() + end, [=](Geometry* a, Geometry* b) {
        glm::dvec3 a_c = (a->getBoundingBox().getMin() + a->getBoundingBox().getMax()) / glm::dvec3 { 2.0, 2.0, 2.0 };
        glm::dvec3 b_c = (b->getBoundingBox().getMin() + b->getBoundingBox().getMax()) / glm::dvec3 { 2.0, 2.0, 2.0 };
        return a_c[centerIndex] < b_c[centerIndex];
    });

    // for(int i = 0; i < geo.size(); ++i){
    //     if (i < geo.size() / 2) {
    //         l.push_back(geo[i]);
    //     } else{
    //         r.push_back(geo[i]);
    //     }
    // }

    int mid = st + (end - st) / 2;

    // cout << "starting children" << endl;
    // cout << "l: " << mid - st << endl;
    // cout << "r: " << end - mid << endl;
    // cout << "entering l" << endl;
    node->left = buildHelper(geo, st, mid);
    // cout << "entering r" << endl;
    node->right = buildHelper(geo, mid, end);
    return node;
}


bool BVHTree::traverse(ray& r, isect& i)
{
    traverseCount = 0;
    misses = 0;
    if(debugMode) cout << "--- starting traverse --- " << endl;
    i = traverseHelper(r, root);
    bool res = i.getT() != -1;
    if(!res){
        i.setT(1000.0);
    }
    if (TraceUI::m_debug) {
        scene->addToIntersectCache(std::make_pair(new ray(r), new isect(i)));
    }

    // cout << "prune percentage: " << (double) misses / traverseCount << endl;
    return res;
}

isect BVHTree::traverseHelper(ray& r, BVHNode* n){

    if(debugMode) cout << "traversing" << endl;
    traverseCount++;
    isect res;
    res.setT(-1);

    if(!n){
        if(debugMode) cout << "off tree" << endl;
        return res;
    }
    

    // If leaf node
    if (n->geometry) {
        if(debugMode) cout << "in leaf" << endl;
        if(debugMode) cout << "ray" << r.getDirection() << " pos: " << r.getPosition() << endl;
        if(n->geometry->intersect(r, res)){
            if(debugMode) cout << "got valid int" << endl;
            return res;
        }
        res.setT(-1);
        return res;
    }

    // Check bounding box of current
    double tMin;
    double tMax;
    if(debugMode){
        cout << "checking current bb" << endl;
        cout << n->bb.getMax() << endl;
        cout << n->bb.getMin() << endl;
    }
    if(!n->bb.intersect(r, tMin, tMax)){
        if(debugMode) cout << "skipping bb" << endl;
        misses++;
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