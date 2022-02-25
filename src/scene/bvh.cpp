#include "bvh.h"
#include "../SceneObjects/trimesh.h"
#include "../SceneObjects/SplitGeo.h"
#include "../ui/TraceUI.h"
#include <glm/gtx/io.hpp>
#include <iostream>
#include <limits>
#include <numeric>

extern bool debugMode;

using namespace std;

BVHTree::BVHTree(){
    built = false;
    size = 0;
    leaves = 0;
    misses = 0;
    percentSum = 0;
    traverseCount = 0;
    traversals = 0;
    visitBoth = 0;
}

void splitLargeGeo(vector<Geometry*>& geo){
    vector<double> v;
    for (Geometry* g : geo) {  

        v.push_back(g->getBoundingBox().volume());
    }
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();
    cout << "mean: " << mean << endl;
    double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / v.size() - mean * mean);
    cout << "stdev: " << stdev << endl;
    double maxVol = *max_element(v.begin(), v.end());
    cout << "max: " << maxVol << endl;

    // vector<Geometry*> temp(geo);
    while(maxVol > mean + stdev * 5){
        maxVol = 0;
        int n = geo.size();
        for(int i = 0; i < n; i++){
            // Do split
            if(geo[i]->getBoundingBox().volume() > mean + stdev * 5){
                // cout << "splitting" << endl;
                glm::dvec3 mins = geo[i]->getBoundingBox().getMin();
                glm::dvec3 maxs = geo[i]->getBoundingBox().getMax();

                auto diff = maxs - mins;

                // Find longest axis
                double center;
                int centerIndex;
                if (diff[0] >= diff[1] && diff[0] >= diff[2]) {
                    // x
                    center = diff[0] / 2 + mins[0];
                    centerIndex = 0;
                } else if (diff[1] >= diff[0] && diff[1] >= diff[2]) {
                    // y
                    center = diff[1] / 2 + mins[1];
                    centerIndex = 1;
                } else {
                    // z
                    center = diff[2] / 2 + mins[2];
                    centerIndex = 2;
                }

                BoundingBox left, right;
                left.setMin(mins);
                right.setMax(maxs);

                mins[centerIndex] = center;
                maxs[centerIndex] = center;

                left.setMax(maxs);
                right.setMin(mins);

                MaterialSceneObject* cur = (MaterialSceneObject* ) geo[i];
                SplitGeo* splitLeft = new SplitGeo(cur->getScene(), cur->material.get(), cur);
                SplitGeo* splitRight = new SplitGeo(cur->getScene(), cur->material.get(), cur);
                splitLeft->bounds = left;
                splitRight->bounds = right;

                geo[i] = splitLeft;
                geo.push_back(splitRight);

                maxVol = max(left.volume(), maxVol);
                maxVol = max(right.volume(), maxVol);
            } else {
                maxVol = max(geo[i]->getBoundingBox().volume(), maxVol);
            }
        }
        cout << "maxVol: " << maxVol << endl;

    }
    cout << "returning" << endl;
}

void BVHTree::build(unique_ptr<Scene>& scene){
    this->scene = scene.get();

    if(built) {
        cout << "built" << endl; 
        return;
    }
    vector<Geometry*> geo;
    cout << "scene obj size: " << scene->objects.size() << endl;
    for(Geometry* g: scene->objects){
        auto x = g->getAll();
        geo.insert(geo.end(), x.begin(), x.end());
    }
    cout << geo.size() << endl;

    // splitLargeGeo(geo);

    root = buildHelper(geo, 0, geo.size());
    built = true;

    cout << leaves << " " << geo.size() << endl;
    if(leaves != geo.size()){
        cout << "missing elements" << endl;
        exit(0);
    }
}

BoundingBox getBox(vector<Geometry*>& geo, int st, int end){
    glm::dvec3 bestMin = geo[st]->getBoundingBox().getMin();
    glm::dvec3 bestMax = geo[st]->getBoundingBox().getMax();

    for(int i = st + 1; i < end; i++){
        if(!geo[i]->hasBoundingBoxCapability()){
            cout << "rip" << endl;
            exit(0);
        }

        glm::dvec3 mins = geo[i]->getBoundingBox().getMin();
        glm::dvec3 maxs = geo[i]->getBoundingBox().getMax();

        bestMin = glm::min(bestMin, mins);
        bestMax = glm::max(bestMax, maxs);
    }

    return BoundingBox(bestMin, bestMax);
}

int split(vector<Geometry*>& geo, int st, int end){
    int mid = st + (end - st) / 2;
    // return mid;
    const static int samples = 500;
    const static double trimeshCost = 1;
    const static double bbCost = 1;

    int best = mid;
    double cost = numeric_limits<double>::max();
    BoundingBox total = getBox(geo, st, end);
    for (int i = max(mid - samples, st + 1); i < min(mid + samples, end - 1); i++) {
    // for(int i = 1; i < 10; i++){
        BoundingBox sA = getBox(geo, st, i);
        BoundingBox sB = getBox(geo, i, end);

        double myCost = bbCost + (sA.area() / total.area()) * (i - st) * trimeshCost + (sB.area() / total.area()) * (end - i) * trimeshCost;
        // cout << "\t" << myCost << " " << sA.area() / total.area() << " " << sB.area() / total.area() << endl;
        // double myCost = sA.area() / total.area() + sB.area() / total.area();
        // double myCost = sA.volume() + sB.volume();
        // cout << "\t" << myCost << " " << sA.volume() << " " << sB.volume() << endl;
        if(myCost < cost){
            cost = myCost;
            best = i;
        }
    }
    // cout << st << " " << best << " " << end << endl;
    // exit(0);
    return best;
}

BVHNode* BVHTree::buildHelper(vector<Geometry*>& geo, int st, int end){
    auto node = new BVHNode(0, 0, nullptr);
    size++;

    if(st >= end){
        cout << "geo 0 " << endl;
        return nullptr;
    }

    // Compute bounding boxes
    node->bb = getBox(geo, st, end);

    // Leaf node
    if(end - st <= 1){
        node->geometry = geo[st];
        leaves++;
        return node;
    }

    glm::dvec3 mins = node->bb.getMin();
    glm::dvec3 maxs = node->bb.getMax();

    auto diff = maxs - mins;

    // Find longest axis
    int centerIndex;
    if(diff[0] >= diff[1] && diff[0] >= diff[2]){
        centerIndex = 0;
    } else if(diff[1] >= diff[0] && diff[1] >= diff[2]){
        centerIndex = 1;
    } else {
        centerIndex = 2;
    }

    sort(geo.begin() + st, geo.begin() + end, [=](Geometry* a, Geometry* b) {
        return a->center()[centerIndex] < b->center()[centerIndex];
    });
    int mid = split(geo, st, end);

    node->left = buildHelper(geo, st, mid);
    node->right = buildHelper(geo, mid, end);
    return node;
}


bool BVHTree::traverse(ray& r, isect& i)
{
    // traverseCount = 0;
    // misses = 0;
    // visitBoth = 0;
    // if(debugMode) cout << "--- starting traverse --- " << endl;
    i = traverseHelper(r, root);
    bool res = i.getT() >= 0;
    // if(!res){
    //     i.setT(1000.0);
    // }
    // if (TraceUI::m_debug) {
    //     scene->addToIntersectCache(std::make_pair(new ray(r), new isect(i)));
    // }
    // traversals++;
    // cout << traverseCount << " " << size << endl;
    // percentSum += (double) traverseCount / size;
    // cout << "prune percentage: " << (double) misses / traverseCount << endl;
    return res;
}

isect BVHTree::traverseHelper(ray& r, BVHNode* n){

    // if(debugMode) cout << "traversing" << endl;
    // traverseCount++;
    isect res;
    res.setT(-1);

    if (!n) {
        // if (debugMode)
        //     cout << "off tree" << endl;
        return res;
    }

    // Check bounding box of current
    double tMin;
    double tMax;
    // if(debugMode){
    //     cout << "checking current bb" << endl;
    //     cout << n->bb.getMax() << endl;
    //     cout << n->bb.getMin() << endl;
    // }
    if(!n->bb.intersect(r, tMin, tMax)){
        // if(debugMode) cout << "skipping bb" << endl;
        // misses++;
        res.setT(-2);
        return res;
    }

    // If leaf node
    if (n->geometry) {
        // if(debugMode) cout << "in leaf" << endl;
        // if(debugMode) cout << "ray" << r.getDirection() << " pos: " << r.getPosition() << endl;
        if(n->geometry->intersect(r, res)){
            // if(debugMode) cout << "got valid int" << endl;
            return res;
        }
        res.setT(-1);
        return res;
    }

    isect li = traverseHelper(r, n->left);
    isect ri = traverseHelper(r, n->right);

    // if(li.getT() == -1 && ri.getT() == -1)
    //     visitBoth++;

    if(li.getT() < 0){
        return ri;
    } else if(ri.getT() < 0){
        return li;
    } else {
        return li.getT() < ri.getT() ? li : ri;
    }
}

int BVHTree::height(){
    return heightHelper(root);
}

int BVHTree::heightHelper(BVHNode* n){
    if(!n) return 0;
    return 1 + max(heightHelper(n->left), heightHelper(n->right));
}