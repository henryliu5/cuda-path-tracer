#ifndef GPU_SCENE_H
#define GPU_SCENE_H

#include "GPUManaged.cuh"
#include "../scene/scene.h"
#include "GPUGeometry.cuh"
#include "GPUTrimesh.cuh"
#include "GPUCamera.cuh"
#include <cuda.h>
#include <unordered_map>
#include "GPUBVH.cuh"
#include "../scene/bvh.h"

namespace GPU {

class Scene : public GPUManaged {
public:
    std::unordered_map<::Geometry*, GPU::Geometry*> cpuToGpuGeo;
    GPU::Geometry** objects;
    GPU::Camera camera;
    int n_objects;

    GPU::BVH* gpuBVH;

    __host__ Scene(::Scene* cpuScene) : camera(cpuScene->getCamera()) {

        // // Update camera
        // camera.setEye(cpuScene->getCamera().getEye());
        // camera.setLook(cpuScene->getCamera().getV(), cpuScene->getCamera().getU());

        // Copy all objects in the scene
        n_objects = cpuScene->objects.size();
        gpuErrchk(cudaMallocManaged(&objects, n_objects * sizeof(GPU::Geometry*)));
        for(int i = 0; i < n_objects; i++){

            switch(cpuScene->objects[i]->myType){
                // Add trimesh
                case Geometry::GeometryType::TRIMESH: {
                    cout << "Found trimesh" << endl;
                    ::Trimesh* t = (::Trimesh*) cpuScene->objects[i];
                    
                    cout << "Trimesh " << i << " has " << t->faces.size() << " faces" << endl;
                    objects[i] = new GPU::Trimesh(*t, cpuToGpuGeo);
                }
                break;
                case Geometry::GeometryType::NONE:
                    cout << "Unsupported object ---------------------------------- " << endl;
                break;
            }
            cpuToGpuGeo[cpuScene->objects[i]] = objects[i];
        }
    }

    // Copy BVH Tree to GPU construct
    __host__ void copyBVH(BVHTree* cpuTree){
        gpuBVH = new GPU::BVH(cpuTree, cpuToGpuGeo);
    }

    CUDA_CALLABLE_MEMBER bool intersect(GPU::Ray& r, GPU::Isect& i) const {
        return gpuBVH->traverseIterative(r, i);
        // double tmin = 0.0;
        // double tmax = 0.0;
        // bool have_one = false;
        // for(int idx = 0; idx < n_objects; idx++) {
        //     Geometry* obj = objects[idx];
        //     GPU::Isect cur;
        //     if( obj->intersect(r, cur) ) {
        //         if(!have_one || (cur.getT() < i.getT())) {
        //             i = cur;
        //             have_one = true;
        //         }
        //     }
        // }
        // if(!have_one)
        //     i.setT(1000.0);
        // return have_one;
    }
};

}

#endif