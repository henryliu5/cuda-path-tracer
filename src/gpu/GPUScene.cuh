#ifndef GPU_SCENE_H
#define GPU_SCENE_H

#include "GPUManaged.cuh"
#include "../scene/scene.h"
#include "GPUTrimesh.cuh"
#include "GPUCamera.cuh"
#include <cuda.h>

namespace GPU {

class Scene : public GPUManaged {
public:
    GPU::Trimesh** objects;
    GPU::Camera camera;
    int n_objects;

    __host__ Scene(::Scene* cpuScene) : camera() {

        // Update camera
        camera.setEye(cpuScene->getCamera().getEye());

        // Copy all objects in the scene
        n_objects = cpuScene->objects.size();
        gpuErrchk(cudaMallocManaged(&objects, n_objects * sizeof(GPU::Trimesh*)));
        for(int i = 0; i < n_objects; i++){
            // TODO - assumes all trimeshs currently
            ::Trimesh* t = (::Trimesh*) cpuScene->objects[i];
            cout << "Trimesh " << i << " has " << t->faces.size() << " faces" << endl;

            objects[i] = new GPU::Trimesh(*t);
        }
    }

    CUDA_CALLABLE_MEMBER bool intersect(GPU::Ray& r, GPU::Isect& i) const {
        double tmin = 0.0;
        double tmax = 0.0;
        bool have_one = false;
        for(int idx = 0; idx < n_objects; idx++) {
            Trimesh* obj = objects[idx];
            GPU::Isect cur;
            if( obj->intersect(r, cur) ) {
                if(!have_one || (cur.getT() < i.getT())) {
                    i = cur;
                    have_one = true;
                }
            }
        }
        if(!have_one)
            i.setT(1000.0);
        return have_one;
    }
};

}

#endif