#ifndef GPU_BB_H
#define GPU_BB_H

#include "GPUManaged.cuh"
#include "GPURay.cuh"
#include <cuda.h>

namespace GPU {

class BoundingBox : public GPUManaged {
public:
    double RAY_EPSILON = 1e-12;
    CUDA_CALLABLE_MEMBER BoundingBox(){}
    CUDA_CALLABLE_MEMBER BoundingBox(glm::dvec3 bMin, glm::dvec3 bMax) : bmin(bMin), bmax(bMax){}

    CUDA_CALLABLE_MEMBER bool intersect(const GPU::Ray& r, double& tMin, double& tMax) const
    {
        /*
        * Kay/Kajiya algorithm.
        */
        glm::dvec3 R0 = r.getPosition();
        glm::dvec3 Rd = r.getDirection();
        tMin = -1.0e308; // 1.0e308 is close to infinity... close enough
                        // for us!
        tMax = 1.0e308;
        double ttemp;

        for (int currentaxis = 0; currentaxis < 3; currentaxis++) {
            double vd = Rd[currentaxis];
            // if the ray is parallel to the face's plane (=0.0)
            if (vd == 0.0)
                continue;
            double v1 = bmin[currentaxis] - R0[currentaxis];
            double v2 = bmax[currentaxis] - R0[currentaxis];
            // two slab intersections
            double t1 = v1 / vd;
            double t2 = v2 / vd;
            if (t1 > t2) { // swap t1 & t2
                ttemp = t1;
                t1    = t2;
                t2    = ttemp;
            }
            if (t1 > tMin)
                tMin = t1;
            if (t2 < tMax)
                tMax = t2;
            if (tMin > tMax)
                return false; // box is missed
            if (tMax < RAY_EPSILON)
                return false; // box is behind ray
        }
        return true; // it made it past all 3 axes.
    }

	glm::dvec3 bmin;
	glm::dvec3 bmax;
};

}

#endif