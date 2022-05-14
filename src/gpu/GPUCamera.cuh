#ifndef GPU_CAMERA_H
#define GPU_CAMERA_H

#include "GPUManaged.cuh"
#include <cuda.h>
// #include "GPURay.cuh"
#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>

namespace GPU {

class Ray;

class Camera : public GPUManaged {
public:
    Camera();

    CUDA_CALLABLE_MEMBER void rayThrough( double x, double y, GPU::Ray &r );
    void setEye( const glm::dvec3 &eye );
    void setLook( double, double, double, double );
    void setLook( const glm::dvec3 &viewDir, const glm::dvec3 &upDir );
    void setFOV( double );
    void setAspectRatio( double );

    double getAspectRatio() { return aspectRatio; }

	CUDA_CALLABLE_MEMBER const glm::dvec3& getEye() const			{ return eye; }
	const glm::dvec3& getLook() const		{ return look; }
	CUDA_CALLABLE_MEMBER const glm::dvec3& getU() const			{ return u; }
	CUDA_CALLABLE_MEMBER const glm::dvec3& getV() const			{ return v; }
private:
    glm::dmat3 m;                     // rotation matrix
    double normalizedHeight;    // dimensions of image place at unit dist from eye
    double aspectRatio;
    
    void update();              // using the above three values calculate look,u,v
    
    glm::dvec3 eye;
    glm::dvec3 look;                  // direction to look
    glm::dvec3 u,v;                   // u and v in the 
};

}

#endif