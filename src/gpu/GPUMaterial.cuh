#ifndef GPU_MATERIAL_H
#define GPU_MATERIAL_H

#include "GPUManaged.cuh"
#include "../scene/material.h"
#include <cuda.h>

namespace GPU {

class Material : public GPUManaged {
public:
    CUDA_CALLABLE_MEMBER Material(glm::dvec3 kd, glm::dvec3 ke, glm::dvec3 ks, glm::dvec3 kr, glm::dvec3 kt, glm::dvec3 sh, glm::dvec3 in) : 
                                  _kd(kd), _ke(ke), _ks(ks), _kr(kr), _kt(kt), _shininess(sh), _index(in) { setBools(); }
    
    CUDA_CALLABLE_MEMBER glm::dvec3 ke() { return _ke; }
    CUDA_CALLABLE_MEMBER glm::dvec3 kd() { return _kd; }
    CUDA_CALLABLE_MEMBER glm::dvec3 ks() { return _ks; }
    CUDA_CALLABLE_MEMBER glm::dvec3 kr() { return _kr; }
    CUDA_CALLABLE_MEMBER glm::dvec3 kt() { return _kt; }

    CUDA_CALLABLE_MEMBER double index() { return (0.299 * _index[0]) + (0.587 * _index[1]) + (0.114 * _index[2]); }
    CUDA_CALLABLE_MEMBER double shininess() { return (0.299 * _shininess[0]) + (0.587 * _shininess[1]) + (0.114 * _shininess[2]); }

    // get booleans for reflection and refraction
    CUDA_CALLABLE_MEMBER bool Diff() const { return _diff; }
    CUDA_CALLABLE_MEMBER bool Refl() const { return _refl; }
    CUDA_CALLABLE_MEMBER bool Trans() const { return _trans; }
    CUDA_CALLABLE_MEMBER bool Recur() const { return _recur; }
    CUDA_CALLABLE_MEMBER bool Spec() const { return _spec; }
    CUDA_CALLABLE_MEMBER bool Both() const { return _both; }


    CUDA_CALLABLE_MEMBER void setBools() {
        _refl = !(glm::length(_kr) == 0); _trans = !(glm::length(_kt) == 0); _recur = _refl || _trans;
		_spec = _refl || !(glm::length(_ks) == 0);
		_both = _refl && _trans;
        _diff = !(glm::length(_kd) == 0);
    }



protected:
    glm::dvec3 _ke;
    glm::dvec3 _kd;
    glm::dvec3 _ks;
    glm::dvec3 _kr;
    glm::dvec3 _kt;

    glm::dvec3 _shininess;
    glm::dvec3 _index;

    bool _diff;
	bool _refl;								  // specular reflector?
	bool _trans;							  // specular transmitter?
	bool _recur;							  // either one
	bool _spec;								  // any kind of specular?
	bool _both;								  // reflection and transmission
};

}

#endif