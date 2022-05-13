#ifndef GPU_TRIMESH_H
#define GPU_TRIMESH_H

#include "GPUManaged.cuh"
#include <cuda.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/vec3.hpp>
#include "GPURay.cuh"
#include "GPUIsect.cuh"
#include "GPUMaterial.cuh"
#include "../SceneObjects/trimesh.h"

namespace GPU {

class TrimeshFace : public GPUManaged {
public:
    glm::dvec3* vertices;
	glm::dvec3& a_coords;
	glm::dvec3& b_coords;
	glm::dvec3& c_coords;
    glm::dvec3 normal;
    GPU::Material* material;
    bool degen;

    CUDA_CALLABLE_MEMBER TrimeshFace(glm::dvec3* vertices, GPU::Material* material, int a, int b, int c) :
        material(material)
        , a_coords(vertices[a])
        , b_coords(vertices[b])
        , c_coords(vertices[c])
    {
		glm::dvec3 vab = (b_coords - a_coords);
		glm::dvec3 vac = (c_coords - a_coords);
		glm::dvec3 vcb = (b_coords - c_coords);

		if (glm::length(vab) == 0.0 || glm::length(vac) == 0.0 ||
		    glm::length(vcb) == 0.0)
			degen = true;
		else {
			degen  = false;
			normal = glm::cross(b_coords - a_coords,
			                    c_coords - a_coords);
			normal = glm::normalize(normal);
		}
    }

    CUDA_CALLABLE_MEMBER bool intersect(GPU::Ray& r, GPU::Isect& i)
    {
        // Ray - plane intersection
        double numerator = -glm::dot(r.getPosition() - a_coords, normal);
        double denominator = glm::dot(r.getDirection(), normal);

        if(denominator >= 0){
            if(!material->Recur() || denominator == 0){
                return false;
            }
        }

        double t = numerator / denominator;

        if(t < 0){
            return false;
        }

        glm::dvec3 p = r.getPosition() + r.getDirection() * t;

        glm::dvec3 vba = (b_coords - a_coords);
        glm::dvec3 vcb = (c_coords - b_coords);
        glm::dvec3 vac = (a_coords - c_coords);

        glm::dvec3 vpa = (p - a_coords);
        glm::dvec3 vpb = (p - b_coords);
        glm::dvec3 vpc = (p - c_coords);

        bool res = glm::dot(glm::cross(vba, vpa), normal) >= 0 &&
                glm::dot(glm::cross(vcb, vpb), normal) >= 0 &&
                glm::dot(glm::cross(vac, vpc), normal) >= 0;

        if(res){
            i.setN(glm::normalize(normal));
            i.setMaterial(material);
            i.setT(t);
        }
        return res;
    }
};

class Trimesh : public GPUManaged {

public:
    glm::dvec3* vertices;
    GPU::TrimeshFace** faces;
    GPU::Material** materials;

    int n_vertices;
    int n_faces;
    int n_materials;

    Trimesh(::Trimesh& other){
        // Copy vertices (easiest)
        n_vertices = other.vertices.size();
        gpuErrchk(cudaMallocManaged(&vertices, n_vertices * sizeof(glm::dvec3)));
        for(int i = 0; i < n_vertices; i++){
            vertices[i] = glm::dvec3(other.vertices[i]);
        }

        // Copy materials
        n_materials = other.faces.size();
        cout << n_materials << endl;
        gpuErrchk(cudaMallocManaged(&materials, n_materials * sizeof(GPU::Material*)));
        for(int i = 0; i < n_materials; i++){
            ::Material* m = other.faces[i]->material.get();
            materials[i] = new GPU::Material(m->_kd._value, m->_ke._value);
        }       

        // Copy faces
        n_faces = other.faces.size();
        gpuErrchk(cudaMallocManaged(&faces, n_faces * sizeof(GPU::TrimeshFace*)));
        for(int i = 0; i < n_faces; i++){
            int a = other.faces[i]->ids[0];
            int b = other.faces[i]->ids[1];
            int c = other.faces[i]->ids[2];
            faces[i] = new GPU::TrimeshFace(vertices, materials[i], a, b, c);
        }
    }

    // Test intersection against all trimesh faces
    CUDA_CALLABLE_MEMBER bool intersect(GPU::Ray& r, GPU::Isect& i) const
    {
        bool have_one = false;
        for (int idx = 0; idx < n_faces; idx++){
            TrimeshFace* face = faces[idx];
            GPU::Isect cur;
            if (face->intersect(r, cur)) {
                if (!have_one || (cur.getT() < i.getT())) {
                    i = cur;
                    have_one = true;
                }
            }
        }
        if (!have_one)
            i.setT(1000.0);
        return have_one;
    }

};

}

#endif