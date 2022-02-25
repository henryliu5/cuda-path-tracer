#include <cmath>
#include <iostream>

#include "light.h"
#include <glm/glm.hpp>
#include <glm/gtx/io.hpp>
#include <unordered_set>


using namespace std;

double DirectionalLight::distanceAttenuation(const glm::dvec3& P) const
{
	// distance to light is infinite, so f(di) goes to 0.  Return 1.
	return 1.0;
}

glm::dvec3 DirectionalLight::getColor() const
{
	return color;
}

glm::dvec3 DirectionalLight::getDirection(const glm::dvec3& P) const
{
	return -orientation;
}

double PointLight::distanceAttenuation(const glm::dvec3& P) const
{

	// YOUR CODE HERE

	// You'll need to modify this method to attenuate the intensity 
	// of the light based on the distance between the source and the 
	// point P.  For now, we assume no attenuation and just return 1.0
	double d = glm::length(P - this->position);

	return min(1.0, 1.0 / (constantTerm + linearTerm * d + quadraticTerm * d * d));
}

glm::dvec3 PointLight::getColor() const
{
	return color;
}

glm::dvec3 PointLight::getDirection(const glm::dvec3& P) const
{
	return glm::normalize(position - P);
}

extern bool debugMode;

glm::dvec3 Light::shadowAttenuation(const ray& r, const glm::dvec3& p, const isect& originalI) const
{
    // YOUR CODE HERE:
    // You should implement shadow-handling code here.
    glm::dvec3 result(1, 1, 1);

    // Shadow ray
	glm::dvec3 dir = r.getDirection();
    double t = 1e-6;

    // vector<pair<glm::dvec3, const SceneObject*>> v;
    isect lastI = originalI;

    while (true) {
        ray shadowR(p + t * dir, dir, r.getAtten(), ray::SHADOW);
        isect curI;

        if (scene->bvhTree->traverse(shadowR, curI)) {
            // v.push_back(make_pair(shadowR.at(curI), curI.getObject()));

            // if(debugMode) cout << "intersected" << endl;
            
            double t1 = 0;
            double t2 = curI.getT();
            glm::dvec3 kt = curI.getObject()->getMaterial().kt(curI);

            bool ok = curI.getObject() == lastI.getObject() || kt == glm::dvec3(0.0,0.0,0.0);

            glm::dvec3 oldPos = p + t * dir;
            // for(int i = v.size() - 2; i >= max((int)v.size() - 1000, 0); --i){
            //     if(v[i].second == curI.getObject()){
            //         oldPos = v[i].first;
            //         ok = true;
            //         v.pop_back();
            //         v.erase(v.begin() + i);
            //         break;
            //     }
            // }

            if (ok) {
                // Check if behind light
                glm::dvec3 lightDir = getDirection(shadowR.at(curI));
                glm::dvec3 lightDir2 = getDirection(shadowR.at(0));
                if (glm::dot(lightDir, lightDir2) <= 0) {
                    // if(debugMode) cout << "past light" << endl;
                    break;
                }
                double d = glm::length(shadowR.at(curI) - oldPos);
                
                result *= glm::pow(kt, { d, d, d });

                if (debugMode)
                    cout << "applying kt: " << kt << " d: " << d << " t1: " << t1 << " t2: " << t2 << endl;

                if (debugMode)
                    cout << "t1 pos: " << shadowR.at(t1) << " t2 pos: " << shadowR.at(t2) << endl;
                    
                // Threshold if time small
                if (t2 <= 1e-6) {
                    break;
                }

                if (kt == glm::dvec3(0, 0, 0)) {
                    break;
                }
            }
            lastI = curI;
            t += t2 + 1e-6;
        } else {
            break;
        }
    }

    return result;
}

glm::dvec3 Light::shade(const ray& r, const isect& i) const {
    Material curMat = i.getMaterial();
    // if(debugMode) cout << "doing light: " << getColor() << endl;
    // TODO sus?
    glm::dvec3 i_in = getColor();
    i_in *= distanceAttenuation(r.at(i));

    // cout << ambient[0] << " " << ambient[1] << " " << ambient[2] << endl;
    glm::dvec3 normal = i.getN();
    // if(r.currentIndex != 1){
    // 	normal *= -1;
    // }
    // TODO really sus
    // Check if light can shine thru
    if(glm::dot(getDirection(r.at(i)), normal) <= 0 && curMat.Trans()){
        normal *= -1;
    }

    // Diffuse
    glm::dvec3 l = getDirection(r.at(i));
    double m = max(glm::dot(l, normal), 0.0);
    if(debugMode)
        cout << "kd: " << curMat.kd(i) << endl;
    glm::dvec3 diffuse = curMat.kd(i) * m * i_in;

    // Specular
    glm::dvec3 v = -r.getDirection();

    glm::dvec3 w_in = -l;
    glm::dvec3 w_normal = glm::dot(w_in, normal) * normal;
    glm::dvec3 w_tan = w_in - w_normal;
    glm::dvec3 w_ref = -w_normal + w_tan;
    w_ref = glm::normalize(w_ref);

    double m2 = max(glm::dot(v, w_ref), 0.0);
    glm::dvec3 specular = curMat.ks(i) * pow(m2, i.getMaterial().shininess(i)) * i_in;

    glm::dvec3 phong = diffuse + specular;

    glm::dvec3 p = r.at(i.getT() - 1e-12); // shift in direction of normal
    glm::dvec3 dir = getDirection(r.at(i));
    ray shadowR(p, dir, r.getAtten(), ray::SHADOW);
    phong *= shadowAttenuation(shadowR, p, i);

    // if(debugMode) cout << "phong: " << phong << endl;
    return phong;
}

#define VERBOSE 0

