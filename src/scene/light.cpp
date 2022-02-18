#include <cmath>
#include <iostream>

#include "light.h"
#include <glm/glm.hpp>
#include <glm/gtx/io.hpp>


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

glm::dvec3 Light::shadowAttenuation(const ray& r, const glm::dvec3& p) const
{
    // YOUR CODE HERE:
    // You should implement shadow-handling code here.
    glm::dvec3 result(1, 1, 1);

    // Shadow ray
	glm::dvec3 dir = r.getDirection();
    double t = 0.0;

    while (true) {
        ray shadowR(p + t * dir, dir, r.getAtten(), ray::SHADOW);
        isect shadowI;
        isect shadowI2;

        if (scene->intersect2(shadowR, shadowI, shadowI2)) {
            if (shadowI.getT() < 0 || shadowI2.getT() < 0) {
                cout << "x" << endl;
                exit(0);
            }
            double t1;
            double t2;
            // cout << "shadowI T: " << shadowI.getT() << " shadowI2 T: " << shadowI2.getT() << endl;
            // Intersected twice
            if (shadowI.getObject() == shadowI2.getObject()) {
                // if(shadowI.getObject() != i.getObject()){
                t1 = shadowI.getT();
                t2 = shadowI.getT() + shadowI2.getT();
            } else { // Once
                t1 = 0;
                t2 = shadowI.getT();
            }

            // Check if behind light
            glm::dvec3 lightDir = getDirection(shadowR.at(t2));
            glm::dvec3 lightDir2 = getDirection(shadowR.at(t1));
            if (glm::dot(lightDir, lightDir2) <= 0) {
				// return result;
                break;
            }

            double d = glm::length(shadowR.at(t2) - shadowR.at(t1));
            glm::dvec3 kt = shadowI.getObject()->getMaterial().kt(shadowI);
            result *= glm::pow(kt, { d, d, d });
            if (debugMode)
                cout << "applying kt: " << kt << " d: " << d << " t1: " << t1 << " t2: " << t2 << endl;

            if (debugMode)
                cout << "t1 pos: " << shadowR.at(t1) << " t2 pos: " << shadowR.at(t2) << endl;
            // Threshold if time small
            if (t2 <= 1e-6) {
                // return result;
                break;
            }

            if (kt == glm::dvec3(0, 0, 0)) {
                // return result;
                break;
            }
            t += t2;
        } else {
            // return result;
            break;
        }
        // TODO REMOVE
        // break;
    }

    return result;
}

#define VERBOSE 0

