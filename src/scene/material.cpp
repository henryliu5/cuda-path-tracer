#include "material.h"
#include "../ui/TraceUI.h"
#include "light.h"
#include "ray.h"
extern TraceUI* traceUI;

#include <glm/gtx/io.hpp>
#include <iostream>
#include "../fileio/images.h"

using namespace std;
extern bool debugMode;

Material::~Material()
{
}

// Apply the phong model to this point on the surface of the object, returning
// the color of that point.
glm::dvec3 Material::shade(Scene* scene, const ray& r, const isect& i) const
{
	// YOUR CODE HERE

	// For now, this method just returns the diffuse color of the object.
	// This gives a single matte color for every distinct surface in the
	// scene, and that's it.  Simple, but enough to get you started.
	// (It's also inconsistent with the phong model...)

	// Your mission is to fill in this method with the rest of the phong
	// shading model, including the contributions of all the light sources.
	// You will need to call both distanceAttenuation() and
	// shadowAttenuation()
	// somewhere in your code in order to compute shadows and light falloff.
	//	if( debugMode )
	//		std::cout << "Debugging Phong code..." << std::endl;

	// When you're iterating through the lights,
	// you'll want to use code that looks something
	// like this:
	//

	// Ambient
	glm::dvec3 ambient = ka(i) * scene->ambient();
	glm::dvec3 sum = ke(i) + ambient;

	for ( const auto& pLight : scene->getAllLights() )
	{
		// TODO sus?
		glm::dvec3 i_in = pLight->getColor();
		i_in *= pLight->distanceAttenuation(r.at(i));

		// cout << ambient[0] << " " << ambient[1] << " " << ambient[2] << endl;

		// Diffuse
		glm::dvec3 l = pLight->getDirection(r.at(i));
		double m = max(glm::dot(l, i.getN()), 0.0);

		glm::dvec3 diffuse = kd(i) * m * i_in;

		// Specular
		glm::dvec3 v = -r.getDirection();

		glm::dvec3 w_in = -l;
		glm::dvec3 w_normal = glm::dot(w_in, i.getN()) * i.getN();
		glm::dvec3 w_tan = w_in - w_normal;
		glm::dvec3 w_ref = -w_normal + w_tan;
		w_ref = glm::normalize(w_ref);

		double m2 = max(glm::dot(v, w_ref), 0.0);
		glm::dvec3 specular = ks(i) * pow(m2, i.getMaterial().shininess(i)) * i_in;

		// Shadow ray
		glm::dvec3 p = r.at(i) + i.getN() * 1e-12; // shift in direction of normal
		glm::dvec3 dir = pLight->getDirection(r.at(i));
		double t = 0.0;
		
		glm::dvec3 phong = diffuse + specular;
		while(true){
			ray shadowR(p + t * dir, dir, r.getAtten(), ray::SHADOW);
			isect shadowI;
			isect shadowI2;

			if(scene->intersect2(shadowR, shadowI, shadowI2)){
				if(shadowI.getT() < 0 || shadowI2.getT() < 0){
					cout << "x" << endl;
					exit(0);
				}
				double t1;
				double t2;
				// cout << "shadowI T: " << shadowI.getT() << " shadowI2 T: " << shadowI2.getT() << endl;
				// Intersected twice
				if(shadowI.getObject() == shadowI2.getObject()){
				// if(shadowI.getObject() != i.getObject()){
					t1 = shadowI.getT();
					t2 = shadowI.getT() + shadowI2.getT();
				} else { // Once
					t1 = 0;
					t2 = shadowI.getT();
				}
				
				// Check if behind light
				glm::dvec3 lightDir = pLight->getDirection(shadowR.at(t2));
				glm::dvec3 lightDir2 = pLight->getDirection(shadowR.at(t1));
				if(glm::dot(lightDir, lightDir2) <= 0){
					break;
				}


				double d = glm::length(shadowR.at(t2) - shadowR.at(t1));
				glm::dvec3 kt = shadowI.getObject()->getMaterial().kt(shadowI);
				phong *= glm::pow(kt, {d,d,d});
				if(debugMode) cout << "applying kt: " << kt << " d: " << d << " t1: " << t1 << " t2: " << t2 << endl;
				
				if(debugMode) cout << "t1 pos: " << shadowR.at(t1) << " t2 pos: " << shadowR.at(t2) << endl;
				// Threshold if time small
				if(t2 <= 1e-6){
					break;
				}

				if(kt == glm::dvec3(0,0,0)){
					break;
				}
				t += t2;
			} else {
				break;
			}
			// TODO REMOVE
			// break;
		}
		if(debugMode) cout << "phong: " << phong << endl;
		sum += phong;
	}
	if(debugMode) cout << "sum: " << sum << endl;
	return sum;
}

TextureMap::TextureMap(string filename)
{
	data = readImage(filename.c_str(), width, height);
	if (data.empty()) {
		width = 0;
		height = 0;
		string error("Unable to load texture map '");
		error.append(filename);
		error.append("'.");
		throw TextureMapException(error);
	}
}

glm::dvec3 TextureMap::getMappedValue(const glm::dvec2& coord) const
{
	// YOUR CODE HERE
	//
	// In order to add texture mapping support to the
	// raytracer, you need to implement this function.
	// What this function should do is convert from
	// parametric space which is the unit square
	// [0, 1] x [0, 1] in 2-space to bitmap coordinates,
	// and use these to perform bilinear interpolation
	// of the values.

	return glm::dvec3(1, 1, 1);
}

glm::dvec3 TextureMap::getPixelAt(int x, int y) const
{
	// YOUR CODE HERE
	//
	// In order to add texture mapping support to the
	// raytracer, you need to implement this function.

	return glm::dvec3(1, 1, 1);
}

glm::dvec3 MaterialParameter::value(const isect& is) const
{
	if (0 != _textureMap)
		return _textureMap->getMappedValue(is.getUVCoordinates());
	else
		return _value;
}

double MaterialParameter::intensityValue(const isect& is) const
{
	if (0 != _textureMap) {
		glm::dvec3 value(
		        _textureMap->getMappedValue(is.getUVCoordinates()));
		return (0.299 * value[0]) + (0.587 * value[1]) +
		       (0.114 * value[2]);
	} else
		return (0.299 * _value[0]) + (0.587 * _value[1]) +
		       (0.114 * _value[2]);
}
