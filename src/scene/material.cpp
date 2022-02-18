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
		if(debugMode) cout << "doing light: " << pLight->getColor() << endl;
		// TODO sus?
		glm::dvec3 i_in = pLight->getColor();
		i_in *= pLight->distanceAttenuation(r.at(i));

		// cout << ambient[0] << " " << ambient[1] << " " << ambient[2] << endl;
		glm::dvec3 normal = i.getN();
		// if(r.currentIndex != 1){
		// 	normal *= -1;
		// }
		// TODO really sus
		// Check if light can shine thru
		if(glm::dot(pLight->getDirection(r.at(i)), normal) <= 0 && Trans()){
			normal *= -1;
		}

		// Diffuse
		glm::dvec3 l = pLight->getDirection(r.at(i));
		double m = max(glm::dot(l, normal), 0.0);

		glm::dvec3 diffuse = kd(i) * m * i_in;

		// Specular
		glm::dvec3 v = -r.getDirection();

		glm::dvec3 w_in = -l;
		glm::dvec3 w_normal = glm::dot(w_in, normal) * normal;
		glm::dvec3 w_tan = w_in - w_normal;
		glm::dvec3 w_ref = -w_normal + w_tan;
		w_ref = glm::normalize(w_ref);

		double m2 = max(glm::dot(v, w_ref), 0.0);
		glm::dvec3 specular = ks(i) * pow(m2, i.getMaterial().shininess(i)) * i_in;

		glm::dvec3 phong = diffuse + specular;

		glm::dvec3 p = r.at(i.getT() - 1e-12); // shift in direction of normal
		glm::dvec3 dir = pLight->getDirection(r.at(i));
		ray shadowR(p, dir, r.getAtten(), ray::SHADOW);
		phong *= pLight->shadowAttenuation(shadowR, p);

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
