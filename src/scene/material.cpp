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
		sum += pLight->shade(r, i);
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
	int u1 = coord.x * (width - 1);
	int v1 = coord.y * (height - 1);
	int u2 = u1 + 1;
	int v2 = v1 + 1;

	double u = coord.x * (width - 1);
	double v = coord.y * (height - 1);

	double alpha = (u2 - u) / (double) (u2 - u1);
	double beta = (u - u1) / (double) (u2 - u1);

	double lhsC = ((v2 - v) / (double) (v2-v1));
	glm::dvec3 lhs(0,0,0);
	if(abs(lhsC) > 1e-12){
        glm::dvec3 a = getPixelAt(u1, v1);
        glm::dvec3 b = getPixelAt(u2, v1);
        lhs = lhsC * (alpha * a + beta * b);
	}
	double rhsC = ((v - v1) / (double) (v2-v1));
	glm::dvec3 rhs(0,0,0);
	if(abs(rhsC) > 1e-12){
		glm::dvec3 c = getPixelAt(u2, v2);
		glm::dvec3 d = getPixelAt(u1, v2);
		rhs = rhsC * (alpha * d + beta * c);
	}

	glm::dvec3 res = lhs + rhs;
	// return getPixelAt(coord.x * (width - 1), coord.y * (height - 1));
	return res;
}

glm::dvec3 TextureMap::getPixelAt(int x, int y) const
{
	// cout << "x: " << x << " y: " << y << endl;
	// YOUR CODE HERE
	//
	// In order to add texture mapping support to the
	// raytracer, you need to implement this function. 
	const uint8_t* p = data.data() + (x + y * width) * 3;
	auto res =  glm::dvec3(*(p) / 255.0, *(p + 1) / 255.0, *(p + 2) / 255.0);
	return res;
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
