#include "cubeMap.h"
#include "ray.h"
#include "../ui/TraceUI.h"
#include "../scene/material.h"
#include <glm/gtx/io.hpp>
#include <iostream>
using namespace std;
extern TraceUI* traceUI;


glm::dvec3 CubeMap::getColor(ray r1) const
{
	glm::dvec3 res(0,0,0);
	int i = -1;
	glm::dvec3 dir = r1.getDirection();
	// cout << "ray: " << dir << endl;
	double x = dir.x;
	double y = dir.y;
	double z = -dir.z;
	double r, s;


	if(abs(x) > abs(y) && abs(x) > abs(z)){
		r = z / abs(x);
		s = y / abs(x);
		if(x > 0){
			i = 0;
			r *= -1;
		} else {
			i = 1;
		}
	} else if(abs(y) >= abs(x) && abs(y) >= abs(z)){
		r = x / abs(y);
		s = z / abs(y);
		if(y > 0){
			i = 2;
			s *= -1;
		} else {
			i = 3;
		}
	} else {
		r = x / abs(z);
		s = y / abs(z);
		if(z > 0){
			i = 4;
		} else {
			i = 5;
			r *= -1;
		}
	}
	// cout << "r,s : " << r << " " << s << endl;
	// r = (r + 1) * 0.5;
	// s = (s + 1) * 0.5;
	r = r * 0.5 + 0.5;
	s = s * 0.5 + 0.5;
	

	// cout << "r,s2: " << r << " " << s << endl;
	// cout << "i " << i << endl;

	return tMap[i].get()->getMappedValue(glm::dvec2(r,s));
}

CubeMap::CubeMap()
{
}

CubeMap::~CubeMap()
{
}

void CubeMap::setNthMap(int n, TextureMap* m)
{
	if (m != tMap[n].get())
		tMap[n].reset(m);
}
