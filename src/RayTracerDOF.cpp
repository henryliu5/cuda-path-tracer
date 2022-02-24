#include "RayTracer.h"
#include "scene/light.h"
#include "scene/material.h"
#include "scene/ray.h"

#include "parser/Tokenizer.h"
#include "parser/Parser.h"

#include "ui/TraceUI.h"
#include <cmath>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtx/io.hpp>
#include <string.h> // for memset

#include <chrono>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <random>

extern TraceUI* traceUI;

// Trace a top-level ray through pixel(i,j), i.e. normalized window coordinates (x,y),
// through the projection plane, and out into the scene.  All we do is
// enter the main ray-tracing method, getting things started by plugging
// in an initial ray weight of (0.0,0.0,0.0) and an initial recursion depth of 0.

glm::dvec3 RayTracer::traceDOF(double x, double y)
{
	// Clear out the ray cache in the scene for debugging purposes,
	if (TraceUI::m_debug)
	{
		scene->clearIntersectCache();		
	}

	ray r(glm::dvec3(0,0,0), glm::dvec3(0,0,0), glm::dvec3(1,1,1), ray::VISIBILITY);
	scene->getCamera().rayThrough(x,y,r);
	double dummy;
	glm::dvec3 ret = traceRay(r, glm::dvec3(1.0,1.0,1.0), traceUI->getDepth(), dummy);
	// clamp color
	ret = glm::clamp(ret, 0.0, 1.0);
	return ret;
}

// trace pixel thru window coord
void RayTracer::tracePixelDOF(int i, int j, double FOCAL_DISTANCE, unsigned int SAMPLES, uniform_real_distribution<double>& unif, default_random_engine& re)
{
    glm::dvec3 eye = scene->getCamera().getEye();
    glm::dvec3 colorSum(0.0, 0.0, 0.0);
    glm::dvec3 focalPoint = getFocalPoint(i, j, FOCAL_DISTANCE);

    // Clear out the ray cache in the scene for debugging purposes,
    if (TraceUI::m_debug) {
        scene->clearIntersectCache();
    }
    // cout << "focalPoint: " << focalPoint << endl;
    for (unsigned int k = 0; k < SAMPLES; ++k) {
        double xShift = unif(re);
        double yShift = unif(re);

        glm::dvec3 jitteredEye = eye + scene->getCamera().getU() * xShift + scene->getCamera().getV() * yShift;
        // cout << "jitteredEye: " << jitteredEye << endl;
        // exit(0);
        ray r(jitteredEye, glm::normalize(focalPoint - jitteredEye), glm::dvec3(1, 1, 1), ray::VISIBILITY);
        double dummy;
        colorSum += traceRay(r, glm::dvec3(1.0, 1.0, 1.0), traceUI->getDepth(), dummy);
        // clamp color
    }
    colorSum /= glm::dvec3(SAMPLES, SAMPLES, SAMPLES);
    colorSum = glm::clamp(colorSum, 0.0, 1.0);
    // update pixel buffer w color
    unsigned char* pixel = buffer.data() + (i + j * buffer_width) * 3;

    pixel[0] = (int)(255.0 * colorSum.x);
    pixel[1] = (int)(255.0 * colorSum.y);
    pixel[2] = (int)(255.0 * colorSum.z);
}

// Get focal point for this pixel
glm::dvec3 RayTracer::getFocalPoint(int i, int j, double focalDistance){
	double x = double(i)/double(buffer_width);
	double y = double(j)/double(buffer_height);
    ray r(glm::dvec3(0, 0, 0), glm::dvec3(0, 0, 0), glm::dvec3(1, 1, 1), ray::VISIBILITY);
    scene->getCamera().rayThrough(x, y, r);
    return r.at(focalDistance);
}