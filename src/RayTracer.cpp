// The main ray tracer.

#pragma warning (disable: 4786)

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

#include <fstream>
#include <iostream>
#include <pthread.h>
#include <chrono>


#define PI 3.14159265358979311600

using namespace std;
extern TraceUI* traceUI;

// Do recursive ray tracing!  You'll want to insert a lot of code here
// (or places called from here) to handle reflection, refraction, etc etc.
// glm::dvec3 pathTraceRay(ray& r, int depth, BVHTree* bvhTree) {
// 	static uniform_real_distribution<double> unif(0, 1);
// 	static default_random_engine re;

// 	static uniform_real_distribution<double> unboundedUnif;

// 	isect i;
// 	glm::dvec3 myContrib;
// 	glm::dvec3 colorC;
// 	#if VERBOSE
// 		std::cerr << "== current depth: " << depth << std::endl;
// 	#endif


// 	double coeffRR = 1.0;
// 	if(traceUI->getDepth() - depth >= 3) {
// 		double probRR = 0.30;
// 		double sampleRR = unif(re);
// 		if (sampleRR >= probRR) {
// 			return colorC;
// 		}
// 		coeffRR = 1 / probRR;
// 	}

// 	if(bvhTree->traverse(r, i)) {
// 		// cout << "Emissive" << endl;
// 		const Material& m = i.getMaterial();
// 		myContrib += (m.ke(i) * coeffRR);		
// 		colorC += (m.ke(i) * coeffRR);
// 		// colorC += m.shade(scene.get(), r, i);	
// 		// if (depth > 0) {
// 			// cout << "Kd of Material at i is : " << m.kd(i).x << " " << m.kd(i).y << " " << m.kd(i).z << " " << m.Diff() << " \n" << endl;
// 				if (m.Diff()) {
// 					//DIFFUSE BRDF
// 					glm::dvec3 normal = glm::normalize(i.getN());

// 					// Cosine weighted sample
// 					double r1 = 2 * PI * unif(re);
// 					double r2 = unif(re);
// 					double r2s = sqrt(r2);

// 					auto w = normal;
// 					auto u = glm::normalize(glm::cross((abs(w.x) > .1 ? glm::dvec3(0, 1, 0) : glm::dvec3(1, 0, 0)), w));
// 					auto v = glm::cross(w,u);
// 					auto rand_dir = glm::normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1 - r2));
// 					double p = 1 / PI;
// 					double theta = 1;

// 					ray ray2(r.at(i) + normal * 1e-12, rand_dir, r.getAtten(), ray::REFLECTION);

					
// 					glm::dvec3 brdf = m.kd(i) / PI;

// 					glm::dvec3 res = pathTraceRay(ray2, depth - 1, bvhTree);
// 					colorC += ((brdf * res * theta / p) * coeffRR);
// 				}
// 				if (m.Spec()) {
// 					// cout << "Specular" << endl;
// 					//SPECULAR BRDF (assuming glossy surface with no roughness)
// 					glm::dvec3 w_in = r.getDirection();
// 					glm::dvec3 normal = i.getN();
// 					if(r.currentIndex != 1.0){
// 						normal *= -1.0;
// 					}
// 					glm::dvec3 w_normal = glm::dot(w_in, normal) * normal;
// 					glm::dvec3 w_tan = w_in - w_normal;
// 					glm::dvec3 w_ref = -w_normal + w_tan;
// 					w_ref = glm::normalize(w_ref);

// 					ray reflect(r.at(i) + normal * 1e-12, w_ref, r.getAtten(), ray::REFLECTION);
// 					// ray reflect(r.at(i) + normal * 1e-12, w_ref + (rand_dir*.1), r.getAtten(), ray::REFLECTION);
// 					reflect.currentIndex = r.currentIndex;

// 					glm::dvec3 temp = pathTraceRay(reflect, depth - 1, bvhTree);
// 					colorC += ((m.ks(i) * temp) * coeffRR);

// 				}
// 				if (m.Trans()) {
// 					// cout << "Transmissive" << endl;
// 					glm::dvec3 w_in = r.getDirection();
// 					glm::dvec3 normal = i.getN();

// 					double n1;
// 					double n2;
// 					glm::dvec3 trans = {1,1,1};
// 					//Based on entering/exiting a material, set n1, n2, and trans
// 					if(r.currentIndex == 1.0){
// 						n1 = r.currentIndex;
// 						n2 = m.index(i);
// 					} else{
// 						n1 = m.index(i);
// 						n2 = 1.0;
// 						normal *= -1;
// 						trans = glm::pow(m.kt(i), {i.getT(),i.getT(),i.getT()});
// 					}

// 					double r0 = (n1 - n2)/(n1 + n2);
// 					r0 *= r0;

// 					double n = n1/n2;

// 					w_in = -glm::normalize(w_in);
					
// 					double cosI = glm::dot(normal, w_in); //cosine of theta_1 (incident angle)

// 					double cosR = 1 - n*n * (1-cosI*cosI); //cosine of theta_2 (refraction angle)
// 					if(cosR >= 0){
// 						//Potentially Refract/Reflect
// 						double reflCoeff = r0 + (1 - r0)*pow((1 - cosI), 5.0); //Schlick's Approximation of Fresnel's Constant

// 						double sampleFresnel = unboundedUnif(re);

// 						if (sampleFresnel > reflCoeff) {
// 							//refraction
// 							double cosT = glm::sqrt(cosR);
// 							glm::dvec3 refrac = (n*cosI - cosT) * normal - n*w_in; //direction of the refracted ray

// 							ray r2(r.at(i) - normal * 1e-12, refrac, r.getAtten(), ray::REFRACTION);
// 							r2.currentIndex = n2;

// 							glm::dvec3 temp = pathTraceRay(r2, depth - 1, bvhTree);

// 							colorC += ((trans * temp) * coeffRR);
// 						} else {
// 							//reflection
// 							glm::dvec3 w_in = r.getDirection();
// 							glm::dvec3 normal = i.getN();
// 							if(r.currentIndex != 1.0){
// 								normal *= -1.0;
// 							}
// 							glm::dvec3 w_normal = glm::dot(w_in, normal) * normal;
// 							glm::dvec3 w_tan = w_in - w_normal;
// 							glm::dvec3 w_ref = -w_normal + w_tan;
// 							w_ref = glm::normalize(w_ref);

// 							ray reflect(r.at(i) + normal * 1e-12, w_ref, r.getAtten(), ray::REFLECTION);
// 							// ray reflect(r.at(i) + normal * 1e-12, w_ref + (rand_dir*.1), r.getAtten(), ray::REFLECTION);
// 							reflect.currentIndex = r.currentIndex;

// 							glm::dvec3 temp = pathTraceRay(reflect, depth - 1, bvhTree);
// 							colorC += ((m.ks(i) * temp) * coeffRR);
// 						}	
// 					} else {
// 						//Total Internal Reflection 
// 						glm::dvec3 w_in = r.getDirection();
// 						glm::dvec3 w_normal = glm::dot(w_in, normal) * normal;
// 						glm::dvec3 w_tan = w_in - w_normal;
// 						glm::dvec3 w_ref = -w_normal + w_tan;

// 						w_ref = glm::normalize(w_ref);
// 						ray reflect(r.at(i) + normal * 1e-12, w_ref, r.getAtten(), ray::REFLECTION);
// 						reflect.currentIndex = r.currentIndex;

// 						glm::dvec3 temp = pathTraceRay(reflect, depth - 1, bvhTree);
// 						colorC += ((m.kr(i) * trans * temp) * coeffRR);
// 					}
// 				}	
				
// 		// } 
// 	} else {
// 			// No intersection.  This ray travels to infinity, so we color
// 			// it according to the background color, which in this (simple) case
// 			// is just black.
// 			colorC = glm::dvec3(0.0, 0.0, 0.0);
// 			if(traceUI->cubeMap()){
// 				CubeMap* cubeMap = traceUI->getCubeMap();
// 				glm::dvec3 kd = cubeMap->getColor(r);
// 				colorC += kd;
// 			}
// 	}
// 	// if(depth >= 1){
// 	// 	return colorC - myContrib;
// 	// }
// 	return colorC;
// }


glm::dvec3 tracePixelPath(unsigned char* buffer, Scene* scene, BVHTree* bvhTree, int pixelI, int pixelJ, int bufferWidth, int bufferHeight, int maxDepth, int samplesPerPixel);

// Use this variable to decide if you want to print out
// debugging messages.  Gets set in the "trace single ray" mode
// in TraceGLWindow, for example.
bool debugMode = false;

// Trace a top-level ray through pixel(i,j), i.e. normalized window coordinates (x,y),
// through the projection plane, and out into the scene.  All we do is
// enter the main ray-tracing method, getting things started by plugging
// in an initial ray weight of (0.0,0.0,0.0) and an initial recursion depth of 0.

glm::dvec3 RayTracer::trace(double x, double y)
{
	// Clear out the ray cache in the scene for debugging purposes,
	if (TraceUI::m_debug)
	{
		scene->clearIntersectCache();		
	}

	ray r(glm::dvec3(0,0,0), glm::dvec3(0,0,0), glm::dvec3(1,1,1), ray::VISIBILITY);
	scene->getCamera().rayThrough(x,y,r);

	// glm::dvec3 ret = pathTraceRay(r, glm::dvec3(1.0,1.0,1.0), traceUI->getDepth());
	double dummy;
	glm::dvec3 ret = traceRay(r, glm::dvec3(1.0,1.0,1.0), traceUI->getDepth(), dummy);
	// clamp color
	ret = glm::clamp(ret, 0.0, 1.0);
	return ret;
}

// trace pixel thru window coord
extern int intersectCallCount;
extern int trimeshCount;
glm::dvec3 RayTracer::tracePixel(int i, int j)
{
	intersectCallCount = 0;
	trimeshCount = 0;
	glm::dvec3 col(0,0,0);

	if( ! sceneLoaded() ) return col;

	double x = double(i)/double(buffer_width);
	double y = double(j)/double(buffer_height);

	// update pixel buffer w color
	unsigned char *pixel = buffer.data() + ( i + j * buffer_width ) * 3;
	col = trace(x, y);

	pixel[0] = (int)( 255.0 * col[0]);
	pixel[1] = (int)( 255.0 * col[1]);
	pixel[2] = (int)( 255.0 * col[2]);
	// cout << "intersectCallCount: " << intersectCallCount << endl;
	// cout << "visitBoth: " << bvhTree.visitBoth << endl;
	// cout << "traverseCount: " << bvhTree.traverseCount << endl;
	// cout << "trimeshCount: " << trimeshCount << endl;
	return col;
}

glm::dvec3 RayTracer::tracePixelAA(int i, int j){
    glm::dvec3 colSum(0, 0, 0);

    int subpixels = 1;
    if (traceUI->aaSwitch()) {
        subpixels = samples;
    }

    for (int subX = i * subpixels; subX < i * subpixels + subpixels; subX++) {
        for (int subY = j * subpixels; subY < j * subpixels + subpixels; subY++) {
            glm::dvec3 col(0, 0, 0);

            if (!sceneLoaded())
                return colSum;

            double x = double(subX + fRand(0, 1)) / double(buffer_width * subpixels);
            double y = double(subY + fRand(0, 1)) / double(buffer_height * subpixels);

            col = trace(x, y);

            colSum += col;
        }
    }
    // update pixel buffer w color
    unsigned char* pixel = buffer.data() + (i + j * buffer_width) * 3;
    pixel[0] = (int)(255.0 * colSum.x / (subpixels * subpixels));
    pixel[1] = (int)(255.0 * colSum.y / (subpixels * subpixels));
    pixel[2] = (int)(255.0 * colSum.z / (subpixels * subpixels));

    return colSum;
}

//getting random vector in hemisphere: https://www.scratchapixel.com/lessons/3d-basic-rendering/global-illumination-path-tracing/global-illumination-path-tracing-practical-implementation
glm::dvec3 RayTracer::sampleHemisphere(glm::dvec3& curNorm) {
	//Generate Sample Vector (Y component is "up")
	static uniform_real_distribution<double> unif(0, 1);
	static default_random_engine re;

	double p1 = unif(re);
	double p2 = unif(re);

	double sinTheta = sqrt(1 - p1 * p1);
	double phi = 2 * PI * p2;
	double x = sinTheta * cos(phi);
	double z = sinTheta * sin(phi);
	glm::dvec3 sample(x, p1, z);

	//Get Transformation Matrix to transform sample to world (Y vector is "up" -> normal vector is "up")
	glm::dvec3 normAxis1 = abs(curNorm.x) > abs(curNorm.y) ?  
						   glm::dvec3(curNorm.z, 0, -curNorm.x) / sqrt(curNorm.x * curNorm.x + curNorm.z * curNorm.z): 
						   glm::dvec3(0, -curNorm.z, curNorm.y) / sqrt(curNorm.y * curNorm.y + curNorm.z * curNorm.z);

	glm::dvec3 normAxis2 = glm::cross(curNorm, normAxis1);

	glm::dvec3 actualSample(
		sample.x * normAxis2.x + sample.y * curNorm.x + sample.z * normAxis1.x, 
		sample.x * normAxis2.y + sample.y * curNorm.y + sample.z * normAxis1.y, 
		sample.x * normAxis2.z + sample.y * curNorm.z + sample.z * normAxis1.z
	);
	
	return actualSample;
}



#define VERBOSE 0

// Do recursive ray tracing!  You'll want to insert a lot of code here
// (or places called from here) to handle reflection, refraction, etc etc.
glm::dvec3 RayTracer::traceRay(ray& r, const glm::dvec3& thresh, int depth, double& t)
{
	isect i;
	glm::dvec3 colorC;
	#if VERBOSE
		std::cerr << "== current depth: " << depth << std::endl;
	#endif
	// if(scene->intersect(r, i)) {
	if(bvhTree.traverse(r, i)) {
		// YOUR CODE HERE

		// An intersection occurred!  We've got work to do.  For now,
		// this code gets the material for the surface that was intersected,
		// and asks that material to provide a color for the ray.

		// This is a great place to insert code for recursive ray tracing.
		// Instead of just returning the result of shade(), add some
		// more steps: add in the contributions from reflected and refracted
		// rays.
		const Material& m = i.getMaterial();
		colorC = m.shade(scene.get(), r, i);	
		// if(debugMode) cout << "tracing" << endl;
		if(depth > 0){
			//Path Tracing based on https://www.cs.rpi.edu/~cutler/classes/advancedgraphics/S10/final_projects/carr_hulcher.pdf
			// if(m.Diff()) {
			// 	glm::dvec3 normal = i.getN();
			// 	glm::dvec3 rand_dir = glm::normalize(sampleHemisphere(normal));
			// 	ray r2(r.at(i) + normal * 1e-12, rand_dir, r.getAtten(), ray::REFLECTION);
			// 	r2.currentIndex = r.currentIndex;
			// 	double dum;
			// 	colorC += traceRay(r2, glm::dvec3(1.0,1.0,1.0), depth - 1, dum) * glm::dot(rand_dir, normal);
			// }
			if(m.Refl()){
				// if(debugMode) cout << "shooting reflection" << endl;
				glm::dvec3 w_in = r.getDirection();
				glm::dvec3 normal = i.getN();
				if(r.currentIndex != 1.0){
					normal *= -1.0;
				}
				glm::dvec3 w_normal = glm::dot(w_in, normal) * normal;
				glm::dvec3 w_tan = w_in - w_normal;
				glm::dvec3 w_ref = -w_normal + w_tan;
				w_ref = glm::normalize(w_ref);

				
				glm::dvec3 rand_dir = glm::normalize(sampleHemisphere(normal));
				// ray reflect(r.at(i) + normal * 1e-12, rand_dir, r.getAtten(), ray::REFLECTION);
				ray reflect(r.at(i) + normal * 1e-12, w_ref, r.getAtten(), ray::REFLECTION);
				// ray reflect(r.at(i) + normal * 1e-12, w_ref + (rand_dir*.1), r.getAtten(), ray::REFLECTION);
				reflect.currentIndex = r.currentIndex;

				double dum;
				glm::dvec3 temp = traceRay(reflect, glm::dvec3(1.0,1.0,1.0), depth - 1, dum);
				colorC += m.kr(i) * temp;
				// colorC += ((m.kr(i) / PI) * temp * glm::dot(rand_dir, normal) / (1 / (2 * PI)));

				// for(int curSample = 0; curSample < SAMPLES_PER_PIXEL; ++curSample) {
				// 	double p1 = distrib(engine);
				// 	double p2 = distrib(engine);
				// 	glm::dvec3 rand_dir = glm::normalize(sampleHemisphere(normal));
				// 	// cout << "Sample " << curSample << " random vector: " << rand_dir.x << " " << rand_dir.y << " " << rand_dir.z << "\n";
				// ray reflect(r.at(i) + normal * 1e-12, rand_dir, r.getAtten(), ray::REFLECTION);
				// 	//ray reflect(r.at(i) + normal * 1e-12, w_ref, r.getAtten(), ray::REFLECTION);

				// 	reflect.currentIndex = r.currentIndex;

				// 	double dum;
				// 	glm::dvec3 temp = traceRay(reflect, glm::dvec3(1.0,1.0,1.0), depth - 1, dum);
				// 	colorC += m.kr(i) * temp;
				// }
				// colorC *= ((2 * PI) / SAMPLES_PER_PIXEL);
			}
			if(m.Trans()){
				// if(debugMode) cout << "shooting refraction" << endl;
				glm::dvec3 w_in = r.getDirection();
				glm::dvec3 normal = i.getN();

				double n1;
				double n2;
				glm::dvec3 trans = {1,1,1};
				if(r.currentIndex == 1.0){
					n1 = r.currentIndex;
					n2 = m.index(i);
				} else{
					n1 = m.index(i);
					n2 = 1.0;
					normal *= -1;
					trans = glm::pow(m.kt(i), {i.getT(),i.getT(),i.getT()});
				}
				
				double n = n1/n2;
				// if(debugMode) cout << "n1: " << n1 << " n2: " << n2 << endl;
				w_in = -glm::normalize(w_in);
				double cosI = glm::dot(normal, w_in);
				double x = 1 - n*n * (1-cosI*cosI);
				if(x >= 0){
					double cosT = glm::sqrt(x);

					glm::dvec3 refrac = (n*cosI - cosT) * normal - n*w_in;

					glm::dvec3 rand_dir = glm::normalize(sampleHemisphere(normal));
					ray r2(r.at(i) - normal * 1e-12, rand_dir, r.getAtten(), ray::REFRACTION);
					// ray r2(r.at(i) - normal * 1e-12, glm::normalize(refrac), r.getAtten(), ray::REFRACTION);
					// ray r2(r.at(i) - normal * 1e-12, glm::normalize(refrac) + (rand_dir*.1), r.getAtten(), ray::REFRACTION);
					r2.currentIndex = n2;


					double dum;
					glm::dvec3 temp = traceRay(r2, glm::dvec3(1.0,1.0,1.0), depth - 1, dum);
					// if(debugMode) cout << "temp : " << temp << endl;
					colorC += trans * temp;		
					// if(debugMode) cout << "colorC: " << colorC << endl;

					// for(int curSample = 0; curSample < SAMPLES_PER_PIXEL; ++curSample) {
					// 	double p1 = distrib(engine);
					// 	double p2 = distrib(engine);
					// 	glm::dvec3 rand_dir = glm::normalize(sampleHemisphere(normal));
					// 	// cout << "Sample " << curSample << " random vector: " << rand_dir.x << " " << rand_dir.y << " " << rand_dir.z << "\n";
						
					// 	// if(debugMode) cout << "refrac: " << refrac << endl;
					// 	ray r2(r.at(i) - normal * 1e-12, glm::normalize(refrac), r.getAtten(), ray::REFRACTION);
					// 	//ray r2(r.at(i) - normal * 1e-12, rand_dir, r.getAtten(), ray::REFRACTION);
					// 	r2.currentIndex = n2;

					// 	double dum;
					// 	glm::dvec3 temp = traceRay(r2, glm::dvec3(1.0,1.0,1.0), depth - 1, dum);
					// 	// if(debugMode) cout << "temp : " << temp << endl;
					// 	colorC += trans * temp;		
					// 	// if(debugMode) cout << "colorC: " << colorC << endl;
					// }	
					// colorC *= ((2 * PI) / SAMPLES_PER_PIXEL);
					
				} else {
					// if(debugMode) cout << "total internal reflection" << endl;
					glm::dvec3 w_in = r.getDirection();
					glm::dvec3 w_normal = glm::dot(w_in, normal) * normal;
					glm::dvec3 w_tan = w_in - w_normal;
					glm::dvec3 w_ref = -w_normal + w_tan;
					w_ref = glm::normalize(w_ref);

					glm::dvec3 rand_dir = glm::normalize(sampleHemisphere(normal));

					// ray reflect(r.at(i) + normal * 1e-12, rand_dir, r.getAtten(), ray::REFLECTION);
					ray reflect(r.at(i), w_ref, r.getAtten(), ray::REFLECTION);
					// ray reflect(r.at(i), w_ref + (rand_dir*.1), r.getAtten(), ray::REFLECTION);
					reflect.currentIndex = r.currentIndex;

					double dum;
					glm::dvec3 temp = traceRay(reflect, glm::dvec3(1.0,1.0,1.0), depth - 1, dum);
					colorC += m.kr(i) * trans * temp;

					// for(int curSample = 0; curSample < SAMPLES_PER_PIXEL; ++curSample) {
					// 	double p1 = distrib(engine);
					// 	double p2 = distrib(engine);
					// 	glm::dvec3 rand_dir = glm::normalize(sampleHemisphere(normal));
					// 	// cout << "Sample " << curSample << " random vector: " << rand_dir.x << " " << rand_dir.y << " " << rand_dir.z << "\n";

					// 	ray reflect(r.at(i) + normal * 1e-12, rand_dir, r.getAtten(), ray::REFLECTION);
					// 	//ray reflect(r.at(i), w_ref, r.getAtten(), ray::REFLECTION);
					// 	reflect.currentIndex = r.currentIndex;
					// 	double dum;
					// 	glm::dvec3 temp = traceRay(reflect, glm::dvec3(1.0,1.0,1.0), depth - 1, dum);
					// 	colorC += m.kr(i) * trans * temp;
					// }
					// colorC *= ((2 * PI)/ SAMPLES_PER_PIXEL);
				}
				
			}

		}
	} else {
		// No intersection.  This ray travels to infinity, so we color
		// it according to the background color, which in this (simple) case
		// is just black.
		//
		// FIXME: Add CubeMap support here.
		// TIPS: CubeMap object can be fetched from traceUI->getCubeMap();
		//       Check traceUI->cubeMap() to see if cubeMap is loaded
		//       and enabled.
		colorC = glm::dvec3(0.0, 0.0, 0.0);
		if(traceUI->cubeMap()){
			CubeMap* cubeMap = traceUI->getCubeMap();
			glm::dvec3 kd = cubeMap->getColor(r);
			// YOUR CODE HERE
			// FIXME: Implement Cube Map here

			colorC += kd;
		}
	}
	#if VERBOSE
		std::cerr << "== depth: " << depth+1 << " done, returning: " << colorC << std::endl;
	#endif
	return colorC;
}

RayTracer::RayTracer()
	: scene(nullptr), buffer(0), thresh(0), buffer_width(0), buffer_height(0), m_bBufferReady(false)
{
}

RayTracer::~RayTracer()
{
}

void RayTracer::getBuffer( unsigned char *&buf, int &w, int &h )
{
	buf = buffer.data();
	w = buffer_width;
	h = buffer_height;
}

double RayTracer::aspectRatio()
{
	return sceneLoaded() ? scene->getCamera().getAspectRatio() : 1;
}

bool RayTracer::loadScene(const char* fn)
{
	ifstream ifs(fn);
	if( !ifs ) {
		string msg( "Error: couldn't read scene file " );
		msg.append( fn );
		traceUI->alert( msg );
		return false;
	}

	// Strip off filename, leaving only the path:
	string path( fn );
	if (path.find_last_of( "\\/" ) == string::npos)
		path = ".";
	else
		path = path.substr(0, path.find_last_of( "\\/" ));

	// Call this with 'true' for debug output from the tokenizer
	Tokenizer tokenizer( ifs, false );
	Parser parser( tokenizer, path );
	try {
		scene.reset(parser.parseScene());
	}
	catch( SyntaxErrorException& pe ) {
		traceUI->alert( pe.formattedMessage() );
		return false;
	} catch( ParserException& pe ) {
		string msg( "Parser: fatal exception " );
		msg.append( pe.message() );
		traceUI->alert( msg );
		return false;
	} catch( TextureMapException e ) {
		string msg( "Texture mapping exception: " );
		msg.append( e.message() );
		traceUI->alert( msg );
		return false;
	}

	if (!sceneLoaded())
		return false;

	return true;
}

void RayTracer::traceSetup(int w, int h, bool useGPU, int pathTracingSamples)
{
	size_t newBufferSize = w * h * 3;
	if (newBufferSize != buffer.size()) {
		bufferSize = newBufferSize;
		buffer.resize(bufferSize);
	}
	buffer_width = w;
	buffer_height = h;
	std::fill(buffer.begin(), buffer.end(), 0);
	m_bBufferReady = true;

	/*
	 * Sync with TraceUI
	 */

	threads = traceUI->getThreads();
	block_size = traceUI->getBlockSize();
	thresh = traceUI->getThreshold();
	samples = traceUI->getSuperSamples();
	aaThresh = traceUI->getAaThreshold();

	// YOUR CODE HERE
	// FIXME: Additional initializations
	bvhTree.build(scene);
	scene->bvhTree = &bvhTree;

	// cout << "max: " << bvhTree.root->bb.getMax() << endl;
	// cout << "min: " << bvhTree.root->bb.getMin() << endl;

	// cout << "max: " << bvhTree.root->left->bb.getMax() << endl;
	// cout << "min: " << bvhTree.root->left->bb.getMin() << endl;
	// cout << "max: " << bvhTree.root->right->bb.getMax() << endl;
	// cout << "min: " << bvhTree.root->right->bb.getMin() << endl;

    pixThreads = new thread[threads];
    pixThreadsDone = new bool[threads];
	for(int i = 0; i < threads; ++i){
		pixThreadsDone[i] = false;
	}
    // cout << "threads: " << threads << "\n";
	this->useGPU = useGPU;
	if(this->useGPU){
		cout << "GPU Enabled\n";
	}
	SAMPLES_PER_PIXEL = pathTracingSamples;
	std::cout << "Current samples per pixel: " << SAMPLES_PER_PIXEL << endl;
}

/*
 * RayTracer::traceImage
 *
 *	Trace the image and store the pixel data in RayTracer::buffer.
 *
 *	Arguments:
 *		w:	width of the image buffer
 *		h:	height of the image buffer
 *
 */
void RayTracer::traceImageCPU(int w, int h)
{
	// Enable/disable depth of field with below config
	const bool USE_DOF = false;
    // Good config for easy3: fd = 8.5, a = 0.2
	// Good config for reflection2: fd = 6, a = 0.3
    const double FOCAL_DISTANCE = 8.5;
    // Side length of camera aperture (square)
    const double APERTURE = 0.2;
    // Number of samples per pixel
    const unsigned int SAMPLES = 128;

	// int debugInterval = 16;
	static uniform_real_distribution<double> unif(-APERTURE, APERTURE);
	static default_random_engine re;

	for(int threadId = 0; threadId < threads; ++threadId){
		pixThreads[threadId] = std::thread([=]() {
			auto start = chrono::steady_clock::now();
			
			// Compute pixels for this thread
			for (int index1d = threadId; index1d < w * h; index1d += threads) {
				int i = index1d / h;
				int j = index1d % h;

				if (USE_DOF) {
					tracePixelDOF(i, j, FOCAL_DISTANCE, SAMPLES, unif, re);
				} else if (traceUI->aaSwitch()) { 
					tracePixelAA(i, j);
				} else {
					tracePixelPath(buffer.data(), scene.get(), &bvhTree, i, j, w, h, traceUI->getDepth(), SAMPLES_PER_PIXEL);
					//tracePixel(i, j);
				}
			}
			pixThreadsDone[threadId] = true;
		});
	}
}



void RayTracer::traceImage(int w, int h){
	// Always call traceSetup before rendering anything.
	traceSetup(w,h, useGPU, SAMPLES_PER_PIXEL);
	if(useGPU){
		traceImageGPU(w, h);
		// betterPathTracing(w, h);
	} else {
		traceImageCPU(w, h);
	}
}

double RayTracer::fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

int RayTracer::aaImage(){ return 0; }

bool RayTracer::checkRender()
{
	// YOUR CODE HERE
	// FIXME: Return true if tracing is done.
	//        This is a helper routine for GUI.
	//
	// TIPS: Introduce an array to track the status of each worker thread.
	//       This array is maintained by the worker threads.
	if(!pixThreadsDone) return true;
	for (int i = 0; i < threads; ++i) {
		if(!pixThreadsDone[i]){
			return false;
		}
	}
	waitRender();
	return true;
}

void RayTracer::waitRender()
{
	// YOUR CODE HERE
	// FIXME: Wait until the rendering process is done.
	//        This function is essential if you are using an asynchronous
	//        traceImage implementation.
	//
	// TIPS: Join all worker threads here.
	for(int i = 0; i < threads; ++i){
		if(pixThreads[i].joinable())
			pixThreads[i].join();
	}
	// delete[] pixThreads;
	// delete[] pixThreadsDone;
	// pixThreads = 0;
	// pixThreadsDone = 0;
	// cout << " ------------- done" << endl;
}


glm::dvec3 RayTracer::getPixel(int i, int j)
{
	unsigned char *pixel = buffer.data() + ( i + j * buffer_width ) * 3;
	return glm::dvec3((double)pixel[0]/255.0, (double)pixel[1]/255.0, (double)pixel[2]/255.0);
}

void RayTracer::setPixel(int i, int j, glm::dvec3 color)
{
	unsigned char *pixel = buffer.data() + ( i + j * buffer_width ) * 3;

	pixel[0] = (int)( 255.0 * color[0]);
	pixel[1] = (int)( 255.0 * color[1]);
	pixel[2] = (int)( 255.0 * color[2]);
}

