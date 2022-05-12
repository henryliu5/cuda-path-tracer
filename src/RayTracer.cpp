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
#define SAMPLES_PER_PIXEL 128

using namespace std;
extern TraceUI* traceUI;

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

glm::dvec3 RayTracer::tracePixelPath(int pixelI, int pixelJ, int samplesPerPixel = SAMPLES_PER_PIXEL) {
	intersectCallCount = 0;
	trimeshCount = 0;
	
	glm::dvec3 colSum(0, 0, 0);

	if( ! sceneLoaded() ) return colSum;

	static uniform_real_distribution<double> unif(-0.5, 0.5);
	static default_random_engine re;

	// update pixel buffer w color
	for(int sample = 0; sample < samplesPerPixel; ++sample) {
		if (!sceneLoaded())
		        return colSum;
		// Shoot ray through camera
		ray r(glm::dvec3(0,0,0), glm::dvec3(0,0,0), glm::dvec3(1,1,1), ray::VISIBILITY);

		// double iShift = unif(re);
		// double jShift = unif(re);
		// double newI = pixelI + iShift;
		// double newJ = pixelJ + jShift;
		// if(newI < 0) newI = 0; if(newI >= buffer_width) newI = buffer_width - 1;
		// if(newJ < 0) newJ = 0; if(newJ >= buffer_height) newJ = buffer_height - 1;
		// double x = newI/double(buffer_width);
		// double y = newJ/double(buffer_height);
		double x = double(pixelI)/double(buffer_width);
		double y = double(pixelJ)/double(buffer_height);

		scene->getCamera().rayThrough(x,y,r);
		auto color = pathTraceRay(r, traceUI->getDepth());
		// color = glm::clamp(color, 0.0, 1.0);
		colSum += color;
	}

	// colSum = glm::clamp(colSum, 0.0, (double) samplesPerPixel);
	colSum = glm::dvec3(colSum.x / samplesPerPixel, colSum.y / samplesPerPixel, colSum.z / samplesPerPixel);
	colSum = glm::clamp(colSum, 0.0, 1.0);

	unsigned char *pixel = buffer.data() + ( pixelI + pixelJ * buffer_width ) * 3;
	pixel[0] = (int)( 255.0 * colSum.x);
	pixel[1] = (int)( 255.0 * colSum.y);
	pixel[2] = (int)( 255.0 * colSum.z);
	return colSum;
}

//getting random vector in hemisphere: https://www.scratchapixel.com/lessons/3d-basic-rendering/global-illumination-path-tracing/global-illumination-path-tracing-practical-implementation
glm::dvec3 RayTracer::sampleHemisphere(glm::dvec3& curNorm) {
	//Generate Sample Vector (Y component is "up")
	uniform_real_distribution<double> unif(0, 1);
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
				ray reflect(r.at(i) + normal * 1e-12, rand_dir, r.getAtten(), ray::REFLECTION);
				// ray reflect(r.at(i) + normal * 1e-12, w_ref, r.getAtten(), ray::REFLECTION);
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

// Do recursive ray tracing!  You'll want to insert a lot of code here
// (or places called from here) to handle reflection, refraction, etc etc.
glm::dvec3 RayTracer::pathTraceRay(ray& r, int depth) {
	isect i;
	glm::dvec3 myContrib;
	glm::dvec3 colorC;
	#if VERBOSE
		std::cerr << "== current depth: " << depth << std::endl;
	#endif

	if(bvhTree.traverse(r, i)) {
		const Material& m = i.getMaterial();
		myContrib += m.ke(i);		
		colorC += m.ke(i);
		// colorC += m.shade(scene.get(), r, i);	
		if (depth > 0) {
			glm::dvec3 normal = glm::normalize(i.getN());
			glm::dvec3 rand_dir = glm::normalize(sampleHemisphere(normal));
			// cout << rand_dir << " " << glm::dot(normal, rand_dir) << endl;
            ray r2(r.at(i) + normal * 1e-12, rand_dir, r.getAtten(), ray::REFLECTION);

			double p = 1 / (2 * PI); 
			double theta = glm::dot(r2.getDirection(), normal); 
			glm::dvec3 brdf = m.kd(i) / PI;

			glm::dvec3 res = pathTraceRay(r2, depth - 1);
			colorC += (brdf * res * theta / p);
		} 
	} else {
			// No intersection.  This ray travels to infinity, so we color
			// it according to the background color, which in this (simple) case
			// is just black.
			colorC = glm::dvec3(0.0, 0.0, 0.0);
			if(traceUI->cubeMap()){
				CubeMap* cubeMap = traceUI->getCubeMap();
				glm::dvec3 kd = cubeMap->getColor(r);
				colorC += kd;
			}
	}
	#if VERBOSE
		std::cerr << "== depth: " << depth+1 << " done, returning: " << colorC << std::endl;
	#endif
	// if(depth >= 1){
	// 	return colorC - myContrib;
	// }
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

void RayTracer::traceSetup(int w, int h)
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
void RayTracer::traceImage(int w, int h)
{
	// Always call traceSetup before rendering anything.
	traceSetup(w,h);

	// Enable/disable depth of field with below config
	const bool USE_DOF = false;
    // Good config for easy3: fd = 8.5, a = 0.2
	// Good config for reflection2: fd = 6, a = 0.3
    const double FOCAL_DISTANCE = 8.5;
    // Side length of camera aperture (square)
    const double APERTURE = 0.2;
    // Number of samples per pixel
    const unsigned int SAMPLES = 128;

    // YOUR CODE HERE
    // FIXME: Start one or more threads for ray tracing
    //
    // TIPS: Ideally, the traceImage should be executed asynchronously,
    //       i.e. returns IMMEDIATELY after working threads are launched.
    //
    //       An asynchronous traceImage lets the GUI update your results
    //       while rendering.

	int debugInterval = 16;
	static uniform_real_distribution<double> unif(-APERTURE, APERTURE);
	static default_random_engine re;

	std::cout << "Current samples per pixel: " << SAMPLES_PER_PIXEL << endl;

	for(int threadId = 0; threadId < threads; ++threadId){
		pixThreads[threadId] = std::thread([=]() {
			auto start = chrono::steady_clock::now();
			
			int count = 0;
			// Compute pixels for this thread
			// for (int index1d = threadId; index1d < w * h; index1d += threads) {
			// int n = w*h;
			for (int index1d = threadId; index1d < w * h; index1d += threads) {
				int i = index1d / h;
				int j = index1d % h;
// 				if(threadId == 0){
// 					if(count % (512*64) == 0){
// 						cout << "count: " << count << endl;
// std::cout << "Avg. elapsed(ms) = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count() / (double)count << "\n";
// 						cout << "thread 0 finished col: " << i << endl;
// 						start = chrono::steady_clock::now();
// 						count = 0;

// 						cout << "total traverals: " << bvhTree.traverseCount << endl;
// 						cout << "visit %: " << bvhTree.percentSum / bvhTree.traversals << endl;

// 						bvhTree.traversals = 0;
// 						bvhTree.percentSum = 0;
// 					}
					
// 				}
				if (USE_DOF) {
					tracePixelDOF(i, j, FOCAL_DISTANCE, SAMPLES, unif, re);
				} else if (traceUI->aaSwitch()) { 
					tracePixelAA(i, j);
				} else {
					tracePixelPath(i, j);
					// tracePixel(i, j);
				}
			}
			pixThreadsDone[threadId] = true;
		});
	}
}

double RayTracer::fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

int RayTracer::aaImage()
{
    // YOUR CODE HERE
    // FIXME: Implement Anti-aliasing here
    //
    // TIP: samples and aaThresh have been synchronized with TraceUI by
    //      RayTracer::traceSetup() function

	// Done in tracePixelAA() above
	return 0;
}

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

