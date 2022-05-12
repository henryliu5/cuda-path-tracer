#ifndef __RAYTRACER_H__
#define __RAYTRACER_H__

#define MAX_THREADS 32

// The main ray tracer.

#include <time.h>
#include <glm/vec3.hpp>
#include <queue>
#include <thread>
#include "scene/cubeMap.h"
#include "scene/ray.h"
#include <mutex>
#include "scene/bvh.h"
#include <random>
#include <thread>

class Scene;
class Pixel {
public:
	Pixel(int i, int j, unsigned char* ptr) : ix(i), jy(j), value(ptr) {}

	int ix;
	int jy;
	unsigned char* value;
};


class RayTracer {
public:
	RayTracer();
	~RayTracer();


	glm::dvec3 sampleHemisphere(glm::dvec3& curNorm);

	glm::dvec3 tracePixelPath(int i, int j, int samples);
	glm::dvec3 tracePixelAA(int i, int j);
	glm::dvec3 tracePixel(int i, int j);
	glm::dvec3 traceRay(ray& r, const glm::dvec3& thresh, int depth,
	                    double& length);
	glm::dvec3 pathTraceRay(ray& r, const glm::dvec3& thresh, int depth,
						double& length);

	glm::dvec3 getPixel(int i, int j);
	void setPixel(int i, int j, glm::dvec3 color);
	void getBuffer(unsigned char*& buf, int& w, int& h);
	double aspectRatio();

	void traceImage(int w, int h);
	int aaImage();
	bool checkRender();
	void waitRender();

	void traceSetup(int w, int h);

	bool loadScene(const char* fn);
	bool sceneLoaded() { return scene != 0; }

	void setReady(bool ready) { m_bBufferReady = ready; }
	bool isReady() const { return m_bBufferReady; }

	const Scene& getScene() { return *scene; }

	void tracePixelDOF(int i, int j, double FOCAL_DISTANCE, unsigned int SAMPLES, std::uniform_real_distribution<double>& unif, std::default_random_engine& re);
	glm::dvec3 traceDOF(double x, double y);

	bool stopTrace;
	BVHTree bvhTree;
private:
	double fRand(double fMin, double fMax);
	glm::dvec3 getFocalPoint(int x, int y, double focalDistance);
	glm::dvec3 trace(double x, double y);

	std::vector<unsigned char> buffer;
	int buffer_width, buffer_height;
	int bufferSize;
	unsigned int threads;
	int block_size;
	double thresh;
	double aaThresh;
	int samples;
	std::unique_ptr<Scene> scene;
	std::thread* pixThreads;
	bool* pixThreadsDone;
	bool m_bBufferReady;

};

#endif // __RAYTRACER_H__
