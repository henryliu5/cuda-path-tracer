# CS378H Final Project - Path Tracing in CUDA
## Henry Liu and Soham Patel

To build this project,

`mkdir build`
`cd build`
`cmake ..`
`make -j`

And an executable will be made in ./build/bin/ray

usage: ./bin/ray [options] [input.ray output.png]
  -r <#>      set recursion level (default 0)
  -w <#>      set output image width (default 512)
  -j <FILE>   set parameters from JSON file
  -c <FILE>   one Cubemap file, the remainings will be detected automatically
  -g          Enable CUDA rendering
  -s <#>      Set number of samples per pixel (path tracing)

Note that the GPU path tracer only supports .ray scenes consisting of only polymeshs.
To emulate area lights set the ke constant of a polymesh.

## Features
Our path tracer implements
- Monte Carlo estimation of rendering equation
- Diffuse, perfectly specular, dielectric/conductor BRDFs
- Importance sampling of cosine weighted hemisphere
- Russian roulette termination
- Depth of field
- CUDA iterative BVH traversal
- CUDA iterative path tracing
- 2 CUDA kernel implementations
  - CUDA work queue/coalescing


We have included scenes and example renders in important_assets
  ## special_scenes 
  This folder includes more interesting scenes with significantly more faces (such as the dragon scenes) as well as caustic scenes (such as the teapot scenes). The naming convention in this folder contains information about if it is diffuse (no indication), specular (contains a '-spec-'), or transmissive ('-trans-') as well as whether it was setup for ray-tracer rendering (contains a '-ray'). For example, the scene 'teapot-trans.ray' is the teapot scene with a transmissive constant meant for the path tracer. 
  ## scenes
  This folder contains the standard cornell-box2 scenes that are either diffuse, specular (contains a '-specular'), or transmissive (contains a 'transmissive') as well as whether it was setup for ray-tracer rendering (contains a '-ray'). These were used mostly to affirm that our path tracer was working (which was affirmed through evidence of global illumination, soft shadows, etc.) as well as to get benchmarks on our implementations. These timings are in the timings folder along with the script used to get the timings. Both the CPU and GPU have two timing txt files, one without BVH and one with BVH (file name contains a '_bvh'). Each line in all 4 of these files be setup as follows:
  {file_name} - {Implementation}|{SPP},{Depth}: {End-to-End Time}
 Where file_name is the target scene, Implementation is either CPU/GPU, SPP is the samples per pixel, Depth is the number of bounces (or recursion depth), and End-to-End Time is the total time for the scene to render. You can find these benchmark renderings under ./Renderings/benchmark_renderings/ where there are two folders for the bvh or no bvh rendering. The image files can be interpreted as follows:
  {file_name}_{Depth}_{SPP}_{Implementation}.png
Note that the file_name does include the .ray extension and the scene is still intended for path tracing.
  ## ray-traced
  This contains the ray-traced versions of the scenes to have a side by side comparision of the ray tracer and the path tracer.
  ## Miscellaneous
  Other renderings in the folder include a 'dragon-split' image where the floor on which the dragon rests on is two colors, allowing us to see global illumination as well as soft shadows. The other two that are important are the cornell-box-DOF image (which displays our path tracing depth of field implementation) and the cornell-box-16384 image (which displays our path tracing taking 16384 samples per pixel and rendered in approximately 4 minutes).
  
  
  
  
  
We affirm that we have completed the ECIS for this course.
