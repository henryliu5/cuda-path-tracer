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

We have included scenes and example renders in important_assets