# CS378H Ray Tracer Milestone 2
## Henry Liu and Soham Patel

We believe we have implemented all required features for Milestone 2 correctly, including
- Bounding Volume Hierarchy Tree
- Texture and Cube Mapping
- Depth of Field through Distributed/Stochastic Ray Tracing (Extra Credit: 20)
- Jittered Supersampling (Extra Credit: 5)
- Multithreading (Extra Credit?)

We have moved our anti-aliasing code from aaImage() to tracePixelAA().

We have moved our Whitted Illumination Model code from shade() into the Light Class.

Depth of Field Explanation:

Our depth of field implementation is located in the tracePixelDOF() function. We have three main parameters that can be set: the focal distance, the apeture, and the number of samples. To simulate the depth of field effect, we implemented distributed ray tracing. We first shot a ray through the camera's center to acquire a focal point (determined using the focal distance), and then randomly jittered the rays (by adjusting their origin points across the plane of our "square" lens) for a fixed number of samples. Finally, before "writing" these colors to the buffer, we perform a box filter on the color's sum.

To see our depth of field implementation, we have added two examples: easy3 and reflection2. The easy3_no_dof and reflection2_no_dof are the resulting renders without distributed ray tracing for depth of field, and the easy3_dof and reflection2_dof are the resulting renders with distributed ray tracing. The parameters for both depth of field renders are:

easy3: Focal Distance = 8.5, Aperture = 0.2
reflection2: Focal Distance = 6, Aperture = 0.3

For both renders we set the number of samples to be 128.

Jittered Supersampling:

For each subpixel, we randomly jitter the ray within the subpixel as opposed to a fixed pattern (like anti-aliasing) for a fixed number of samples. Once again, before writing the ray's color contribution to the buffer, we perform a box filter on the color's sum.


Multithreading:

For multithreading, we spawn n number of threads and stride the buffer to ensure each thread receives roughly the same number of pixels to trace. 

# Milestone 1 Features Below

We believe we have implemented all required features for Milestone 1 correctly, including
- Triangle-Ray intersection
- Phong Interpolation of Normals
- Whitted Illumination Model
    - Phong shading (emissive, ambient, diffuse, specular)
    - Light and shadow attenuation
    - Reflection and refraction
- Anti-aliasing

Our implementation differs from the reference solution in how it handles overlapping translucent objects. In particular we make the assumption that all objects do not intersect and are separated from each other by air. As a result we toggle the the refractive index between the value of air and the value of the object which it enters next. We store the "current" refractive index in the ray and check this to see whether the refractive index of the next intersection or the refractive index of air should be used for the next refraction. This leads to differences in scenes such as sphere_box and easy3a (minor) where objects intersect.

We chose to have reflection rays reflect off the back faces of objects with both a reflective and refractive material, and we correctly flip the normals for this case. 

We do not implement backface culling.

We implement total internal reflection by treating the ray as a reflective ray. i.e. we do apply the reflective constant, kr, to the ray's intensity.

Our code does not use the shadowAttenuation methods in light.cpp and instead directly performs the shadow attenuation in shade().
