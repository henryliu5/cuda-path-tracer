# CS378H Ray Tracer Milestone 1
## Henry Liu and Soham Patel

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
