# SPH Fluid Simulation

A real-time **Smoothed Particle Hydrodynamics (SPH)** fluid simulation developed progressively from a CPU-based 2D implementation to a GPU-accelerated 3D simulation with **ray-marched surface rendering**.

The project was built as a hands-on implementation and study of SPH-based fluid simulation, drawing from published research, technical resources, and Sebastian Lague's fluid simulation project. The main goal was to understand both the **physics simulation** and the **GPU techniques required to scale it to large numbers of particles**.

## Demo

<p align="center">
  <img src="Content/README/demo.gif" width="700"> 

https://github.com/user-attachments/assets/733e5c4c-5698-4525-a501-34083e9062d8


</p>

---

# Project Evolution

The project was developed in four major stages:

```text
CPU 2D
  ↓
GPU 2D
  ↓
GPU 3D
  ↓
GPU 3D + Surface Rendering
```

Each stage introduced a different challenge, progressing from understanding the SPH equations to parallelizing the simulation and finally reconstructing a continuous fluid surface from particles.

---

# 1. CPU 2D SPH

The project began with a **2D CPU implementation** of Smoothed Particle Hydrodynamics.

Instead of representing the fluid as a continuous volume, SPH represents it as a collection of particles. Each particle stores properties such as position, velocity, density, and pressure.

The density of each particle is estimated by evaluating the contribution of neighboring particles through a smoothing kernel.

### Simulation Pipeline

For each simulation step:

1. Find neighboring particles.
2. Calculate particle density.
3. Calculate pressure from density.
4. Calculate pressure forces.
5. Calculate viscosity forces.
6. Apply gravity and other forces.
7. Integrate velocity and position.
8. Resolve boundary collisions.

The resulting particle positions are then rendered to visualize the fluid.
<img width="1206" height="680" alt="CPU 2D" src="https://github.com/user-attachments/assets/3f243d7c-4997-404d-b328-5a30e15e19b3" />

---

# 2. GPU 2D SPH

The next stage was to move the computationally expensive SPH simulation from the CPU to the GPU using **Unity Compute Shaders**.

SPH is highly parallelizable because many particle calculations can be performed independently.

Instead of processing particles sequentially on the CPU:

```text
CPU

Particle 1 → Calculate
Particle 2 → Calculate
Particle 3 → Calculate
...
```

the GPU version distributes particles across many compute shader threads:

```text
GPU

Thread 1 → Particle 1
Thread 2 → Particle 2
Thread 3 → Particle 3
Thread 4 → Particle 4
...
```

### Compute Shader Stages

The GPU simulation separates the major parts of the SPH calculation into compute shader stages:

```text
Particle Data
     ↓
Density
     ↓
Pressure
     ↓
Forces
     ↓
Integration
     ↓
Updated Particles
```

This stage demonstrated the performance advantage of moving the particle simulation onto the GPU and provided the foundation for scaling the simulation to substantially larger particle counts.

https://github.com/user-attachments/assets/f69410ec-ad89-4013-b977-e859e0cdbfa0

---

# 3. GPU 3D SPH

After validating the GPU implementation in 2D, the simulation was extended into **three dimensions**.

The fundamental SPH formulation remains the same, but the dimensionality introduces additional computational and implementation challenges:

* 3D particle neighborhoods
* 3D boundary constraints
* Increased number of neighboring particles
* Larger simulation domains
* Increased memory requirements
* More expensive neighbor searches

The particle state is now represented in 3D:

```text
Position → (x, y, z)
Velocity → (vx, vy, vz)
```

The same density, pressure, viscosity, and integration stages operate over the 3D particle neighborhood.



https://github.com/user-attachments/assets/b9f8bdc8-ed3b-4750-b03c-ed818005b0c0



---

# 4. Surface Rendering

Particle-based simulation produces a collection of discrete particles, but real fluids appear as a **continuous surface**.

Simply rendering the particles as spheres does not produce a convincing fluid appearance.

To address this, the project introduces a **ray-marched surface representation**.


### From Particles to Surface

The particle positions are used to construct an implicit representation of the fluid.

During rendering, rays are cast through the scene and sampled against the fluid representation to determine where the ray intersects the surface.

The surface normal can then be estimated from the local density field and used for lighting.


```text
SPH Particles
      ↓
Density Field
      ↓
Implicit Fluid Surface
      ↓
Ray Marching
      ↓
Surface Intersection
      ↓
Normal Estimation
      ↓
Lighting
      ↓
Final Fluid Image
```

This separates the project into two distinct systems:

**Simulation**

> Compute the physical state of the particles.

**Rendering**

> Reconstruct and shade a continuous surface from that particle state.

This separation allows the same particle simulation to be visualized using different rendering techniques.

---

# SPH Simulation

The simulation is based on the idea of approximating fluid quantities from neighboring particles using smoothing kernels.

For a particle \(i\), density is estimated from its neighboring particles:

$$
\rho_i = \sum_j m_j W(\mathbf{r}_i-\mathbf{r}_j,h)
$$

where:

* \(\rho_i\) is the density of particle \(i\)
* \(m_j\) is the mass of neighboring particle \(j\)
* \(W\) is the smoothing kernel
* \(h\) is the smoothing radius

Pressure is then derived from the density error relative to the target rest density, and pressure and viscosity forces are used to update particle motion.

The implementation follows the standard SPH formulation described in the referenced literature and educational material.

---

# Technologies

**C# · Unity · Compute Shaders · HLSL · SPH · GPU Programming · Ray Marching**

---

# Project Structure

```text
Fluid-Sim/
│
├── Assets/
│   ├── Scripts/
│   │   ├── Simulation/
│   │   ├── Rendering/
│   │   └── ...
│   │
│   ├── Compute/
│   │   ├── SPH
│   │   └── Ray Marching
│   │
│   ├── Shaders/
│   └── ...
│
└── ProjectSettings/
```

---

# Setup

### Requirements

* Unity
* A GPU supporting compute shaders
* Windows recommended

### Running

Clone the repository:

```bash
git clone https://github.com/ChiranjivMalhi/Fluid-Sim.git
```

Open the project using Unity and run the relevant scene.

---

# Future Improvements

### Simulation

* **Spatial Hashing / Uniform Grid** — Improve neighbor search efficiency and allow significantly larger particle counts.
* **Adaptive Time Stepping** — Adjust the simulation timestep based on particle velocity and stability constraints.
* **Improved Boundary Handling** — Develop more robust particle-boundary interaction methods.
* **Surface Tension** — Add surface tension forces for more realistic free-surface behavior.
* **Viscosity Models** — Experiment with more physically accurate viscosity formulations.
* **Fluid-Object Interaction** — Couple the fluid simulation with rigid bodies and moving obstacles.

### GPU Performance

* **Neighbor Search Optimization** — Reduce the cost of particle neighborhood queries.
* **Memory Optimization** — Improve GPU memory layout and access patterns.
* **Compute Shader Optimization** — Investigate thread-group sizing, occupancy, and memory bandwidth.
* **GPU Profiling** — Use GPU profiling tools to identify the dominant simulation and rendering bottlenecks.

### Rendering

* **Screen-Space Fluid Rendering** — Explore screen-space techniques for faster fluid surface reconstruction.
* **Refraction** — Add physically motivated refraction through the fluid surface.
* **Absorption** — Simulate wavelength-dependent attenuation through the fluid.
* **Fresnel Effects** — Improve the interaction between reflection and refraction at grazing angles.
* **Foam and Spray** — Generate secondary particles for splashes and turbulent regions.
* **Temporal Filtering** — Reduce ray-marching noise and improve temporal stability.
* **Higher-Quality Ray Marching** — Improve surface reconstruction and reduce the number of samples required.

---

# References

This project was developed as an implementation and learning exercise based on SPH research, technical resources, and educational implementations.

### SPH

**Müller, Charypar & Gross — Particle-Based Fluid Simulation for Interactive Applications (2003)**
https://matthias-research.github.io/pages/publications/sca03.pdf

This work provides the foundation for the particle-based fluid formulation used by the project.

### Sebastian Lague — Coding Adventure: Simulating Fluids

https://www.youtube.com/watch?v=rSKMYc1CQHE

The project was heavily influenced by Sebastian Lague's educational implementation and development process. His implementation was particularly useful for understanding how the SPH equations can be translated into a practical real-time simulation.

His project also provides references to the underlying SPH literature and GPU particle simulation resources.

### NVIDIA — GPU Particle Simulation

https://docs.nvidia.com/cuda/samples/5_simulations/particles/doc/particles.pdf

Used as a reference for GPU-based particle simulation techniques.

---

# Author

**Chiranjiv Malhi**

Computer Graphics · GPU Programming · Fluid Simulation

https://github.com/ChiranjivMalhi
