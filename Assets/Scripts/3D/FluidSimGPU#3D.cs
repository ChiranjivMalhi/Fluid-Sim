using System.Collections.Generic;
using UnityEngine;

public class FluidSimGPU3D : MonoBehaviour
{
    [Header("Compute")]
    [SerializeField] private ComputeShader simulationCompute;
    [SerializeField] private Material particleMaterial;

    [Header("Scene Setup")]
    [SerializeField] private Vector3 simulationBounds = new Vector3(20f, 12f, 10f);

    [Header("Particle Grid")]
    [SerializeField, Min(1)] private int particlesX = 10;
    [SerializeField, Min(1)] private int particlesY = 10;
    [SerializeField, Min(1)] private int particlesZ = 10;
    [SerializeField, Min(0.01f)] private float particleSpacing = 0.15f;
    [SerializeField, Min(0.001f)] private float particleRadius = 0.05f;
    [SerializeField, Tooltip("Starting center of the particle grid relative to the simulation bounds center.")]
    private Vector3 spawnCenter;
    [SerializeField, Min(3)] private int sphereLongitudeSegments = 8;
    [SerializeField, Min(2)] private int sphereLatitudeSegments = 6;

    [Header("Boundary Particles")]
    [SerializeField, Min(0.1f), Tooltip("Height of the static side walls above the floor.")]
    private float boundaryWallHeight = 6f;

    [Header("Physics")]
    [SerializeField] private float gravity = 9.81f;
    [SerializeField, Range(0f, 1f)] private float dampingFactor = 0.9f;

    [Header("Simulation Time")]
    [SerializeField, Min(0.001f)] private float simulationTimeStep = 1f / 120f;
    [SerializeField, Min(1)] private int maxSubstepsPerFrame = 8;

    [Header("Density")]
    [SerializeField, Min(0.01f)] private float smoothingRadius = 0.45f;
    [SerializeField, Min(0.01f), Tooltip("Recomputed at startup from the particle spacing and smoothing radius.")]
    private float restDensity = 4.0f;
    [SerializeField, Min(0.01f)] private float densityScale = 1.5f;

    [Header("Pressure")]
    [SerializeField] private float pressureMultiplier = 50f;
    [SerializeField, Min(0f), Tooltip("Caps the magnitude of attractive pressure below rest density. Set to zero to disable attraction.")]
    private float maxNegativePressure;
    [SerializeField] private float nearPressureMultiplier = 50f;

    [Header("Viscosity")]
    [SerializeField] private float viscosityMultiplier = 0.1f;

    [Header("Surface Tension")]
    [SerializeField] private float surfaceTensionCoefficient = 0.0f;
    [SerializeField] private float surfaceTensionThreshold = 0.05f;

    [Header("Debug Rendering")]
    [SerializeField] private bool drawParticleSpheres = true;

    private int particleCount;
    private int fluidCount;
    private int kernelIndex;
    private int densityKernelIndex;
    private int pressureKernelIndex;
    private int nearDensityKernelIndex;
    private int nearPressureKernelIndex;
    private int viscosityKernelIndex;
    private int surfaceFieldKernelIndex;
    private int surfaceTensionKernelIndex;

    private int UpdateVelocitiesKernelIndex;
    private int UpdatePositionsKernelIndex;

    private const int SortGroupSize = 128;
    private int clearBucketCountsKernel;
    private int countParticlesKernel;
    private int scanBucketCountsKernel;
    private int addBlockOffsetsKernel;
    private int scatterParticlesKernel;
    private float accumulatedTime;

    // Spatial Hashing
    private ComputeBuffer spatialKeysBuffer;
    private ComputeBuffer spatialIndicesBuffer;
    private ComputeBuffer spatialOffsetsBuffer;
    private ComputeBuffer particleKeysBuffer;
    private ComputeBuffer bucketCountsBuffer;
    private readonly List<ComputeBuffer> scanOutputs = new();
    private readonly List<ComputeBuffer> scanBlockSums = new();

    // ComputeBuffers
    private ComputeBuffer positionsBuffer;
    private ComputeBuffer velocitiesBuffer;
    private ComputeBuffer argsBuffer;
    private ComputeBuffer predictedPositionsBuffer;

    // Density
    private ComputeBuffer densitiesBuffer;
    private ComputeBuffer nearDensitiesBuffer;

    // Pressure
    private ComputeBuffer pressureBuffer;
    private ComputeBuffer pressureForcesBuffer;
    private ComputeBuffer nearPressureAccelerationsBuffer;

    // Viscosity
    private ComputeBuffer viscosityBuffer;

    // Surface Tension
    private ComputeBuffer colorFieldNormalsBuffer;
    private ComputeBuffer colorFieldLaplaciansBuffer;
    private ComputeBuffer surfaceTensionBuffer;
    private Mesh sphereMesh;
    private Bounds drawBounds;

    private static readonly int PositionsID = Shader.PropertyToID("_Positions");
    private static readonly int DensitiesID = Shader.PropertyToID("_Densities");
    private static readonly int VelocitiesID = Shader.PropertyToID("_Velocities");
    private static readonly int SimulationCenterID = Shader.PropertyToID("_SimulationCenter");

    // Public properties
    public ComputeBuffer PositionsBuffer => positionsBuffer;
    public ComputeBuffer PredictedPositionsBuffer => predictedPositionsBuffer;
    public ComputeBuffer SpatialKeysBuffer => spatialKeysBuffer;
    public ComputeBuffer SpatialIndicesBuffer => spatialIndicesBuffer;
    public ComputeBuffer SpatialOffsetsBuffer => spatialOffsetsBuffer;
    public int ParticleCount => particleCount;
    public int FluidCount => fluidCount;
    public Vector3 SimulationBounds => simulationBounds;
    public float SmoothingRadius => smoothingRadius;
    public Vector3 SimulationCenter => transform.position;

    private void BuildSpatialLookup()
    {
        int groups = SortGroupCount(particleCount);
        simulationCompute.SetInt("_ParticleCount", particleCount);
        simulationCompute.SetFloat("_SmoothingRadius", smoothingRadius);
        simulationCompute.SetBuffer(clearBucketCountsKernel, "_BucketCounts", bucketCountsBuffer);
        simulationCompute.Dispatch(clearBucketCountsKernel, groups, 1, 1);

        simulationCompute.SetBuffer(countParticlesKernel, "_PredictedPositions", predictedPositionsBuffer);
        simulationCompute.SetBuffer(countParticlesKernel, "_ParticleKeys", particleKeysBuffer);
        simulationCompute.SetBuffer(countParticlesKernel, "_BucketCounts", bucketCountsBuffer);
        simulationCompute.Dispatch(countParticlesKernel, groups, 1, 1);

        ComputeBuffer input = bucketCountsBuffer;
        for (int level = 0; level < scanOutputs.Count; level++)
        {
            ComputeBuffer sums = scanBlockSums[level];
            simulationCompute.SetInt("_ScanCount", input.count);
            simulationCompute.SetBuffer(scanBucketCountsKernel, "_ScanInput", input);
            simulationCompute.SetBuffer(scanBucketCountsKernel, "_ScanOutput", scanOutputs[level]);
            simulationCompute.SetBuffer(scanBucketCountsKernel, "_BlockSums", sums);
            simulationCompute.Dispatch(scanBucketCountsKernel, sums.count, 1, 1);
            input = sums;
        }

        for (int level = scanOutputs.Count - 2; level >= 0; level--)
        {
            simulationCompute.SetInt("_ScanCount", scanOutputs[level].count);
            simulationCompute.SetBuffer(addBlockOffsetsKernel, "_ScanOutput", scanOutputs[level]);
            simulationCompute.SetBuffer(addBlockOffsetsKernel, "_BlockOffsets", scanOutputs[level + 1]);
            simulationCompute.Dispatch(addBlockOffsetsKernel, scanBlockSums[level].count, 1, 1);
        }

        simulationCompute.SetBuffer(scatterParticlesKernel, "_ParticleKeys", particleKeysBuffer);
        simulationCompute.SetBuffer(scatterParticlesKernel, "_BucketCounts", bucketCountsBuffer);
        simulationCompute.SetBuffer(scatterParticlesKernel, "_SpatialOffsets", spatialOffsetsBuffer);
        simulationCompute.SetBuffer(scatterParticlesKernel, "_SortKeys", spatialKeysBuffer);
        simulationCompute.SetBuffer(scatterParticlesKernel, "_SortIndices", spatialIndicesBuffer);
        simulationCompute.Dispatch(scatterParticlesKernel, groups, 1, 1);
    }

    private static int SortGroupCount(int count) => (count - 1) / SortGroupSize + 1;

    private void Awake()
    {
        restDensity = ComputeLatticeRestDensity();
        fluidCount = particlesX * particlesY * particlesZ;
        List<Vector3> boundary = BuildBoundaryParticles();
        particleCount = fluidCount + boundary.Count;
        Debug.Log($"Fluid particles: {fluidCount}, boundary particles: {boundary.Count}, rest density: {restDensity:F2}", this);

        kernelIndex = simulationCompute.FindKernel("ApplyForcesIntegrateCollide");
        densityKernelIndex = simulationCompute.FindKernel("CalculateDensity");
        pressureKernelIndex = simulationCompute.FindKernel("CalculatePressure");
        nearDensityKernelIndex = simulationCompute.FindKernel("CalculateNearDensities");
        nearPressureKernelIndex = simulationCompute.FindKernel("CalculateNearPressure");
        viscosityKernelIndex = simulationCompute.FindKernel("CalculateViscosity");
        surfaceFieldKernelIndex = simulationCompute.FindKernel("CalculateColorField");
        surfaceTensionKernelIndex = simulationCompute.FindKernel("CalculateSurfaceTension");
        UpdateVelocitiesKernelIndex = simulationCompute.FindKernel("UpdateVelocities");
        UpdatePositionsKernelIndex = simulationCompute.FindKernel("UpdatePositions");
        clearBucketCountsKernel = simulationCompute.FindKernel("ClearBucketCounts");
        countParticlesKernel = simulationCompute.FindKernel("CountParticles");
        scanBucketCountsKernel = simulationCompute.FindKernel("ScanBucketCounts");
        addBlockOffsetsKernel = simulationCompute.FindKernel("AddBlockOffsets");
        scatterParticlesKernel = simulationCompute.FindKernel("ScatterParticles");

        CreateBuffers();
        SpawnParticles(boundary);
        CreateSphereMesh();
        SetupArgsBuffer();

        if (particleMaterial != null)
        {
            particleMaterial.SetBuffer(PositionsID, positionsBuffer);
            particleMaterial.SetBuffer(DensitiesID, densitiesBuffer);
            particleMaterial.SetBuffer(VelocitiesID, velocitiesBuffer);
            particleMaterial.SetVector(SimulationCenterID, transform.position);
        }

        drawBounds = new Bounds(transform.position,
            new Vector3(simulationBounds.x + 10f, simulationBounds.y + 10f, simulationBounds.z + 10f));

        // The renderer can run before the first simulation substep.
        BuildSpatialLookup();
    }

    private void CreateBuffers()
    {
        positionsBuffer = new ComputeBuffer(particleCount, sizeof(float) * 3);
        velocitiesBuffer = new ComputeBuffer(particleCount, sizeof(float) * 3);
        predictedPositionsBuffer = new ComputeBuffer(particleCount, sizeof(float) * 3);
        densitiesBuffer = new ComputeBuffer(particleCount, sizeof(float));
        nearDensitiesBuffer = new ComputeBuffer(particleCount, sizeof(float));
        pressureBuffer = new ComputeBuffer(particleCount, sizeof(float));
        pressureForcesBuffer = new ComputeBuffer(particleCount, sizeof(float) * 3);
        nearPressureAccelerationsBuffer = new ComputeBuffer(particleCount, sizeof(float) * 3);
        viscosityBuffer = new ComputeBuffer(particleCount, sizeof(float) * 3);
        surfaceTensionBuffer = new ComputeBuffer(particleCount, sizeof(float) * 3);
        colorFieldNormalsBuffer = new ComputeBuffer(particleCount, sizeof(float) * 3);
        colorFieldLaplaciansBuffer = new ComputeBuffer(particleCount, sizeof(float));

        spatialKeysBuffer = new ComputeBuffer(particleCount, sizeof(uint));
        spatialIndicesBuffer = new ComputeBuffer(particleCount, sizeof(uint));
        spatialOffsetsBuffer = new ComputeBuffer(particleCount, sizeof(uint));
        particleKeysBuffer = new ComputeBuffer(particleCount, sizeof(uint));
        bucketCountsBuffer = new ComputeBuffer(particleCount, sizeof(uint));

        scanOutputs.Add(spatialOffsetsBuffer);
        int blocks = SortGroupCount(particleCount);
        scanBlockSums.Add(new ComputeBuffer(blocks, sizeof(uint)));
        while (blocks > 1)
        {
            scanOutputs.Add(new ComputeBuffer(blocks, sizeof(uint)));
            blocks = SortGroupCount(blocks);
            scanBlockSums.Add(new ComputeBuffer(blocks, sizeof(uint)));
        }
    }

    private void SpawnParticles(List<Vector3> boundary)
    {
        Vector3[] positions = new Vector3[particleCount];
        Vector3[] velocities = new Vector3[particleCount];

        float gridWidth = (particlesX - 1) * particleSpacing;
        float gridHeight = (particlesY - 1) * particleSpacing;
        float gridDepth = (particlesZ - 1) * particleSpacing;
        Vector3 gridHalfSize = new Vector3(gridWidth, gridHeight, gridDepth) * 0.5f;
        Vector3 centerLimit = Vector3.Max(Vector3.zero,
            simulationBounds * 0.5f - gridHalfSize - Vector3.one * particleRadius);
        Vector3 center = new Vector3(
            Mathf.Clamp(spawnCenter.x, -centerLimit.x, centerLimit.x),
            Mathf.Clamp(spawnCenter.y, -centerLimit.y, centerLimit.y),
            Mathf.Clamp(spawnCenter.z, -centerLimit.z, centerLimit.z));
        Vector3 gridMin = center - gridHalfSize;

        int index = 0;
        for (int z = 0; z < particlesZ; z++)
        {
            for (int y = 0; y < particlesY; y++)
            {
                for (int x = 0; x < particlesX; x++)
                {
                    positions[index] = gridMin + new Vector3(x * particleSpacing, y * particleSpacing, z * particleSpacing);
                    index++;
                }
            }
        }
        boundary.CopyTo(positions, fluidCount);
        positionsBuffer.SetData(positions);
        predictedPositionsBuffer.SetData(positions);
        velocitiesBuffer.SetData(velocities);
    }

    private float ComputeLatticeRestDensity()
    {
        int cells = Mathf.CeilToInt(smoothingRadius / particleSpacing);
        float radiusSq = smoothingRadius * smoothingRadius;
        float spacingSq = particleSpacing * particleSpacing;
        float sum = 0f;

        for (int x = -cells; x <= cells; x++)
        for (int y = -cells; y <= cells; y++)
        for (int z = -cells; z <= cells; z++)
        {
            float distanceSq = spacingSq * (x * x + y * y + z * z);
            if (distanceSq < radiusSq)
            {
                float weight = smoothingRadius - Mathf.Sqrt(distanceSq);
                sum += weight * weight;
            }
        }

        return sum * (15f / (2f * Mathf.PI * Mathf.Pow(smoothingRadius, 5f)));
    }

    private List<Vector3> BuildBoundaryParticles()
    {
        int layers = Mathf.CeilToInt(smoothingRadius / particleSpacing);
        Vector3 half = simulationBounds * 0.5f;
        float wallTop = Mathf.Min(half.y, -half.y + boundaryWallHeight);
        var xs = BuildBoundaryAxis(-half.x, half.x, particleSpacing, layers, true);
        var ys = BuildBoundaryAxis(-half.y, wallTop, particleSpacing, layers, false);
        var zs = BuildBoundaryAxis(-half.z, half.z, particleSpacing, layers, true);

        var boundary = new List<Vector3>();
        foreach (var x in xs)
        foreach (var y in ys)
        foreach (var z in zs)
        {
            if (x.wall || y.wall || z.wall)
                boundary.Add(new Vector3(x.position, y.position, z.position));
        }

        return boundary;
    }

    private static List<(float position, bool wall)> BuildBoundaryAxis(
        float min, float max, float spacing, int layers, bool upperWall)
    {
        // The innermost wall samples are half a spacing outside the collision planes.
        float first = min - 0.5f * spacing;
        float last = upperWall ? max + 0.5f * spacing : max;
        var axis = new List<(float position, bool wall)>();

        for (int layer = layers - 1; layer >= 0; layer--)
            axis.Add((first - layer * spacing, true));

        int steps = Mathf.Max(1, Mathf.RoundToInt((last - first) / spacing));
        for (int step = 1; step < steps; step++)
            axis.Add((Mathf.Lerp(first, last, (float)step / steps), false));

        axis.Add((last, upperWall));
        for (int layer = 1; upperWall && layer < layers; layer++)
            axis.Add((last + layer * spacing, true));

        return axis;
    }

    private void CreateSphereMesh()
    {
        sphereMesh = new Mesh();

        float r = particleRadius;
        int longSegments = sphereLongitudeSegments;
        int latSegments = sphereLatitudeSegments;

        var vertices = new System.Collections.Generic.List<Vector3>();
        var normals = new System.Collections.Generic.List<Vector3>();
        var uvs = new System.Collections.Generic.List<Vector2>();
        var triangles = new System.Collections.Generic.List<int>();

        vertices.Add(new Vector3(0, r, 0));
        normals.Add(Vector3.up);
        uvs.Add(new Vector2(0.5f, 1f));

        for (int lat = 1; lat < latSegments; lat++)
        {
            float theta = Mathf.PI * lat / latSegments;
            float sinTheta = Mathf.Sin(theta);
            float cosTheta = Mathf.Cos(theta);

            for (int lon = 0; lon <= longSegments; lon++)
            {
                float phi = 2f * Mathf.PI * lon / longSegments;
                float sinPhi = Mathf.Sin(phi);
                float cosPhi = Mathf.Cos(phi);

                Vector3 normal = new Vector3(sinTheta * cosPhi, cosTheta, sinTheta * sinPhi);
                vertices.Add(normal * r);
                normals.Add(normal);
                uvs.Add(new Vector2((float)lon / longSegments, 1f - (float)lat / latSegments));
            }
        }

        vertices.Add(new Vector3(0, -r, 0));
        normals.Add(Vector3.down);
        uvs.Add(new Vector2(0.5f, 0f));

        int bottomPoleIndex = vertices.Count - 1;
        int ringVertCount = longSegments + 1;

        for (int lon = 0; lon < longSegments; lon++)
        {
            triangles.Add(0);
            triangles.Add(lon + 1);
            triangles.Add(lon + 2);
        }

        for (int lat = 0; lat < latSegments - 2; lat++)
        {
            int ringStart = 1 + lat * ringVertCount;
            int nextRingStart = ringStart + ringVertCount;

            for (int lon = 0; lon < longSegments; lon++)
            {
                int a = ringStart + lon;
                int b = nextRingStart + lon;

                triangles.Add(a);
                triangles.Add(b);
                triangles.Add(a + 1);

                triangles.Add(a + 1);
                triangles.Add(b);
                triangles.Add(b + 1);
            }
        }
        int lastRingStart = 1 + (latSegments - 2) * ringVertCount;
        for (int lon = 0; lon < longSegments; lon++)
        {
            triangles.Add(bottomPoleIndex);
            triangles.Add(lastRingStart + lon + 1);
            triangles.Add(lastRingStart + lon);
        }

        sphereMesh.SetVertices(vertices);
        sphereMesh.SetNormals(normals);
        sphereMesh.SetUVs(0, uvs);
        sphereMesh.SetTriangles(triangles, 0);
        sphereMesh.RecalculateBounds();
    }

    private void SetupArgsBuffer()
    {
        uint[] args = new uint[5];
        args[0] = sphereMesh.GetIndexCount(0);
        args[1] = (uint)fluidCount;
        args[2] = sphereMesh.GetIndexStart(0);
        args[3] = sphereMesh.GetBaseVertex(0);
        args[4] = 0;

        argsBuffer = new ComputeBuffer(1, args.Length * sizeof(uint), ComputeBufferType.IndirectArguments);
        argsBuffer.SetData(args);
    }

    private void Update()
    {
        accumulatedTime += Time.deltaTime;

        int substeps = 0;
        while (accumulatedTime >= simulationTimeStep && substeps < maxSubstepsPerFrame)
        {
            DispatchSimulationStep(simulationTimeStep);
            accumulatedTime -= simulationTimeStep;
            substeps++;
        }

        if (substeps == maxSubstepsPerFrame)
        {
            accumulatedTime = Mathf.Min(accumulatedTime, simulationTimeStep);
        }

        if (drawParticleSpheres && particleMaterial != null)
        {
            drawBounds.center = transform.position;
            particleMaterial.SetVector(SimulationCenterID, transform.position);
            Graphics.DrawMeshInstancedIndirect(sphereMesh, 0, particleMaterial, drawBounds, argsBuffer);
        }
    }

    private void DispatchSimulationStep(float deltaTime)
    {
        int groups = SortGroupCount(fluidCount);

        simulationCompute.SetFloat("_DeltaTime", deltaTime);
        simulationCompute.SetFloat("_Gravity", Mathf.Abs(gravity));
        simulationCompute.SetFloat("_DampingFactor", dampingFactor);
        simulationCompute.SetFloat("_ParticleRadius", particleRadius);
        simulationCompute.SetVector("_SimulationBounds", simulationBounds);
        simulationCompute.SetInt("_FluidCount", fluidCount);
        simulationCompute.SetFloat("_SmoothingRadius", smoothingRadius);
        simulationCompute.SetFloat("_RestDensity", restDensity);
        simulationCompute.SetFloat("_PressureMultiplier", pressureMultiplier);
        simulationCompute.SetFloat("_MaxNegativePressure", Mathf.Max(0f, maxNegativePressure));
        simulationCompute.SetFloat("_NearPressureMultiplier", nearPressureMultiplier);
        simulationCompute.SetFloat("_ViscosityCoefficient", viscosityMultiplier);
        simulationCompute.SetFloat("_SurfaceTensionCoefficient", surfaceTensionCoefficient);
        simulationCompute.SetFloat("_SurfaceTensionThreshold", surfaceTensionThreshold);

        simulationCompute.SetBuffer(kernelIndex, "_Positions", positionsBuffer);
        simulationCompute.SetBuffer(kernelIndex, "_Velocities", velocitiesBuffer);
        simulationCompute.SetBuffer(kernelIndex, "_PredictedPositions", predictedPositionsBuffer);
        simulationCompute.Dispatch(kernelIndex, groups, 1, 1);

        BuildSpatialLookup();

        simulationCompute.SetBuffer(densityKernelIndex, "_PredictedPositions", predictedPositionsBuffer);
        simulationCompute.SetBuffer(densityKernelIndex, "_Densities", densitiesBuffer);
        simulationCompute.SetBuffer(densityKernelIndex, "_NearDensities", nearDensitiesBuffer);
        simulationCompute.SetBuffer(densityKernelIndex, "_SortKeys", spatialKeysBuffer);
        simulationCompute.SetBuffer(densityKernelIndex, "_SortIndices", spatialIndicesBuffer);
        simulationCompute.SetBuffer(densityKernelIndex, "_SpatialOffsets", spatialOffsetsBuffer);
        simulationCompute.Dispatch(densityKernelIndex, groups, 1, 1);

        simulationCompute.SetBuffer(pressureKernelIndex, "_PredictedPositions", predictedPositionsBuffer);
        simulationCompute.SetBuffer(pressureKernelIndex, "_Velocities", velocitiesBuffer);
        simulationCompute.SetBuffer(pressureKernelIndex, "_Densities", densitiesBuffer);
        simulationCompute.SetBuffer(pressureKernelIndex, "_NearDensities", nearDensitiesBuffer);
        simulationCompute.SetBuffer(pressureKernelIndex, "_Pressure", pressureBuffer);
        simulationCompute.SetBuffer(pressureKernelIndex, "_PressureForces", pressureForcesBuffer);
        simulationCompute.SetBuffer(pressureKernelIndex, "_NearPressureAccelerations", nearPressureAccelerationsBuffer);
        simulationCompute.SetBuffer(pressureKernelIndex, "_ViscosityForces", viscosityBuffer);
        simulationCompute.SetBuffer(pressureKernelIndex, "_SortKeys", spatialKeysBuffer);
        simulationCompute.SetBuffer(pressureKernelIndex, "_SortIndices", spatialIndicesBuffer);
        simulationCompute.SetBuffer(pressureKernelIndex, "_SpatialOffsets", spatialOffsetsBuffer);
        simulationCompute.Dispatch(pressureKernelIndex, groups, 1, 1);

        if (surfaceTensionCoefficient > 0.0001f)
        {
            simulationCompute.SetBuffer(surfaceFieldKernelIndex, "_PredictedPositions", predictedPositionsBuffer);
            simulationCompute.SetBuffer(surfaceFieldKernelIndex, "_Densities", densitiesBuffer);
            simulationCompute.SetBuffer(surfaceFieldKernelIndex, "_ColorFieldNormals", colorFieldNormalsBuffer);
            simulationCompute.SetBuffer(surfaceFieldKernelIndex, "_ColorFieldLaplacians", colorFieldLaplaciansBuffer);
            simulationCompute.SetBuffer(surfaceFieldKernelIndex, "_SortKeys", spatialKeysBuffer);
            simulationCompute.SetBuffer(surfaceFieldKernelIndex, "_SortIndices", spatialIndicesBuffer);
            simulationCompute.SetBuffer(surfaceFieldKernelIndex, "_SpatialOffsets", spatialOffsetsBuffer);
            simulationCompute.Dispatch(surfaceFieldKernelIndex, groups, 1, 1);

            simulationCompute.SetBuffer(surfaceTensionKernelIndex, "_PredictedPositions", predictedPositionsBuffer);
            simulationCompute.SetBuffer(surfaceTensionKernelIndex, "_Densities", densitiesBuffer);
            simulationCompute.SetBuffer(surfaceTensionKernelIndex, "_ColorFieldNormals", colorFieldNormalsBuffer);
            simulationCompute.SetBuffer(surfaceTensionKernelIndex, "_ColorFieldLaplacians", colorFieldLaplaciansBuffer);
            simulationCompute.SetBuffer(surfaceTensionKernelIndex, "_SurfaceTensionForces", surfaceTensionBuffer);
            simulationCompute.Dispatch(surfaceTensionKernelIndex, groups, 1, 1);
        }

        simulationCompute.SetBuffer(UpdateVelocitiesKernelIndex, "_Velocities", velocitiesBuffer);
        simulationCompute.SetBuffer(UpdateVelocitiesKernelIndex, "_PressureForces", pressureForcesBuffer);
        simulationCompute.SetBuffer(UpdateVelocitiesKernelIndex, "_NearPressureAccelerations", nearPressureAccelerationsBuffer);
        simulationCompute.SetBuffer(UpdateVelocitiesKernelIndex, "_ViscosityForces", viscosityBuffer);
        simulationCompute.SetBuffer(UpdateVelocitiesKernelIndex, "_SurfaceTensionForces", surfaceTensionBuffer);
        simulationCompute.SetBuffer(UpdateVelocitiesKernelIndex, "_Densities", densitiesBuffer);
        simulationCompute.Dispatch(UpdateVelocitiesKernelIndex, groups, 1, 1);

        simulationCompute.SetBuffer(UpdatePositionsKernelIndex, "_Positions", positionsBuffer);
        simulationCompute.SetBuffer(UpdatePositionsKernelIndex, "_Velocities", velocitiesBuffer);
        simulationCompute.Dispatch(UpdatePositionsKernelIndex, groups, 1, 1);
    }

    private void OnDestroy()
    {
        positionsBuffer?.Release();
        velocitiesBuffer?.Release();
        argsBuffer?.Release();
        predictedPositionsBuffer?.Release();
        densitiesBuffer?.Release();
        pressureBuffer?.Release();
        pressureForcesBuffer?.Release();
        nearPressureAccelerationsBuffer?.Release();
        nearDensitiesBuffer?.Release();
        viscosityBuffer?.Release();
        surfaceTensionBuffer?.Release();
        colorFieldLaplaciansBuffer?.Release();
        colorFieldNormalsBuffer?.Release();
        spatialKeysBuffer?.Release();
        spatialIndicesBuffer?.Release();
        spatialOffsetsBuffer?.Release();
        particleKeysBuffer?.Release();
        bucketCountsBuffer?.Release();
        // The first output is spatialOffsetsBuffer, already released above.
        for (int level = 1; level < scanOutputs.Count; level++)
            scanOutputs[level].Release();
        foreach (ComputeBuffer sums in scanBlockSums)
            sums.Release();
    }
}
