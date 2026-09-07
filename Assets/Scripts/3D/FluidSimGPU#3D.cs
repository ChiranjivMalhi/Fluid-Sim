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
    [SerializeField, Min(3)] private int sphereLongitudeSegments = 8;
    [SerializeField, Min(2)] private int sphereLatitudeSegments = 6;

    [Header("Physics")]
    [SerializeField] private float gravity = 9.81f;
    [SerializeField, Range(0f, 1f)] private float dampingFactor = 0.9f;

    [Header("Simulation Time")]
    [SerializeField, Min(0.001f)] private float simulationTimeStep = 1f / 120f;
    [SerializeField, Min(1)] private int maxSubstepsPerFrame = 8;

    [Header("Density")]
    [SerializeField, Min(0.01f)] private float smoothingRadius = 0.45f;
    [SerializeField, Min(0.01f)] private float restDensity = 4.0f;
    [SerializeField, Min(0.01f)] private float densityScale = 1.5f;

    [Header("Pressure")]
    [SerializeField] private float pressureMultiplier = 50f;
    [SerializeField] private float nearPressureMultiplier = 50f;

    [Header("Viscosity")]
    [SerializeField] private float viscosityMultiplier = 0.1f;

    [Header("Surface Tension")]
    [SerializeField] private float surfaceTensionCoefficient = 0.0f;
    [SerializeField] private float surfaceTensionThreshold = 0.05f;

    [Header("Debug Rendering")]
    [SerializeField] private bool drawParticleSpheres = true;

    private int particleCount;
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

    private int sortKernelIndex;
    private int calculateOffsetsKernelIndex;
    private int hashParticlesKernelIndex;

    private int clearOffsetsKernelIndex;
    private float accumulatedTime;
    private int paddedCount;

    // Spatial Hashing
    private ComputeBuffer spatialKeysBuffer;
    private ComputeBuffer spatialIndicesBuffer;
    private ComputeBuffer spatialOffsetsBuffer;

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
    private ComputeBuffer nearPressureForcesBuffer;

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
    public Vector3 SimulationBounds => simulationBounds;
    public float SmoothingRadius => smoothingRadius;
    public Vector3 SimulationCenter => transform.position;

    private void DispatchSort()
    {
        int numStages = (int)Mathf.Log(paddedCount, 2);
        simulationCompute.SetBuffer(sortKernelIndex, "_SortKeys", spatialKeysBuffer);
        simulationCompute.SetBuffer(sortKernelIndex, "_SortIndices", spatialIndicesBuffer);
        simulationCompute.SetInt("_SortNumEntries", paddedCount);

        int groups = Mathf.CeilToInt(paddedCount / 2f / 128f);

        for (int stageIndex = 0; stageIndex < numStages; stageIndex++)
        {
            for (int stepIndex = 0; stepIndex <= stageIndex; stepIndex++)
            {
                int groupWidth = 1 << (stageIndex - stepIndex);
                int groupHeight = 2 * groupWidth - 1;
                simulationCompute.SetInt("_GroupWidth", groupWidth);
                simulationCompute.SetInt("_GroupHeight", groupHeight);
                simulationCompute.SetInt("_StepIndex", stepIndex);
                simulationCompute.Dispatch(sortKernelIndex, groups, 1, 1);
            }
        }
    }

    private void Awake()
    {
        particleCount = particlesX * particlesY * particlesZ;
        paddedCount = NextPowerOfTwo(particleCount);

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
        hashParticlesKernelIndex = simulationCompute.FindKernel("HashParticles");
        sortKernelIndex = simulationCompute.FindKernel("BitonicSort");
        calculateOffsetsKernelIndex = simulationCompute.FindKernel("CalculateOffsets");
        clearOffsetsKernelIndex = simulationCompute.FindKernel("ClearOffsets");

        CreateBuffers();
        SpawnParticles();
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
        nearPressureForcesBuffer = new ComputeBuffer(particleCount, sizeof(float) * 3);
        viscosityBuffer = new ComputeBuffer(particleCount, sizeof(float) * 3);
        surfaceTensionBuffer = new ComputeBuffer(particleCount, sizeof(float) * 3);
        colorFieldNormalsBuffer = new ComputeBuffer(particleCount, sizeof(float) * 3);
        colorFieldLaplaciansBuffer = new ComputeBuffer(particleCount, sizeof(float));

        spatialKeysBuffer = new ComputeBuffer(paddedCount, sizeof(uint));
        spatialIndicesBuffer = new ComputeBuffer(paddedCount, sizeof(uint));
        spatialOffsetsBuffer = new ComputeBuffer(particleCount, sizeof(uint));
    }

    private void SpawnParticles()
    {
        Vector3[] positions = new Vector3[particleCount];
        Vector3[] velocities = new Vector3[particleCount];

        float gridWidth = (particlesX - 1) * particleSpacing;
        float gridHeight = (particlesY - 1) * particleSpacing;
        float gridDepth = (particlesZ - 1) * particleSpacing;
        Vector3 bottomLeft = new Vector3(-gridWidth * 0.5f, -gridHeight * 0.5f, -gridDepth * 0.5f);

        int index = 0;
        for (int z = 0; z < particlesZ; z++)
        {
            for (int y = 0; y < particlesY; y++)
            {
                for (int x = 0; x < particlesX; x++)
                {
                    positions[index] = bottomLeft + new Vector3(x * particleSpacing, y * particleSpacing, z * particleSpacing);
                    velocities[index] = Vector3.zero;
                    index++;
                }
            }
        }
        positionsBuffer.SetData(positions);
        predictedPositionsBuffer.SetData(positions);
        velocitiesBuffer.SetData(velocities);
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
        args[1] = (uint)particleCount;
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
        int groups = Mathf.CeilToInt(particleCount / 128f);
        int paddedGroups = Mathf.CeilToInt(paddedCount / 128f);

        simulationCompute.SetFloat("_DeltaTime", deltaTime);
        simulationCompute.SetFloat("_Gravity", Mathf.Abs(gravity));
        simulationCompute.SetFloat("_DampingFactor", dampingFactor);
        simulationCompute.SetFloat("_ParticleRadius", particleRadius);
        simulationCompute.SetVector("_SimulationBounds", simulationBounds);
        simulationCompute.SetInt("_ParticleCount", particleCount);
        simulationCompute.SetFloat("_SmoothingRadius", smoothingRadius);
        simulationCompute.SetFloat("_RestDensity", restDensity);
        simulationCompute.SetFloat("_PressureMultiplier", pressureMultiplier);
        simulationCompute.SetFloat("_NearPressureMultiplier", nearPressureMultiplier);
        simulationCompute.SetFloat("_ViscosityCoefficient", viscosityMultiplier);
        simulationCompute.SetFloat("_SurfaceTensionCoefficient", surfaceTensionCoefficient);
        simulationCompute.SetFloat("_SurfaceTensionThreshold", surfaceTensionThreshold);

        simulationCompute.SetBuffer(kernelIndex, "_Positions", positionsBuffer);
        simulationCompute.SetBuffer(kernelIndex, "_Velocities", velocitiesBuffer);
        simulationCompute.SetBuffer(kernelIndex, "_PredictedPositions", predictedPositionsBuffer);
        simulationCompute.Dispatch(kernelIndex, groups, 1, 1);

        simulationCompute.SetBuffer(hashParticlesKernelIndex, "_PredictedPositions", predictedPositionsBuffer);
        simulationCompute.SetBuffer(hashParticlesKernelIndex, "_SortKeys", spatialKeysBuffer);
        simulationCompute.SetBuffer(hashParticlesKernelIndex, "_SortIndices", spatialIndicesBuffer);
        simulationCompute.Dispatch(hashParticlesKernelIndex, paddedGroups, 1, 1);

        DispatchSort();

        simulationCompute.SetBuffer(clearOffsetsKernelIndex, "_SpatialOffsets", spatialOffsetsBuffer);
        simulationCompute.Dispatch(clearOffsetsKernelIndex, groups, 1, 1);

        simulationCompute.SetBuffer(calculateOffsetsKernelIndex, "_SortKeys", spatialKeysBuffer);
        simulationCompute.SetBuffer(calculateOffsetsKernelIndex, "_SpatialOffsets", spatialOffsetsBuffer);
        simulationCompute.Dispatch(calculateOffsetsKernelIndex, groups, 1, 1);

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
        simulationCompute.SetBuffer(pressureKernelIndex, "_NearPressureForces", nearPressureForcesBuffer);
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
        simulationCompute.SetBuffer(UpdateVelocitiesKernelIndex, "_NearPressureForces", nearPressureForcesBuffer);
        simulationCompute.SetBuffer(UpdateVelocitiesKernelIndex, "_ViscosityForces", viscosityBuffer);
        simulationCompute.SetBuffer(UpdateVelocitiesKernelIndex, "_SurfaceTensionForces", surfaceTensionBuffer);
        simulationCompute.SetBuffer(UpdateVelocitiesKernelIndex, "_Densities", densitiesBuffer);
        simulationCompute.Dispatch(UpdateVelocitiesKernelIndex, groups, 1, 1);

        simulationCompute.SetBuffer(UpdatePositionsKernelIndex, "_Positions", positionsBuffer);
        simulationCompute.SetBuffer(UpdatePositionsKernelIndex, "_Velocities", velocitiesBuffer);
        simulationCompute.SetBuffer(UpdatePositionsKernelIndex, "_PredictedPositions", predictedPositionsBuffer);
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
        nearPressureForcesBuffer?.Release();
        nearDensitiesBuffer?.Release();
        viscosityBuffer?.Release();
        surfaceTensionBuffer?.Release();
        colorFieldLaplaciansBuffer?.Release();
        colorFieldNormalsBuffer?.Release();
        spatialKeysBuffer?.Release();
        spatialIndicesBuffer?.Release();
        spatialOffsetsBuffer?.Release();
    }

    private int NextPowerOfTwo(int n)
    {
        int p = 1;
        while (p < n) p *= 2;
        return p;
    }
}
