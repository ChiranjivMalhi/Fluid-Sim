using UnityEngine;
public class FluidSimGPU : MonoBehaviour
{
    [Header("Compute")]
    [SerializeField] private ComputeShader simulationCompute;
    [SerializeField] private Material particleMaterial;

    [Header("Scene Setup")]
    [SerializeField] private Vector2 simulationBounds = new Vector2(20f, 12f);

    [Header("Particle Grid")]
    [SerializeField, Min(1)] private int particlesPerRow = 100;
    [SerializeField, Min(1)] private int particlesPerColumn = 100;
    [SerializeField, Min(0.01f)] private float particleSpacing = 0.15f;
    [SerializeField, Min(0.001f)] private float particleRadius = 0.05f;

    [Header("Physics")]
    [SerializeField] private float gravity = 9.81f;
    [SerializeField, Range(0f, 1f)] private float dampingFactor = 0.5f;

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
    [SerializeField] private float surfaceTensionCoefficient = 0.5f; // σ
    [SerializeField] private float surfaceTensionThreshold = 0.05f;

    private int particleCount;
    private int kernelIndex;
    private int densityKernelIndex;
    private int pressureKernelIndex;
    private int nearPressureKernelIndex;
    private int viscosityKernelIndex;
    private int surfaceFieldKernelIndex;
    private int surfaceTensionKernelIndex;

    private int UpdateVelocitiesKernelIndex;
    private int UpdatePositionsKernelIndex;
    private float accumulatedTime;

    // ComputeBuffers
    private ComputeBuffer positionsBuffer;
    private ComputeBuffer velocitiesBuffer;
    private ComputeBuffer argsBuffer;
    private ComputeBuffer predictedPositionsBuffer;

    //Density
    private ComputeBuffer densitiesBuffer;
    private ComputeBuffer nearDensitiesBuffer;

    //Pressure
    private ComputeBuffer pressureBuffer;
    private ComputeBuffer pressureForcesBuffer;
    private ComputeBuffer nearPressureForcesBuffer;

    //Viscosity
    private ComputeBuffer viscosityBuffer;

    //Surface Tension
    private ComputeBuffer colorFieldNormalsBuffer;
    private ComputeBuffer colorFieldLaplaciansBuffer;
    private ComputeBuffer surfaceTensionBuffer;
    private Mesh quadMesh;
    private Bounds drawBounds;

    private static readonly int PositionsID = Shader.PropertyToID("_Positions");
    private static readonly int DensitiesID = Shader.PropertyToID("_Densities");
    private static readonly int VelocitiesID = Shader.PropertyToID("_Velocities");

    private void Awake()
    {
        particleCount = particlesPerRow * particlesPerColumn;
        kernelIndex = simulationCompute.FindKernel("ApplyForcesIntegrateCollide");
        densityKernelIndex = simulationCompute.FindKernel("CalculateDensity");
        pressureKernelIndex = simulationCompute.FindKernel("CalculatePressure");
        nearPressureKernelIndex = simulationCompute.FindKernel("CalculateNearPressure");
        viscosityKernelIndex = simulationCompute.FindKernel("CalculateViscosity");
        surfaceFieldKernelIndex = simulationCompute.FindKernel("CalculateColorField");
        surfaceTensionKernelIndex = simulationCompute.FindKernel("CalculateSurfaceTension");
        UpdateVelocitiesKernelIndex = simulationCompute.FindKernel("UpdateVelocities");
        UpdatePositionsKernelIndex = simulationCompute.FindKernel("UpdatePositions");

        CreateBuffers();
        SpawnParticles();
        CreateQuadMesh();
        SetupArgsBuffer();

        particleMaterial.SetBuffer(PositionsID, positionsBuffer);
        particleMaterial.SetBuffer(DensitiesID, densitiesBuffer);
        particleMaterial.SetBuffer(VelocitiesID, velocitiesBuffer);

        drawBounds = new Bounds(Vector3.zero,
            new Vector3(simulationBounds.x + 5f, simulationBounds.y + 5f, 5f));
    }

    private void CreateBuffers()
    {
        positionsBuffer = new ComputeBuffer(particleCount, sizeof(float) * 2);
        velocitiesBuffer = new ComputeBuffer(particleCount, sizeof(float) * 2);
        predictedPositionsBuffer = new ComputeBuffer(particleCount, sizeof(float) * 2);
        densitiesBuffer = new ComputeBuffer(particleCount, sizeof(float));
        nearDensitiesBuffer = new ComputeBuffer(particleCount, sizeof(float));
        pressureBuffer = new ComputeBuffer(particleCount, sizeof(float));
        pressureForcesBuffer = new ComputeBuffer(particleCount, sizeof(float) * 2);
        nearPressureForcesBuffer = new ComputeBuffer(particleCount, sizeof(float) * 2);
        viscosityBuffer = new ComputeBuffer(particleCount, sizeof(float) * 2);
        surfaceTensionBuffer = new ComputeBuffer(particleCount, sizeof(float) * 2);
        colorFieldNormalsBuffer = new ComputeBuffer(particleCount, sizeof(float) * 2);
        colorFieldLaplaciansBuffer = new ComputeBuffer(particleCount, sizeof(float));
    }

    private void SpawnParticles()
    {
        Vector2[] positions = new Vector2[particleCount];
        Vector2[] velocities = new Vector2[particleCount];

        float gridWidth = (particlesPerRow - 1) * particleSpacing;
        float gridHeight = (particlesPerColumn - 1) * particleSpacing;
        Vector2 bottomLeft = new Vector2(-gridWidth * 0.5f, -gridHeight * 0.5f);

        int index = 0;
        for (int y = 0; y < particlesPerColumn; y++)
        {
            for (int x = 0; x < particlesPerRow; x++)
            {
                positions[index] = bottomLeft + new Vector2(x * particleSpacing, y * particleSpacing);
                velocities[index] = Vector2.zero;
                index++;
            }
        }

        positionsBuffer.SetData(positions);
        velocitiesBuffer.SetData(velocities);
    }

    private void CreateQuadMesh()
    {

        quadMesh = new Mesh();
        float r = particleRadius;
        quadMesh.vertices = new Vector3[]
        {
            new Vector3(-r, -r, 0),
            new Vector3(-r,  r, 0),
            new Vector3( r,  r, 0),
            new Vector3( r, -r, 0),
        };
        quadMesh.uv = new Vector2[]
        {
            new Vector2(0, 0),
            new Vector2(0, 1),
            new Vector2(1, 1),
            new Vector2(1, 0),
        };
        quadMesh.triangles = new int[] { 0, 1, 2, 0, 2, 3 };
        quadMesh.RecalculateBounds();
    }

    private void SetupArgsBuffer()
    {
        uint[] args = new uint[5];
        args[0] = quadMesh.GetIndexCount(0);
        args[1] = (uint)particleCount;
        args[2] = quadMesh.GetIndexStart(0);
        args[3] = quadMesh.GetBaseVertex(0);
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

        Graphics.DrawMeshInstancedIndirect(quadMesh, 0, particleMaterial, drawBounds, argsBuffer);
    }

    private void DispatchSimulationStep(float deltaTime)
    {
        simulationCompute.SetFloat("_DeltaTime", deltaTime);
        simulationCompute.SetFloat("_Gravity", gravity);
        simulationCompute.SetFloat("_DampingFactor", dampingFactor);
        simulationCompute.SetFloat("_ParticleRadius", particleRadius);
        simulationCompute.SetVector("_SimulationBounds", simulationBounds);
        simulationCompute.SetInt("_ParticleCount", particleCount);
        simulationCompute.SetFloat("_SmoothingRadius", smoothingRadius);

        simulationCompute.SetBuffer(kernelIndex, "_Positions", positionsBuffer);
        simulationCompute.SetBuffer(kernelIndex, "_Velocities", velocitiesBuffer);
        simulationCompute.SetBuffer(kernelIndex, "_PredictedPositions", predictedPositionsBuffer);

        simulationCompute.SetBuffer(densityKernelIndex, "_PredictedPositions", predictedPositionsBuffer);
        simulationCompute.SetBuffer(densityKernelIndex, "_Densities", densitiesBuffer);

        simulationCompute.SetBuffer(pressureKernelIndex, "_Densities", densitiesBuffer);
        simulationCompute.SetBuffer(pressureKernelIndex, "_Pressure", pressureBuffer);
        simulationCompute.SetBuffer(pressureKernelIndex, "_PressureForces", pressureForcesBuffer);
        simulationCompute.SetBuffer(pressureKernelIndex, "_PredictedPositions", predictedPositionsBuffer);
        simulationCompute.SetFloat("_RestDensity", restDensity);
        simulationCompute.SetFloat("_PressureMultiplier", pressureMultiplier);

        simulationCompute.SetBuffer(nearPressureKernelIndex, "_NearDensities", nearDensitiesBuffer);
        simulationCompute.SetBuffer(nearPressureKernelIndex, "_Densities", densitiesBuffer);
        simulationCompute.SetBuffer(nearPressureKernelIndex, "_NearPressureForces", nearPressureForcesBuffer);
        simulationCompute.SetBuffer(nearPressureKernelIndex, "_PredictedPositions", predictedPositionsBuffer);
        simulationCompute.SetFloat("_NearPressureMultiplier", nearPressureMultiplier);

        simulationCompute.SetBuffer(viscosityKernelIndex, "_Velocities", velocitiesBuffer);
        simulationCompute.SetBuffer(viscosityKernelIndex, "_ViscosityForces", viscosityBuffer);
        simulationCompute.SetBuffer(viscosityKernelIndex, "_PredictedPositions", predictedPositionsBuffer);
        simulationCompute.SetBuffer(viscosityKernelIndex, "_Densities", densitiesBuffer);
        simulationCompute.SetFloat("_ViscosityCoefficient", viscosityMultiplier);

        simulationCompute.SetBuffer(surfaceFieldKernelIndex, "_PredictedPositions", predictedPositionsBuffer);
        simulationCompute.SetBuffer(surfaceFieldKernelIndex, "_Densities", densitiesBuffer);
        simulationCompute.SetBuffer(surfaceFieldKernelIndex, "_ColorFieldNormals", colorFieldNormalsBuffer);
        simulationCompute.SetBuffer(surfaceFieldKernelIndex, "_ColorFieldLaplacians", colorFieldLaplaciansBuffer);

        simulationCompute.SetBuffer(surfaceTensionKernelIndex, "_PredictedPositions", predictedPositionsBuffer);
        simulationCompute.SetBuffer(surfaceTensionKernelIndex, "_Densities", densitiesBuffer);
        simulationCompute.SetBuffer(surfaceTensionKernelIndex, "_ColorFieldNormals", colorFieldNormalsBuffer);
        simulationCompute.SetBuffer(surfaceTensionKernelIndex, "_ColorFieldLaplacians", colorFieldLaplaciansBuffer);
        simulationCompute.SetBuffer(surfaceTensionKernelIndex, "_SurfaceTensionForces", surfaceTensionBuffer);
        simulationCompute.SetFloat("_SurfaceTensionCoefficient", surfaceTensionCoefficient);
        simulationCompute.SetFloat("_SurfaceTensionThreshold", surfaceTensionThreshold);

        simulationCompute.SetBuffer(UpdateVelocitiesKernelIndex, "_Velocities", velocitiesBuffer);
        simulationCompute.SetBuffer(UpdateVelocitiesKernelIndex, "_PressureForces", pressureForcesBuffer);
        simulationCompute.SetBuffer(UpdateVelocitiesKernelIndex, "_ViscosityForces", viscosityBuffer);
        simulationCompute.SetBuffer(UpdateVelocitiesKernelIndex, "_SurfaceTensionForces", surfaceTensionBuffer);
        simulationCompute.SetBuffer(UpdateVelocitiesKernelIndex, "_Densities", densitiesBuffer);
        simulationCompute.SetBuffer(UpdateVelocitiesKernelIndex, "_NearPressureForces", nearPressureForcesBuffer);

        simulationCompute.SetBuffer(UpdatePositionsKernelIndex, "_Positions", positionsBuffer);
        simulationCompute.SetBuffer(UpdatePositionsKernelIndex, "_Velocities", velocitiesBuffer);
        simulationCompute.SetBuffer(UpdatePositionsKernelIndex, "_PredictedPositions", predictedPositionsBuffer);

        int groups = Mathf.CeilToInt(particleCount / 128f);
        simulationCompute.Dispatch(kernelIndex, groups, 1, 1);
        simulationCompute.Dispatch(densityKernelIndex, groups, 1, 1);
        simulationCompute.Dispatch(pressureKernelIndex, groups, 1, 1);
        simulationCompute.Dispatch(nearPressureKernelIndex, groups, 1, 1);
        simulationCompute.Dispatch(viscosityKernelIndex, groups, 1, 1);
        simulationCompute.Dispatch(surfaceFieldKernelIndex, groups, 1, 1);
        simulationCompute.Dispatch(surfaceTensionKernelIndex, groups, 1, 1);
        simulationCompute.Dispatch(UpdateVelocitiesKernelIndex, groups, 1, 1);
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
    }
}
