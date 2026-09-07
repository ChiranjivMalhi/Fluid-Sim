using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

public class FluidRaymarchVolume3D : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private FluidSimGPU3D simulation;
    [SerializeField] private ComputeShader volumeCompute;
    [SerializeField] private Material raymarchMaterial;

    [Header("Density Volume")]
    [SerializeField, Range(16, 128)] private int volumeResolution = 64;

    [Header("Raymarch")]
    [SerializeField, Min(8)] private int maxRaySteps = 256;
    [SerializeField, Min(0.001f)] private float rayStepSize = 0.05f;
    [SerializeField] private float surfaceThreshold = 2.0f;
    [SerializeField] private Color waterColor = new Color(0.08f, 0.45f, 0.9f, 0.92f);

    [Header("Optics (Reflection, Refraction & Fresnel)")]
    [SerializeField, Range(1f, 2f)] private float indexOfRefraction = 1.333f;
    [SerializeField, Range(0f, 2f)] private float reflectionStrength = 1.0f;
    [SerializeField, Range(0f, 2f)] private float refractionStrength = 1.0f;
    [SerializeField, Range(1f, 10f)] private float fresnelPower = 5.0f;
    [SerializeField, Range(8f, 256f)] private float specularPower = 64f;
    [SerializeField, Range(0f, 5f)] private float sunSpecular = 1.5f;
    [SerializeField, Range(0f, 5f)] private float absorptionDensity = 1.0f;

    [Header("Scene Interaction")]
    [SerializeField] private bool interactWithSceneGeometry = true;
    [SerializeField] private Cubemap customEnvironmentMap;

    private static readonly int DensityVolumeID = Shader.PropertyToID("_DensityVolume");
    private static readonly int VolumeResolutionID = Shader.PropertyToID("_VolumeResolution");
    private static readonly int VolumeBoundsID = Shader.PropertyToID("_VolumeBounds");
    private static readonly int VolumeCenterID = Shader.PropertyToID("_VolumeCenter");
    private static readonly int ParticleCountID = Shader.PropertyToID("_ParticleCount");
    private static readonly int SmoothingRadiusID = Shader.PropertyToID("_SmoothingRadius");
    private static readonly int PositionsID = Shader.PropertyToID("_Positions");
    private static readonly int SortKeysID = Shader.PropertyToID("_SortKeys");
    private static readonly int SortIndicesID = Shader.PropertyToID("_SortIndices");
    private static readonly int SpatialOffsetsID = Shader.PropertyToID("_SpatialOffsets");

    private static readonly int CameraWorldPositionID = Shader.PropertyToID("_CameraWorldPosition");
    private static readonly int LightDirectionID = Shader.PropertyToID("_LightDirection");
    private static readonly int SurfaceThresholdID = Shader.PropertyToID("_SurfaceThreshold");
    private static readonly int RayStepSizeID = Shader.PropertyToID("_RayStepSize");
    private static readonly int MaxRayStepsID = Shader.PropertyToID("_MaxRaySteps");
    private static readonly int WaterColorID = Shader.PropertyToID("_WaterColor");

    private static readonly int IndexOfRefractionID = Shader.PropertyToID("_IndexOfRefraction");
    private static readonly int ReflectionStrengthID = Shader.PropertyToID("_ReflectionStrength");
    private static readonly int RefractionStrengthID = Shader.PropertyToID("_RefractionStrength");
    private static readonly int FresnelPowerID = Shader.PropertyToID("_FresnelPower");
    private static readonly int SpecularPowerID = Shader.PropertyToID("_SpecularPower");
    private static readonly int SunSpecularID = Shader.PropertyToID("_SunSpecular");
    private static readonly int AbsorptionDensityID = Shader.PropertyToID("_AbsorptionDensity");
    private static readonly int InteractWithSceneID = Shader.PropertyToID("_InteractWithScene");
    private static readonly int HasEnvironmentMapID = Shader.PropertyToID("_HasEnvironmentMap");
    private static readonly int EnvironmentMapID = Shader.PropertyToID("_EnvironmentMap");

    private RenderTexture densityVolume;
    private int buildDensityKernel = -1;
    private int allocatedResolution;
    private Mesh cubeMesh;

    private void Awake()
    {
        if (volumeCompute != null)
        {
            buildDensityKernel = volumeCompute.FindKernel("BuildDensityVolume");
        }
        cubeMesh = CreateCubeMesh();
    }

    private void OnEnable()
    {
        RenderPipelineManager.beginCameraRendering += OnBeginCameraRendering;
        if (cubeMesh == null)
        {
            cubeMesh = CreateCubeMesh();
        }
    }

    private void OnDisable()
    {
        RenderPipelineManager.beginCameraRendering -= OnBeginCameraRendering;
    }

    private void OnBeginCameraRendering(ScriptableRenderContext context, Camera camera)
    {
        if (raymarchMaterial != null && camera != null)
        {
            raymarchMaterial.SetVector(CameraWorldPositionID, camera.transform.position);
        }
    }

    private void LateUpdate()
    {
        if (!Application.isPlaying) return;
        if (simulation == null || simulation.PositionsBuffer == null || volumeCompute == null || raymarchMaterial == null)
        {
            return;
        }

        EnsureDensityVolume();
        BuildDensityVolume();
        RenderVolume();
    }

    private void EnsureDensityVolume()
    {
        if (densityVolume != null && allocatedResolution == volumeResolution)
        {
            return;
        }

        if (densityVolume != null)
        {
            densityVolume.Release();
            DestroyImmediate(densityVolume);
        }

        RenderTextureDescriptor descriptor = new RenderTextureDescriptor(
            volumeResolution,
            volumeResolution,
            GraphicsFormat.R16_SFloat,
            0)
        {
            dimension = TextureDimension.Tex3D,
            volumeDepth = volumeResolution,
            enableRandomWrite = true,
            msaaSamples = 1,
            sRGB = false
        };

        densityVolume = new RenderTexture(descriptor)
        {
            name = "Fluid Density Volume",
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp
        };
        densityVolume.Create();
        allocatedResolution = volumeResolution;
    }

    private void BuildDensityVolume()
    {
        if (buildDensityKernel < 0 && volumeCompute != null)
        {
            buildDensityKernel = volumeCompute.FindKernel("BuildDensityVolume");
        }

        Vector3 bounds = simulation.SimulationBounds;

        volumeCompute.SetTexture(buildDensityKernel, DensityVolumeID, densityVolume);
        volumeCompute.SetInts(VolumeResolutionID, volumeResolution, volumeResolution, volumeResolution);
        volumeCompute.SetVector(VolumeBoundsID, bounds);
        volumeCompute.SetInt(ParticleCountID, simulation.ParticleCount);
        volumeCompute.SetFloat(SmoothingRadiusID, simulation.SmoothingRadius);

        volumeCompute.SetBuffer(buildDensityKernel, PositionsID, simulation.PositionsBuffer);
        volumeCompute.SetBuffer(buildDensityKernel, SortKeysID, simulation.SpatialKeysBuffer);
        volumeCompute.SetBuffer(buildDensityKernel, SortIndicesID, simulation.SpatialIndicesBuffer);
        volumeCompute.SetBuffer(buildDensityKernel, SpatialOffsetsID, simulation.SpatialOffsetsBuffer);

        int groups = Mathf.CeilToInt(volumeResolution / 4f);
        volumeCompute.Dispatch(buildDensityKernel, groups, groups, groups);
    }

    private void RenderVolume()
    {
        if (cubeMesh == null)
        {
            cubeMesh = CreateCubeMesh();
        }

        Vector3 bounds = simulation.SimulationBounds;
        Vector3 center = simulation.transform.position;

        raymarchMaterial.SetTexture(DensityVolumeID, densityVolume);
        raymarchMaterial.SetVector(VolumeBoundsID, bounds);
        raymarchMaterial.SetVector(VolumeCenterID, center);
        raymarchMaterial.SetFloat(SurfaceThresholdID, surfaceThreshold);
        raymarchMaterial.SetFloat(RayStepSizeID, rayStepSize);
        raymarchMaterial.SetInt(MaxRayStepsID, maxRaySteps);
        raymarchMaterial.SetColor(WaterColorID, waterColor);

        
        raymarchMaterial.SetFloat(IndexOfRefractionID, indexOfRefraction);
        raymarchMaterial.SetFloat(ReflectionStrengthID, reflectionStrength);
        raymarchMaterial.SetFloat(RefractionStrengthID, refractionStrength);
        raymarchMaterial.SetFloat(FresnelPowerID, fresnelPower);
        raymarchMaterial.SetFloat(SpecularPowerID, specularPower);
        raymarchMaterial.SetFloat(SunSpecularID, sunSpecular);
        raymarchMaterial.SetFloat(AbsorptionDensityID, absorptionDensity);
        raymarchMaterial.SetFloat(InteractWithSceneID, interactWithSceneGeometry ? 1.0f : 0.0f);

        if (customEnvironmentMap != null)
        {
            raymarchMaterial.SetFloat(HasEnvironmentMapID, 1.0f);
            raymarchMaterial.SetTexture(EnvironmentMapID, customEnvironmentMap);
        }
        else
        {
            raymarchMaterial.SetFloat(HasEnvironmentMapID, 0.0f);
        }

        if (RenderSettings.sun != null)
        {
            raymarchMaterial.SetVector(LightDirectionID, -RenderSettings.sun.transform.forward);
        }
        else
        {
            raymarchMaterial.SetVector(LightDirectionID, new Vector3(0.4f, 0.8f, -0.4f).normalized);
        }

        if (Camera.main != null)
        {
            raymarchMaterial.SetVector(CameraWorldPositionID, Camera.main.transform.position);
        }

       
        Graphics.DrawMesh(
            cubeMesh,
            Matrix4x4.TRS(center, Quaternion.identity, bounds),
            raymarchMaterial,
            0,
            null,
            0,
            null,
            ShadowCastingMode.Off,
            false
        );
    }

    private Mesh CreateCubeMesh()
    {
        Mesh mesh = new Mesh { name = "FluidRaymarchCube" };

        Vector3[] vertices = new Vector3[]
        {
            new Vector3(-0.5f, -0.5f, -0.5f), 
            new Vector3( 0.5f, -0.5f, -0.5f), 
            new Vector3( 0.5f,  0.5f, -0.5f), 
            new Vector3(-0.5f,  0.5f, -0.5f), 
            new Vector3(-0.5f, -0.5f,  0.5f), 
            new Vector3( 0.5f, -0.5f,  0.5f), 
            new Vector3( 0.5f,  0.5f,  0.5f), 
            new Vector3(-0.5f,  0.5f,  0.5f)  
        };

        int[] triangles = new int[]
        {
            // Front (-Z)
            0, 2, 1, 0, 3, 2,
            // Back (+Z)
            5, 7, 4, 5, 6, 7,
            // Left (-X)
            4, 3, 0, 4, 7, 3,
            // Right (+X)
            1, 6, 5, 1, 2, 6,
            // Top (+Y)
            3, 6, 2, 3, 7, 6,
            // Bottom (-Y)
            4, 1, 5, 4, 0, 1
        };

        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateBounds();
        return mesh;
    }

    private void OnDestroy()
    {
        if (densityVolume != null)
        {
            densityVolume.Release();
            DestroyImmediate(densityVolume);
            densityVolume = null;
        }

        if (cubeMesh != null)
        {
            DestroyImmediate(cubeMesh);
            cubeMesh = null;
        }
    }
}
