using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;
public class FluidSim : MonoBehaviour
{
    [Header("Scene Setup")]
    [Tooltip("The intended width and height of the future simulation container, in world units.")]
    [SerializeField] private Vector2 simulationBounds = new Vector2(12f, 7f);

    [Tooltip("A single manually placed particle. It is only a Transform for now.")]
    [SerializeField] private Transform starterParticle;

    [Header("Temporary 2D Preview")]
    [SerializeField] private Color borderColor = new Color(0.2f, 0.85f, 1f);
    [SerializeField] private Color particleColor = new Color(0.55f, 0.9f, 1f);
    [SerializeField, Min(0.01f)] private float borderThickness = 0.05f;
    [SerializeField, Min(0.01f)] private float particleRadius = 0.14f;
    [SerializeField, Min(0f)] private float dampingFactor = 0.5f;
    [SerializeField] private float cellSize = 0.7f;

    [Header("Simulation Time")]
    [SerializeField, Min(0.001f)] private float simulationTimeStep = 1f / 120f;
    [SerializeField, Min(1)] private int maxSubstepsPerFrame = 8;

    [Header("Particle Grid")]
    [SerializeField, Min(1)] private int particlesPerRow = 10;
    [SerializeField, Min(1)] private int particlesPerColumn = 8;
    [SerializeField, Min(0.01f)] private float particleSpacing = 0.35f;

    [Header("Density")]
    [SerializeField, Min(0.01f)] private float smoothingRadius = 0.7f;
    [SerializeField, Min(0.01f)] private float densityColorScale = 8f;
    [SerializeField] private Color lowDensityColor = new Color(0.1f, 0.3f, 1f);
    [SerializeField] private Color highDensityColor = new Color(1f, 0.2f, 0.1f);

    [SerializeField] private float targetDensity = 6f;
    [SerializeField] private float pressureMultiplier = 30f;
    [SerializeField] private float viscosity = 0.1f;

    private Vector2[] colorFieldNormals;
    private float[] colorFieldLaplacians;
    private Vector2[] surfaceTensionForces;

    [Header("Surface Tension")]
    [SerializeField] private float surfaceTensionCoefficient = 0.5f; 
    [SerializeField] private float surfaceTensionThreshold = 0.05f;  

    private float[] pressures;
    private Vector2[] pressureForces;
    private Vector2[] predictedPositions;

    private float[] densities;

    private readonly Dictionary<Vector2Int, List<int>> spatialGrid = new();
    

    private readonly List<Transform> particles = new();

    private Texture2D previewTexture;
    private Sprite previewSprite;
    private Texture2D particleTexture;
    private Sprite particleSprite;

    private Vector2[] velocity;
    private float accumulatedTime;

    private Transform borderTop;
    private Transform borderBottom;
    private Transform borderLeft;
    private Transform borderRight;
    private Vector2 lastSimulationBounds;

    private void Awake()
    {
        SetupCamera();
        CreateStaticPreview();
    }

    private Vector2Int GetCell(Vector2 position)
    {
        return new Vector2Int(
            Mathf.FloorToInt(position.x / cellSize),
            Mathf.FloorToInt(position.y / cellSize)
        );
    }

    public Vector2 SimulationBounds => simulationBounds;
    public Transform StarterParticle => starterParticle;

    private void SetupCamera()
    {
        Camera camera = Camera.main;
        if (camera == null)
        {
            return;
        }

        camera.orthographic = true;
        camera.orthographicSize = 5f;
        camera.clearFlags = CameraClearFlags.SolidColor;
        camera.backgroundColor = Color.black;
    }

    private void CreateStaticPreview()
    {
        previewTexture = new Texture2D(1, 1);
        previewTexture.SetPixel(0, 0, Color.white);
        previewTexture.Apply();
        previewSprite = Sprite.Create(previewTexture, new Rect(0, 0, 1, 1), new Vector2(0.5f, 0.5f), 1f);

        particleTexture = new Texture2D(32, 32, TextureFormat.RGBA32, false);
        for (int y = 0; y < particleTexture.height; y++)
        {
            for (int x = 0; x < particleTexture.width; x++)
            {
                float dx = x - 15.5f;
                float dy = y - 15.5f;
                particleTexture.SetPixel(x, y, dx * dx + dy * dy <= 15.5f * 15.5f ? Color.white : Color.clear);
            }
        }
        particleTexture.Apply();
        particleSprite = Sprite.Create(particleTexture, new Rect(0, 0, 32, 32), new Vector2(0.5f, 0.5f), 32f);

        CreateBorder();
        CreateParticleGrid();
    }

    private void CreateBorder()
    {
        Transform border = new GameObject("Preview Border").transform;
        border.SetParent(transform, false);

        float halfWidth = simulationBounds.x * 0.5f;
        float halfHeight = simulationBounds.y * 0.5f;

        borderTop = CreatePreviewRectangle(border, "Top", Vector2.zero, Vector2.zero);
        borderBottom = CreatePreviewRectangle(border, "Bottom", Vector2.zero, Vector2.zero);
        borderLeft = CreatePreviewRectangle(border, "Left", Vector2.zero, Vector2.zero);
        borderRight = CreatePreviewRectangle(border, "Right", Vector2.zero, Vector2.zero);

        UpdateBorder();

    }

    private void UpdateBorder()
    {
        float halfWidth = simulationBounds.x * 0.5f;
        float halfHeight = simulationBounds.y * 0.5f;

        borderTop.localPosition = new Vector2(0, halfHeight);
        borderTop.localScale = new Vector2(simulationBounds.x + borderThickness, borderThickness);

        borderBottom.localPosition = new Vector2(0, -halfHeight);
        borderBottom.localScale = new Vector2(simulationBounds.x + borderThickness, borderThickness);

        borderLeft.localPosition = new Vector2(-halfWidth, 0);
        borderLeft.localScale = new Vector2(borderThickness, simulationBounds.y + borderThickness);

        borderRight.localPosition = new Vector2(halfWidth, 0);
        borderRight.localScale = new Vector2(borderThickness, simulationBounds.y + borderThickness);
    }

    private Transform CreatePreviewRectangle(Transform parent, string objectName, Vector2 position, Vector2 size)
    {
        GameObject rectangle = new GameObject(objectName);
        rectangle.transform.SetParent(parent, false);
        rectangle.transform.localPosition = position;
        rectangle.transform.localScale = size;

        SpriteRenderer renderer = rectangle.AddComponent<SpriteRenderer>();
        renderer.sprite = previewSprite;
        renderer.color = borderColor;
        renderer.sortingOrder = 1;

        return rectangle.transform;
    }

    private void CreateParticleGrid()
    {
        Transform gridParent = new GameObject("Particles").transform;
        gridParent.SetParent(transform, false);

        float gridWidth = (particlesPerRow - 1) * particleSpacing;
        float gridHeight = (particlesPerColumn - 1) * particleSpacing;
        Vector2 bottomLeft = new Vector2(-gridWidth * 0.5f, -gridHeight * 0.5f);

        for (int y = 0; y < particlesPerColumn; y++)
        {
            for (int x = 0; x < particlesPerRow; x++)
            {
                GameObject particle = new GameObject($"Particle {x}, {y}");
                particle.transform.SetParent(gridParent, false);
                particle.transform.localPosition =
                    bottomLeft + new Vector2(x * particleSpacing, y * particleSpacing);
                particle.transform.localScale = Vector3.one * (particleRadius * 2f);

                SpriteRenderer renderer = particle.AddComponent<SpriteRenderer>();
                renderer.sprite = particleSprite;
                renderer.color = particleColor;
                renderer.sortingOrder = 2;

                particles.Add(particle.transform);
            }
        }

        velocity = new Vector2[particles.Count];
        densities = new float[particles.Count];
        pressures = new float[particles.Count];
        pressureForces = new Vector2[particles.Count];
        predictedPositions = new Vector2[particles.Count];
        colorFieldNormals = new Vector2[particles.Count];
        colorFieldLaplacians = new float[particles.Count];
        surfaceTensionForces = new Vector2[particles.Count];
    }

    private void OnDestroy()
    {
        if (previewSprite != null)
        {
            Destroy(previewSprite);
        }

        if (previewTexture != null)
        {
            Destroy(previewTexture);
        }

        if (particleSprite != null)
        {
            Destroy(particleSprite);
        }

        if (particleTexture != null)
        {
            Destroy(particleTexture);
        }
    }

    private void Update()
    {
        

        if (simulationBounds != lastSimulationBounds)
        {
            UpdateBorder();
            lastSimulationBounds = simulationBounds;
        }

        accumulatedTime += Time.deltaTime;

        int substeps = 0;
        while (accumulatedTime >= simulationTimeStep && substeps < maxSubstepsPerFrame)
        {
            SimulateStep(simulationTimeStep);
            accumulatedTime -= simulationTimeStep;
            substeps++;
        }

        if (substeps == maxSubstepsPerFrame)
        {
            accumulatedTime = Mathf.Min(accumulatedTime, simulationTimeStep);
        }
    }

    private void ApplyExternalForcesAndPredict(float deltaTime)
    {
        for (int i = 0; i < particles.Count; i++)
        {
            velocity[i] += Vector2.down * 9.81f * deltaTime;

            Vector2 position = particles[i].position;
            predictedPositions[i] = position + velocity[i] * deltaTime;
        }
    }

    private void SimulateStep(float deltaTime)
    {
        ApplyExternalForcesAndPredict(deltaTime);
        BuildSpatialGrid();
        CalculateDensitiesWithSpatialGrid();
        //CalculateDensities();
        CalculatePressures();
        CalculateSurfaceTensionFields();
        CalculateSurfaceTensionForces();

        for (int i = 0; i < particles.Count; i++)
        {
          
            //velocity[i] += Vector2.down * 9.81f * deltaTime;
            velocity[i] += (pressureForces[i] + surfaceTensionForces[i])/densities[i] * deltaTime;

            Transform particle = particles[i];
            Vector2 position = particle.position;
            position += velocity[i] * deltaTime;
            particle.position = position;
            float densityT = Mathf.Clamp01(densities[i] / densityColorScale);
            float velocityT = Mathf.Clamp01(velocity[i].magnitude / 8f);
            particle.GetComponent<SpriteRenderer>().color = Color.Lerp(lowDensityColor, highDensityColor, velocityT);
        }

        ResolveCollision();
       

    }

    private void BuildSpatialGrid()
    {
        spatialGrid.Clear();

        for (int i = 0; i < particles.Count; i++)
        {
            Vector2Int cell = GetCell(predictedPositions[i]);

            if (!spatialGrid.TryGetValue(cell, out List<int> cellParticles))
            {
                cellParticles = new List<int>();
                spatialGrid.Add(cell, cellParticles);
            }

            cellParticles.Add(i);
        }
    }

    private void ResolveCollision()
    {
        for(int i = 0; i < particles.Count; i++)
        {
            Transform particle = particles[i];
            Vector2 pos = particle.position;
            float halfWidth = simulationBounds.x * 0.5f;
            float halfHeight = simulationBounds.y * 0.5f;
            if (pos.x - particleRadius < -halfWidth)
            {
                pos.x = -halfWidth + particleRadius;
                velocity[i].x *= -dampingFactor; 
            }
            else if (pos.x + particleRadius > halfWidth)
            {
                pos.x = halfWidth - particleRadius;
                velocity[i].x *= -dampingFactor;
            }
            if (pos.y - particleRadius < -halfHeight)
            {
                pos.y = -halfHeight + particleRadius;
                velocity[i].y *= -dampingFactor; 
            }
            else if (pos.y + particleRadius > halfHeight)
            {
                pos.y = halfHeight - particleRadius;
                velocity[i].y *= -dampingFactor;
            }
            particle.position = pos;
        }
    }

    private void CalculateDensities()
    {
        float radiusSquared = smoothingRadius * smoothingRadius;

        for (int i = 0; i < particles.Count; i++)
        {
            Vector2 position = predictedPositions[i];
            float density = 0f;

            for (int j = 0; j < particles.Count; j++)
            {
                Vector2 offset = new Vector2(predictedPositions[j].x, predictedPositions[j].y) - position;
                float distanceSquared = offset.sqrMagnitude;

                if (distanceSquared >= radiusSquared)
                {
                    continue;
                }

                float distance = Mathf.Sqrt(distanceSquared);
                density += DensityKernel(distance);
            }

            densities[i] = density;
        }
    }

    private void CalculateDensitiesWithSpatialGrid()
    {
        float radiusSquared = smoothingRadius * smoothingRadius;

        for (int i = 0; i < particles.Count; i++)
        {
            Vector2 position = predictedPositions[i];
            Vector2Int originCell = GetCell(position);
            densities[i] = 0f;

            for (int offsetY = -1; offsetY <= 1; offsetY++)
            {
                for (int offsetX = -1; offsetX <= 1; offsetX++)
                {
                    Vector2Int neighbourCell = originCell + new Vector2Int(offsetX, offsetY);

                    if (!spatialGrid.TryGetValue(neighbourCell, out List<int> cellParticles))
                    {
                        continue;
                    }

                    foreach (int j in cellParticles)
                    {
                        Vector2 offset = new Vector2(predictedPositions[j].x, predictedPositions[j].y) - position;
                        float distanceSquared = offset.sqrMagnitude;
                        if (distanceSquared >= radiusSquared)
                        {
                            continue;
                        }
                        float distance = Mathf.Sqrt(distanceSquared);
                        densities[i] += DensityKernel(distance);
                    }
                }
            }
        }
    }
    private void CalculatePressures()
    {

        for (int i = 0; i < particles.Count; i++)
        {
            pressures[i] = Mathf.Max(0f, densities[i] - targetDensity)
                           * pressureMultiplier;

            pressureForces[i] = Vector2.zero;
        }

        float radiusSquared = smoothingRadius * smoothingRadius;

        for (int i = 0; i < particles.Count; i++)
        {
            Vector2 position = predictedPositions[i];
            Vector2Int originCell = GetCell(position);

            for (int offsetY = -1; offsetY <= 1; offsetY++)
            {
                for (int offsetX = -1; offsetX <= 1; offsetX++)
                {
                    Vector2Int neighbourCell = originCell + new Vector2Int(offsetX, offsetY);


                    if (!spatialGrid.TryGetValue(neighbourCell, out List<int> cellParticles))
                    {
                        continue;
                    }

                    foreach (int j in cellParticles)
                    {
                        if (j <= i)
                        {
                            continue;
                        }
                        Vector2 offset = predictedPositions[j] - predictedPositions[i];
                        float distanceSquared = offset.sqrMagnitude;
                        float distance = Mathf.Sqrt(distanceSquared);
                        Vector2 directionToNeighbour = offset / distance;
                        if (distanceSquared >= radiusSquared || distanceSquared < 0.000001f)
                        {
                            continue;
                        }
                        float sharedPressure = (pressures[i] + pressures[j]) / 2f;
                        float derivative = DensityKernelDerivative(distance);
                        Vector2 force = directionToNeighbour * (derivative * sharedPressure);
                        Vector2 viscoscity = - viscosity * (velocity[j] - velocity[i]) * DensityKernelDerivative(distance) / densities[j];
                        force += viscoscity;
                        pressureForces[i] += force;
                        pressureForces[j] -= force;
                    }
                }
            }
        }
    }
    private float DensityKernel(float distance)
    {
        if (distance >= smoothingRadius)
        {
            return 0f;
        }

        float q = 1f - distance / smoothingRadius;
        return q * q;
    }

    private float DensityKernelDerivative(float distance)
    {
        if (distance >= smoothingRadius)
        {
            return 0f;
        }

        float q = 1f - distance / smoothingRadius;
        return -2f * q / smoothingRadius;
    }


    private void CalculateSurfaceTensionFields()
    {
        for (int i = 0; i < particles.Count; i++)
        {
            colorFieldNormals[i] = Vector2.zero;
            colorFieldLaplacians[i] = 0f;
        }

        float radiusSquared = smoothingRadius * smoothingRadius;

        for (int i = 0; i < particles.Count; i++)
        {
            Vector2 position = predictedPositions[i];
            Vector2Int originCell = GetCell(position);

            for (int offsetY = -1; offsetY <= 1; offsetY++)
            {
                for (int offsetX = -1; offsetX <= 1; offsetX++)
                {
                    Vector2Int neighbourCell = originCell + new Vector2Int(offsetX, offsetY);


                    if (!spatialGrid.TryGetValue(neighbourCell, out List<int> cellParticles))
                    {
                        continue;
                    }


                    foreach (int j in cellParticles)
                    {
                        Vector2 offset = predictedPositions[j] - predictedPositions[i];
                        float distanceSquared = offset.sqrMagnitude;

                        if (distanceSquared >= radiusSquared || distanceSquared < 0.000001f)
                        {
                            continue;
                        }

                        
                        Vector2 gradIJ = Poly6Gradient(offset, distanceSquared, smoothingRadius) / densities[j];
                        Vector2 gradJI = Poly6Gradient(-offset, distanceSquared, smoothingRadius) / densities[i];
                        colorFieldNormals[i] += gradIJ;
                        colorFieldNormals[j] += gradJI;

                        float lap = Poly6Laplacian(distanceSquared, smoothingRadius);
                        colorFieldLaplacians[i] += lap / densities[j];
                        colorFieldLaplacians[j] += lap / densities[i];
                    }
                }
            }
        }
    }

    private void CalculateSurfaceTensionForces()
    {
        for (int i = 0; i < particles.Count; i++)
        {
            surfaceTensionForces[i] = Vector2.zero;

            float nMagnitude = colorFieldNormals[i].magnitude;
            if (nMagnitude < surfaceTensionThreshold)
            {
                continue; 
            }

            Vector2 nHat = colorFieldNormals[i] / nMagnitude;
            surfaceTensionForces[i] = -surfaceTensionCoefficient * colorFieldLaplacians[i] * nHat;
        }
    }

    private float Poly6Kernel(float distanceSquared, float h)
    {
        if (distanceSquared >= h * h) return 0f;
        float diff = h * h - distanceSquared;
        return (4f / (Mathf.PI * Mathf.Pow(h, 8))) * diff * diff * diff;
    }

    private Vector2 Poly6Gradient(Vector2 offset, float distanceSquared, float h)
    {
        if (distanceSquared >= h * h) return Vector2.zero;
        float diff = h * h - distanceSquared;
        float coeff = -24f / (Mathf.PI * Mathf.Pow(h, 8));
        return coeff * diff * diff * offset;
    }

    private float Poly6Laplacian(float distanceSquared, float h)
    {
        if (distanceSquared >= h * h) return 0f;
        float diff = h * h - distanceSquared;
        float coeff = -48f / (Mathf.PI * Mathf.Pow(h, 8));
        return coeff * diff * (h * h - 3f * distanceSquared);
    }
}

  
