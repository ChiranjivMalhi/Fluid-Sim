Shader "FluidSim/RaymarchVolume3D"
{
    Properties
    {
        _DensityVolume ("Density Volume", 3D) = "black" {}
        _SurfaceThreshold ("Surface Threshold", Float) = 2.0
        _RayStepSize ("Ray Step Size", Float) = 0.05
        _MaxRaySteps ("Max Ray Steps", Int) = 256
        _WaterColor ("Water Color", Color) = (0.08, 0.45, 0.9, 0.92)

        [Header(Optics)]
        _IndexOfRefraction ("Index of Refraction (IOR)", Range(1.0, 2.0)) = 1.333
        _ReflectionStrength ("Reflection Strength", Range(0.0, 2.0)) = 1.0
        _RefractionStrength ("Refraction Strength", Range(0.0, 2.0)) = 1.0
        _FresnelPower ("Fresnel Power", Range(1.0, 10.0)) = 5.0
        _SpecularPower ("Specular Power (Gloss)", Range(8.0, 256.0)) = 64.0
        _SunSpecular ("Sun Specular Intensity", Range(0.0, 5.0)) = 1.5
        _AbsorptionDensity ("Absorption Density", Range(0.0, 5.0)) = 1.0
        _AbsorptionCoefficients ("Absorption Coefficients (RGB)", Vector) = (1, 1, 1, 0)
        _NormalSampleDistance ("Normal Sample Distance", Float) = 0.05
        [Toggle] _InteractWithScene ("Use Scene Depth and Color", Float) = 1
        _EnvironmentMap ("Environment Map (Optional)", Cube) = "" {}
    }

    SubShader
    {
        Tags { "RenderPipeline"="UniversalPipeline" "Queue"="Transparent" "RenderType"="Transparent" }
        Cull Front
        ZWrite Off
        ZTest Always
        Blend SrcAlpha OneMinusSrcAlpha

        Pass
        {
            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex Vert
            #pragma fragment Frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            TEXTURE3D(_DensityVolume);
            SAMPLER(sampler_DensityVolume);

            TEXTURECUBE(_EnvironmentMap);
            SAMPLER(sampler_EnvironmentMap);

            TEXTURE2D(_CameraDepthTexture);
            SAMPLER(sampler_CameraDepthTexture);

            TEXTURE2D(_CameraOpaqueTexture);
            SAMPLER(sampler_CameraOpaqueTexture);

            float3 _VolumeBounds;
            float3 _VolumeCenter;
            float3 _CameraWorldPosition;
            float3 _LightDirection;
            float _SurfaceThreshold;
            float _RayStepSize;
            int _MaxRaySteps;
            float4 _WaterColor;

            float _IndexOfRefraction;
            float _ReflectionStrength;
            float _RefractionStrength;
            float _FresnelPower;
            float _SpecularPower;
            float _SunSpecular;
            float _AbsorptionDensity;
            float3 _AbsorptionCoefficients;
            float _NormalSampleDistance;
            float _InteractWithScene;
            float _HasEnvironmentMap;

            struct Attributes
            {
                float4 positionOS : POSITION;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 worldPos   : TEXCOORD0;
            };

            Varyings Vert(Attributes input)
            {
                Varyings output;
                output.positionCS = TransformObjectToHClip(input.positionOS.xyz);
                output.worldPos   = TransformObjectToWorld(input.positionOS.xyz);
                return output;
            }

            
            float2 RayBoxDistance(float3 boxMin, float3 boxMax, float3 rayOrigin, float3 rayDirection)
            {
                float3 safeDirection = float3(
                    rayDirection.x >= 0 ? max(rayDirection.x, 1e-6) : min(rayDirection.x, -1e-6),
                    rayDirection.y >= 0 ? max(rayDirection.y, 1e-6) : min(rayDirection.y, -1e-6),
                    rayDirection.z >= 0 ? max(rayDirection.z, 1e-6) : min(rayDirection.z, -1e-6));
                float3 invDir = 1.0 / safeDirection;
                float3 t0 = (boxMin - rayOrigin) * invDir;
                float3 t1 = (boxMax - rayOrigin) * invDir;
                float3 tMin = min(t0, t1);
                float3 tMax = max(t0, t1);

                float dstA = max(max(tMin.x, tMin.y), tMin.z);
                float dstB = min(min(tMax.x, tMax.y), tMax.z);

                float dstToBox = max(0.0, dstA);
                float dstInsideBox = max(0.0, dstB - dstToBox);
                return float2(dstToBox, dstInsideBox);
            }

            float SampleDensity(float3 worldPosition)
            {
                float3 uvw = (worldPosition - _VolumeCenter) / _VolumeBounds + 0.5;
                if (any(uvw < 0.0) || any(uvw > 1.0))
                {
                    return 0.0;
                }
                return SAMPLE_TEXTURE3D_LOD(_DensityVolume, sampler_DensityVolume, uvw, 0).r;
            }

           
            float3 CalculateNormal(float3 worldPosition)
            {
                uint width, height, depth;
                _DensityVolume.GetDimensions(width, height, depth);
                float3 voxelSize = _VolumeBounds / float3(width, height, depth);
                float spacing = max(max(voxelSize.x, voxelSize.y), voxelSize.z);
                float3 sampleStep = max(max(_NormalSampleDistance, spacing), 0.001).xxx;
                float dx = SampleDensity(worldPosition - float3(sampleStep.x, 0, 0)) - SampleDensity(worldPosition + float3(sampleStep.x, 0, 0));
                float dy = SampleDensity(worldPosition - float3(0, sampleStep.y, 0)) - SampleDensity(worldPosition + float3(0, sampleStep.y, 0));
                float dz = SampleDensity(worldPosition - float3(0, 0, sampleStep.z)) - SampleDensity(worldPosition + float3(0, 0, sampleStep.z));
                float3 normal = float3(dx, dy, dz);
                float len = length(normal);
                return len > 1e-5 ? normal / len : float3(0, 1, 0);
            }

            float3 SampleProceduralSky(float3 dir, float3 sunDir)
            {
                const float3 colGround = float3(0.22, 0.24, 0.28) * 0.7;
                const float3 colSkyHorizon = float3(0.85, 0.90, 0.95);
                const float3 colSkyZenith = float3(0.12, 0.42, 0.82);

                float groundToSky = smoothstep(-0.02, 0.05, dir.y);
                float zenithT = pow(saturate(dir.y), 0.4);
                float3 sky = lerp(colSkyHorizon, colSkyZenith, zenithT);
                float3 envColor = lerp(colGround, sky, groundToSky);

                float sunDot = max(0.0, dot(dir, sunDir));
                float sunDisc = pow(sunDot, 400.0) * 2.5;
                return envColor + sunDisc * float3(1.0, 0.95, 0.85);
            }

            float3 SampleEnvironment(float3 dir, float3 sunDir)
            {
                if (_HasEnvironmentMap > 0.5)
                {
                    return SAMPLE_TEXTURECUBE_LOD(_EnvironmentMap, sampler_EnvironmentMap, dir, 0).rgb;
                }
                return SampleProceduralSky(dir, sunDir);
            }

            float CalculateFresnel(float3 normal, float3 viewDir, float ior)
            {
                float n1 = 1.0; 
                float n2 = max(ior, 1.0001);
                float r0 = (n1 - n2) / (n1 + n2);
                r0 = r0 * r0;

                float cosTheta = saturate(dot(normal, viewDir));
                return r0 + (1.0 - r0) * pow(1.0 - cosTheta, _FresnelPower);
            }

            bool TraceFluidExit(float3 entry, float3 direction, float3 boxMin, float3 boxMax,
                float stepSize, out float3 exitPosition, out float opticalDepth)
            {
                float maxDistance = RayBoxDistance(boxMin, boxMax, entry, direction).y;
                float previousDistance = 0;
                float previousExcess = max(0, SampleDensity(entry) - _SurfaceThreshold);
                opticalDepth = 0;
                exitPosition = entry;

                [loop]
                for (int step = 0; step < _MaxRaySteps; step++)
                {
                    float distance = min(previousDistance + stepSize, maxDistance);
                    float density = SampleDensity(entry + direction * distance);
                    bool reachedBoundary = distance >= maxDistance;
                    if (density < _SurfaceThreshold || reachedBoundary)
                    {
                        float inside = previousDistance;
                        float outside = distance;
                        if (density < _SurfaceThreshold)
                        {
                            for (int refine = 0; refine < 4; refine++)
                            {
                                float midpoint = (inside + outside) * 0.5;
                                if (SampleDensity(entry + direction * midpoint) >= _SurfaceThreshold)
                                    inside = midpoint;
                                else
                                    outside = midpoint;
                            }
                        }
                        float exitDistance = density < _SurfaceThreshold ? (inside + outside) * 0.5 : distance;
                        opticalDepth += previousExcess * (exitDistance - previousDistance) * 0.5;
                        exitPosition = entry + direction * exitDistance;
                        return true;
                    }

                    float excess = max(0, density - _SurfaceThreshold);
                    opticalDepth += (previousExcess + excess) * 0.5 * (distance - previousDistance);
                    previousDistance = distance;
                    previousExcess = excess;
                }

                exitPosition = entry + direction * previousDistance;
                return false;
            }

            // Project the exiting ray into the opaque scene image and refine its depth estimate.
            float3 SampleRefractedScene(float3 exitPosition, float3 direction,
                float3 cameraForward, float3 lightDir)
            {
                float3 fallback = SampleEnvironment(direction, lightDir);
                if (_InteractWithScene < 0.5)
                    return fallback;

                float forwardAmount = dot(direction, cameraForward);
                if (forwardAmount <= 0.001)
                    return fallback;

                float exitEyeDepth = -TransformWorldToView(exitPosition).z;
                float distance = max(_RayStepSize, 0.01);
                float2 uv = 0;
                for (int iteration = 0; iteration < 4; iteration++)
                {
                    float3 samplePosition = exitPosition + direction * distance;
                    float4 clip = TransformWorldToHClip(samplePosition);
                    if (clip.w <= 0)
                        return fallback;
                    float4 screenPosition = ComputeScreenPos(clip);
                    uv = screenPosition.xy / screenPosition.w;
                    if (any(uv < 0) || any(uv > 1))
                        return fallback;
                    float rawDepth = SAMPLE_TEXTURE2D(_CameraDepthTexture, sampler_CameraDepthTexture, uv).r;
                    float eyeDepth = LinearEyeDepth(rawDepth, _ZBufferParams);
                    if (eyeDepth >= _ProjectionParams.z * 0.999 || eyeDepth <= exitEyeDepth)
                        return fallback;
                    distance = (eyeDepth - exitEyeDepth) / forwardAmount;
                }
                return SAMPLE_TEXTURE2D(_CameraOpaqueTexture, sampler_CameraOpaqueTexture, uv).rgb;
            }

            float4 Frag(Varyings input) : SV_Target
            {
                float3 rayOrigin = _CameraWorldPosition;
                float3 rayDirection = normalize(input.worldPos - rayOrigin);

                float3 boxMin = _VolumeCenter - _VolumeBounds * 0.5;
                float3 boxMax = _VolumeCenter + _VolumeBounds * 0.5;

                float2 boxInfo = RayBoxDistance(boxMin, boxMax, rayOrigin, rayDirection);
                float dstToBox = boxInfo.x;
                float dstInsideBox = boxInfo.y;

                float2 screenUV = input.positionCS.xy / _ScreenParams.xy;
                float rawDepth = SAMPLE_TEXTURE2D(_CameraDepthTexture, sampler_CameraDepthTexture, screenUV).r;

                float sceneEyeDepth = LinearEyeDepth(rawDepth, _ZBufferParams);

                float3 camForward = -UNITY_MATRIX_V[2].xyz; 
                float cosAngle = dot(rayDirection, camForward);
                float sceneDistAlongRay = sceneEyeDepth / max(cosAngle, 1e-4);

                float sceneDistInsideBox = sceneDistAlongRay - dstToBox;
                if (_InteractWithScene > 0.5)
                    dstInsideBox = min(dstInsideBox, max(0.0, sceneDistInsideBox));

                if (dstInsideBox <= 0.0001)
                {
                    discard;
                }

                float travelled = 0.0;
                float previousDensity = 0.0;
                float stepSize = max(_RayStepSize, 0.005);
                float3 lightDir = normalize(_LightDirection);

                [loop]
                for (int step = 0; step < _MaxRaySteps && travelled < dstInsideBox; step++)
                {
                    float3 position = rayOrigin + rayDirection * (dstToBox + travelled);
                    float density = SampleDensity(position);

                    if (density >= _SurfaceThreshold && previousDensity < _SurfaceThreshold)
                    {
                        float t0 = max(0.0, travelled - stepSize);
                        float t1 = travelled;
                        for (int refine = 0; refine < 4; refine++)
                        {
                            float tMid = (t0 + t1) * 0.5;
                            float dMid = SampleDensity(rayOrigin + rayDirection * (dstToBox + tMid));
                            if (dMid >= _SurfaceThreshold)
                                t1 = tMid;
                            else
                                t0 = tMid;
                        }
                        position = rayOrigin + rayDirection * (dstToBox + t1);

                        float3 normal = CalculateNormal(position);
                        float3 viewDir = -rayDirection;

                        float fresnel = CalculateFresnel(normal, viewDir, _IndexOfRefraction);

                        float3 reflectDir = reflect(rayDirection, normal);
                        float3 reflectionColor = SampleEnvironment(reflectDir, lightDir) * _ReflectionStrength;

                        float3 halfDir = normalize(lightDir + viewDir);
                        float NdotH = saturate(dot(normal, halfDir));
                        float specular = pow(NdotH, _SpecularPower) * _SunSpecular;
                        reflectionColor += specular * float3(1.0, 0.98, 0.9);

                        float eta = 1.0 / max(_IndexOfRefraction, 1.0001);
                        float3 refractDir = refract(rayDirection, normal, eta);

                        float3 refractionColor = float3(0, 0, 0);

                        if (dot(refractDir, refractDir) < 0.001)
                        {
                            refractionColor = reflectionColor;
                            fresnel = 1.0;
                        }
                        else
                        {
                            refractDir = normalize(refractDir);

                            float3 exitPosition;
                            float opticalDepth;
                            bool foundExit = TraceFluidExit(position, refractDir, boxMin, boxMax,
                                stepSize, exitPosition, opticalDepth);
                            float3 transmittance = exp(-max(_AbsorptionCoefficients, 0) *
                                max(1.0 - _WaterColor.rgb, 0) * opticalDepth * _AbsorptionDensity);
                            float3 backgroundLight = SampleEnvironment(refractDir, lightDir);

                            if (foundExit)
                            {
                                float3 exitNormal = CalculateNormal(exitPosition);
                                if (dot(exitNormal, refractDir) < 0)
                                    exitNormal = -exitNormal;
                                float3 outgoing = refract(refractDir, -exitNormal, max(_IndexOfRefraction, 1.0001));
                                if (dot(outgoing, outgoing) > 0.001)
                                    backgroundLight = SampleRefractedScene(exitPosition, normalize(outgoing), camForward, lightDir);
                                else
                                    backgroundLight = SampleEnvironment(reflect(refractDir, exitNormal), lightDir);
                            }

                            refractionColor = backgroundLight * transmittance + _WaterColor.rgb * (1.0 - transmittance);
                            refractionColor *= _RefractionStrength;
                        }

                        float3 finalColor = lerp(refractionColor, reflectionColor, saturate(fresnel));

                        float NdotL = saturate(dot(normal, lightDir));
                        finalColor += _WaterColor.rgb * (NdotL * 0.15);

                        float alpha = saturate(_WaterColor.a + fresnel * 0.4);
                        return float4(finalColor, alpha);
                    }

                    previousDensity = density;
                    travelled += stepSize;
                }

                discard;
                return 0;
            }
            ENDHLSL
        }
    }
}
