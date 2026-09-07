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
        _EnvironmentMap ("Environment Map (Optional)", Cube) = "" {}
    }

    SubShader
    {
        Tags { "RenderPipeline"="UniversalPipeline" "Queue"="Transparent" "RenderType"="Transparent" }
        Cull Front
        ZWrite Off
        ZTest LEqual
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
                float3 invDir = 1.0 / (abs(rayDirection) > 1e-6 ? rayDirection : (sign(rayDirection) * 1e-6 + 1e-6));
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
                float3 sampleStep = _VolumeBounds / 64.0;
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

            float2 GetRefractionUV(float3 worldPos, float3 refractDir, float distortStrength)
            {
                float3 offsetWorldPos = worldPos + refractDir * distortStrength;

                float4 originalClip = TransformWorldToHClip(worldPos);
                float4 offsetClip   = TransformWorldToHClip(offsetWorldPos);

                float2 originalUV = (originalClip.xy / originalClip.w) * 0.5 + 0.5;
                float2 offsetUV   = (offsetClip.xy / offsetClip.w) * 0.5 + 0.5;

                #if UNITY_UV_STARTS_AT_TOP
                originalUV.y = 1.0 - originalUV.y;
                offsetUV.y   = 1.0 - offsetUV.y;
                #endif

                return offsetUV;
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

                            float2 refractBox = RayBoxDistance(boxMin, boxMax, position + refractDir * 0.01, refractDir);
                            float maxRefractDist = refractBox.y;
                            float refractStepSize = stepSize * 2.0;
                            float fluidDist = 0.0;

                            for (int rStep = 0; rStep < 16 && fluidDist < maxRefractDist; rStep++)
                            {
                                float3 rPos = position + refractDir * (fluidDist + refractStepSize);
                                if (SampleDensity(rPos) < _SurfaceThreshold)
                                {
                                    break;
                                }
                                fluidDist += refractStepSize;
                            }
                            fluidDist = min(fluidDist, maxRefractDist);

                            float3 absorptionColor = 1.0 - _WaterColor.rgb;
                            float3 transmittance = exp(-absorptionColor * (fluidDist * _AbsorptionDensity));


                            float distortStrength = saturate(fluidDist * 0.15); 
                            float2 refractUV = GetRefractionUV(position, refractDir, distortStrength);

                            float3 sceneColor = SAMPLE_TEXTURE2D(_CameraOpaqueTexture, sampler_CameraOpaqueTexture, refractUV).rgb;
                            float3 skyColor = SampleEnvironment(refractDir, lightDir);
                            float3 backgroundLight = sceneColor;


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
