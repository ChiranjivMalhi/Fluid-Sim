Shader "FluidSim/ParticleInstanced"
{
    Properties
    {
        _Color ("Color", Color) = (0.55, 0.9, 1, 1)
        _RestDensity ("Rest Density", Float) = 1000.0
        _DensityScale ("Density Intensity Scale", Float) = 1.5
        _BaseColor ("Base Color", Color) = (0.55, 0.9, 1, 1)
        _HighColor ("High Density Color", Color) = (0.2, 0.5, 1, 1)
        _MaxSpeed ("Max Speed (for color scaling)", Float) = 8.0
    }
    SubShader
    {
        Tags { "RenderPipeline"="UniversalPipeline" "RenderType"="Opaque" "Queue"="Geometry" }
        Blend Off
        ZWrite On
        Cull Back

        Pass
        {
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            StructuredBuffer<float3> _Positions;
            StructuredBuffer<float> _Densities;
            StructuredBuffer<float3> _Velocities;
            float4 _Color;

            float3 _SimulationCenter;
            float _RestDensity;
            float _DensityScale;
            float4 _BaseColor;
            float4 _HighColor;
            float _MaxSpeed;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
                uint instanceID : SV_InstanceID;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
                float densityRatio : TEXCOORD1;
                float velocityRatio : TEXCOORD2;
            };

            v2f vert(appdata v)
            {
                v2f o;
                float3 worldPos3D = _SimulationCenter + _Positions[v.instanceID] + v.vertex.xyz;

                float density = _Densities[v.instanceID];
                o.densityRatio = saturate((density - _RestDensity) / max(_RestDensity, 0.001) * _DensityScale);

                float3 velocity = _Velocities[v.instanceID];
                o.velocityRatio = saturate(length(velocity) / max(_MaxSpeed, 0.001));

                o.pos = TransformWorldToHClip(worldPos3D);
                o.uv = v.uv;
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                float4 runtimeColor = lerp(_BaseColor, _HighColor, i.velocityRatio);
                return float4(runtimeColor.rgb, 1.0);
            }
            ENDHLSL
        }
    }
}
