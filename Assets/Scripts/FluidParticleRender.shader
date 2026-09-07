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
       
        Tags { "RenderPipeline"="UniversalPipeline" "RenderType"="Transparent" "Queue"="Transparent" }
        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off
        Cull Off

        Pass
        {
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            StructuredBuffer<float2> _Positions;
            StructuredBuffer<float> _Densities;
            StructuredBuffer<float2> _Velocities;
            float4 _Color;

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
                float2 worldPos2D = _Positions[v.instanceID] + v.vertex.xy;
                float3 worldPos = float3(worldPos2D, 0);

                float density = _Densities[v.instanceID];
                o.densityRatio = saturate((density - _RestDensity) / _RestDensity * _DensityScale);

                float2 velocity = _Velocities[v.instanceID];
                o.velocityRatio = saturate(length(velocity) / _MaxSpeed);

                o.pos = TransformWorldToHClip(worldPos);
                o.uv = v.uv;
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                float2 centered = i.uv - 0.5;
                float dist = length(centered) * 2;
                float alpha = smoothstep(1.0, 0.9, dist);
                if (alpha <= 0.001)
                {
                    discard;
                }

                float4 runtimeColor = lerp(_BaseColor, _HighColor, i.velocityRatio);

                return float4(runtimeColor.rgb, runtimeColor.a * alpha);
            }
            ENDHLSL
        }
    }
}
