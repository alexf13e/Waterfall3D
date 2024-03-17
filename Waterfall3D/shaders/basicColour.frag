
#version 460 core

out vec4 outColour;

uniform vec4 colour = vec4(1.0f, 1.0f, 1.0f, 1.0f);

void main()
{
    float depth = gl_FragCoord.z / gl_FragCoord.w;
    float nearCap = 0.5f;
    float farCap = 20.0f;
    float farMul = 0.2f;
    float depthMul = 1.0f;

    if (depth < nearCap) depthMul = 1.0f;
    else if (depth < farCap)
    {
        depthMul = 1.0f - (depth - nearCap) / (farCap - nearCap) * (1.0f - farMul);
    }
    else depthMul = farMul;

    outColour = vec4(colour.xyz * depthMul, colour.w);
}

