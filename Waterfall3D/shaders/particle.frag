
#version 460 core

in vec2 texCoord;

layout (std430) buffer ParticlePositions
{
    vec2 d_positions[];
};

uniform int numParticles;
uniform vec3 coldColour, hotColour;
uniform mat4 matView, matViewInv;

out vec4 outColour;

void main()
{
    //get uv in world space, rather than every particle pos in uv space
    vec2 worldCoord = (matViewInv * vec4(texCoord, 0.0f, 1.0f)).xy;

    float circleRadius = 0.2;
    bool inCircle = false;
    int particleHitIndex = 0;
    
    for (int i = 0; i < numParticles; i++)
    {
        inCircle = inCircle || dot(d_positions[i] - worldCoord, d_positions[i] - worldCoord) < circleRadius * circleRadius;
        if (inCircle)
        {
            particleHitIndex = i;
            break;
        }
    }
    
    vec4 bgColour = vec4(0.1f, 0.1f, 0.1f, 1.0f);
    vec4 particleColour = vec4(coldColour, 1.0f);
    outColour = inCircle ? particleColour : bgColour;

    //int x = int(floor(gl_FragCoord.x / 30));
    //int y = int(floor(gl_FragCoord.y / 30));
    //
    //if (x < 4 && y < 4)
    //{
    //    outColour = vec4(abs(fsq_viewMatrix[y][x]), 0.0f, 0.0f, 1.0f);
    //}
    //else
    //{
    //    outColour = vec4(0.0f);
    //}
}

