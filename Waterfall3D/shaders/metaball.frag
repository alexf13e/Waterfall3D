
#version 460 core

in vec2 texCoord;
out vec4 outColour;

uniform int texWidth, texHeight;

layout (std430) buffer RayData
{
    vec4 d_rayData[];
};

void main()
{
    int x = int(texCoord.x * texWidth);
    int y = int(texCoord.y * texHeight);
    int i = y * texWidth + x;
    vec3 rayCol = d_rayData[i].xyz;
    outColour = vec4(rayCol, 1.0f);
}

