
#version 460 core

layout (location = 0) in vec4 sampleData;

uniform vec4 coldColour, hotColour;

out vec4 vertColour;

void main()
{
    gl_Position = vec4(sampleData.x, sampleData.y, 0.0f, 1.0f); //sample position is already in normalised screen space

    float value = sampleData.z;
    float density = sampleData.w;
    
    vec3 rgb = mix(coldColour, hotColour, density / 15.0f).xyz;
    //vec3 rgb = vec3(0.15f, 0.25f, 0.8f);
    float opacity = value;

    vertColour = vec4(rgb, opacity);
}

