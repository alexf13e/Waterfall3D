
#version 460 core

layout (location = 0) in vec3 vertex;

uniform mat4 matProjView;

void main()
{
    gl_Position = matProjView * vec4(vertex, 1.0f);
    gl_PointSize = 8;
}

