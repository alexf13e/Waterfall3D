
#version 460 core

out vec2 texCoord;

void main()
{
    //https://stackoverflow.com/a/59739538
    vec2 vertices[3] = vec2[3](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
    gl_Position = vec4(vertices[gl_VertexID], 0.0f, 1.0f);
    texCoord = 0.5f * gl_Position.xy + vec2(0.5f); //from (-1, +1) to (0, +1) in x and y
}

