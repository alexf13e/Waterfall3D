#include "ShaderProgram.h"

#include <iostream>
#include <sstream>
#include <fstream>

#include "glad/glad.h"


void checkAndPrintGLerror()
{
    GLenum errNum = glGetError();
    if (errNum != GL_NO_ERROR)
    {
        std::string error;
        switch (errNum)
        {
        case GL_INVALID_ENUM:                  error = "INVALID_ENUM"; break;
        case GL_INVALID_VALUE:                 error = "INVALID_VALUE"; break;
        case GL_INVALID_OPERATION:             error = "INVALID_OPERATION"; break;
        case GL_STACK_OVERFLOW:                error = "STACK_OVERFLOW"; break;
        case GL_STACK_UNDERFLOW:               error = "STACK_UNDERFLOW"; break;
        case GL_OUT_OF_MEMORY:                 error = "OUT_OF_MEMORY"; break;
        case GL_INVALID_FRAMEBUFFER_OPERATION: error = "INVALID_FRAMEBUFFER_OPERATION"; break;
        }

        std::cout << error << std::endl;
    }
}

bool ShaderProgram::init(const std::string& vertDir, const std::string& fragDir)
{
    //read in text from shader files
    std::stringstream tempStream;
    std::string tempLine;

    std::ifstream vertFile(vertDir);
    while (std::getline(vertFile, tempLine))
    {
        tempStream << tempLine << std::endl;
    }
    vertFile.close();
    std::string vertShaderText = tempStream.str();
    const char* vertShaderChars = vertShaderText.c_str();

    tempStream.str(std::string()); //clear line reading stream ready for next file

    std::ifstream fragFile(fragDir);
    while (std::getline(fragFile, tempLine))
    {
        tempStream << tempLine << std::endl;
    }
    fragFile.close();
    std::string fragShaderText = tempStream.str();
    const char* fragShaderChars = fragShaderText.c_str();


    //create shaders in opengl
    unsigned int vertShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertShader, 1, &vertShaderChars, NULL);
    glCompileShader(vertShader);

    int success;
    char infoLog[512];
    infoLog[0] = '\0'; //prevent printing garbage if infolog isnt overwritten
    glGetShaderiv(vertShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertShader, 512, NULL, infoLog);
        std::cerr << "vertex shader error: compilation failed\n" << infoLog << std::endl;
        glDeleteShader(vertShader);
        return false;
    }

    unsigned int fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragShader, 1, &fragShaderChars, NULL);
    glCompileShader(fragShader);

    glGetShaderiv(fragShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragShader, 512, NULL, infoLog);
        std::cerr << "fragment shader error: compilation failed\n" << infoLog << std::endl;
        glDeleteShader(vertShader);
        glDeleteShader(fragShader);
        return false;
    }

    this->id = glCreateProgram();
    glAttachShader(this->id, vertShader);
    glAttachShader(this->id, fragShader);
    glLinkProgram(this->id);

    glGetProgramiv(this->id, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(this->id, 512, NULL, infoLog);
        std::cerr << "shader program error: link failed\n" << infoLog << std::endl;
        glDeleteShader(vertShader);
        glDeleteShader(fragShader);
        glDeleteProgram(this->id);
        return false;
    }


    //individual shaders no longer needed after linked with shader program
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    return true;
}

void ShaderProgram::destroy()
{
    glDeleteProgram(this->id);
}
