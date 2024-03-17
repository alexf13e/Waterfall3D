
#ifndef SHADERPROGRAM_H
#define SHADERPROGRAM_H

#include <string>

class ShaderProgram
{
	unsigned int id;

public:
	unsigned int getID() const { return id; }

	bool init(const std::string& vertDir, const std::string& fragDir);
	void destroy();
};

#endif // !SHADERPROGRAM_H