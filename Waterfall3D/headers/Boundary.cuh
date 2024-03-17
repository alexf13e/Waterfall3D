
#include "glm/glm.hpp"

struct Boundary
{
	//3D plane which forces particles to be on the side pointed to by the normal
	glm::vec3 pos, norm;
};