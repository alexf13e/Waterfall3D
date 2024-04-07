
#ifndef METABALL_SAMPLER_H
#define METABALL_SAMPLER_H

#include "glm/glm.hpp"
#include "glad/glad.h" //even though its not used here, it is used elsewhere and glad gets upset if cuda_gl_interop is included before it.
#include "cuda_gl_interop.h"

class MetaballSampler
{
	unsigned int vao_fullScreenTri;

	cudaGraphicsResource* cuda_rayDataBuffer;

	bool initialised = false;

public:
	unsigned int gl_rayDataSSBO;
	glm::vec4* d_rayData;
	int textureWidth, textureHeight;
	float boundaryRadius, hitEpsilon;
	bool updateFragTexSize = false;
	int maxIterations;

	unsigned int getVAOFullScreenTri() { return vao_fullScreenTri; }
	
	bool init(int textureWidth, int textureHeight, float boundaryRadius, int maxIterations);
	bool updateResolution(int width, int height);
	void destroy();
	
	void mapCudaResources();
	void unmapCudaResources();
};

#endif