
#ifndef METABALL_SAMPLER_H
#define METABALL_SAMPLER_H

#include "glm/glm.hpp"
#include "glad/glad.h" //even though its not used here, it is used elsewhere and glad gets upset if cuda_gl_interop is included before it.
#include "cuda_gl_interop.h"

struct MetaballSampler
{
	unsigned int vao_samplePoints, vbo_samplePoints, ebo_samplePoints;

	cudaGraphicsResource* cuda_samplePointsBuffer;
	glm::vec4* d_sampleData;

	int numSamplePoints, numTriIndices;
	float r1, r0;
};

#endif