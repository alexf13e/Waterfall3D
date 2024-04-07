
#include "MetaballSampler.cuh"


bool MetaballSampler::init(int textureWidth, int textureHeight, float boundaryRadius, int maxIterations)
{
	if (initialised) return false;

	this->textureWidth = textureWidth;
	this->textureHeight = textureHeight;
	this->boundaryRadius = boundaryRadius;
	this->maxIterations = maxIterations;
	hitEpsilon = 0.05f;

	//create empty vao for drawing full screen triangle
	glGenVertexArrays(1, &vao_fullScreenTri);

	//create buffer for storing ray sample data
	glm::vec4* emptyVec3 = new glm::vec4[textureWidth * textureHeight]();

	glGenBuffers(1, &gl_rayDataSSBO);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, gl_rayDataSSBO);
	glBufferData(GL_SHADER_STORAGE_BUFFER, textureWidth * textureHeight * sizeof(glm::vec4), emptyVec3, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	cudaGraphicsGLRegisterBuffer(&cuda_rayDataBuffer, gl_rayDataSSBO, cudaGraphicsMapFlagsWriteDiscard);

	delete[] emptyVec3;

	initialised = true;
	return true;
}

bool MetaballSampler::updateResolution(int width, int height)
{
	if (!initialised) return false;

	textureWidth = width;
	textureHeight = height;

	glm::vec4* emptyVec3 = new glm::vec4[textureWidth * textureHeight]();

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, gl_rayDataSSBO);
	glBufferData(GL_SHADER_STORAGE_BUFFER, textureWidth * textureHeight * sizeof(glm::vec4), emptyVec3, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	delete[] emptyVec3;

	updateFragTexSize = true;
}

void MetaballSampler::destroy()
{
	glDeleteBuffers(1, &gl_rayDataSSBO);
	glDeleteVertexArrays(1, &vao_fullScreenTri);

	initialised = false;
}

void MetaballSampler::mapCudaResources()
{
	size_t _numbytes;
	cudaGraphicsMapResources(1, &cuda_rayDataBuffer);
	cudaGraphicsResourceGetMappedPointer((void**)&d_rayData, &_numbytes, cuda_rayDataBuffer);
}

void MetaballSampler::unmapCudaResources()
{
	cudaGraphicsUnmapResources(1, &cuda_rayDataBuffer, 0);
	d_rayData = nullptr;
}