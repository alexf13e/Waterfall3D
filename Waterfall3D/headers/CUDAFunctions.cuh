
#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H


#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "SPH.cuh"
#include "MetaballSampler.cuh"
#include "Renderer.cuh"

namespace CUDADeviceFunctions
{
	namespace SmoothingKernels
	{
		__device__ float getKernelValue(SmoothingKernelType kernelType, const float distance,
			const float smoothingRadius);
		__device__ float spiky(const float distance, const float smoothingRadius);
		__device__ float spikyGradient(const float distance, const float smoothingRadius);
		__device__ float viscosity(const float distance, const float smoothingRadius);
	}

	__device__ int getParticleDistanceIndex(int i1, int i2, int numParticles);

	__device__ float calculatePressure(const float density, const SPHConfiguration& simSettings);

	__device__ glm::vec3 calculateParticleAcceleration(int particleIndex, SPHSimulationData& simData,
		const SPHConfiguration& simSettings);

	__device__ void handleBoundaryCollisions(int particleIndex, SPHSimulationData simData, SPHConfiguration simSettings);

	__device__ int UGCellIDFromPosition(const glm::vec3& pos, const float& cellSize, const int& dimCells);

	__device__ glm::vec3 UGCalculateParticleAcceleration(int particleIndex, SPHSimulationData simData,
		SPHConfiguration simSettings, UniformGridData ugData, UniformGridSettings ugSettings);

	__device__ glm::vec3 getRayDirection(int threadIndex, int textureWidth, int textureHeight, float fieldOfView,
		float pitch, float yaw);
}

namespace CUDAKernels
{
	__global__ void setInitialParticlePositions(glm::vec3* particlePositions, int numParticles, const int width,
		const int height, const int depth, const float spacing);

	__global__ void updatePredictedParticlePositions(SPHSimulationData simData, SPHConfiguration simSettings);

	__global__ void calculateInterParticleValues(SPHSimulationData simData, SPHConfiguration simSettings);

	__global__ void processTimeStep(SPHSimulationData simData, SPHConfiguration simSettings);

	__global__ void UGUpdateCellParticles(SPHSimulationData simData, SPHConfiguration simSettings,
		UniformGridData ugData, UniformGridSettings ugSettings);

	__global__ void UGUpdateCellStarts(SPHConfiguration simSettings, UniformGridData ugData);

	__global__ void UGCalculateInterParticleValues(SPHSimulationData simData, SPHConfiguration simSettings,
		UniformGridData ugData, UniformGridSettings ugSettings);

	__global__ void UGProcessTimeStep(SPHSimulationData simData, SPHConfiguration simSettings, UniformGridData ugData,
		UniformGridSettings ugSettings);

	__global__ void userInteractParticles(SPHSimulationData simData, int numParticles, glm::vec2 attractionPoint,
		float attractionRadius, float attractionVelocity);

	__global__ void calculateMetaballSamples(SPHSimulationData simdata, SPHConfiguration simSettings,
		UniformGridData ugData, UniformGridSettings ugSettings, MetaballSampler mbSampler, Camera cam);
}

#endif // !CUDA_FUNCTIONS_H