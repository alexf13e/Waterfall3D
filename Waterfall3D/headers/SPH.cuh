
#ifndef SPH2DCUDA_H
#define SPH2DCUDA_H

#include <vector>

#include "DebugOptions.h"
#include "glm/glm.hpp"
#include <vector_types.h>
#include "cuda_runtime.h"

#include "Boundary.cuh"

enum SmoothingKernelType
{
	SPIKY, SPIKY_GRADIENT, VISCOSITY
};

struct SPHConfiguration
{
	float timeStep;

	int numParticles;
	int numParticleDistances;
	float particleMass;

	float gravity;
	float viscosity;
	float smoothingRadius;
	float stiffnessConstant;
	float restDensity;
	float boundaryCollisionDamping;

	SmoothingKernelType kernelPressure, kernelPressureGradient, kernelViscosity;
};

struct SPHSimulationData
{
	unsigned int gl_positionsVBO, gl_positionsVAO, gl_predictedPositionsVBO, gl_predictedPositionsVAO;
	cudaGraphicsResource* cuda_positionsBuffer, * cuda_predictedPositionsBuffer;
	
	glm::vec3* d_positions, * d_velocities, * d_accelerations, * d_predictedPositions;
	float* d_densities, * d_distances;

	Boundary* h_worldBoundaries, * d_worldBoundaries;
	int worldBoundaryCount;
};

//https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/particles/doc/particles.pdf
struct UniformGridSettings
{
	float cellSize;
	int dimCells, numCells, numLinesToDraw;
};

struct UniformGridData
{
	int* d_cellIDs, * d_particleIDs, * d_cellStarts; //items at same index in cellIDs and particleIDs are paired
};

class UniformGrid
{
	UniformGridSettings settings;
	UniformGridData data;

	bool initialised;

public:
	const UniformGridSettings getSettings() const { return settings; }
	const UniformGridData getData() const { return data; }

	void setNumLinesToDraw(int num) { settings.numLinesToDraw = num; }

	~UniformGrid();

	bool init(const float& smoothingRadius, const float& simRegionSize, const int& numParticles);
	void destroy();

	void update(const SPHSimulationData& simData, const SPHConfiguration& simSettings);
};

class SPHSolver
{
	SPHConfiguration simSettings;
	SPHSimulationData simData;
	UniformGrid* uniformGrid;

	float timeElapsed;
	bool initialised = false;

public:
	const Boundary* getWorldBoundaries() const { return simData.h_worldBoundaries; }

	const float getTimeElapsed() const { return timeElapsed; }

	const SPHConfiguration& getSettings() const { return simSettings; }
	const SPHSimulationData& getSimData() const { return simData; }
	UniformGrid& getUniformGrid() const { return *uniformGrid; }  //has be non-const so that renderer can update numLinesToDraw

	~SPHSolver();

	bool init(const SPHConfiguration& settings, const float& simRegionSize);
	void destroy();

	bool update(int iterations);
	bool UGUpdate(int iterations);

	void setInitialParticlePositions(const float spacing);
	void setWorldBoundaries(Boundary* boundaries, int count);
	/*void userInteractParticles(const glm::vec2& attractionPoint, const float attractionRadius,
		const float attractionVelocity);*/

	void mapCudaResources();
	void unmapCudaResources();
};


#endif // !SPH2DCUDA_H