
#include "SPH.cuh"

#include <iostream>
#include <chrono>
#include <string>

#include "glad/glad.h"

#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "thrust/device_ptr.h"
#include "thrust/sort.h"

#include "glm/gtc/constants.hpp"
#include "CUDAFunctions.cuh"

UniformGrid::~UniformGrid()
{
	destroy();
}

bool UniformGrid::init(const float& smoothingRadius, const float& simRegionSize, const int& numParticles)
{
	initialised = false;

	settings.cellSize = 2 * smoothingRadius;
	settings.dimCells = ceil(simRegionSize / settings.cellSize) + 2;
	settings.numCells = settings.dimCells * settings.dimCells * settings.dimCells;

	cudaMalloc((void**)&data.d_cellIDs, sizeof(int) * numParticles);
	cudaMalloc((void**)&data.d_particleIDs, sizeof(int) * numParticles);
	cudaMalloc((void**)&data.d_cellStarts, sizeof(int) * settings.numCells);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Uniform grid init: error during cuda memory allocation: " << cudaGetErrorName(err) << std::endl;
		return false;
	}

	initialised = true;

	return true;
}

void UniformGrid::destroy()
{
	cudaFree(data.d_cellIDs);
	cudaFree(data.d_particleIDs);
	cudaFree(data.d_cellStarts);

	initialised = false;
}

void UniformGrid::update(const SPHSimulationData& simData, const SPHConfiguration& simSettings, bool enableTiming,
	std::vector<std::pair<std::string, float>>& timingValues)
{
	if (!initialised) return;

	int blockSize = glm::min(simSettings.numParticles, 1024);
	int numBlocks = (simSettings.numParticles - 1) / blockSize + 1;

	//update which cells contain which particles
	if (enableTiming)
	{
		std::chrono::steady_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		CUDAKernels::UGUpdateCellParticles << <numBlocks, blockSize >> > (simData, simSettings, this->data, this->settings);
		cudaDeviceSynchronize();
		std::chrono::steady_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		float dt = (t2 - t1).count() * 1e-9;
		timingValues.push_back({ "UGUpdateCellParticles", dt });

		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error in UGUpdateCellParticles: " << cudaGetErrorName(err) << std::endl;
		}

		t1 = std::chrono::high_resolution_clock::now();
		thrust::sort_by_key(thrust::device_ptr<int>(data.d_cellIDs),
			thrust::device_ptr<int>(data.d_cellIDs + simSettings.numParticles),
			thrust::device_ptr<int>(data.d_particleIDs));
		cudaDeviceSynchronize();
		t2 = std::chrono::high_resolution_clock::now();
		dt = (t2 - t1).count() * 1e-9;
		timingValues.push_back({ "UGSort", dt });

		t1 = std::chrono::high_resolution_clock::now();
		cudaMemset(data.d_cellStarts, -1, sizeof(int) * settings.numCells);
		CUDAKernels::UGUpdateCellStarts << <numBlocks, blockSize >> > (simSettings, data);
		cudaDeviceSynchronize();
		t2 = std::chrono::high_resolution_clock::now();
		dt = (t2 - t1).count() * 1e-9;
		timingValues.push_back({ "UGUpdateCellStarts", dt });

		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error in UGUpdateCellStarts: " << cudaGetErrorName(err) << std::endl;
		}
	}
	else
	{
		//update cellIDs for new particle positions (and reset particleIDs)
		CUDAKernels::UGUpdateCellParticles << <numBlocks, blockSize >> > (simData, simSettings, this->data, this->settings);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error in UGUpdateCellParticles: " << cudaGetErrorName(err) << std::endl;
		}

		//cellIDs[i] contains the index of the cell which particle with index i is in
		//particleIDs[i] contains i, just as a list of the all particle IDs, which will be sorted to maintain the
		//pairing with cellIDs when cellIDs is sorted to be ascending

		//sort particles by which cell they are in
		//cellID is key, particleID is the value
		//both keys and values will be sorted
		thrust::sort_by_key(thrust::device_ptr<int>(data.d_cellIDs),
			thrust::device_ptr<int>(data.d_cellIDs + simSettings.numParticles),
			thrust::device_ptr<int>(data.d_particleIDs));

		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error in UGUpdateCellParticles: " << cudaGetErrorName(err) << std::endl;
		}

		//update cellStarts
		//need to be initialised as having no cells, then ones which do have cells will be overwritten
		cudaMemset(data.d_cellStarts, -1, sizeof(int) * settings.numCells);

		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error in UGUpdateCellParticles: " << cudaGetErrorName(err) << std::endl;
		}

		CUDAKernels::UGUpdateCellStarts << <numBlocks, blockSize >> > (simSettings, data);
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error in UGUpdateCellStarts: " << cudaGetErrorName(err) << std::endl;
		}
	}

	//cellStarts[i] contains the index for the first particle in particleIDs which is in cell with id i
	//if no particles in cell i, then it contains -1
}


SPHSolver::~SPHSolver()
{
	destroy();
}

bool SPHSolver::init(const SPHConfiguration& settings, const float& simRegionSize)
{
	if (initialised)
	{
		std::cerr << "attempted to initialise solver which was already initialised" << std::endl;
		return false;
	}

	simSettings = settings;

	//placeholder arrays initialised to 0
	glm::vec3* emptyDataVec3 = new glm::vec3[simSettings.numParticles]();
	float* emptyDataFloat = new float[simSettings.numParticles]();

	//triangular array of distances between pairs of particles, size follows equation 0.5n(n - 1)
	simSettings.numParticleDistances = 0.5 * simSettings.numParticles * (simSettings.numParticles - 1);
	float* emptyDistances = new float[simSettings.numParticleDistances]();

	//create simulation data on GPU
	//particle positions need to be accessible by opengl for rendering, requiring them to be created in openGL and mapped to cuda
	glGenVertexArrays(1, &simData.gl_positionsVAO);
	glBindVertexArray(simData.gl_positionsVAO);
	glGenBuffers(1, &simData.gl_positionsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, simData.gl_positionsVBO);
	glBufferData(GL_ARRAY_BUFFER, simSettings.numParticles * sizeof(glm::vec3), emptyDataVec3, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);
	glEnableVertexAttribArray(0);

	glGenVertexArrays(1, &simData.gl_predictedPositionsVAO);
	glBindVertexArray(simData.gl_predictedPositionsVAO);
	glGenBuffers(1, &simData.gl_predictedPositionsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, simData.gl_predictedPositionsVBO);
	glBufferData(GL_ARRAY_BUFFER, simSettings.numParticles * sizeof(glm::vec3), emptyDataVec3, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);
	glEnableVertexAttribArray(0);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	//tell cuda where the buffer is
	cudaGraphicsGLRegisterBuffer(&simData.cuda_positionsBuffer, simData.gl_positionsVBO, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&simData.cuda_predictedPositionsBuffer, simData.gl_predictedPositionsVBO, cudaGraphicsMapFlagsWriteDiscard);

	//other data can be simply stored only for access by cuda
	cudaMalloc((void**)&simData.d_velocities, sizeof(glm::vec3) * simSettings.numParticles);
	cudaMalloc((void**)&simData.d_accelerations, sizeof(glm::vec3) * simSettings.numParticles);
	cudaMalloc((void**)&simData.d_densities, sizeof(float) * simSettings.numParticles);
	cudaMalloc((void**)&simData.d_distances, sizeof(float) * simSettings.numParticleDistances);

	cudaMemcpy(simData.d_velocities, emptyDataVec3, sizeof(glm::vec3) * simSettings.numParticles, cudaMemcpyHostToDevice);
	cudaMemcpy(simData.d_accelerations, emptyDataVec3, sizeof(glm::vec3) * simSettings.numParticles, cudaMemcpyHostToDevice);
	cudaMemcpy(simData.d_densities, emptyDataFloat, sizeof(float) * simSettings.numParticles, cudaMemcpyHostToDevice);
	cudaMemcpy(simData.d_distances, emptyDistances, sizeof(float) * simSettings.numParticleDistances, cudaMemcpyHostToDevice);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "sph solver init: error during cuda memory allocation: " << cudaGetErrorName(err) << std::endl;
		return false;
	}

	uniformGrid = new UniformGrid();
	if (uniformGrid->init(simSettings.smoothingRadius, simRegionSize, simSettings.numParticles) == false) return false;

	timeElapsed = 0.0f;
	initialised = true;

	return true;
}

void SPHSolver::destroy()
{
	if (initialised)
	{
		glDeleteBuffers(1, &simData.gl_positionsVBO);
		glDeleteBuffers(1, &simData.gl_predictedPositionsVBO);
		glDeleteVertexArrays(1, &simData.gl_positionsVAO);
		glDeleteVertexArrays(1, &simData.gl_predictedPositionsVAO);

		cudaFree(simData.d_velocities);
		cudaFree(simData.d_accelerations);
		cudaFree(simData.d_densities);
		cudaFree(simData.d_distances);

		//uniformGrid deleted automatically when solver class is

		initialised = false;
	}
}

bool SPHSolver::update(int iterations, bool enableTiming, std::vector<std::pair<std::string, float>>& timingValues)
{
	if (!initialised)
	{
		std::cerr << "attempted to update before intialisation" << std::endl;
		return false;
	}

	int blockSize = glm::min(simSettings.numParticles, 1024);
	int numBlocks = (simSettings.numParticles - 1) / blockSize + 1;

	mapCudaResources(); //to use particle position data which is shared with opengl
	if (enableTiming)
	{

		while (iterations > 0)
		{
			std::chrono::steady_clock::time_point t1 = std::chrono::high_resolution_clock::now();
			CUDAKernels::updatePredictedParticlePositions << <numBlocks, blockSize >> > (simData, simSettings);
			cudaDeviceSynchronize();
			std::chrono::steady_clock::time_point t2 = std::chrono::high_resolution_clock::now();
			float dt = (t2 - t1).count() * 1e-9;
			timingValues.push_back({ "updatePredictedParticlePositions", dt });

			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess)
			{
				std::cerr << "CUDA error in updatePredictedParticlePositions: " << cudaGetErrorName(err) << std::endl;
			}

			t1 = std::chrono::high_resolution_clock::now();
			CUDAKernels::calculateInterParticleValues << <numBlocks, blockSize >> > (simData, simSettings);
			cudaDeviceSynchronize();
			t2 = std::chrono::high_resolution_clock::now();
			dt = (t2 - t1).count() * 1e-9;
			timingValues.push_back({ "calculateInterParticleValues", dt });

			err = cudaGetLastError();
			if (err != cudaSuccess)
			{
				std::cerr << "CUDA error in calculateInterParticleValues: " << cudaGetErrorName(err) << std::endl;
			}

			t1 = std::chrono::high_resolution_clock::now();
			CUDAKernels::processTimeStep << <numBlocks, blockSize >> > (simData, simSettings);
			cudaDeviceSynchronize();
			t2 = std::chrono::high_resolution_clock::now();
			dt = (t2 - t1).count() * 1e-9;
			timingValues.push_back({ "processTimeStep", dt });

			err = cudaGetLastError();
			if (err != cudaSuccess)
			{
				std::cerr << "CUDA error in processTimeStep: " << cudaGetErrorName(err) << std::endl;
			}

			iterations--;
		}
	}
	else
	{
		while (iterations > 0)
		{
			CUDAKernels::updatePredictedParticlePositions << <numBlocks, blockSize >> > (simData, simSettings);
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess)
			{
				std::cerr << "CUDA error in updatePredictedParticlePositions: " << cudaGetErrorName(err) << std::endl;
			}

			CUDAKernels::calculateInterParticleValues << <numBlocks, blockSize >> > (simData, simSettings);
			err = cudaGetLastError();
			if (err != cudaSuccess)
			{
				std::cerr << "CUDA error in calculateInterParticleValues: " << cudaGetErrorName(err) << std::endl;
			}

			CUDAKernels::processTimeStep << <numBlocks, blockSize >> > (simData, simSettings);
			err = cudaGetLastError();
			if (err != cudaSuccess)
			{
				std::cerr << "CUDA error in processTimeStep: " << cudaGetErrorName(err) << std::endl;
			}

			iterations--;
		}
	}
	unmapCudaResources(); //must be unmapped before being used by opengl (or else)


	timeElapsed += simSettings.timeStep;

	return true;
}

bool SPHSolver::UGUpdate(int iterations, bool enableTiming, std::vector<std::pair<std::string, float>>& timingValues)
{
	if (!initialised)
	{
		std::cerr << "attempted to update before intialisation" << std::endl;
		return false;
	}

	int blockSize = glm::min(simSettings.numParticles, 1024);
	int numBlocks = (simSettings.numParticles - 1) / blockSize + 1;

	const UniformGridSettings& ugSettings = uniformGrid->getSettings();
	const UniformGridData& ugData = uniformGrid->getData();

	mapCudaResources(); //to use particle position data which is shared with opengl
	if (enableTiming)
	{
		while (iterations > 0)
		{
			std::chrono::steady_clock::time_point t1 = std::chrono::high_resolution_clock::now();
			CUDAKernels::updatePredictedParticlePositions << <numBlocks, blockSize >> > (simData, simSettings);
			cudaDeviceSynchronize();
			std::chrono::steady_clock::time_point t2 = std::chrono::high_resolution_clock::now();
			float dt = (t2 - t1).count() * 1e-9;
			timingValues.push_back({ "updatePredictedParticlePositions", dt });

			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess)
			{
				std::cerr << "CUDA error in updatePredictedParticlePositions: " << cudaGetErrorName(err) << std::endl;
			}

			//predicted positions should be set before being used by uniform grid
			uniformGrid->update(simData, simSettings, enableTiming, timingValues);

			t1 = std::chrono::high_resolution_clock::now();
			CUDAKernels::UGCalculateInterParticleValues << <numBlocks, blockSize >> > (simData, simSettings, ugData, ugSettings);
			cudaDeviceSynchronize();
			t2 = std::chrono::high_resolution_clock::now();
			dt = (t2 - t1).count() * 1e-9;
			timingValues.push_back({ "UGCalculateInterParticleValues", dt });

			err = cudaGetLastError();
			if (err != cudaSuccess)
			{
				std::cerr << "CUDA error in UGCalculateInterParticleValues: " << cudaGetErrorName(err) << std::endl;
			}

			t1 = std::chrono::high_resolution_clock::now();
			CUDAKernels::UGProcessTimeStep << <numBlocks, blockSize >> > (simData, simSettings, ugData, ugSettings);
			cudaDeviceSynchronize();
			t2 = std::chrono::high_resolution_clock::now();
			dt = (t2 - t1).count() * 1e-9;
			timingValues.push_back({ "UGProcessTimeStep", dt });

			err = cudaGetLastError();
			if (err != cudaSuccess)
			{
				std::cerr << "CUDA error in UGProcessTimeStep: " << cudaGetErrorName(err) << std::endl;
			}

			iterations--;
		}
	}
	else
	{
		while (iterations > 0)
		{
			CUDAKernels::updatePredictedParticlePositions << <numBlocks, blockSize >> > (simData, simSettings);
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess)
			{
				std::cerr << "CUDA error in updatePredictedParticlePositions: " << cudaGetErrorName(err) << std::endl;
			}

			uniformGrid->update(simData, simSettings, enableTiming, timingValues);

			CUDAKernels::UGCalculateInterParticleValues << <numBlocks, blockSize >> > (simData, simSettings, ugData, ugSettings);
			err = cudaGetLastError();
			if (err != cudaSuccess)
			{
				std::cerr << "CUDA error in UGCalculateInterParticleValues: " << cudaGetErrorName(err) << std::endl;
			}

			CUDAKernels::UGProcessTimeStep << <numBlocks, blockSize >> > (simData, simSettings, ugData, ugSettings);
			err = cudaGetLastError();
			if (err != cudaSuccess)
			{
				std::cerr << "CUDA error in UGProcessTimeStep: " << cudaGetErrorName(err) << std::endl;
			}

			iterations--;
		}
	}
	unmapCudaResources(); //must be unmapped before being used by opengl (or else)
	

	timeElapsed += simSettings.timeStep;

	return true;
}

void SPHSolver::setInitialParticlePositions(const float spacing)
{
	if (!initialised)
	{
		std::cerr << "attempted to set initial particle positions before intialisation" << std::endl;
		return;
	}

	//generate positions in as close to a square around the origin as possible, filling space in order x, z, then y
	//find minimum width, height and depth to fit the number of particles into a cube
	int target = simSettings.numParticles;
	int w = 1;
	int d = 1;
	int h = 1;

	while (w * d * h < target)
	{
		w++; if (w * d * h >= target) break;
		d++; if (w * d * h >= target) break;
		h++;
	}

	int blockSize = glm::min(simSettings.numParticles, 1024);
	int numBlocks = (simSettings.numParticles - 1) / blockSize + 1;

	mapCudaResources();
		CUDAKernels::setInitialParticlePositions<<<numBlocks, blockSize>>>(simData.d_positions,
			simSettings.numParticles, w, h, d, spacing);
	unmapCudaResources();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error in setInitialParticlePositions: " << cudaGetErrorName(err) << std::endl;
	}
}

void SPHSolver::setWorldBoundaries(Boundary* boundaries, int count)
{
	if (simData.h_worldBoundaries != nullptr)
	{
		delete[] simData.h_worldBoundaries;
		cudaFree(simData.d_worldBoundaries);
	}

	simData.h_worldBoundaries = new Boundary[count];
	simData.worldBoundaryCount = count;
	
	memcpy(simData.h_worldBoundaries, boundaries, count * sizeof(Boundary));

	cudaMalloc((void**)&simData.d_worldBoundaries, count * sizeof(Boundary));
	cudaMemcpy(simData.d_worldBoundaries, simData.h_worldBoundaries, count * sizeof(Boundary), cudaMemcpyHostToDevice);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error in setWorldBoundaries: " << cudaGetErrorName(err) << std::endl;
	}
}

//void SPHSolver::userInteractParticles(const glm::vec2& attractionPoint, const float attractionRadius,
//	const float attractionVelocity)
//{
//	int blockSize = glm::min(simSettings.numParticles, 1024);
//	int numBlocks = (simSettings.numParticles - 1) / blockSize + 1;
//
//	mapCudaResources();
//		CUDAKernels::userInteractParticles<<<numBlocks, blockSize>>>(simData, simSettings.numParticles, attractionPoint,
//			attractionRadius, attractionVelocity);
//	unmapCudaResources();
//
//	cudaError_t err = cudaGetLastError();
//	if (err != cudaSuccess)
//	{
//		std::cerr << "CUDA error in userInteractParticles: " << cudaGetErrorName(err) << std::endl;
//	}
//}

void SPHSolver::mapCudaResources()
{
	size_t _numbytes;
	cudaGraphicsMapResources(1, &simData.cuda_positionsBuffer);
	cudaGraphicsMapResources(1, &simData.cuda_predictedPositionsBuffer);
	cudaGraphicsResourceGetMappedPointer((void**)&simData.d_positions, &_numbytes, simData.cuda_positionsBuffer);
	cudaGraphicsResourceGetMappedPointer((void**)&simData.d_predictedPositions, &_numbytes, simData.cuda_predictedPositionsBuffer);
}

void SPHSolver::unmapCudaResources()
{
	cudaGraphicsUnmapResources(1, &simData.cuda_positionsBuffer, 0);
	cudaGraphicsUnmapResources(1, &simData.cuda_predictedPositionsBuffer, 0);

	//pointers are no longer valid after unmapping, set to null just in case they are used by mistake
	simData.d_positions = nullptr;
	simData.d_predictedPositions = nullptr;
}