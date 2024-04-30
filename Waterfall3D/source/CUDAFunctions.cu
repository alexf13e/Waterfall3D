
#include "CUDAFunctions.cuh"

#include <device_launch_parameters.h>
#include "glm/gtc/constants.hpp"
#include "glm/gtx/vector_angle.hpp"


namespace CUDADeviceFunctions
{
	namespace SmoothingKernels
	{
		__device__ float getKernelValue(SmoothingKernelType kernelType, const float distance,
			const float smoothingRadius)
		{
			switch (kernelType)
			{
			case SmoothingKernelType::SPIKY:
				return spiky(distance, smoothingRadius);

			case SmoothingKernelType::SPIKY_GRADIENT:
				return spikyGradient(distance, smoothingRadius);

			case SmoothingKernelType::VISCOSITY:
				return viscosity(distance, smoothingRadius);

			default:
				return 0.0f;
			}
		}

		//https://matthias-research.github.io/pages/publications/sca03.pdf section 3.5
		__device__ float spiky(const float distance, const float smoothingRadius)
		{
			if (distance > smoothingRadius) return 0;

			float num = 15.0f * glm::pow(smoothingRadius - distance, 3);
			float den = glm::pi<float>() * glm::pow(smoothingRadius, 6);

			return num / den;
		}

		__device__ float spikyGradient(const float distance, const float smoothingRadius)
		{
			if (distance > smoothingRadius) return 0;

			float num = 45.0f * glm::pow(smoothingRadius - distance, 2);
			float den = glm::pi<float>() * glm::pow(smoothingRadius, 6);

			return num / den;
		}

		__device__ float viscosity(const float distance, const float smoothingRadius)
		{
			if (distance > smoothingRadius) return 0;

			/*float part1 = 15.0f / (2.0f * pi * glm::pow(smoothingRadius, 3));
			float part2 = -glm::pow(distance, 3) / (2.0f * glm::pow(smoothingRadius, 3));
			float part3 = glm::pow(distance, 2) / glm::pow(smoothingRadius, 2);
			float part4 = smoothingRadius / (2.0f * distance);

			return part1 * (part2 + part3 + part4 - 1);*/

			return 45.0f / (glm::pi<float>() * glm::pow(smoothingRadius, 6)) * (smoothingRadius - distance);
		}
	}

	__device__ int getParticleDistanceIndex(int i1, int i2, int numParticles)
	{
		if (i1 > i2)
		{
			int t = i1;
			i1 = i2;
			i2 = t;
		}

		return i1 * (numParticles - i1 * 0.5 - 1.5) + i2 - 1;
	}

	__device__ float calculatePressure(const float density, const SPHConfiguration& simSettings)
	{
		return simSettings.stiffnessConstant * (density - simSettings.restDensity);
	}

	__device__ glm::vec3 calculateParticleAcceleration(int particleIndex, SPHSimulationData& simData,
		const SPHConfiguration& simSettings)
	{
		float density = simData.d_densities[particleIndex];
		float pressure = calculatePressure(density, simSettings);

		glm::vec3 forcePressure = glm::vec3(0.0f);
		glm::vec3 forceViscosity = glm::vec3(0.0f);

		for (int j = 0; j < simSettings.numParticles; j++)
		{
			if (j == particleIndex) continue;

			float distance = simData.d_distances[ getParticleDistanceIndex(particleIndex, j, simSettings.numParticles) ];

			float kernelViscosity = CUDADeviceFunctions::SmoothingKernels::getKernelValue(simSettings.kernelViscosity,
				distance, simSettings.smoothingRadius);

			float kernelPressureGradient = CUDADeviceFunctions::SmoothingKernels::getKernelValue(
				simSettings.kernelPressureGradient, distance, simSettings.smoothingRadius);

			float otherDensity = simData.d_densities[j];

			if (otherDensity != 0)
			{
				float otherPressure = calculatePressure(otherDensity, simSettings);
				float sharedPressure = (pressure + otherPressure) * 0.5f;

				//direction to accelerate in
				glm::vec3 pressureDirection;
				if (distance == 0) pressureDirection = glm::vec3(1.0f, 0.0f, 0.0f); //in case particles perfectly overlapped
				else pressureDirection = glm::normalize(simData.d_predictedPositions[particleIndex] -
					simData.d_predictedPositions[j]);

				forcePressure += sharedPressure / otherDensity * kernelPressureGradient * pressureDirection;
				forceViscosity += (simData.d_velocities[j] - simData.d_velocities[particleIndex]) / otherDensity *
					kernelViscosity;
			}
		}

		forceViscosity *= simSettings.viscosity;

		glm::vec3 particleAcceleration = forcePressure + forceViscosity; //mass factored out in force calculation
		particleAcceleration += simSettings.gravity;

		return particleAcceleration;
	}

	__device__ void handleBoundaryCollisions(int particleIndex, SPHSimulationData simData, SPHConfiguration simSettings)
	{
		//check world boundaries and force the particle to be on the side pointed to by the normal. repeat until on
		//correct side
		bool collided = false;
		do
		{
			collided = false;
			for (int j = 0; j < simData.worldBoundaryCount; j++)
			{
				const Boundary& b = simData.d_worldBoundaries[j];
				glm::vec3& particlePos = simData.d_positions[particleIndex];

				//particle is on wrong side of wall
				if (glm::dot(particlePos - b.pos, b.norm) < 0)
				{
					//find 1D distance along normal of particle and plane point, difference between them is particle
					//distance from plane
					float tParticle = glm::dot(particlePos, b.norm);
					float tPlane = glm::dot(b.pos, b.norm);

					//difference between tParticle and tPlane is how far the particle is from the plane along the normal
					//since we know particle is more negative along plane normal than plane position,
					//doing plane - particle will give positive distance
					float distBehindWall = tPlane - tParticle;

					//reflect its position back out of the wall
					particlePos = particlePos + 2.001f * distBehindWall * b.norm;

					//reflect its velocity to be away from the wall
					simData.d_velocities[particleIndex] = glm::reflect(simData.d_velocities[particleIndex], b.norm) *
						(1.0f - simSettings.boundaryCollisionDamping);

					collided = true;
				}
			}
		} while (collided == true);
		
	}

	__device__ int UGCellIDFromPosition(const glm::vec3& pos, const float& cellSize, const int& dimCells)
	{
		//https://www.desmos.com/calculator/s7essmanrk
		int z = pos.z / cellSize + dimCells * 0.5f;
		int y = pos.y / cellSize + dimCells * 0.5f;
		int x = pos.x / cellSize + dimCells * 0.5f;

		//clamp to edges of grid
		if (z < 0) return -1;
		if (z >= dimCells) return -1;
		if (y < 0) return -1;
		if (y >= dimCells) return -1;
		if (x < 0) return -1;
		if (x >= dimCells) return -1;

		return z * dimCells * dimCells + y * dimCells + x;
	}

	__device__ glm::vec3 UGCalculateParticleAcceleration(int particleIndex, SPHSimulationData simData,
		SPHConfiguration simSettings, UniformGridData ugData, UniformGridSettings ugSettings)
	{
		const glm::vec3& particlePos = simData.d_predictedPositions[particleIndex];

		float density = simData.d_densities[particleIndex];
		float pressure = calculatePressure(density, simSettings);

		glm::vec3 forcePressure = glm::vec3(0.0f);
		glm::vec3 forceViscosity = glm::vec3(0.0f);

		//for each grid cell in a 3x3x3 region around this particle
		for (int dz = -1; dz <= 1; dz++)
		for (int dy = -1; dy <= 1; dy++)
		for (int dx = -1; dx <= 1; dx++)
		{
			glm::vec3 tempPos = particlePos + glm::vec3(dx, dy, dz) * ugSettings.cellSize;
			int otherCellID = CUDADeviceFunctions::UGCellIDFromPosition(tempPos, ugSettings.cellSize,
				ugSettings.dimCells);
			if (otherCellID == -1)
			{
				//-1 is returned when cell would be outside of grid area
				continue;
			}

			int cellStart = ugData.d_cellStarts[otherCellID];
			if (cellStart == -1)
			{
				//other cell has no particles in it
				continue;
			}

			int j = cellStart;
			while (j < simSettings.numParticles && ugData.d_cellIDs[j] == otherCellID) //for every particle in this cell
			{
				int otherIndex = ugData.d_particleIDs[j];
				if (otherIndex == particleIndex)
				{
					j++;
					continue;
				}

				float distance = simData.d_distances[getParticleDistanceIndex(particleIndex, otherIndex,
					simSettings.numParticles)];

				float kernelViscosity = CUDADeviceFunctions::SmoothingKernels::getKernelValue(
					simSettings.kernelViscosity, distance, simSettings.smoothingRadius);

				float kernelPressureGradient = CUDADeviceFunctions::SmoothingKernels::getKernelValue(
					simSettings.kernelPressureGradient, distance, simSettings.smoothingRadius);

				float otherDensity = simData.d_densities[otherIndex];

				if (otherDensity != 0)
				{
					float otherPressure = calculatePressure(otherDensity, simSettings);
					float sharedPressure = (pressure + otherPressure) * 0.5f;

					//direction to accelerate in
					glm::vec3 pressureDirection;
					if (distance == 0) pressureDirection = glm::vec3(1.0f, 0.0f, 0.0f); //in case perfectly overlapped
					else pressureDirection = glm::normalize(simData.d_predictedPositions[particleIndex] -
						simData.d_predictedPositions[otherIndex]);

					forcePressure += sharedPressure / otherDensity * kernelPressureGradient * pressureDirection;
					forceViscosity += (simData.d_velocities[otherIndex] - simData.d_velocities[particleIndex]) /
						otherDensity * kernelViscosity;
				}

				j++;
			}
		}

		forceViscosity *= simSettings.viscosity;

		glm::vec3 particleAcceleration = forcePressure + forceViscosity; //mass factored out in force calculation
		particleAcceleration += simSettings.gravity;

		return particleAcceleration;
	}

	__device__ glm::vec3 getRayDirection(int threadIndex, int textureWidth, int textureHeight, float fieldOfView,
		float pitch, float yaw)
	{
		//calculate texture coordinate of this thread
		float xPix = threadIndex % textureWidth;
		float yPix = threadIndex / textureWidth;

		//ray target generation https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/generating-camera-rays.html
		//target position of ray in 2D grid of pixels 1 unit in front of camera
		float texAR = (float)textureWidth / textureHeight;
		glm::vec2 gridTarget = glm::vec2((xPix + 0.5f) / textureWidth, (yPix + 0.5f) / textureHeight) * 2.0f - 1.0f;
		gridTarget.x *= texAR;
		gridTarget *= glm::tan(fieldOfView / 2.0f);
		glm::vec3 worldTarget = glm::vec3(gridTarget, -1.0f);

		//find rotation matrix to rotate from grid (forward = -Z) to camera (forward = cam.direction)
		glm::vec3 gridRight = glm::vec3(1.0f, 0.0f, 0.0f);
		glm::vec3 gridUp = glm::vec3(0.0f, 1.0f, 0.0f);
		glm::mat4 I = glm::mat4(1.0f);
		glm::mat4 matPitch = glm::rotate(I, pitch, gridRight);
		glm::mat4 matYaw = glm::rotate(I, -yaw - glm::half_pi<float>(), gridUp);
		glm::mat4 matGridRotation = matYaw * matPitch;

		glm::vec3 rayDirection = matGridRotation * glm::vec4(worldTarget, 1.0f);
		rayDirection = glm::normalize(rayDirection);

		return rayDirection;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace CUDAKernels
{
	__global__ void setInitialParticlePositions(glm::vec3* particlePositions, int numParticles, const int width,
		const int height, const int depth, const float spacing)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= numParticles) return;

		//get normalised xyz within volume for placing particles
		//want to fill up in horizontal layers, so x, z, y

		/*
		i = y * maxX * maxZ + z * maxX + x
		x = index % maxX
		z = (index / maxX) % maxZ
		y = index / maxZ / maxX
		*/

		//integer divisions
		float x = i % width;
		float z = (i / width) % depth;
		float y = i / depth / width;
		
		x = (x - width * 0.5f) * spacing;
		y = (y - height * 0.5f) * spacing;
		z = (z - depth * 0.5f) * spacing;

		particlePositions[i] = glm::vec3(x, y, z);
	}

	__global__ void updatePredictedParticlePositions(SPHSimulationData simData, SPHConfiguration simSettings)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= simSettings.numParticles) return;

		simData.d_predictedPositions[i] = simData.d_positions[i] + simData.d_velocities[i] * 0.01f;
	}

	__global__ void calculateInterParticleValues(SPHSimulationData simData, SPHConfiguration simSettings)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= simSettings.numParticles) return;

		simData.d_densities[i] = 0;

		for (int j = 0; j < simSettings.numParticles; j++)
		{
			if (i == j) continue;

			float distance = glm::distance(simData.d_predictedPositions[i], simData.d_predictedPositions[j]);

			int pdi = CUDADeviceFunctions::getParticleDistanceIndex(i, j, simSettings.numParticles);
			simData.d_distances[pdi] = distance;
			
			float kernelValue = CUDADeviceFunctions::SmoothingKernels::getKernelValue(simSettings.kernelPressure,
				distance, simSettings.smoothingRadius);
			simData.d_densities[i] += kernelValue;
		}

		simData.d_densities[i] *= simSettings.particleMass;
	}

	__global__ void processTimeStep(SPHSimulationData simData, SPHConfiguration simSettings)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= simSettings.numParticles) return;



		//not quite right as acceleration isn't constant
		simData.d_accelerations[i] = CUDADeviceFunctions::calculateParticleAcceleration(i, simData, simSettings);
		simData.d_velocities[i] += simData.d_accelerations[i] * simSettings.timeStep;
		simData.d_positions[i] += simData.d_velocities[i] * simSettings.timeStep;

		CUDADeviceFunctions::handleBoundaryCollisions(i, simData, simSettings);
	}

	__global__ void UGUpdateCellParticles(SPHSimulationData simData, SPHConfiguration simSettings,
		UniformGridData ugData, UniformGridSettings ugSettings)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= simSettings.numParticles) return;

		const glm::vec3& pos = simData.d_positions[i];
		size_t cellIndex = CUDADeviceFunctions::UGCellIDFromPosition(pos, ugSettings.cellSize, ugSettings.dimCells);

		ugData.d_cellIDs[i] = cellIndex;
		ugData.d_particleIDs[i] = i;
	}

	__global__ void UGUpdateCellStarts(SPHConfiguration simSettings, UniformGridData ugData)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= simSettings.numParticles) return;

		int cellID = ugData.d_cellIDs[i];
		if (cellID == -1)
		{
			//particle somehow broke out of boundary
			return;
		}

		if (i == 0)
		{
			//must be the first occurrence of a cell id if i is 0
			ugData.d_cellStarts[cellID] = i;
		}
		else
		{
			int cellIDPrev = ugData.d_cellIDs[i - 1];
			if (cellIDPrev != cellID)
			{
				//this cellID is different to previous ID. they are sorted, so this must be the first
				ugData.d_cellStarts[cellID] = i;
			}
			else
			{
				//i is not the first occurrence of this cell id
			}
		}

		//may be worth saving cellEnds as well, so that when iterating through particles in a given cell, the for range
		//can be set directly rather than continuing until the next cell doesn't match the current (requiring an if each
		//time)
	}

	__global__ void UGCalculateInterParticleValues(SPHSimulationData simData, SPHConfiguration simSettings,
		UniformGridData ugData, UniformGridSettings ugSettings)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= simSettings.numParticles) return;

		const int& particleIndex = i;
		const glm::vec3& particlePos = simData.d_predictedPositions[particleIndex];

		simData.d_densities[particleIndex] = 0;

		for (int dz = -1; dz <= 1; dz++)
		for (int dy = -1; dy <= 1; dy++)
		for (int dx = -1; dx <= 1; dx++)
		{
			glm::vec3 tempPos = particlePos + glm::vec3(dx, dy, dz) * ugSettings.cellSize;
			int otherCellID = CUDADeviceFunctions::UGCellIDFromPosition(tempPos, ugSettings.cellSize, ugSettings.dimCells);
			if (otherCellID == -1)
			{
				//-1 is returned when cell would be outside of grid area
				continue;
			}

			int cellStart = ugData.d_cellStarts[otherCellID];
			if (cellStart == -1)
			{
				//other cell has no particles in it
				continue;
			}

			int j = cellStart;
			while (j < simSettings.numParticles && ugData.d_cellIDs[j] == otherCellID)
			{
				int otherIndex = ugData.d_particleIDs[j];
				if (otherIndex == particleIndex)
				{
					j++;
					continue;
				}

				float distance = glm::distance(particlePos, simData.d_predictedPositions[otherIndex]);

				int pdi = CUDADeviceFunctions::getParticleDistanceIndex(particleIndex, otherIndex,
					simSettings.numParticles);
				simData.d_distances[pdi] = distance;

				float kernelValue = CUDADeviceFunctions::SmoothingKernels::getKernelValue(
					simSettings.kernelPressure, distance, simSettings.smoothingRadius);
				simData.d_densities[particleIndex] += kernelValue;

				j++;
			}
		}

		simData.d_densities[particleIndex] *= simSettings.particleMass;
	}

	__global__ void UGProcessTimeStep(SPHSimulationData simData, SPHConfiguration simSettings,
		UniformGridData ugData, UniformGridSettings ugSettings)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= simSettings.numParticles) return;


		//not quite right as acceleration isn't constant
		simData.d_accelerations[i] = CUDADeviceFunctions::UGCalculateParticleAcceleration(i, simData, simSettings,
			ugData, ugSettings);
		simData.d_velocities[i] += simData.d_accelerations[i] * simSettings.timeStep;
		simData.d_positions[i] += simData.d_velocities[i] * simSettings.timeStep;

		CUDADeviceFunctions::handleBoundaryCollisions(i, simData, simSettings);
	}

	__global__ void userInteractParticles(SPHSimulationData simData, int numParticles, glm::vec3 attractionPoint,
		float attractionRadius, float attractionVelocity)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= numParticles) return;

		glm::vec3 difference = attractionPoint - simData.d_positions[i];
		float distance = glm::length(difference);
		if (distance < attractionRadius)
		{
			glm::vec3 direction = glm::normalize(difference);
			simData.d_velocities[i] += attractionVelocity * direction * (1.0f - distance / attractionRadius);
		}
	}

	__global__ void calculateRaymarchSamples(SPHSimulationData simdata, SPHConfiguration simSettings,
		UniformGridData ugData, UniformGridSettings ugSettings, MetaballSampler mbSampler,
		Camera cam)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= mbSampler.textureWidth * mbSampler.textureHeight) return;

		//get origin and direction of ray
		glm::vec3 rayOrigin = cam.position;
		glm::vec3 rayDirection = CUDADeviceFunctions::getRayDirection(i, mbSampler.textureWidth, mbSampler.textureHeight,
			cam.fov, cam.angles.x, cam.angles.y);

		glm::vec3 skyColour = glm::vec3(0.7f, 0.7f, 1.0f) * (rayDirection.y + 1.0f) / 2.0f;
		glm::vec3 lightDir = glm::vec3(0.44023f, -0.88045f, -0.17609f);

		//if ray is starting outside of the uniform grid
		int cellID = CUDADeviceFunctions::UGCellIDFromPosition(rayOrigin, ugSettings.cellSize, ugSettings.dimCells);
		if (cellID == -1)
		{
			//move ray forward until inside uniform grid, or return sky colour if grid is never hit
			//box intersection check: https://tavianator.com/2022/ray_box_boundary.html
			float boxSize = ugSettings.dimCells * ugSettings.cellSize;
			glm::vec3 boxMin = glm::vec3(-boxSize * 0.5f);
			glm::vec3 boxMax = glm::vec3(boxSize * 0.5f);

			float tmin = 0.0f;
			float tmax = FLT_MAX;

			glm::vec3 rayDirectionInverse = 1.0f / rayDirection;
			float t1 = (boxMin.x - rayOrigin.x) * rayDirectionInverse.x;
			float t2 = (boxMax.x - rayOrigin.x) * rayDirectionInverse.x;
			tmin = glm::max(tmin, glm::min(t1, t2));
			tmax = glm::min(tmax, glm::max(t1, t2));

			t1 = (boxMin.y - rayOrigin.y) * rayDirectionInverse.y;
			t2 = (boxMax.y - rayOrigin.y) * rayDirectionInverse.y;
			tmin = glm::max(tmin, glm::min(t1, t2));
			tmax = glm::min(tmax, glm::max(t1, t2));

			t1 = (boxMin.z - rayOrigin.z) * rayDirectionInverse.z;
			t2 = (boxMax.z - rayOrigin.z) * rayDirectionInverse.z;
			tmin = glm::max(tmin, glm::min(t1, t2));
			tmax = glm::min(tmax, glm::max(t1, t2));

			if (tmin > tmax)
			{
				//no intersection
				mbSampler.d_rayData[i] = glm::vec4(skyColour, 0.0f);
				return;
			}

			cellID = CUDADeviceFunctions::UGCellIDFromPosition(rayOrigin + (tmin + 0.001f) * rayDirection, ugSettings.cellSize, ugSettings.dimCells);
		}


		//set up values for jumping to next grid cell (DDA)
		//adapted to 3d from https://github.com/OneLoneCoder/Javidx9/blob/master/PixelGameEngine/SmallerProjects/OneLoneCoder_PGE_RayCastDDA.cpp
		glm::vec3 rayUnitStepSize = glm::abs(1.0f / rayDirection);
		int cellX = cellID % ugSettings.dimCells;
		int cellY = (cellID / ugSettings.dimCells) % ugSettings.dimCells;
		int cellZ = cellID / (ugSettings.dimCells * ugSettings.dimCells);
		glm::vec3 cellPosIndex = glm::vec3(cellX, cellY, cellZ);
		glm::vec3 rayLength1D = glm::vec3(0.0f);
		glm::vec3 step = glm::vec3(0.0f);

		//cellPosIndex goes from 0,0,0 in bottom corner to ugSettings.dimCells in top corner
		//actual 0,0,0 (which rayOrigin is relative to) is in centre
		//raylength needs rayorigin relative to bottom corner 0,0,0
		//int z = pos.z / cellSize + dimCells * 0.5f;
		glm::vec3 rayOriginDDA = rayOrigin / ugSettings.cellSize + ugSettings.dimCells * 0.5f;
		if (rayDirection.x < 0)
		{
			step.x = -1;
			rayLength1D.x = (rayOriginDDA.x - cellPosIndex.x) * rayUnitStepSize.x;
		}
		else
		{
			step.x = 1;
			rayLength1D.x = (cellPosIndex.x + 1 - rayOriginDDA.x) * rayUnitStepSize.x;
		}

		if (rayDirection.y < 0)
		{
			step.y = -1;
			rayLength1D.y = (rayOriginDDA.y - cellPosIndex.y) * rayUnitStepSize.y;
		}
		else
		{
			step.y = 1;
			rayLength1D.y = (cellPosIndex.y + 1 - rayOriginDDA.y) * rayUnitStepSize.y;
		}

		if (rayDirection.z < 0)
		{
			step.z = -1;
			rayLength1D.z = (rayOriginDDA.z - cellPosIndex.z) * rayUnitStepSize.z;
		}
		else
		{
			step.z = 1;
			rayLength1D.z = (cellPosIndex.z + 1 - rayOriginDDA.z) * rayUnitStepSize.z;
		}


		int iterations = mbSampler.maxIterations;
		float t = FLT_MAX;
		int hitParticleIndex = -1;


		//while the ray is still within the uniform grid
		while (iterations > 0 && hitParticleIndex == -1)
		{
			//check 3x3x3 cells including current cell
			float tMinCurrentCells = FLT_MAX;
			for (int dz = -1; dz <= 1; dz++)
			for (int dy = -1; dy <= 1; dy++)
			for (int dx = -1; dx <= 1; dx++)
			{
				glm::vec3 otherCellPosIndex = cellPosIndex + glm::vec3(dx, dy, dz);
				if (   otherCellPosIndex.x < 0 || otherCellPosIndex.x >= ugSettings.dimCells
					|| otherCellPosIndex.y < 0 || otherCellPosIndex.y >= ugSettings.dimCells
					|| otherCellPosIndex.z < 0 || otherCellPosIndex.z >= ugSettings.dimCells)
				{
					//outside of uniform grid
					continue;
				}

				int otherCellID = otherCellPosIndex.z * ugSettings.dimCells * ugSettings.dimCells +
					otherCellPosIndex.y * ugSettings.dimCells + otherCellPosIndex.x;
				int cellStart = ugData.d_cellStarts[otherCellID];
				if (cellStart == -1)
				{
					//cell has no particles in it
					continue;
				}

				int j = cellStart;
				while (j < simSettings.numParticles && ugData.d_cellIDs[j] == otherCellID)
				{
					//for every particle in this cell, check if ray would intersect it

					int particleIndex = ugData.d_particleIDs[j];
					glm::vec3& particlePos = simdata.d_positions[particleIndex];
					//ray-sphere intersection https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection.html
					glm::vec3 L = rayOrigin - particlePos;
					float a = 1.0f;
					float b = 2.0f * glm::dot(rayDirection, L);
					float c = glm::dot(L, L) - mbSampler.boundaryRadius * mbSampler.boundaryRadius;
					float d = b * b - 4.0f * a * c;
					
					if (d < 0)
					{
						//no intersection
						j++;
						continue;
					}
					
					if (d == 0)
					{
						//one intersection
						float tParticle = -0.5f * b / a;
						if (tParticle > 0.0f && tParticle < tMinCurrentCells)
						{
							tMinCurrentCells = tParticle;
							hitParticleIndex = particleIndex;
						}
					}
					else
					{
						//two intersections, use the lowest one which is above 0 (i.e. not behind camera)
						float q = (b > 0) ? -0.5f * (b + glm::sqrt(d)) : -0.5f * (b - glm::sqrt(d));
						float t1 = q / a;
						float t2 = c / q;
							
						float tParticle = FLT_MAX;
						if (t1 > 0)
						{
							if (t2 > 0)
							{
								tParticle = glm::min(t1, t2);
							}
							else
							{
								tParticle = t1;
							}
						}
						else
						{
							if (t2 > 0)
							{
								tParticle = t2;
							}
							else
							{
								//no intersection
							}
						}

						if (tParticle < tMinCurrentCells)
						{
							tMinCurrentCells = tParticle;
							hitParticleIndex = particleIndex;
						}
					}

					j++;
				}
			}

			if (hitParticleIndex != -1)
			{
				t = tMinCurrentCells;
				break;
			}
			else
			{
				//calculate which cell the ray would hit next
				float minRayLength = glm::min(glm::min(rayLength1D.x, rayLength1D.y), rayLength1D.z);
				if (minRayLength == rayLength1D.x)
				{
					cellPosIndex.x += step.x;
					rayLength1D.x += rayUnitStepSize.x;
				}
				else if (minRayLength == rayLength1D.y)
				{
					cellPosIndex.y += step.y;
					rayLength1D.y += rayUnitStepSize.y;
				}
				else
				{
					cellPosIndex.z += step.z;
					rayLength1D.z += rayUnitStepSize.z;
				}

				if (   cellPosIndex.x < 0 || cellPosIndex.x >= ugSettings.dimCells
					|| cellPosIndex.y < 0 || cellPosIndex.y >= ugSettings.dimCells
					|| cellPosIndex.z < 0 || cellPosIndex.z >= ugSettings.dimCells)
				{
					//outside of uniform grid
					break;
				}
			}

			iterations--;
		}

		if (hitParticleIndex != -1)
		{
			//use hit particle index to get the normal of the ray intersection
			glm::vec3 hitPos = rayOrigin + t * rayDirection;
			glm::vec3 particlePos = simdata.d_positions[hitParticleIndex];
			glm::vec3 normal = glm::normalize(hitPos - particlePos);
			float brightness = (glm::dot(-normal, lightDir) + 1.0f) / 2.0f;
			mbSampler.d_rayData[i] = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f) * brightness;
		}
		else
		{
			if (iterations == 0)
			{
				//took too long to hit a particle, give up
				mbSampler.d_rayData[i] = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
			}
			else
			{
				//exited the uniform grid without hitting any sphere, return sky colour
				mbSampler.d_rayData[i] = glm::vec4(skyColour, 0.0f);
			}
		}
	}
}