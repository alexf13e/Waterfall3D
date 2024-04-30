
#include "Renderer.cuh"

#include <iostream>
#include <chrono>

#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#include "CUDAFunctions.cuh"


void Camera::updateDirection()
{
	direction = glm::vec3(
		glm::cos(angles.y) * glm::cos(angles.x),
		glm::sin(angles.x),
		glm::sin(angles.y) * glm::cos(angles.x)
	);
	updateViewMatrix();
}

void Camera::updateViewMatrix()
{
	matView = glm::lookAt(position, position + direction, glm::vec3(0.0f, 1.0f, 0.0f));
	updatedSinceLastFrame = true;
}

void Camera::updatePerspective()
{
	matProj = glm::perspective(fov, ar, clipNear, clipFar);
	updatedSinceLastFrame = true;
}

void Camera::setFieldOfView(const float fieldOfView)
{
	fov = fieldOfView;
	updatePerspective();
}

void Camera::setAspectRatio(const float width, const float height)
{
	ar = width / height;
	updatePerspective();
}

void Camera::init(const float aspectRatio)
{
	initialised = false;

	position = glm::vec3(0.0f, 0.0f, 20.0f);
	angles = glm::vec2(0.0f, -glm::half_pi<float>()); //0,0 = positive X
	
	updateDirection();

	//default perspective values
	fov = glm::radians(90.0f);
	ar = aspectRatio;
	clipNear = 0.1f;
	clipFar = 1000.0f;
	updatePerspective();

	initialised = true;
}

void Camera::updatePosition(const glm::vec3& deltaPos)
{
	position += deltaPos;
	updateViewMatrix();
}

void Camera::updateViewAngle(const glm::vec2& deltaAngles)
{
	float rad90deg = glm::radians(90.0f);
	angles.x = glm::min(glm::max(angles.x + deltaAngles.x, -rad90deg), rad90deg);
	angles.y = glm::mod(angles.y + deltaAngles.y, 2.0f * glm::pi<float>());
	updateDirection();
}


bool Renderer::init(const uint32_t& resx, const uint32_t& resy, const glm::vec4& coldColour,
	const glm::vec4& hotColour, const SPHConfiguration& simSettings, const SPHSimulationData& simData,
	UniformGrid& uniformGrid)
{
	this->windowResolution = glm::vec2(resx, resy);
	glViewport(0, 0, resx, resy);

	this->cam.init((float)resx / resy);

	if (!this->shWorldToScreen.init("shaders/basicProjection.vert", "shaders/basicColour.frag")) return false;

//////////////// BOUNDARIES ////////////////////////////////////////////////////////////////////////////////////////////
	//create vertex data for boundaries
	int numBoundaryPoints = simData.worldBoundaryCount * 2; //start and end
	glm::vec3* boundaryPoints = new glm::vec3[numBoundaryPoints];
	for (int i = 0; i < simData.worldBoundaryCount; i++)
	{
		assert(i * 2 + 1 < numBoundaryPoints);
		const Boundary& b = simData.h_worldBoundaries[i];
		boundaryPoints[i * 2] = b.pos;
		boundaryPoints[i * 2 + 1] = b.pos - b.norm * 1.0f;
	}

	glGenVertexArrays(1, &this->vao_boundaryLines);
	glBindVertexArray(this->vao_boundaryLines);
	glGenBuffers(1, &this->vbo_boundaryLines);
	glBindBuffer(GL_ARRAY_BUFFER, this->vbo_boundaryLines);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numBoundaryPoints, boundaryPoints, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);
	glEnableVertexAttribArray(0);

	delete[] boundaryPoints;


//////////////// UNIFORM GRID //////////////////////////////////////////////////////////////////////////////////////////
	//create vertex data for uniform grid
	const int& ugCellDim = uniformGrid.getSettings().dimCells;

	//number of points to draw edges of grid. points need to cover surfaces of perimeter e.g. 3x3x3 cells needs
	//4x4 points on each of the 6 cube faces. edges will have duplicate points for simplicity
	int numUGPoints = 6 * (ugCellDim + 1) * (ugCellDim + 1);
	uniformGrid.setNumLinesToDraw(numUGPoints);
	glm::vec3* ugPoints = new glm::vec3[numUGPoints];

	const float& ugCellSize = uniformGrid.getSettings().cellSize;
	float halfUGWidth = ugCellDim * ugCellSize * 0.5f; //for centering the grid around (0,0,0)

	for (int col = 0; col < ugCellDim + 1; col++)
	{
		for (int row = 0; row < ugCellDim + 1; row++)
		{
			int lineIndex = (col * (ugCellDim + 1) + row) * 6;
			float r = row * ugCellSize - halfUGWidth;
			float c = col * ugCellSize - halfUGWidth;

			//left-right lines
			ugPoints[lineIndex + 0] = glm::vec3(-halfUGWidth, r, c);
			ugPoints[lineIndex + 1] = glm::vec3(halfUGWidth, r, c);

			//bottom-top lines
			ugPoints[lineIndex + 2] = glm::vec3(r, -halfUGWidth, c);
			ugPoints[lineIndex + 3] = glm::vec3(r, halfUGWidth, c);

			//front-back lines
			ugPoints[lineIndex + 4] = glm::vec3(r, c, -halfUGWidth);
			ugPoints[lineIndex + 5] = glm::vec3(r, c, halfUGWidth);
		}
	}

	glGenVertexArrays(1, &this->vao_ugLines);
	glBindVertexArray(this->vao_ugLines);
	glGenBuffers(1, &this->vbo_ugLines);
	glBindBuffer(GL_ARRAY_BUFFER, this->vbo_ugLines);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numUGPoints, ugPoints, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);
	glEnableVertexAttribArray(0);

	delete[] ugPoints;

	
//////////////// METABALL SAMPLING /////////////////////////////////////////////////////////////////////////////////////
	if (!this->shMetaballs.init("shaders/fullScreenTri.vert", "shaders/metaball.frag")) return false;

	mbSampler.init(this->windowResolution.x / 4, this->windowResolution.y / 4, 1.0f, 10);

	glUseProgram(shMetaballs.getID());
	glUniform4fv(glGetUniformLocation(shMetaballs.getID(), "coldColour"), 1, glm::value_ptr(coldColour));
	glUniform4fv(glGetUniformLocation(shMetaballs.getID(), "hotColour"), 1, glm::value_ptr(hotColour));
	glUniform1i(glGetUniformLocation(shMetaballs.getID(), "texWidth"), mbSampler.textureWidth);
	glUniform1i(glGetUniformLocation(shMetaballs.getID(), "texHeight"), mbSampler.textureHeight);

	//tell shader where shader storage block for ray data is
	int gl_rayDataBufferBlockBinding = 0;
	int gl_rayDataBufferBlockIndex = glGetProgramResourceIndex(shMetaballs.getID(), GL_SHADER_STORAGE_BLOCK, "RayData");
	glShaderStorageBlockBinding(shMetaballs.getID(), gl_rayDataBufferBlockIndex, gl_rayDataBufferBlockBinding);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, gl_rayDataBufferBlockBinding, mbSampler.gl_rayDataSSBO);
	glUseProgram(0);	


//////////////// MISC //////////////////////////////////////////////////////////////////////////////////////////////////
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	this->coldColour = coldColour;
	this->hotColour = hotColour;
	this->showUniformGrid = false;

	this->initialised = true;

	glGenQueries(1, &timeQueryID);

	return true;
}

void Renderer::visualise(SPHSolver& solver, bool enableTiming, std::vector<std::pair<std::string, float>>& timingValues,
	RenderMode renderMode)
{
	if (cam.getUpdated())
	{
		//give updated matrices to shader
		glUseProgram(this->shWorldToScreen.getID());
			int loc = glGetUniformLocation(this->shWorldToScreen.getID(), "matProjView");
			glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(cam.getMatrixWorldToScreen()));
		glUseProgram(0);

		cam.clearUpdated();
	}

	switch (renderMode)
	{
	case POINTS:
		if (enableTiming)
		{
			//https://www.lighthouse3d.com/tutorials/opengl-timer-query/
			glBeginQuery(GL_TIME_ELAPSED, timeQueryID);
		}

		glUseProgram(this->shWorldToScreen.getID());
		//draw particles predicted positions
		/*glBindVertexArray(solver.getSimData().gl_predictedPositionsVAO);
		glUniform4fv(glGetUniformLocation(this->shWorldToScreen.getID(), "colour"), 1, glm::value_ptr(hotColour));
		glDrawArrays(GL_POINTS, 0, solver.getSettings().numParticles);*/

		//draw particles current positions
		glBindVertexArray(solver.getSimData().gl_positionsVAO);
		glUniform4fv(glGetUniformLocation(this->shWorldToScreen.getID(), "colour"), 1, glm::value_ptr(coldColour));
		glDrawArrays(GL_POINTS, 0, solver.getSettings().numParticles);
		break;

	case RAYMARCH:
		//call cuda to update density values at metaball grid points
		if (enableTiming)
		{
			mbSampler.mapCudaResources();
			solver.mapCudaResources();
			int numPixels = mbSampler.textureWidth * mbSampler.textureHeight;
			int blockSize = glm::min(numPixels, 1024);
			int numBlocks = (numPixels - 1) / blockSize + 1;

			std::chrono::steady_clock::time_point t1 = std::chrono::high_resolution_clock::now();
			CUDAKernels::calculateRaymarchSamples << <numBlocks, blockSize >> > (solver.getSimData(), solver.getSettings(),
				solver.getUniformGrid().getData(), solver.getUniformGrid().getSettings(), mbSampler, cam);
			cudaDeviceSynchronize();
			std::chrono::steady_clock::time_point t2 = std::chrono::high_resolution_clock::now();
			float dt = (t2 - t1).count() * 1e-9;
			timingValues.push_back({ "calculateRaymarchSamples", dt });

			solver.unmapCudaResources();
			mbSampler.unmapCudaResources();

			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess)
			{
				std::cerr << "CUDA error in renderer visualise with raymarch: " << cudaGetErrorName(err) << std::endl;
			}

			//don't want metaball kernel counted towards shader render time
			glBeginQuery(GL_TIME_ELAPSED, timeQueryID);
		}
		else
		{
			mbSampler.mapCudaResources();
			solver.mapCudaResources();
			int numPixels = mbSampler.textureWidth * mbSampler.textureHeight;
			int blockSize = glm::min(numPixels, 1024);
			int numBlocks = (numPixels - 1) / blockSize + 1;

			//cudaProfilerStart();
			CUDAKernels::calculateRaymarchSamples << <numBlocks, blockSize >> > (solver.getSimData(), solver.getSettings(),
				solver.getUniformGrid().getData(), solver.getUniformGrid().getSettings(), mbSampler, cam);
			//cudaProfilerStop();

			solver.unmapCudaResources();
			mbSampler.unmapCudaResources();

			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess)
			{
				std::cerr << "CUDA error in renderer visualise with raymarch: " << cudaGetErrorName(err) << std::endl;
			}
		}

		//render metaball texture to fullscreen tri
		glUseProgram(this->shMetaballs.getID());

		//check if metaballs render has been resized
		if (mbSampler.updateFragTexSize)
		{
			glUniform1i(glGetUniformLocation(shMetaballs.getID(), "texWidth"), mbSampler.textureWidth);
			glUniform1i(glGetUniformLocation(shMetaballs.getID(), "texHeight"), mbSampler.textureHeight);
			mbSampler.updateFragTexSize = false;
		}

		glBindVertexArray(mbSampler.getVAOFullScreenTri());
		glDrawArrays(GL_TRIANGLES, 0, 3);

		glUseProgram(this->shWorldToScreen.getID()); //for drawing boundaries, which wants to be done after metaballs
		break;
	}
	
	//draw boundaries
	float boundColour[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
	glUniform4fv(glGetUniformLocation(this->shWorldToScreen.getID(), "colour"), 1, boundColour);
	glBindVertexArray(this->vao_boundaryLines);
	glDrawArrays(GL_LINES, 0, solver.getSimData().worldBoundaryCount * 2);

	if (showUniformGrid)
	{
		//draw uniform grid
		glBindVertexArray(this->vao_ugLines);
		float gridColour[4] = { 0.0f, 1.0f, 0.0f, 1.0f };
		glUniform4fv(glGetUniformLocation(this->shWorldToScreen.getID(), "colour"), 1, gridColour);
		glDrawArrays(GL_LINES, 0, solver.getUniformGrid().getSettings().numLinesToDraw);
	}
	
	if (enableTiming)
	{
		glEndQuery(GL_TIME_ELAPSED);

		int queryReady = 0;
		while (!queryReady)
		{
			glGetQueryObjectiv(timeQueryID, GL_QUERY_RESULT_AVAILABLE, &queryReady);
		}

		glGetQueryObjectui64v(timeQueryID, GL_QUERY_RESULT, &elapsedTime);
		float dt = elapsedTime * 1e-9;
		timingValues.push_back({ "Render", dt });
	}
	
	glUseProgram(0);
	glBindVertexArray(0);
}

void Renderer::destroy()
{
	if (initialised)
	{
		glDeleteBuffers(1, &vbo_boundaryLines);
		glDeleteBuffers(1, &vbo_ugLines);

		glDeleteVertexArrays(1, &vao_boundaryLines);
		glDeleteVertexArrays(1, &vao_ugLines);

		mbSampler.destroy();

		this->shMetaballs.destroy();
		this->shWorldToScreen.destroy();

		initialised = false;
	}
}

void Renderer::updateWindowRes(const int& width, const int& height)
{
	this->windowResolution = glm::vec2(width, height);
	glViewport(0, 0, width, height);

	cam.setAspectRatio(width, height);
}

void Renderer::updateCam(const glm::vec3& deltaPos, const glm::vec2& deltaAngles)
{
	cam.updatePosition(deltaPos);
	cam.updateViewAngle(deltaAngles);
}