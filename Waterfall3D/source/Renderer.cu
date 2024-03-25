
#include "Renderer.cuh"

#include <iostream>

#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "cuda_runtime.h"

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
	angles = glm::vec2(0.0f, -glm::pi<float>() * 0.5f);
	
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

#if RENDERMODE == RM_METABALLS
	mbSampler.r1 = 0.03f; //below this distance, value contributions capped to 1
	mbSampler.r0 = simSettings.smoothingRadius; //above this distance, value contributions capped to 0

	if (!this->shMetaballs.init("shaders/metaball.vert", "shaders/metaball.frag")) return false;

	glUseProgram(shMetaballs.getID());
	glUniform4fv(glGetUniformLocation(this->shMetaballs.getID(), "coldColour"), 1, glm::value_ptr(coldColour));
	glUniform4fv(glGetUniformLocation(this->shMetaballs.getID(), "hotColour"), 1, glm::value_ptr(hotColour));
	glUseProgram(0);

	int mbGridDimPoints = 513; //how many rows/columns of points (minimum 2)
	mbSampler.numSamplePoints = mbGridDimPoints * mbGridDimPoints; //how many actual points
	glm::vec4* mbGridMeshPoints = new glm::vec4[mbSampler.numSamplePoints]; //normalised screen positions (-1 to +1), sampleValue, density
	
	for (float y = 0; y < mbGridDimPoints; y++)
	{
		for (float x = 0; x < mbGridDimPoints; x++)
		{
			float normFactor = 1.0f / (mbGridDimPoints - 1); //turn range (0 to n-1) to (-1 to +1)
			float yScreen = y * normFactor * 2.0f - 1.0f;
			float xScreen = x * normFactor * 2.0f - 1.0f;

			int i = y * mbGridDimPoints + x;
			mbGridMeshPoints[i] = glm::vec4(xScreen, yScreen, 0.0f, 0.0f);
		}
	}

	//triangles for quads which will display metaballs 
	//e.g. first (bottom left) quad will have points 0, 1, mbGridDimPoints, mbGridDimPoints + 1 as the corners
	
	int mbGridDimQuads = mbGridDimPoints - 1;
	mbSampler.numTriIndices = 6 * mbGridDimQuads * mbGridDimQuads;
	unsigned int* mbGridMeshIndices = new unsigned int[mbSampler.numTriIndices];

	for (int y = 0; y < mbGridDimQuads; y++)
	{
		for (int x = 0; x < mbGridDimQuads; x++)
		{
			unsigned int BL = y * mbGridDimPoints + x; //index of point in bottom left of current quad
			unsigned int BR = y * mbGridDimPoints + x + 1;
			unsigned int TL = (y + 1) * mbGridDimPoints + x;
			unsigned int TR = (y + 1) * mbGridDimPoints + x + 1;

			int gridMeshIndicesOffset = (y * mbGridDimQuads + x) * 6;

			mbGridMeshIndices[gridMeshIndicesOffset + 0] = BL;
			mbGridMeshIndices[gridMeshIndicesOffset + 1] = TL;
			mbGridMeshIndices[gridMeshIndicesOffset + 2] = TR;

			mbGridMeshIndices[gridMeshIndicesOffset + 3] = BL;
			mbGridMeshIndices[gridMeshIndicesOffset + 4] = TR;
			mbGridMeshIndices[gridMeshIndicesOffset + 5] = BR;
		}
	}

	glGenVertexArrays(1, &mbSampler.vao_samplePoints);
	glBindVertexArray(mbSampler.vao_samplePoints);
	
	glGenBuffers(1, &mbSampler.vbo_samplePoints);
	glBindBuffer(GL_ARRAY_BUFFER, mbSampler.vbo_samplePoints);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec4) * mbSampler.numSamplePoints, mbGridMeshPoints, GL_STATIC_DRAW);
	
	glGenBuffers(1, &mbSampler.ebo_samplePoints);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mbSampler.ebo_samplePoints);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * mbSampler.numTriIndices, mbGridMeshIndices, GL_STATIC_DRAW);
	
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), 0);
	glEnableVertexAttribArray(0);
	
	cudaGraphicsGLRegisterBuffer(&mbSampler.cuda_samplePointsBuffer, mbSampler.vbo_samplePoints,
		cudaGraphicsMapFlagsWriteDiscard);

	delete[] mbGridMeshPoints;
	delete[] mbGridMeshIndices;
#endif

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


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

void Renderer::visualise(SPHSolver& solver, bool enableTiming, std::vector<std::pair<std::string, float>>& timingValues)
{
	if (cam.getUpdated())
	{
		//give updated matrices to shader
		glUseProgram(this->shWorldToScreen.getID());
			int loc = glGetUniformLocation(this->shWorldToScreen.getID(), "matProjView");
			glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(cam.getMatrix()));
		glUseProgram(0);

		cam.clearUpdated();
	}


#if RENDERMODE == RM_METABALLS
	//call cuda to update density values at metaball grid points
	mapCudaResources();
	solver.mapCudaResources();
	int blockSize = glm::min(mbSampler.numSamplePoints, 1024);
	int numBlocks = (mbSampler.numSamplePoints - 1) / blockSize + 1;

	CUDAKernels::calculateMetaballSamples << <numBlocks, blockSize >> > (solver.getSimData(), solver.getSettings(),
		solver.getUniformGrid().getData(), solver.getUniformGrid().getSettings(), mbSampler, matScreenNormToWorld);
	solver.unmapCudaResources();
	unmapCudaResources();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error in renderer visualise with metaballs: " << cudaGetErrorName(err) << std::endl;
	}
#endif

	if (enableTiming)
	{
		//https://www.lighthouse3d.com/tutorials/opengl-timer-query/
		glBeginQuery(GL_TIME_ELAPSED, timeQueryID);
	}

#if RENDERMODE == RM_POINTS
	glUseProgram(this->shWorldToScreen.getID());
	//draw particles predicted positions
	/*glBindVertexArray(solver.getSimData().gl_predictedPositionsVAO);
	glUniform4fv(glGetUniformLocation(this->shWorldToScreen.getID(), "colour"), 1, glm::value_ptr(hotColour));
	glDrawArrays(GL_POINTS, 0, solver.getSettings().numParticles);*/

	//draw particles current positions
	glBindVertexArray(solver.getSimData().gl_positionsVAO);
	glUniform4fv(glGetUniformLocation(this->shWorldToScreen.getID(), "colour"), 1, glm::value_ptr(coldColour));
	glDrawArrays(GL_POINTS, 0, solver.getSettings().numParticles);
	
#elif RENDERMODE == RM_METABALLS	
	//render metaball grid
	glUseProgram(this->shMetaballs.getID());
	glBindVertexArray(mbSampler.vao_samplePoints);
	glDrawElements(GL_TRIANGLES, mbSampler.numTriIndices, GL_UNSIGNED_INT, 0);

	glUseProgram(this->shWorldToScreen.getID()); //for drawing boundaries, which wants to be done after metaballs
#endif

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
		glDeleteBuffers(1, &mbSampler.vbo_samplePoints);
		glDeleteBuffers(1, &mbSampler.ebo_samplePoints);

		glDeleteVertexArrays(1, &vao_boundaryLines);
		glDeleteVertexArrays(1, &vao_ugLines);
		glDeleteVertexArrays(1, &mbSampler.vao_samplePoints);

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

void Renderer::mapCudaResources()
{
	size_t _numbytes;
	cudaGraphicsMapResources(1, &mbSampler.cuda_samplePointsBuffer);
	cudaGraphicsResourceGetMappedPointer((void**)&mbSampler.d_sampleData, &_numbytes, mbSampler.cuda_samplePointsBuffer);
}

void Renderer::unmapCudaResources()
{
	cudaGraphicsUnmapResources(1, &mbSampler.cuda_samplePointsBuffer, 0);
	mbSampler.d_sampleData = nullptr;
}