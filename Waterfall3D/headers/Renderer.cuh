
#ifndef RENDERER_H
#define RENDERER_H


#include "glm/glm.hpp"

#include "DebugOptions.h"
#include "SPH.cuh"
#include "ShaderProgram.h"
#include "MetaballSampler.cuh"


class Camera
{
	//https://learnopengl.com/Getting-started/Camera

	glm::vec3 position, direction;
	glm::vec2 angles; //angles are x: pitch, y: yaw, (no roll)
	glm::mat4 matView, matProj;

	float fov, ar, clipNear, clipFar;

	bool initialised, updatedSinceLastFrame;

	void updateDirection();
	void updateViewMatrix();
	void updatePerspective();

public:
	const glm::vec2& getAngles() const { return angles; }
	const glm::mat4& getMatrix() const { return matProj * matView; }
	const float& getFieldOfView() const { return fov; }
	const bool getUpdated() const { return updatedSinceLastFrame; }

	void setFieldOfView(const float fieldOfView);
	void setAspectRatio(const float width, const float height);

	void clearUpdated() { updatedSinceLastFrame = false; }

	void init(const float aspectRatio);

	void updatePosition(const glm::vec3& deltaPos);
	void updateViewAngle(const glm::vec2& deltaAngles);

};


class Renderer
{
	glm::vec2 windowResolution;

	ShaderProgram shWorldToScreen, shMetaballs;
	unsigned int vao_boundaryLines, vbo_boundaryLines;
	unsigned int vao_ugLines, vbo_ugLines;
	
	MetaballSampler mbSampler;

	glm::vec4 coldColour, hotColour;

	bool showUniformGrid;

	unsigned int timeQueryID;
	uint64_t elapsedTime;

	bool initialised = false;

public:
	Camera cam;

	glm::vec2 getWindowResolution() const { return windowResolution; }
	bool getShowUniformGrid() const { return showUniformGrid; }

	void setShowUniformGrid(bool val) { showUniformGrid = val; }

	bool init(const uint32_t& resx, const uint32_t& resy, const glm::vec4& coldColour, const glm::vec4& hotColour,
		const SPHConfiguration& simSettings, const SPHSimulationData& simData, UniformGrid& uniformGrid);
	void visualise(SPHSolver& solver, bool enableTiming, std::vector<std::pair<std::string, float>>& timingValues);
	void destroy();

	void updateWindowRes(const int& width, const int& height);
	void updateCam(const glm::vec3& deltaPos, const glm::vec2& deltaAngles);

	void mapCudaResources();
	void unmapCudaResources();
};


#endif // !RENDERER_H