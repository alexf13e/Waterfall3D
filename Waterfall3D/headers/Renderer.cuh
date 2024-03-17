
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
	glm::vec2 angles; //angles are x: pitch, y: yaw (no roll)
	glm::mat4 matView, matProj;

	bool initialised, updatedSinceLastFrame;

	void updateDirection();
	void updateViewMatrix();

public:
	const glm::vec2& getAngles() const { return angles; }
	const glm::mat4& getMatrix() const { return matProj * matView; }
	const bool getUpdated() const { return updatedSinceLastFrame; }

	void clearUpdated() { updatedSinceLastFrame = false; }

	void init();

	void updatePosition(const glm::vec3& deltaPos);
	void updateViewAngle(const glm::vec2& deltaAngles);

};


class Renderer
{
	glm::vec2 windowResolution;

	Camera cam;

	ShaderProgram shWorldToScreen, shMetaballs;
	unsigned int vao_fullscreenTri;
	unsigned int vao_boundaryLines, vbo_boundaryLines;
	unsigned int vao_ugLines, vbo_ugLines;
	
	MetaballSampler mbSampler;

	glm::vec4 coldColour, hotColour;

	bool initialised = false;

public:
	glm::vec2 getWindowResolution() const { return windowResolution; }

	bool init(const uint32_t& resx, const uint32_t& resy, const glm::vec4& coldColour, const glm::vec4& hotColour,
		const SPHConfiguration& simSettings, const SPHSimulationData& simData, UniformGrid& uniformGrid);
	void visualise(SPHSolver& solver);
	void destroy();

	void updateWindowRes(const int& width, const int& height);
	void updateCam(const glm::vec3& deltaPos, const glm::vec2& deltaAngles);
	const glm::vec2& getCamAngles() const { return cam.getAngles(); }

	void mapCudaResources();
	void unmapCudaResources();
};


#endif // !RENDERER_H