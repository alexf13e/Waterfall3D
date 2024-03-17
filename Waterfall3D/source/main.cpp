
#include <Windows.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <map>

#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/vector_angle.hpp"
#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include "DebugOptions.h"
#include "SPH.cuh"
#include "CUDAFunctions.cuh"
#include "Renderer.cuh"
#include "Key.h"


constexpr float WINDOW_WIDTH = 1024;
constexpr float WINDOW_HEIGHT = 1024;
GLFWwindow* window;

SPHConfiguration settings;
SPHSolver solver;
Renderer renderer;

int simIterationsPerFrame;

glm::vec2 prevMousePosScreen;
float scrollSensitivity;
float scrollDeltaY;
bool scrollOccurred;

static std::map<int, Key> keyMap = {
	{GLFW_KEY_A, Key()},
	{GLFW_KEY_B, Key()},
	{GLFW_KEY_C, Key()},
	{GLFW_KEY_D, Key()},
	{GLFW_KEY_E, Key()},
	{GLFW_KEY_F, Key()},
	{GLFW_KEY_G, Key()},
	{GLFW_KEY_H, Key()},
	{GLFW_KEY_I, Key()},
	{GLFW_KEY_J, Key()},
	{GLFW_KEY_K, Key()},
	{GLFW_KEY_L, Key()},
	{GLFW_KEY_M, Key()},
	{GLFW_KEY_N, Key()},
	{GLFW_KEY_O, Key()},
	{GLFW_KEY_P, Key()},
	{GLFW_KEY_Q, Key()},
	{GLFW_KEY_R, Key()},
	{GLFW_KEY_S, Key()},
	{GLFW_KEY_T, Key()},
	{GLFW_KEY_U, Key()},
	{GLFW_KEY_V, Key()},
	{GLFW_KEY_W, Key()},
	{GLFW_KEY_X, Key()},
	{GLFW_KEY_Y, Key()},
	{GLFW_KEY_Z, Key()},
	{GLFW_KEY_0, Key()},
	{GLFW_KEY_1, Key()},
	{GLFW_KEY_2, Key()},
	{GLFW_KEY_3, Key()},
	{GLFW_KEY_4, Key()},
	{GLFW_KEY_5, Key()},
	{GLFW_KEY_6, Key()},
	{GLFW_KEY_7, Key()},
	{GLFW_KEY_8, Key()},
	{GLFW_KEY_9, Key()},
	{GLFW_KEY_LEFT, Key()},
	{GLFW_KEY_RIGHT, Key()},
	{GLFW_KEY_UP, Key()},
	{GLFW_KEY_DOWN, Key()},
	{GLFW_KEY_LEFT_SHIFT, Key()}
};

std::ofstream timingFile;
std::chrono::high_resolution_clock::time_point prevFrameEndTime;
float frameDuration;


static void windowResize(GLFWwindow* window, int width, int height)
{
	int widthpx, heightpx;
	glfwGetFramebufferSize(window, &widthpx, &heightpx);
	renderer.updateWindowRes(widthpx, heightpx);
}

static void keyPress(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (keyMap.count(key) == 0)
	{
		//key pressed which is not set to be used in the keymap
		return;
	}

	switch (action)
	{
	case GLFW_PRESS:
		keyMap.at(key).setDown();
		break;

	case GLFW_RELEASE:
		keyMap.at(key).setUp();
		break;
	}
}

static void mouseScroll(GLFWwindow* window, double dx, double dy)
{
	scrollDeltaY = dy;
	scrollOccurred = true;
}

void userInteractParticles(const glm::vec2& worldPos, bool attract)
{
	float radius = 2.0f;
	float acceleration = 2.0f;

	if (attract)
	{
		acceleration = -acceleration;
	}

	//solver.userInteractParticles(worldPos, radius, -acceleration);
	//renderer.drawCircle(worldPos, radius);
}

GLFWwindow* initGL()
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Water 2D", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return nullptr;
	}

	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return nullptr;
	}

	glfwSetFramebufferSizeCallback(window, windowResize);
	glfwSetKeyCallback(window, keyPress);
	glfwSetScrollCallback(window, mouseScroll);

	return window;
}

bool init()
{
	srand(std::chrono::system_clock::now().time_since_epoch().count());

	settings.timeStep = 0.003f;
	settings.numParticles = 2000;
	settings.particleMass = 2.0;
	settings.gravity = 9.0f;
	settings.viscosity = 0.3f;
	settings.smoothingRadius = 1.2;
	settings.stiffnessConstant = 300.0;
	settings.restDensity = 1.0f;
	settings.boundaryCollisionDamping = 0.5;
	settings.kernelPressure = SmoothingKernelType::SPIKY;
	settings.kernelPressureGradient = SmoothingKernelType::SPIKY_GRADIENT;
	settings.kernelViscosity = SmoothingKernelType::VISCOSITY;

	simIterationsPerFrame = 1;

	//(half of) the size of the region being simulated
	float boxWidth = 8.0f;
	//float boxWidth = glm::max(sqrtf(settings.numParticles) / 3.0f, 1.0f);

	if (solver.init(settings, 2 * boxWidth) == false)
	{
		std::cerr << "failed to initialise solver, exiting" << std::endl;
		return false;
	}

	solver.setInitialParticlePositions(0.5f);
	Boundary boundaries[6];
	glm::vec3 p1 = glm::vec3(-boxWidth, 0.0f, 0.0f); //left
	glm::vec3 p2 = glm::vec3(boxWidth, 0.0f, 0.0f); //right
	glm::vec3 p3 = glm::vec3(0.0f, -boxWidth, 0.0f); //bottom
	glm::vec3 p4 = glm::vec3(0.0f, boxWidth, 0.0f); //top
	glm::vec3 p5 = glm::vec3(0.0f, 0.0f, -boxWidth); //back
	glm::vec3 p6 = glm::vec3(0.0f, 0.0f, boxWidth); //front

	glm::vec3 n1 = glm::vec3(1.0f, 0.0f, 0.0f);
	glm::vec3 n2 = glm::vec3(-1.0f, 0.0f, 0.0f);
	glm::vec3 n3 = glm::vec3(0.0f, 1.0f, 0.0f);
	glm::vec3 n4 = glm::vec3(0.0f, -1.0f, 0.0f);
	glm::vec3 n5 = glm::vec3(0.0f, 0.0f, 1.0f);
	glm::vec3 n6 = glm::vec3(0.0f, 0.0f, -1.0f);

	boundaries[0] = { p1, n1 };
	boundaries[1] = { p2, n2 };
	boundaries[2] = { p3, n3 };
	boundaries[3] = { p4, n4 };
	boundaries[4] = { p5, n5 };
	boundaries[5] = { p6, n6 };

	solver.setWorldBoundaries(boundaries, 6);


	if (renderer.init(WINDOW_WIDTH, WINDOW_HEIGHT, glm::vec4(0.5f, 0.6f, 1.0f, 1.0f), glm::vec4(0.2f, 0.2f, 1.0f, 1.0f),
		solver.getSettings(), solver.getSimData(), solver.getUniformGrid()) == false)
	{
		std::cerr << "failed to initialise renderer, exiting" << std::endl;
		return false;
	}

	prevMousePosScreen = glm::vec2(0);
	scrollSensitivity = 0.1f;

#if ENABLE_TIMING_SPH || ENABLE_TIMING_GL
	//setup cout to print timings to file
	std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	char timeString[20];
	std::strftime(&timeString[0], sizeof(timeString), "%Y-%m-%d_%H-%M-%S", std::localtime(&t));

	std::string fileName = "timings_" + std::to_string(settings.numParticles) + "_" + std::string(timeString) + ".csv";
	timingFile = std::ofstream(fileName);
	std::cout.rdbuf(timingFile.rdbuf());
#if USE_UNIFORM_GRID
	std::cout << "UG update cell particles, UG sort, UG update cell starts, SPH update predicted positions, SPH calculate inter particle values, SPH accelerate particles, render" << std::endl;
#else
	std::cout << "SPH update predicted positions, SPH calculate inter particle values, SPH accelerate particles, render" << std::endl;
#endif
#endif

	prevFrameEndTime = std::chrono::high_resolution_clock::now();

	return true;
}

bool update()
{
	//mouse interactions
	//double mousex, mousey;
	//int windowHeight = renderer.getWindowResolution().y;
	//glfwGetCursorPos(window, &mousex, &mousey);
	//glm::vec2 currentMousePosScreen = glm::vec2(mousex, windowHeight - mousey);
	//glm::vec2 currentMousePosWorld = renderer.getviewMatrixScreenPixToWorld() * glm::vec4(currentMousePosScreen, 0.0f, 1.0f);

	//if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
	//{
	//	//want to pan in opposite direction to mouse drag, so the same point stays under the mouse
	//	glm::vec2 dragDistanceScreen = prevMousePosScreen - currentMousePosScreen;
	//	glm::vec2 dragDistanceWorld = glm::vec2(renderer.scalarScreenToWorld(dragDistanceScreen.x),
	//		renderer.scalarScreenToWorld(dragDistanceScreen.y));
	//	renderer.panView(dragDistanceWorld);
	//}

	//if (scrollOccurred && scrollDeltaY != 0)
	//{
	//	float amount = glm::pow(2, scrollDeltaY * scrollSensitivity);
	//	renderer.changeScaleFromPosition(amount, currentMousePosScreen);
	//	scrollOccurred = false;
	//}

	//currentMousePosWorld = renderer.getviewMatrixScreenPixToWorld() * glm::vec4(currentMousePosScreen, 0.0f, 1.0f);

	//if (keyMap[GLFW_KEY_F].getHeld())
	//{
	//	bool attract = keyMap[GLFW_KEY_LEFT_SHIFT].getHeld();
	//	userInteractParticles(glm::vec2(0), attract);
	//}

	//prevMousePosScreen = currentMousePosScreen;

	////key interactions
	//if (keyMap.at(GLFW_KEY_F).getHeld())
	//{
	//	float attractionRadius = 5.0f;
	//	float attractionVelocity = 5.0f;
	//	if (keyMap.at(GLFW_KEY_LEFT_SHIFT).getHeld())
	//	{
	//		attractionVelocity = -attractionVelocity;
	//	}

	//	solver.userInteractParticles(currentMousePosWorld, attractionRadius, attractionVelocity);
	//}

	//camera controls
	glm::vec3 moveInputs = glm::vec3(0.0f);
	glm::vec2 lookInputs = glm::vec2(0.0f);
	float moveMult = 1.0f;

	if (keyMap.at(GLFW_KEY_W).getHeld()) moveInputs.z -= 1;
	if (keyMap.at(GLFW_KEY_S).getHeld()) moveInputs.z += 1;
	if (keyMap.at(GLFW_KEY_A).getHeld()) moveInputs.x -= 1;
	if (keyMap.at(GLFW_KEY_D).getHeld()) moveInputs.x += 1;
	if (keyMap.at(GLFW_KEY_Q).getHeld()) moveInputs.y -= 1;
	if (keyMap.at(GLFW_KEY_E).getHeld()) moveInputs.y += 1;
	
	if (keyMap.at(GLFW_KEY_UP).getHeld()) lookInputs.x += 1;
	if (keyMap.at(GLFW_KEY_DOWN).getHeld()) lookInputs.x -= 1;
	if (keyMap.at(GLFW_KEY_LEFT).getHeld()) lookInputs.y -= 1;
	if (keyMap.at(GLFW_KEY_RIGHT).getHeld()) lookInputs.y += 1;

	if (keyMap.at(GLFW_KEY_C).getHeld()) moveMult = 4.0f;

	float moveSpeed = 5.0f * moveMult * frameDuration;
	float lookSpeed = glm::pi<float>() * frameDuration;
	float camYaw = renderer.getCamAngles().y;

	renderer.updateCam(glm::rotateY(moveInputs, -camYaw - glm::radians(90.0f)) * moveSpeed, lookInputs * lookSpeed);



	//simulation
#if USE_UNIFORM_GRID
	solver.UGUpdate(simIterationsPerFrame);
#else
	solver.update(simIterationsPerFrame);
#endif

	//visualisation
	renderer.visualise(solver);

#if ENABLE_TIMING_SPH || ENABLE_TIMING_GL
	std::cout << std::endl;
#endif

	std::chrono::high_resolution_clock::time_point currentFrameEndTime = std::chrono::high_resolution_clock::now();
	frameDuration = (currentFrameEndTime - prevFrameEndTime).count() * 1e-9;
	prevFrameEndTime = currentFrameEndTime;

#if !(ENABLE_TIMING_SPH || ENABLE_TIMING_GL)
	std::cout << std::to_string(frameDuration) << '\r';
#endif

	return true;
}

void destroy()
{
	solver.destroy();
	renderer.destroy();
}

int main()
{
	window = initGL();
	if (window == nullptr) return -1;

	if (!init())
	{
		return -1;
	}

	while (!glfwWindowShouldClose(window))
	{
		//process key presses, mouse movements, window resizes etc.
		glfwPollEvents();
		
		//make things happen
		update();

		glfwSwapBuffers(window);
	}

	destroy();
	glfwTerminate();
	return 0;
}