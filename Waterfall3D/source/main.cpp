
#include <Windows.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <map>

#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/vector_angle.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

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
bool useUniformGrid;
float simBoxSize;

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
bool etSimulation, etRender; //enable timings
bool saveTimings; //capture timings and write to file
std::vector<std::pair<std::string, float>> timingValues; //string for name of time value, float for value itself

bool simulationInitialised;
bool paused, step;


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

void initImGui()
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init();
	ImGui::GetStyle().ScaleAllSizes(2.0f);

	//default simulation settings for use in ui
	settings.timeStep = 0.003f;
	settings.numParticles = 2000;
	settings.particleMass = 3.0;
	settings.gravity = glm::vec3(0.0f, -9.81f, 0.0f);
	settings.viscosity = 0.3f;
	settings.smoothingRadius = 2;
	settings.stiffnessConstant = 300.0;
	settings.restDensity = 1.0f;
	settings.boundaryCollisionDamping = 0.5;
	settings.kernelPressure = SmoothingKernelType::SPIKY;
	settings.kernelPressureGradient = SmoothingKernelType::SPIKY_GRADIENT;
	settings.kernelViscosity = SmoothingKernelType::VISCOSITY;

	simIterationsPerFrame = 1;
	useUniformGrid = true;

	//(half of) the size of the region being simulated
	simBoxSize = 8.0f;
	//float boxWidth = glm::max(sqrtf(settings.numParticles) / 3.0f, 1.0f);

	paused = false;
	step = false;

	etSimulation = false;
	etRender = false;
	saveTimings = false;
}

bool initSim()
{
	if (solver.init(settings, 2 * simBoxSize) == false)
	{
		std::cerr << "failed to initialise solver, exiting" << std::endl;
		return false;
	}

	solver.setInitialParticlePositions(0.5f);
	Boundary boundaries[6];
	glm::vec3 p1 = glm::vec3(-simBoxSize, 0.0f, 0.0f); //left
	glm::vec3 p2 = glm::vec3(simBoxSize, 0.0f, 0.0f); //right
	glm::vec3 p3 = glm::vec3(0.0f, -simBoxSize, 0.0f); //bottom
	glm::vec3 p4 = glm::vec3(0.0f, simBoxSize, 0.0f); //top
	glm::vec3 p5 = glm::vec3(0.0f, 0.0f, -simBoxSize); //back
	glm::vec3 p6 = glm::vec3(0.0f, 0.0f, simBoxSize); //front

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

	prevFrameEndTime = std::chrono::high_resolution_clock::now();
	
	simulationInitialised = true;

	return true;
}

void destroySim(); //only function that needs forward declaring...
bool updateSim()
{
	SPHConfiguration runningSettings = solver.getSettings();

	if (ImGui::Button("Stop simulation"))
	{
		destroySim();
		settings = runningSettings;

		return true;
	}
	ImGui::SameLine();
	if (ImGui::Button(paused ? "UnPause" : "Pause"))
	{
		paused = !paused;
	}

	ImGui::SeparatorText("Simulation settings");
	if (ImGui::DragFloat("Time step (seconds)", &runningSettings.timeStep, 0.0001f, 0.0001f, 1.0f))
	{
		solver.setTimeStep(runningSettings.timeStep);
	}

	if (ImGui::DragFloat("Particle mass", &runningSettings.particleMass, 0.005f, 0.001f, 10.0f))
	{
		solver.setParticleMass(runningSettings.particleMass);
	}

	if (ImGui::DragFloat3("Gravity", glm::value_ptr(runningSettings.gravity), 0.01f))
	{
		solver.setGravity(runningSettings.gravity);
	}

	if (ImGui::DragFloat("Viscosity", &runningSettings.viscosity, 0.005f, 0.001f, 10.0f))
	{
		solver.setViscosity(runningSettings.viscosity);
	}

	if (ImGui::DragFloat("Stiffness constant", &runningSettings.stiffnessConstant, 1.0f, 1.0f, 1000.0f))
	{
		solver.setStiffnessConstant(runningSettings.stiffnessConstant);
	}

	if (ImGui::DragFloat("Rest density", &runningSettings.restDensity, 0.005f, 0.001f, 20.0f))
	{
		solver.setRestDensity(runningSettings.restDensity);
	}

	if (ImGui::DragFloat("Boundary collision damping", &runningSettings.boundaryCollisionDamping, 0.001f, 0.0f, 1.0f))
	{
		solver.setBoundaryCollisionDamping(runningSettings.boundaryCollisionDamping);
	}

	int oldSimIterationsPerFrame = simIterationsPerFrame;
	if (ImGui::InputInt("Sim iterations per frame", &simIterationsPerFrame))
	{
		//if timing enabled, only want to run 1 iteration per frame (to prevent display issues)
		if (etSimulation) simIterationsPerFrame = oldSimIterationsPerFrame;

		//cap iterations per frame to 1 or more
		if (simIterationsPerFrame <= 0) simIterationsPerFrame = 1;
	}

	ImGui::Checkbox("Use uniform grid", &useUniformGrid);

	ImGui::SeparatorText("Visual settings");
	float camFov = glm::degrees(renderer.cam.getFieldOfView());
	if (ImGui::SliderFloat("Field of view", &camFov, 10.0f, 110.0f))
	{
		renderer.cam.setFieldOfView(glm::radians(camFov));
	}

	if (useUniformGrid)
	{
		bool imguiShowUniformGrid = renderer.getShowUniformGrid();
		if (ImGui::Checkbox("Show uniform grid", &imguiShowUniformGrid))
		{
			renderer.setShowUniformGrid(imguiShowUniformGrid);
		}
	}
	else
	{
		renderer.setShowUniformGrid(false);
	}

	ImGui::SeparatorText("Timing");
	if (ImGui::Checkbox("Enable timing simulation", &etSimulation))
	{
		if (etSimulation)
		{
			//the only sensible solution I could come up with to not have issues displaying/saving timings
			simIterationsPerFrame = 1;
		}
	}
	ImGui::Checkbox("Enable timing render", &etRender);

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
	if (keyMap.at(GLFW_KEY_P).getReleased())
		paused = !paused;
	if (keyMap.at(GLFW_KEY_O).getReleased()) step = true;

	for (auto it = keyMap.begin(); it != keyMap.end(); it++)
	{
		Key& k = it->second;
		k.updateStates();
	}

	float moveSpeed = 5.0f * moveMult * frameDuration;
	float lookSpeed = glm::pi<float>() * frameDuration;
	float camYaw = renderer.cam.getAngles().y;

	renderer.updateCam(glm::rotateY(moveInputs, -camYaw - glm::radians(90.0f)) * moveSpeed, lookInputs * lookSpeed);

	//simulation
	if (paused == false || step == true)
	{
		if (useUniformGrid) solver.UGUpdate(simIterationsPerFrame, etSimulation, timingValues);
		else solver.update(simIterationsPerFrame, etSimulation, timingValues);

		step = false;
	}

	//visualisation
	renderer.visualise(solver, etRender, timingValues);


	//timing
	std::chrono::high_resolution_clock::time_point currentFrameEndTime = std::chrono::high_resolution_clock::now();
	frameDuration = (currentFrameEndTime - prevFrameEndTime).count() * 1e-9;
	prevFrameEndTime = currentFrameEndTime;
	timingValues.push_back({ "Whole Frame", frameDuration });

	if (ImGui::BeginTable("Timings", timingValues.size()))
	{
		//create header row
		ImGui::TableNextRow();
		for (int col = 0; col < timingValues.size(); col++)
		{
			ImGui::TableSetColumnIndex(col);
			ImGui::Text(timingValues[col].first.c_str());
		}

		//write values in next row
		ImGui::TableNextRow();
		for (int col = 0; col < timingValues.size(); col++)
		{
			ImGui::TableSetColumnIndex(col);
			ImGui::Text(std::to_string(timingValues[col].second).c_str());
		}

		ImGui::EndTable();
	}

	timingValues.clear();
}

bool update()
{
	//ImGui new frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	
	ImGui::Begin("Waterfall 3D");

	if (simulationInitialised)
	{
		//run simulation, display options for during simulation running
		updateSim();
	}
	else
	{
		//display options for simulation config
		if (ImGui::Button("Start simulation"))
		{
			if (!initSim())
			{
				return false;
			}
		}

		ImGui::SeparatorText("Simulation settings");
		//can change while running, but are also here for being set before initialising simulation
		if (ImGui::InputFloat("Time step (seconds)", &settings.timeStep))
		{
			settings.timeStep = glm::max(settings.timeStep, 0.00001f);
		}

		if (ImGui::InputFloat("Particle mass", &settings.particleMass))
		{
			settings.particleMass = glm::max(settings.particleMass, 0.001f);
		}

		ImGui::InputFloat3("Gravity", glm::value_ptr(settings.gravity));

		if (ImGui::InputFloat("Viscosity", &settings.viscosity))
		{
			settings.viscosity = glm::max(settings.viscosity, 0.001f);
		}

		if (ImGui::InputFloat("Stiffness constant", &settings.stiffnessConstant))
		{
			settings.stiffnessConstant = glm::max(settings.stiffnessConstant, 0.001f);
		}

		if (ImGui::InputFloat("Rest density", &settings.restDensity))
		{
			settings.restDensity = glm::max(settings.restDensity, 0.001f);
		}

		if (ImGui::InputFloat("Boundary collision damping", &settings.restDensity))
		{
			settings.restDensity = glm::max(settings.restDensity, 0.001f);
		}

		//cannot change while running
		if (ImGui::InputInt("Number of particles", &settings.numParticles))
		{
			settings.numParticles = glm::max(settings.numParticles, 1);
		}

		if (ImGui::InputFloat("Smoothing radius", &settings.smoothingRadius))
		{
			settings.smoothingRadius = glm::max(settings.smoothingRadius, 0.001f);
		}
	}

	ImGui::End();

	//ImGui rendering
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	return true;
}

void destroySim()
{
	if (simulationInitialised)
	{
		solver.destroy();
		renderer.destroy();

		simulationInitialised = false;
	}
}

void destroy()
{
	destroySim();

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

int main()
{
	window = initGL();
	if (window == nullptr) return -1;

	initImGui();

	while (!glfwWindowShouldClose(window))
	{
		//process key presses, mouse movements, window resizes etc.
		glfwPollEvents();
		
		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		//make things happen
		if (!update())
		{
			//there was an error
			break;
		}

		glfwSwapBuffers(window);
	}

	destroy();
	glfwTerminate();
	return 0;
}