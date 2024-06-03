cmake_minimum_required(VERSION 3.28)

# For easy printing for script debug. example: cmake_print_variables(CMAKE_CXX_STANDARD)
include(CMakePrintHelpers)

set(PROJECT_NAME "optix")

# Generates a compile_commands.json that some text editors use for their language servers symbol search
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(VCPKG_TARGET_TRIPLET x64-windows)

set(DEPENDENCIES_DIR ${PROJECT_SOURCE_DIR}/dep)

project(${PROJECT_NAME} VERSION 1.0 LANGUAGES CUDA CXX)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS True)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_EXTENSIONS True)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/dep/CMake)
cmake_print_variables(CMAKE_MODULE_PATH)

find_package(CUDAToolkit REQUIRED)
find_package(OptiX REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

add_subdirectory(example01_helloOptix)
add_subdirectory(example02_pipelineAndRayGen)
add_subdirectory(example03_inGLFWindow)
add_subdirectory(example04_firstTriangleMesh)
add_subdirectory(example05_firstSBTData)
