cmake_minimum_required(VERSION 3.28)

set(PROJECT_NAME "optix")

# For easy printing for script debug. example cmake_print_variables(CMAKE_CXX_STANDARD)
include(CMakePrintHelpers)

# Generates a compile_commands.json that some text editors use for their language servers symbol search
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(VCPKG_TARGET_TRIPLET x64-windows)

set(EXTERNAL ${CMAKE_SOURCE_DIR}/ext)

project(${PROJECT_NAME} VERSION 1.0 LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

find_package(CUDAToolkit)
find_package(OptiX REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
add_subdirectory(src)
