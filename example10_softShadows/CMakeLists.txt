set(SUBPROJECT_NAME "ex10")


 # Include Directories
list(APPEND INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${OPTIX_INCLUDE_DIR}
    ${Stb_INCLUDE_DIR}
)

# Cuda Compile Options
set(CUDA_COMPILE_OPTIONS
	--machine=64
    --gpu-architecture=compute_80
    --use_fast_math
	--relocatable-device-code=true
    -Wno-deprecated-gpu-targets
    -diag-suppress 20012
    --extended-lambda
    $<$<AND:$<CONFIG:Debug>>:-G>
    $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:-O3>
)

# Create object library for OptiX files
file(
    GLOB
    CUDA_OPTIXIR_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/optixir/*.cu
)
set(OPTIXIR_TARGET ${SUBPROJECT_NAME}_OptixIR_Compilation)
add_library(${OPTIXIR_TARGET} OBJECT ${CUDA_OPTIXIR_FILES})
set_property(TARGET ${OPTIXIR_TARGET} PROPERTY CUDA_OPTIX_COMPILATION ON)
target_include_directories(${OPTIXIR_TARGET} PRIVATE ${INCLUDE_DIRS})
target_compile_options(${OPTIXIR_TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_OPTIONS}>)
set_target_properties(${OPTIXIR_TARGET} PROPERTIES
	CUDA_ARCHITECTURES OFF
	CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
	CUDA_VERBOSE_BUILD ON
)
target_compile_definitions(${OPTIXIR_TARGET} PRIVATE $<$<CONFIG:Debug>:DEBUG>)

# Main Executable Target
add_executable(${SUBPROJECT_NAME} main.cpp)

file(
    GLOB
    COMMON_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/common/OptixUtil.*
    ${CMAKE_SOURCE_DIR}/common/CudaUtil.*
    ${CMAKE_SOURCE_DIR}/common/IOUtil.*
    ${CMAKE_SOURCE_DIR}/common/CUDABuffer.*
    ${CMAKE_SOURCE_DIR}/common/LaunchParams.*
    ${CMAKE_SOURCE_DIR}/common/GLFWindow.*
    ${CMAKE_SOURCE_DIR}/common/GLFCameraWindow.*
    ${CMAKE_SOURCE_DIR}/common/TriangleMesh.*
    ${CMAKE_SOURCE_DIR}/common/Camera.*
    ${CMAKE_SOURCE_DIR}/common/CameraFrame.*
    ${CMAKE_SOURCE_DIR}/common/CameraFrameManip.*
    ${CMAKE_SOURCE_DIR}/common/InspectModeManip.*
    ${CMAKE_SOURCE_DIR}/common/FlyModeManip.*
    ${CMAKE_SOURCE_DIR}/common/TriangleMeshSBTData.*
    ${CMAKE_SOURCE_DIR}/common/Model.*
    ${CMAKE_SOURCE_DIR}/common/Box.*
    ${CMAKE_SOURCE_DIR}/common/Texture.*
    ${CMAKE_SOURCE_DIR}/common/QuadLight.*
    ${CMAKE_SOURCE_DIR}/common/Random.*
)

file(
    GLOB
    SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)

target_sources(
    ${SUBPROJECT_NAME}
    PRIVATE
    ${SOURCE_FILES}
    ${COMMON_SOURCE_FILES}
)

add_dependencies(
    ${SUBPROJECT_NAME}
	${OPTIXIR_TARGET}
)

target_include_directories(
    ${SUBPROJECT_NAME}
    PRIVATE
    ${INCLUDE_DIRS}
)

target_compile_options(
    ${SUBPROJECT_NAME}
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_OPTIONS}>
)

target_compile_definitions(${SUBPROJECT_NAME} PRIVATE $<$<CONFIG:Debug>:DEBUG>)

set_target_properties(${SUBPROJECT_NAME} PROPERTIES
	CUDA_ARCHITECTURES OFF
	CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
	CUDA_VERBOSE_BUILD ON
)

target_link_libraries(
	${SUBPROJECT_NAME}
    PRIVATE
	imgui::imgui
    glm::glm-header-only
	glfw
	glad::glad
	spdlog::spdlog_header_only
	tinyobjloader::tinyobjloader
	CUDA::cudart
	CUDA::cuda_driver
    CUDA::nvtx3
)

set(usePrecompiledHeaders 0)
if(usePrecompiledHeaders)
  target_precompile_headers(
    ${SUBPROJECT_NAME}
    PRIVATE
    common/stdafx.h
  )
endif()

# Generated files
set(GENERATED_H_FILE CMakeGenerated.h)
configure_file(
  ${CMAKE_SOURCE_DIR}/config/${GENERATED_H_FILE}.in
  ${GENERATED_H_FILE} # NOTE: This will get generated in build/generated
  @ONLY # If @ONLY is specified, only variables of the form @VAR@ will be replaced and ${VAR} will be ignored. This is useful for configuring scripts that use ${VAR}.
)
file(COPY_FILE ${CMAKE_CURRENT_BINARY_DIR}/${GENERATED_H_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/${GENERATED_H_FILE})

#############################################################
##################   POST_BUILD   ###########################
#############################################################
add_custom_command(TARGET ${SUBPROJECT_NAME} POST_BUILD
				   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/../compile_commands.json ${CMAKE_SOURCE_DIR}
                   COMMENT "Copying compile_commands.json to root for non-msvc text editors..."
)
add_custom_command(TARGET ${SUBPROJECT_NAME} POST_BUILD
				   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${OPTIXIR_TARGET}.dir/optixir ${CMAKE_CURRENT_SOURCE_DIR}/optixir
                   COMMENT "Copying .optixir files"
)
