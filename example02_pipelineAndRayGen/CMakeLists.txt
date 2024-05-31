set(SUBPROJECT_NAME "ex02")

file(
    GLOB
    COMMON_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/common/*.cpp
    ${PROJECT_SOURCE_DIR}/common/*.h
)

file(
    GLOB
    SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)

file(
    GLOB
	CUDA_FILES
	${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu
)

file(
    GLOB
    CUDA_OPTIXIR_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/optixir/*.cu
)

file(
    GLOB
    CUDA_PTX_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/ptx/*.cu
)

 # Include Directories
list(APPEND INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/common
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${OPTIX_INCLUDE_DIR}
    ${Stb_INCLUDE_DIR}
)

# Cuda Compile Options
set(CUDA_COMPILE_OPTIONS
    --use_fast_math
    --extended-lambda
    $<$<AND:$<CONFIG:Debug>>:-G>
    $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:-O3>
)

# Create object library for OptiX files
set(OPTIXIR_TARGET "OptixIR_Compilation")
add_library(${OPTIXIR_TARGET} OBJECT ${CUDA_OPTIXIR_FILES})
set_property(TARGET ${OPTIXIR_TARGET} PROPERTY CUDA_OPTIX_COMPILATION ON)
target_include_directories(${OPTIXIR_TARGET} PRIVATE ${INCLUDE_DIRS})
target_compile_options(${OPTIXIR_TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_OPTIONS}>)
set_target_properties(${OPTIXIR_TARGET} PROPERTIES
	CUDA_ARCHITECTURES native
	CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
	CUDA_VERBOSE_BUILD ON
)
target_compile_definitions(${OPTIXIR_TARGET} PRIVATE $<$<CONFIG:Debug>:DEBUG>)

# Create object library for PTX files
set(PTX_TARGET "PTX_Compilation")
add_library(${PTX_TARGET} OBJECT ${CUDA_PTX_FILES})
set_property(TARGET ${PTX_TARGET} PROPERTY CUDA_PTX_COMPILATION ON)
target_include_directories(${PTX_TARGET} PRIVATE ${INCLUDE_DIRS})
target_compile_options(${PTX_TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_OPTIONS}>)
set_target_properties(${PTX_TARGET} PROPERTIES
	CUDA_ARCHITECTURES native
	CUDA_SEPARABLE_COMPILATION ON
	POSITION_INDEPENDENT_CODE ON
	CUDA_VERBOSE_BUILD ON
)
target_compile_definitions(${PTX_TARGET} PRIVATE $<$<CONFIG:Debug>:DEBUG>)

# Main Executable Target
add_executable(${SUBPROJECT_NAME} main.cpp)

target_sources(
    ${SUBPROJECT_NAME}
    PRIVATE
    ${SOURCE_FILES}
    ${COMMON_SOURCE_FILES}
    ${CUDA_FILES}
)

add_dependencies(${SUBPROJECT_NAME} ${OPTIXIR_TARGET} ${PTX_TARGET})

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
	CUDA_ARCHITECTURES native
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
	CUDA::cudart
	CUDA::cuda_driver
    CUDA::nvtx3
    #CUDA::cufft CUDA::curand CUDA::cublas CUDA::cublasLt
)

# Post Build Commands
add_custom_command(TARGET ${SUBPROJECT_NAME} POST_BUILD
				   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/../compile_commands.json ${PROJECT_SOURCE_DIR}
                   COMMENT "Copying compile_commands.json to root for non-msvc text editors..."
)