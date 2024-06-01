set(SUBPROJECT_NAME "ex02")


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
    #--generate-line-info
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
set(OPTIXIR_TARGET "OptixIR_Compilation")
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
    ${CMAKE_SOURCE_DIR}/common/*.cpp
    ${CMAKE_SOURCE_DIR}/common/*.h
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
	#NOTE: file is copied to /src at PRE_BUILD
    #TODO: is it needed if its getting copied
	${CMAKE_CURRENT_BINARY_DIR}/generated # NOTE: this is out/build/<target>/generated
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
	CUDA::cudart
	CUDA::cuda_driver
    CUDA::nvtx3
    #CUDA::cufft CUDA::curand CUDA::cublas CUDA::cublasLt
)

set(usePrecompiledHeaders 0)
if(usePrecompiledHeaders)
  target_precompile_headers(
    ${SUBPROJECT_NAME}
    PRIVATE
    src/stdafx.h
  )
endif()

# Generated files
configure_file(
  ${CMAKE_SOURCE_DIR}/config/CMakeGenerated.h.in
  generated/CMakeGenerated.h # NOTE: This will get generated in build/generated
  @ONLY # If @ONLY is specified, only variables of the form @VAR@ will be replaced and ${VAR} will be ignored. This is useful for configuring scripts that use ${VAR}.
)
#############################################################
##################   PRE_BUILD   ###########################
#############################################################
# NOTE: build step can be PRE_BUILD PRE_LINK or POST_BUILD
list(
  APPEND PreBuildFilesToCopyToProjectSrc
  ${CMAKE_CURRENT_BINARY_DIR}/generated/CMakeGenerated.h
)
foreach(_file ${PreBuildFilesToCopyToProjectSrc})
  add_custom_command(
    TARGET ${SUBPROJECT_NAME} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${_file} ${CMAKE_CURRENT_SOURCE_DIR}
  )
endforeach()

#############################################################
##################   POST_BUILD   ###########################
#############################################################
add_custom_command(TARGET ${SUBPROJECT_NAME} POST_BUILD
				   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/../compile_commands.json ${CMAKE_SOURCE_DIR}
                   COMMENT "Copying compile_commands.json to root for non-msvc text editors..."
)
add_custom_command(TARGET ${SUBPROJECT_NAME} POST_BUILD
				   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/OptixIR_Compilation.dir/optixir ${CMAKE_CURRENT_SOURCE_DIR}/optixir
                   COMMENT "Copying .optixir files"
)
