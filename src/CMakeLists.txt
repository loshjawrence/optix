set(DEPENDENCIES_DIR ${PROJECT_SOURCE_DIR}/dep)

file(
    GLOB SOURCE_FILES
    *.cpp
    *.h
    ${DEPENDENCIES_DIR}/dx/DDSTextureLoader.h
    ${DEPENDENCIES_DIR}/dx/DDSTextureLoader.cpp
)
file(GLOB CUDA_FILES cuda/*.cu)
file(GLOB CUDA_OPTIXIR_FILES optixir/*.cu)
file(GLOB CUDA_PTX_FILES ptx/*.cu)


 # Include Directories
list(APPEND INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/${PROJECT_NAME}/src
    ${CMAKE_SOURCE_DIR}/assets/fonts
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${OPTIX_INCLUDE_DIR}
    ${Stb_INCLUDE_DIR}
)

# Cuda Compile Options
set(CUDA_COMPILE_OPTIONS
    #--relocatable-device-code=true
    #-gencode=arch=compute_70,code=\"sm_70,compute_70\"
    #-x cu
    #--maxrregcount=0
    #--machine 64
    #--compile
    #-cudart static
    --use_fast_math
    --extended-lambda
    --use-local-env
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

##ensure the files are copied even if the main target is not built
#add_custom_target(${OPTIXIR_TARGET}_copyFiles ALL
#    COMMAND ${CMAKE_COMMAND} -E copy_directory
#    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/CMakeFiles/${OPTIXIR_TARGET}.dir/OptixIr/"
#    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/optixir/"
#    COMMENT "Copying OptiX IR files to destination directory"
#    DEPENDS ${OPTIXIR_TARGET}
#)
#add_custom_target(${PTX_TARGET}_copyFiles ALL
#    COMMAND ${CMAKE_COMMAND} -E copy_directory
#    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/CMakeFiles/${PTX_TARGET}.dir/Ptx/"
#    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/ptx/"
#    COMMENT "Copying PTX files to destination directory"
#    DEPENDS ${PTX_TARGET}
#)


# Main Executable Target
add_executable(${PROJECT_NAME} main.cpp)


# TODO: make sure this is correct
target_sources(${PROJECT_NAME} PRIVATE ${SOURCE_FILES} ${CUDA_FILES} )

#add_dependencies(${PROJECT_NAME} ${OPTIXIR_TARGET} ${PTX_TARGET} ${OPTIXIR_TARGET}_copyFiles ${PTX_TARGET}_copyFiles)
add_dependencies(${PROJECT_NAME} ${OPTIXIR_TARGET} ${PTX_TARGET})

target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_OPTIONS}>)

target_compile_definitions(${PROJECT_NAME} PRIVATE $<$<CONFIG:Debug>:DEBUG>)

set_target_properties(${PROJECT_NAME} PROPERTIES
	CUDA_ARCHITECTURES native
	CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
	CUDA_VERBOSE_BUILD ON
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  imgui::imgui
  glfw
  glad::glad
  #CUDA::cudart_static
  CUDA::cudart
  CUDA::cuda_driver
  CUDA::nvtx3
  CUDA::cufft
  CUDA::curand
  CUDA::cublas
  CUDA::cublasLt
)

# Post Build Commands
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
				   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/../compile_commands.json ${PROJECT_SOURCE_DIR}
                   COMMENT "Copying compile_commands.json to root for non-msvc text editors..."
)
