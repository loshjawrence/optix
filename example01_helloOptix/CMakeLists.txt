set(SUBPROJECT_NAME "ex01")
file(
    GLOB COMMON_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/common/*.cpp
    ${PROJECT_SOURCE_DIR}/common/*.h
)

file(
    GLOB SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)

 # Include Directories
list(APPEND INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/common
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${OPTIX_INCLUDE_DIR}
    ${Stb_INCLUDE_DIR}
)

# Main Executable Target
add_executable(${SUBPROJECT_NAME} main.cpp)

target_sources(${SUBPROJECT_NAME} PRIVATE ${SOURCE_FILES} ${COMMON_SOURCE_FILES})

target_include_directories(${SUBPROJECT_NAME} PRIVATE ${INCLUDE_DIRS})

target_link_libraries(${SUBPROJECT_NAME} PRIVATE
  spdlog::spdlog_header_only
  CUDA::cudart
)

# Post Build Commands
add_custom_command(TARGET ${SUBPROJECT_NAME} POST_BUILD
				   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/../compile_commands.json ${PROJECT_SOURCE_DIR}
                   COMMENT "Copying compile_commands.json to root for non-msvc text editors..."
)