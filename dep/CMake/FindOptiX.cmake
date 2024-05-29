# Looks for the environment variable:
# OPTIX_PATH

# Sets the variables :
# OPTIX_INCLUDE_DIR

# OptiX_FOUND

# For easy printing for script debug. example: cmake_print_variables(CMAKE_CXX_STANDARD)
include(CMakePrintHelpers)

set(OPTIX_PATH $ENV{OPTIX_PATH})

if ("${OPTIX_PATH}" STREQUAL "")
    set(OPTIX_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0")
endif()

find_path(OPTIX_INCLUDE_DIR optix_host.h ${OPTIX_PATH}/include)

cmake_print_variables(OPTIX_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OptiX DEFAULT_MSG OPTIX_INCLUDE_DIR)

mark_as_advanced(OPTIX_INCLUDE_DIR)

cmake_print_variables(OPTIX_FOUND)
