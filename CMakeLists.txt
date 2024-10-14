cmake_minimum_required(VERSION 3.16...3.26)

project(
    FastNeurons
    VERSION 0.0.1
    DESCRIPTION "A simple, fast, CUDA-parallelized neural network and machine learning library."
    LANGUAGES CXX CUDA
)


# Check CUDA compatibility

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)

enable_language(CUDA)



# Setup basics

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

set(NEURONS_OUT_DIR "${CMAKE_SOURCE_DIR}/out")
set(NEURONS_BIN_DIR "${NEURONS_OUT_DIR}/bin")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)



# Setup dependencies

# include(FetchContent)

# FetchContent_Declare(
#     TOOL
#     GIT_REPOSITORY https://github.com/crazysissen/TOOL.git
#     GIT_TAG origin/main
# )

# FetchContent_MakeAvailable(TOOL)



# Configure build

add_compile_definitions()



# Add project code

add_subdirectory(source)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_subdirectory(test)
endif()



# End

endif() # For check_language