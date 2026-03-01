cmake_minimum_required(VERSION 3.16)
project(libint_cpp LANGUAGES CXX)

# --------------------------------------------------
# Settings
# --------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --------------------------------------------------
# Find dependencies
# --------------------------------------------------

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Libint2 REQUIRED)

# --------------------------------------------------
# Build module
# --------------------------------------------------

pybind11_add_module(libint_cpp
    bindings.cpp
    integrals_core.cpp
    cp2k_parser.cpp
)

target_include_directories(libint_cpp PRIVATE
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(libint_cpp PRIVATE
    Libint2::cxx
    Eigen3::Eigen
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

target_compile_options(libint_cpp PRIVATE
    -O3
    -march=native
)

# --------------------------------------------------
# IMPORTANT: Respect setuptools output directory
# --------------------------------------------------

# If setuptools passes CMAKE_LIBRARY_OUTPUT_DIRECTORY,
# use it. Otherwise fall back to binary dir.

if(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(libint_cpp PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    )
else()
    set_target_properties(libint_cpp PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

