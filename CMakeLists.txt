cmake_minimum_required(VERSION 3.23)
set(CMAKE_CXX_COMPILER "/opt/rocm/bin/amdclang++")
project(run_colony LANGUAGES CXX)

# Find necessary packages
find_package(Python 3.6 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(hip REQUIRED)

# Enable HIP language after finding the package
enable_language(HIP)

# Create pybind11 module
pybind11_add_module(run_colony src/run_colony.cpp)

# Link libraries
target_link_libraries(run_colony PRIVATE hip::host hip::device)
