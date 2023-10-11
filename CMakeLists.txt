cmake_minimum_required(VERSION 3.21)
project(run_colony LANGUAGES CXX)

find_package(Python 3.6 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(hip REQUIRED)

find_package(pybind11)
pybind11_add_module(run_colony src/run_colony.cpp)
target_link_libraries(run_colony PRIVATE hip::host hip::device)
