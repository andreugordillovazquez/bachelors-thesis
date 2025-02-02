cmake_minimum_required(VERSION 3.26)
project(untitled12)

set(CMAKE_CXX_STANDARD 17)
    set(EIGEN3_INCLUDE_DIR "C:/Users/TESTER/Downloads/eigen-3.4.0/eigen-3.4.0")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O2")

    include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(untitled12 main.cpp
        functions.cpp
)
