cmake_minimum_required(VERSION 3.24)
project(sspp_assignment)

set(CMAKE_CXX_STANDARD 14)

include_directories(/usr/local/include)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fopenmp")
find_library(OPENMP_LIB libomp.dylib /usr/local/lib)

add_executable(sspp_assignment main.cpp src/mmio.h src/mmio.c)
target_link_libraries(sspp_assignment LINK_PUBLIC ${OPENMP_LIB})
