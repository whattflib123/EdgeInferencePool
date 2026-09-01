cmake_minimum_required(VERSION 3.16)
project(EdgeInferencePool CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(edge_inference
    src/DpuBackend.cpp
    src/main.cpp
)

target_include_directories(edge_inference PRIVATE
    include
    /usr/include/vart
    /usr/include/xir
)

target_link_libraries(edge_inference PRIVATE
    vart-runner
    xir
)
