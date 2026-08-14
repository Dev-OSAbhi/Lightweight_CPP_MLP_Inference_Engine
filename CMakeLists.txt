cmake_minimum_required(VERSION 3.16)

project(lightweight_mlp_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

add_library(mlp_runtime
    src/model.cpp
    src/quantization.cpp
    src/runtime.cpp
)

target_include_directories(mlp_runtime PUBLIC include)

add_executable(mlp_demo src/demo.cpp)
target_link_libraries(mlp_demo PRIVATE mlp_runtime)

add_executable(mlp_benchmark src/benchmark.cpp)
target_link_libraries(mlp_benchmark PRIVATE mlp_runtime)

enable_testing()

add_executable(mlp_tests tests/test_main.cpp)
target_link_libraries(mlp_tests PRIVATE mlp_runtime)

add_test(NAME mlp_tests COMMAND mlp_tests)
set_tests_properties(mlp_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
