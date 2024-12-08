cmake_minimum_required(VERSION 3.10)
project(MNISTNeuralNetwork VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_TOOLCHAIN_FILE "D:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

# Use the all-keyword style here
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ws2_32)

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
