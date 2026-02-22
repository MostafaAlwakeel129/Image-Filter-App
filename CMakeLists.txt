cmake_minimum_required(VERSION 3.12)
project(MyVisionApp)

find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

file(GLOB SRC_FILES "backend/*.cpp")

pybind11_add_module(cv_backend ${SRC_FILES})

target_link_libraries(cv_backend PRIVATE ${OpenCV_LIBS})