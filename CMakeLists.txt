cmake_minimum_required(VERSION 3.12)
project(MyVisionApp)

set(pybind11_DIR "D:/Python11/Lib/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

file(GLOB SRC_FILES "backend/*.cpp")

pybind11_add_module(cv_backend ${SRC_FILES})

target_link_libraries(cv_backend PRIVATE ${OpenCV_LIBS})

set_target_properties(cv_backend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/frontend"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/frontend"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/frontend"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/frontend"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/frontend"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/frontend"
)