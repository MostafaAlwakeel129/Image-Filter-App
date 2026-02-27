cmake_minimum_required(VERSION 3.12)
project(MyVisionApp)
cmake_policy(SET CMP0148 NEW)
set(CMAKE_CXX_STANDARD 17)
set(PYBIND11_FINDPYTHON ON)
# Path to your OpenCV installation
set(OpenCV_DIR "C:/opencv/build")
set(pybind11_DIR "C:/Users/Sarah/AppData/Local/Programs/Python/Python312/Lib/site-packages/pybind11/share/cmake/pybind11")
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