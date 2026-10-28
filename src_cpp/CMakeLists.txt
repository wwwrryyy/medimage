cmake_minimum_required(VERSION 3.15)
project(scan_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Python (interpreter + extension-module development files) -- portable across Windows/macOS/Linux
find_package(Python 3 REQUIRED COMPONENTS Interpreter Development.Module)

# pybind11 installed in that Python environment (pip install pybind11)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_RESULT
)
if(NOT PYBIND11_RESULT EQUAL 0)
    message(FATAL_ERROR "pybind11 not found for ${Python_EXECUTABLE}. Run: pip install pybind11")
endif()
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

# OpenCV C++ development package (headers + libs). If it is not found automatically, pass
# -DCMAKE_PREFIX_PATH=<prefix containing lib/cmake/opencv4 or opencv5> (e.g. your conda env).
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)

# Build C++ Zero-Copy pybind11 module (correct suffix per platform: .so / .pyd)
pybind11_add_module(scan_core
    src/pybind_bridge.cpp
    src/video_ingestor.cpp
    src/anomaly_calculus.cpp
)

target_include_directories(scan_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(scan_core PRIVATE ${OpenCV_LIBS})

set_target_properties(scan_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../src_python"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../src_python"
)
foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(scan_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${cfg} "${CMAKE_CURRENT_SOURCE_DIR}/../src_python"
        RUNTIME_OUTPUT_DIRECTORY_${cfg} "${CMAKE_CURRENT_SOURCE_DIR}/../src_python"
    )
endforeach()
