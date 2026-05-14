cmake_minimum_required(VERSION 3.18)
project(yolov8_hand_detect CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# ── OpenCV ────────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)

# ── ONNX Runtime ─────────────────────────────────────────────────────────────
# Adjust ONNXRUNTIME_ROOT if you installed it somewhere else.
# Default: /usr/local  (from the tar.gz install script below)
set(ONNXRUNTIME_ROOT "/usr/local" CACHE PATH "ONNX Runtime install prefix")

find_path(ORT_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT}/include/onnxruntime
          ${ONNXRUNTIME_ROOT}/include
    REQUIRED)

find_library(ORT_LIB
    NAMES onnxruntime
    HINTS ${ONNXRUNTIME_ROOT}/lib
    REQUIRED)

# ── Target ────────────────────────────────────────────────────────────────────
add_executable(yolov8_hand main.cpp)

target_include_directories(yolov8_hand PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ORT_INCLUDE_DIR})

find_package(Threads REQUIRED)

target_link_libraries(yolov8_hand PRIVATE
    ${OpenCV_LIBS}
    ${ORT_LIB}
    Threads::Threads)
