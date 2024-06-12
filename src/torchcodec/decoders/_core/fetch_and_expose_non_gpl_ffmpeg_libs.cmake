# This file fetches the non-GPL ffmpeg libraries from the torchcodec S3 bucket,
# and exposes them as CMake targets so we can dynamically link against them.
# These libraries were built on the CI via the build_ffmpeg.yaml workflow.

# Avoid warning: see https://cmake.org/cmake/help/latest/policy/CMP0135.html
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
set(
    base_url
    https://pytorch.s3.amazonaws.com/torchcodec/ffmpeg/2024-06-11/linux_x86_64
)
FetchContent_Declare(
    f4
    URL ${base_url}/ffmpeg_4.4.4.tar.gz
    URL_HASH
    SHA256=a564721e51038d01ead4bbc7a482398929101ca4c80e5ce5c42042637235a297
)
FetchContent_Declare(
    f5
    URL ${base_url}/ffmpeg_5.1.4.tar.gz
    URL_HASH
    SHA256=d9c2d3a355c091ddc3205ae73426d0d6402ad8a31212dc920daabbaa5fdae944
)
FetchContent_Declare(
    f6
    URL ${base_url}/ffmpeg_6.1.1.tar.gz
    URL_HASH
    SHA256=7ee5830dc09fed7270aa575650474ab16e18477551e5511f256ce92daa30b136
)
FetchContent_Declare(
    f7
    URL ${base_url}/ffmpeg_7.0.1.tar.gz
    URL_HASH
    SHA256=fa4cda7aa67fcd58428017f7ebd2a981b0c6babba7ec89f71d6840877712ddcd
)

FetchContent_MakeAvailable(f4 f5 f6 f7)

add_library(ffmpeg4 INTERFACE)
add_library(ffmpeg5 INTERFACE)
add_library(ffmpeg6 INTERFACE)
add_library(ffmpeg7 INTERFACE)

# Note: the f?_SOURCE_DIR variables were set by FetchContent_MakeAvailable
target_include_directories(ffmpeg4 INTERFACE ${f4_SOURCE_DIR}/include)
target_include_directories(ffmpeg5 INTERFACE ${f5_SOURCE_DIR}/include)
target_include_directories(ffmpeg6 INTERFACE ${f6_SOURCE_DIR}/include)
target_include_directories(ffmpeg7 INTERFACE ${f7_SOURCE_DIR}/include)

target_link_libraries(
    ffmpeg4
    INTERFACE
    ${f4_SOURCE_DIR}/lib/libavutil.so.56
    ${f4_SOURCE_DIR}/lib/libavcodec.so.58
    ${f4_SOURCE_DIR}/lib/libavformat.so.58
    ${f4_SOURCE_DIR}/lib/libavdevice.so.58
    ${f4_SOURCE_DIR}/lib/libavfilter.so.7
)
target_link_libraries(
    ffmpeg5
    INTERFACE
    ${f5_SOURCE_DIR}/lib/libavutil.so.57
    ${f5_SOURCE_DIR}/lib/libavcodec.so.59
    ${f5_SOURCE_DIR}/lib/libavformat.so.59
    ${f5_SOURCE_DIR}/lib/libavdevice.so.59
    ${f5_SOURCE_DIR}/lib/libavfilter.so.8
)
target_link_libraries(
    ffmpeg6
    INTERFACE
    ${f6_SOURCE_DIR}/lib/libavutil.so.58
    ${f6_SOURCE_DIR}/lib/libavcodec.so.60
    ${f6_SOURCE_DIR}/lib/libavformat.so.60
    ${f6_SOURCE_DIR}/lib/libavdevice.so.60
    ${f6_SOURCE_DIR}/lib/libavfilter.so.9
)
target_link_libraries(
    ffmpeg7
    INTERFACE
    ${f7_SOURCE_DIR}/lib/libavutil.so.59
    ${f7_SOURCE_DIR}/lib/libavcodec.so.61
    ${f7_SOURCE_DIR}/lib/libavformat.so.61
    ${f7_SOURCE_DIR}/lib/libavdevice.so.61
    ${f7_SOURCE_DIR}/lib/libavfilter.so.10
)
