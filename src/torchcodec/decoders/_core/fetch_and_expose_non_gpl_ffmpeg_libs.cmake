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
    https://pytorch.s3.amazonaws.com/torchcodec/ffmpeg/2024-09-13
)

if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(
        platform_url
        ${base_url}/linux_x86_64
    )

    set(
        f4_sha256
        07d3e33281f0dce04d3e987d20cce03b155b0c39965333960689c625f451f93a
    )
    set(
        f5_sha256
        1a2227445f513deb8f4f339050a160fa2419ca494a7f981df93e747d00eeaa69
    )
    set(
        f6_sha256
        63320ec05ae9341ba307ff0005ac853bcec0b9d2cb55a580d1a72731de2bb5d8
    )
    set(
        f7_sha256
        0b7c983b5d675441a1c1756eefa23cb24450af6bae5ae2011d9e5807a315d7df
    )

    set(
       f4_library_file_names
       libavutil.so.56
       libavcodec.so.58
       libavformat.so.58
       libavdevice.so.58
       libavfilter.so.7
    )
    set(
       f5_library_file_names
       libavutil.so.57
       libavcodec.so.59
       libavformat.so.59
       libavdevice.so.59
       libavfilter.so.8
    )
    set(
       f6_library_file_names
       libavutil.so.58
       libavcodec.so.60
       libavformat.so.60
       libavdevice.so.60
       libavfilter.so.9
    )
    set(
       f7_library_file_names
       libavutil.so.59
       libavcodec.so.61
       libavformat.so.61
       libavdevice.so.61
       libavfilter.so.10
    )
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set(
        platform_url
        ${base_url}/macos_arm64
    )
    set(
        f4_sha256
        7839bebecb9a25f470405a745225d29a5a7f43f4e6d9a57868732aa897ce32be
    )
    set(
        f5_sha256
        df204c89ae52d3af16eb23604955e8cfbee649845d3ae737778a264346ab0063
    )
    set(
        f6_sha256
        8a82e9ae2eabb23ba546e2c96ba7f1bd656b4db38679876df936db7a92c15677
    )
    set(
        f7_sha256
        39d96d8191c58ff439d674701d83c775b2b57019a1c2436aa78e7bc9ab74445b
    )
    set(
       f4_library_file_names
       libavutil.56.dylib
       libavcodec.58.dylib
       libavformat.58.dylib
       libavdevice.58.dylib
       libavfilter.7.dylib
    )
    set(
       f5_library_file_names
       libavutil.57.dylib
       libavcodec.59.dylib
       libavformat.59.dylib
       libavdevice.59.dylib
       libavfilter.8.dylib
    )
    set(
       f6_library_file_names
       libavutil.58.dylib
       libavcodec.60.dylib
       libavformat.60.dylib
       libavdevice.60.dylib
       libavfilter.9.dylib
    )
    set(
       f7_library_file_names
       libavutil.59.dylib
       libavcodec.61.dylib
       libavformat.61.dylib
       libavdevice.61.dylib
       libavfilter.10.dylib
    )
else()
    message(
        FATAL_ERROR
        "Unsupported operating system: ${CMAKE_SYSTEM_NAME}"
    )
endif()

FetchContent_Declare(
    f4
    URL ${platform_url}/4.4.4.tar.gz
    URL_HASH
    SHA256=${f4_sha256}
)
FetchContent_Declare(
    f5
    URL ${platform_url}/5.1.4.tar.gz
    URL_HASH
    SHA256=${f5_sha256}
)
FetchContent_Declare(
    f6
    URL ${platform_url}/6.1.1.tar.gz
    URL_HASH
    SHA256=${f6_sha256}
)
FetchContent_Declare(
    f7
    URL ${platform_url}/7.0.1.tar.gz
    URL_HASH
    SHA256=${f7_sha256}
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

list(
    TRANSFORM f4_library_file_names
    PREPEND ${f4_SOURCE_DIR}/lib/
    OUTPUT_VARIABLE f4_library_paths
)
list(
    TRANSFORM f5_library_file_names
    PREPEND ${f5_SOURCE_DIR}/lib/
    OUTPUT_VARIABLE f5_library_paths
)
list(
    TRANSFORM f6_library_file_names
    PREPEND ${f6_SOURCE_DIR}/lib/
    OUTPUT_VARIABLE f6_library_paths
)
list(
    TRANSFORM f7_library_file_names
    PREPEND ${f7_SOURCE_DIR}/lib/
    OUTPUT_VARIABLE f7_library_paths
)

target_link_libraries(
    ffmpeg4
    INTERFACE
    ${f4_library_paths}
)
target_link_libraries(
    ffmpeg5
    INTERFACE
    ${f5_library_paths}
)
target_link_libraries(
    ffmpeg6
    INTERFACE
    ${f6_library_paths}
)
target_link_libraries(
    ffmpeg7
    INTERFACE
    ${f7_library_paths}
)
