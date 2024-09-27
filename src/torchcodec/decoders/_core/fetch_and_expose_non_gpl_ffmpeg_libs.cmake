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
    https://pytorch.s3.amazonaws.com/torchcodec/ffmpeg/2024-09-23
)

if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(
        platform_url
        ${base_url}/linux_x86_64
    )

    set(
        f4_sha256
        c0311e983c426a7f5abcffc3056f0d64a93bfcb69a9db8e40b81d5c976d84952
    )
    set(
        f5_sha256
        9a48dbe7912a0c3dbbac0c906a840754caf147d37dad2f1b3906de7441e1483a
    )
    set(
        f6_sha256
        337cd2ce671a69737e246c73bf69e2c36732d89b7d2c37eefaca8601cad272ca
    )
    set(
        f7_sha256
        b7df528b1c66eb37b926c1336c89a63b3b784165f6f30bd0932a39b82469f0e9
    )

    set(
       f4_library_file_names
       libavutil.so.56
       libavcodec.so.58
       libavformat.so.58
       libavdevice.so.58
       libavfilter.so.7
       libswscale.so.5
    )
    set(
       f5_library_file_names
       libavutil.so.57
       libavcodec.so.59
       libavformat.so.59
       libavdevice.so.59
       libavfilter.so.8
       libswscale.so.6
    )
    set(
       f6_library_file_names
       libavutil.so.58
       libavcodec.so.60
       libavformat.so.60
       libavdevice.so.60
       libavfilter.so.9
       libswscale.so.7
    )
    set(
       f7_library_file_names
       libavutil.so.59
       libavcodec.so.61
       libavformat.so.61
       libavdevice.so.61
       libavfilter.so.10
       libswscale.so.8
    )
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set(
        platform_url
        ${base_url}/macos_arm64
    )
    set(
        f4_sha256
        57622392af27bf479e18afb9d79ccf3deddaef153048b34ce518bd477c12d1e6
    )
    set(
        f5_sha256
        7bc5a70ac66d45857372ccabdcd15594aa9a39a86bc396f92724435e5c4de54e
    )
    set(
        f6_sha256
        0214733bc987c2deeabfc779331108c19964dcdac2c5e2db12960f0febcea2c4
    )
    set(
        f7_sha256
        c28925bb423383c0c37d9f3106fa7768c8733153a33154c8bedab8acf883366f
    )
    set(
       f4_library_file_names
       libavutil.56.dylib
       libavcodec.58.dylib
       libavformat.58.dylib
       libavdevice.58.dylib
       libavfilter.7.dylib
       libswscale.5.dylib
    )
    set(
       f5_library_file_names
       libavutil.57.dylib
       libavcodec.59.dylib
       libavformat.59.dylib
       libavdevice.59.dylib
       libavfilter.8.dylib
       libswscale.6.dylib
    )
    set(
       f6_library_file_names
       libavutil.58.dylib
       libavcodec.60.dylib
       libavformat.60.dylib
       libavdevice.60.dylib
       libavfilter.9.dylib
       libswscale.7.dylib
    )
    set(
       f7_library_file_names
       libavutil.59.dylib
       libavcodec.61.dylib
       libavformat.61.dylib
       libavdevice.61.dylib
       libavfilter.10.dylib
       libswscale.8.dylib
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
