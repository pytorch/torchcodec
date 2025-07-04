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
    https://pytorch.s3.amazonaws.com/torchcodec/ffmpeg/2025-03-14
)



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

if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(
        platform_url
        ${base_url}/linux_x86_64
    )

    set(
        f4_sha256
        1a083f1922443bedb5243d04896383b8c606778a7ddb9d886c8303e55339fe0c
    )
    set(
        f5_sha256
        65d6ad54082d94dcb3f801d73df2265e0e1bb303c7afbce7723e3b77ccd0e207
    )
    set(
        f6_sha256
        8bd5939c2f4a4b072e837e7870c13fe7d13824e5ff087ab534e4db4e90b7be9c
    )
    set(
        f7_sha256
        1cb946d8b7c6393c2c3ebe1f900b8de7a2885fe614c45d4ec32c9833084f2f26
    )

    set(
       f4_library_paths
       ${f4_SOURCE_DIR}/lib/libavutil.so.56
       ${f4_SOURCE_DIR}/lib/libavcodec.so.58
       ${f4_SOURCE_DIR}/lib/libavformat.so.58
       ${f4_SOURCE_DIR}/lib/libavdevice.so.58
       ${f4_SOURCE_DIR}/lib/libavfilter.so.7
       ${f4_SOURCE_DIR}/lib/libswscale.so.5
       ${f4_SOURCE_DIR}/lib/libswresample.so.3
    )
    set(
       f5_library_paths
       ${f5_SOURCE_DIR}/lib/libavutil.so.57
       ${f5_SOURCE_DIR}/lib/libavcodec.so.59
       ${f5_SOURCE_DIR}/lib/libavformat.so.59
       ${f5_SOURCE_DIR}/lib/libavdevice.so.59
       ${f5_SOURCE_DIR}/lib/libavfilter.so.8
       ${f5_SOURCE_DIR}/lib/libswscale.so.6
       ${f5_SOURCE_DIR}/lib/libswresample.so.4
    )
    set(
       f6_library_paths
       ${f6_SOURCE_DIR}/lib/libavutil.so.58
       ${f6_SOURCE_DIR}/lib/libavcodec.so.60
       ${f6_SOURCE_DIR}/lib/libavformat.so.60
       ${f6_SOURCE_DIR}/lib/libavdevice.so.60
       ${f6_SOURCE_DIR}/lib/libavfilter.so.9
       ${f6_SOURCE_DIR}/lib/libswscale.so.7
       ${f6_SOURCE_DIR}/lib/libswresample.so.4
    )
    set(
       f7_library_paths
       ${f7_SOURCE_DIR}/lib/libavutil.so.59
       ${f7_SOURCE_DIR}/lib/libavcodec.so.61
       ${f7_SOURCE_DIR}/lib/libavformat.so.61
       ${f7_SOURCE_DIR}/lib/libavdevice.so.61
       ${f7_SOURCE_DIR}/lib/libavfilter.so.10
       ${f7_SOURCE_DIR}/lib/libswscale.so.8
       ${f7_SOURCE_DIR}/lib/libswresample.so.5
    )
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set(
        platform_url
        ${base_url}/macos_arm64
    )
    set(
        f4_sha256
        f0335434529d9e19359eae0fe912dd9e747667534a1c92e662f5219a55dfad8c
    )
    set(
        f5_sha256
        cfc3449c9af6863731a431ce89e32c08c5f8ece94b306fb6b695828502a76166
    )
    set(
        f6_sha256
        ec47b4783c342038e720e33b2fdfa55a9a490afb1cf37a26467733983688647e
    )
    set(
        f7_sha256
        48a4fc8ce098305cfd4a58f40889249c523ca3c285f66ba704b5bad0e3ada53a
    )
    set(
       f4_library_paths
       ${f4_SOURCE_DIR}/lib/libavutil.56.dylib
       ${f4_SOURCE_DIR}/lib/libavcodec.58.dylib
       ${f4_SOURCE_DIR}/lib/libavformat.58.dylib
       ${f4_SOURCE_DIR}/lib/libavdevice.58.dylib
       ${f4_SOURCE_DIR}/lib/libavfilter.7.dylib
       ${f4_SOURCE_DIR}/lib/libswscale.5.dylib
       ${f4_SOURCE_DIR}/lib/libswresample.3.dylib
    )
    set(
       f5_library_paths
       ${f5_SOURCE_DIR}/lib/libavutil.57.dylib
       ${f5_SOURCE_DIR}/lib/libavcodec.59.dylib
       ${f5_SOURCE_DIR}/lib/libavformat.59.dylib
       ${f5_SOURCE_DIR}/lib/libavdevice.59.dylib
       ${f5_SOURCE_DIR}/lib/libavfilter.8.dylib
       ${f5_SOURCE_DIR}/lib/libswscale.6.dylib
       ${f5_SOURCE_DIR}/lib/libswresample.4.dylib
    )
    set(
       f6_library_paths
       ${f6_SOURCE_DIR}/lib/libavutil.58.dylib
       ${f6_SOURCE_DIR}/lib/libavcodec.60.dylib
       ${f6_SOURCE_DIR}/lib/libavformat.60.dylib
       ${f6_SOURCE_DIR}/lib/libavdevice.60.dylib
       ${f6_SOURCE_DIR}/lib/libavfilter.9.dylib
       ${f6_SOURCE_DIR}/lib/libswscale.7.dylib
       ${f6_SOURCE_DIR}/lib/libswresample.4.dylib
    )
    set(
       f7_library_paths
       ${f7_SOURCE_DIR}/lib/libavutil.59.dylib
       ${f7_SOURCE_DIR}/lib/libavcodec.61.dylib
       ${f7_SOURCE_DIR}/lib/libavformat.61.dylib
       ${f7_SOURCE_DIR}/lib/libavdevice.61.dylib
       ${f7_SOURCE_DIR}/lib/libavfilter.10.dylib
       ${f7_SOURCE_DIR}/lib/libswscale.8.dylib
       ${f7_SOURCE_DIR}/lib/libswresample.5.dylib
    )
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(
        platform_url
        ${base_url}/windows_x86_64
    )
    set(
        f4_sha256
        270a1aa8892225267e68a7eb87c417931da30dccbf08ee2bde8833e659cab5cb
    )
    set(
        f5_sha256
        b8b2a349a847e56a6da875b066dff1cae53cb8ee7cf5ba9321ec1243dea0cde0
    )
    set(
        f6_sha256
        5d9f8c76dc55f790fa31d825985e9270bf9e498b8bfec21a0ad3a1feb1fa053a
    )
    set(
        f7_sha256
        ae391ace382330e912793b70b68529ee7c91026d2869b4df7e7c3e7d3656bdd5
    )
    set(
       f4_library_paths
       ${f4_SOURCE_DIR}/bin/avutil.lib
       ${f4_SOURCE_DIR}/bin/avcodec.lib
       ${f4_SOURCE_DIR}/bin/avformat.lib
       ${f4_SOURCE_DIR}/bin/avdevice.lib
       ${f4_SOURCE_DIR}/bin/avfilter.lib
       ${f4_SOURCE_DIR}/bin/swscale.lib
       ${f4_SOURCE_DIR}/bin/swresample.lib
    )
    set(
       f5_library_paths
       ${f5_SOURCE_DIR}/bin/avutil.lib
       ${f5_SOURCE_DIR}/bin/avcodec.lib
       ${f5_SOURCE_DIR}/bin/avformat.lib
       ${f5_SOURCE_DIR}/bin/avdevice.lib
       ${f5_SOURCE_DIR}/bin/avfilter.lib
       ${f5_SOURCE_DIR}/bin/swscale.lib
       ${f5_SOURCE_DIR}/bin/swresample.lib
    )
    set(
       f6_library_paths
       ${f6_SOURCE_DIR}/bin/avutil.lib
       ${f6_SOURCE_DIR}/bin/avcodec.lib
       ${f6_SOURCE_DIR}/bin/avformat.lib
       ${f6_SOURCE_DIR}/bin/avdevice.lib
       ${f6_SOURCE_DIR}/bin/avfilter.lib
       ${f6_SOURCE_DIR}/bin/swscale.lib
       ${f6_SOURCE_DIR}/bin/swresample.lib
    )
    set(
       f7_library_paths
       ${f7_SOURCE_DIR}/bin/avutil.lib
       ${f7_SOURCE_DIR}/bin/avcodec.lib
       ${f7_SOURCE_DIR}/bin/avformat.lib
       ${f7_SOURCE_DIR}/bin/avdevice.lib
       ${f7_SOURCE_DIR}/bin/avfilter.lib
       ${f7_SOURCE_DIR}/bin/swscale.lib
       ${f7_SOURCE_DIR}/bin/swresample.lib
    )
else()
    message(
        FATAL_ERROR
        "Unsupported operating system: ${CMAKE_SYSTEM_NAME}"
    )
endif()
target_include_directories(ffmpeg4 INTERFACE ${f4_SOURCE_DIR}/include)
target_include_directories(ffmpeg5 INTERFACE ${f5_SOURCE_DIR}/include)
target_include_directories(ffmpeg6 INTERFACE ${f6_SOURCE_DIR}/include)
target_include_directories(ffmpeg7 INTERFACE ${f7_SOURCE_DIR}/include)


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
