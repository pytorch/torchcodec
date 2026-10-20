# This file fetches a decode-only libavif from the torchcodec S3 bucket and
# exposes it as the `avif` CMake target, so wheel builds can link+bundle a tiny
# self-contained libavif (dav1d + libyuv statically embedded, no encoders)
# instead of conda-forge's libavif (which drags in ~20MB of unused AV1 encoder
# libraries). The tarball is built by packaging/build_libavif.sh.
#
# This mirrors fetch_and_expose_non_gpl_ffmpeg_libs.cmake.

# Avoid warning: see https://cmake.org/cmake/help/latest/policy/CMP0135.html
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)

set(libavif_version "1.4.2")

set(
    base_url
    https://pytorch.s3.amazonaws.com/torchcodec/libavif/2026-07-22
)

if (LINUX)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(platform_url ${base_url}/linux_aarch64)
        set(avif_sha256 9a377d3183bfabdc2e5f2e2f3969d8c3d605de56640b695832827aa7df8a8cce)
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        set(platform_url ${base_url}/linux_x86_64)
        set(avif_sha256 b814cbb53ede0b2e32f4907116f466cd68a789283ba8be149d919fb33d336300)
    else()
        message(FATAL_ERROR
            "No prebuilt libavif available for linux_${CMAKE_SYSTEM_PROCESSOR}. "
            "Set TORCHCODEC_BUILD_AVIF=0 to build without AVIF support")
    endif()
elseif (APPLE)
    set(platform_url ${base_url}/macos_arm64)
    set(avif_sha256 3130f6aa6087d38fb3769073899a17e6c570a88ba68fec97e3d2d90d0ead57f5)
elseif (WIN32)
    set(platform_url ${base_url}/windows_x86_64)
    set(avif_sha256 eb30dad70894e6d8b15a677b1cd2560d900c5333820be944cc3a8584765e24ac)
else()
    message(FATAL_ERROR "Unsupported operating system: ${CMAKE_SYSTEM_NAME}")
endif()

FetchContent_Declare(
    avif_s3
    URL ${platform_url}/${libavif_version}.tar.gz
    URL_HASH
    SHA256=${avif_sha256}
)
FetchContent_MakeAvailable(avif_s3)

# avif_s3_SOURCE_DIR was set by FetchContent_MakeAvailable and contains the usual
# include/ and lib/ (and bin/ on Windows) directories
set(include_dir "${avif_s3_SOURCE_DIR}/include")

# libavif's SOVERSION is 16 (paired with libavif 1.4.2).
if (LINUX)
    set(lib_path "${avif_s3_SOURCE_DIR}/lib/libavif.so.16")
elseif (APPLE)
    set(lib_path "${avif_s3_SOURCE_DIR}/lib/libavif.16.dylib")
elseif (WIN32)
    set(lib_path "${avif_s3_SOURCE_DIR}/lib/libavif.dll.a")
endif()

foreach (path IN LISTS include_dir lib_path)
    if (NOT EXISTS "${path}")
        message(FATAL_ERROR "${path} does not exist")
    endif()
endforeach()

if (WIN32)
    # Windows only: we link the .dll.a import lib above, but the actual runtime
    # DLL (from bin/) is what must be shipped. We expose it to the calling
    # CMakelists.txt for Windows editable builds.
    set(avif_runtime_lib "${avif_s3_SOURCE_DIR}/bin/libavif.dll")
    if (NOT EXISTS "${avif_runtime_lib}")
        message(FATAL_ERROR "${avif_runtime_lib} does not exist")
    endif()
endif()

message(STATUS "Adding libavif (decode-only, from S3) as the `avif` target")
add_library(avif INTERFACE IMPORTED)
target_include_directories(avif INTERFACE ${include_dir})
target_link_libraries(avif INTERFACE ${lib_path})
