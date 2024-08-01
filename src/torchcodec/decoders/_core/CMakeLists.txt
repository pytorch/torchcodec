cmake_minimum_required(VERSION 3.18)
project(TorchCodec)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

if(ENABLE_CUDA)
    find_package(CUDA REQUIRED)

if(ENABLE_NVTX)
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.38.3/CPM.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
  EXPECTED_HASH SHA256=cc155ce02e7945e7b8967ddfaff0b050e958a723ef7aad3766d368940cb15494
)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)
CPMAddPackage(
    NAME NVTX
    GITHUB_REPOSITORY NVIDIA/NVTX
    GIT_TAG v3.1.0-c-cpp
    GIT_SHALLOW TRUE)
endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(Python3 ${PYTHON_VERSION} EXACT COMPONENTS Development)

function(make_torchcodec_library library_name ffmpeg_target)
    set(
        sources
        FFMPEGCommon.h
        FFMPEGCommon.cpp
        VideoDecoder.h
        VideoDecoder.cpp
        VideoDecoderOps.h
        VideoDecoderOps.cpp
    )
    add_library(${library_name} SHARED ${sources})
    set_property(TARGET ${library_name} PROPERTY CXX_STANDARD 17)
    if(ENABLE_CUDA)
        target_compile_definitions(${library_name} PRIVATE ENABLE_CUDA=1)
    endif()
    if(ENABLE_NVTX)
        target_compile_definitions(${library_name} PRIVATE ENABLE_NVTX=1)
    endif()

    target_include_directories(
        ${library_name}
        PRIVATE
        ./../../../../
        "${TORCH_INSTALL_PREFIX}/include"
        ${Python3_INCLUDE_DIRS}
    )

    set(NEEDED_LIBRARIES ${ffmpeg_target} ${TORCH_LIBRARIES} ${Python3_LIBRARIES})
    if(ENABLE_CUDA)
        list(APPEND NEEDED_LIBRARIES ${CUDA_CUDA_LIBRARY})
    endif()
    if(ENABLE_NVTX)
        list(APPEND NEEDED_LIBRARIES nvtx3-cpp)
    endif()
    target_link_libraries(
        ${library_name}
        PUBLIC
        ${NEEDED_LIBRARIES}
    )

    # We already set the library_name to be libtorchcodecN, so we don't want
    # cmake to add another "lib" prefix. We do it this way because it makes it
    # easier to find references to libtorchcodec in the code (e.g. via `git
    # grep`)
    set_target_properties(${library_name} PROPERTIES PREFIX "")

    # The install step is invoked within CMakeBuild.build_library() in
    # setup.py and just copies the built .so files from the temp
    # cmake/setuptools build folder into the CMAKE_INSTALL_PREFIX folder. We
    # still need to manually pass "DESTINATION ..." for cmake to copy those
    # files in CMAKE_INSTALL_PREFIX instead of CMAKE_INSTALL_PREFIX/lib.
    install(
        TARGETS ${library_name}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
    )
endfunction()

if(DEFINED ENV{BUILD_AGAINST_ALL_FFMPEG_FROM_S3})
    message(
        STATUS
        "Building and dynamically linking libtorchcodec against our pre-built
        non-GPL FFmpeg libraries. These libraries are only used at build time,
        you still need a different FFmpeg to be installed for run time!"
    )

    # This will expose the ffmpeg4, ffmpeg5, ffmpeg6, and ffmpeg7 targets
    include(
        ${CMAKE_CURRENT_SOURCE_DIR}/fetch_and_expose_non_gpl_ffmpeg_libs.cmake
    )

    make_torchcodec_library(libtorchcodec4 ffmpeg4)
    make_torchcodec_library(libtorchcodec5 ffmpeg5)
    make_torchcodec_library(libtorchcodec6 ffmpeg6)
    make_torchcodec_library(libtorchcodec7 ffmpeg7)
else()
    message(
        STATUS
        "Building and dynamically linking libtorchcodec against the installed
        FFmpeg libraries. This require pkg-config to be installed. If you have
        installed FFmpeg from conda, make sure pkg-config is installed from
        conda as well."
    )
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
        libavdevice
        libavfilter
        libavformat
        libavcodec
        libavutil
    )

    # Split libavcodec's version string by '.' and convert it to a list
    string(REPLACE "." ";" libavcodec_version_list ${LIBAV_libavcodec_VERSION})
    # Get the first element of the list, which is the major version
    list(GET libavcodec_version_list 0 libavcodec_major_version)

    if (${libavcodec_major_version} STREQUAL "58")
        set(ffmpeg_major_version "4")
    elseif (${libavcodec_major_version} STREQUAL "59")
        set(ffmpeg_major_version "5")
    elseif (${libavcodec_major_version} STREQUAL "60")
        set(ffmpeg_major_version "6")
    elseif (${libavcodec_major_version} STREQUAL "61")
        set(ffmpeg_major_version "7")
    else()
        message(
            FATAL_ERROR
            "Unsupported libavcodec version: ${libavcodec_major_version}"
        )
    endif()

    set(libtorchcodec_target_name libtorchcodec${ffmpeg_major_version})
    # Make libtorchcodec_target_name available in the parent's scope, for the
    # test's CMakeLists.txt
    set(libtorchcodec_target_name ${libtorchcodec_target_name} PARENT_SCOPE)

    make_torchcodec_library(${libtorchcodec_target_name} PkgConfig::LIBAV)
endif()
