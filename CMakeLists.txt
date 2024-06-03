cmake_minimum_required(VERSION 3.18)
project(TorchCodec)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
find_package(PkgConfig REQUIRED)

set(LIBRARY_NAME "torchcodec")

pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavdevice
    libavfilter
    libavformat
    libavcodec
    libswresample
    libswscale
    libavutil
)

set(
  sources
  src/torchcodec/decoders/_core/FFMPEGCommon.h
  src/torchcodec/decoders/_core/FFMPEGCommon.cpp
  src/torchcodec/decoders/_core/VideoDecoder.h
  src/torchcodec/decoders/_core/VideoDecoder.cpp
  src/torchcodec/decoders/_core/VideoDecoderOps.h
  src/torchcodec/decoders/_core/VideoDecoderOps.cpp
  )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(${LIBRARY_NAME}
    SHARED
    "${sources}"
)

set_property(TARGET ${LIBRARY_NAME} PROPERTY CXX_STANDARD 17)

target_include_directories(${LIBRARY_NAME} PRIVATE ./)

target_link_libraries(${LIBRARY_NAME}
    PkgConfig::LIBAV
    "${TORCH_LIBRARIES}"
)

# The install step is invoked within CMakeBuild.build_extension() and just
# copies the built .so files from the temp cmake/setuptools build folder into
# the CMAKE_INSTALL_PREFIX folder.
# We still need to manually pass "DESTINATION ..." for cmake to copy those files
# in CMAKE_INSTALL_PREFIX instead of CMAKE_INSTALL_PREFIX/lib.
install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX})
