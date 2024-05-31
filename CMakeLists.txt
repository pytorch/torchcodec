cmake_minimum_required(VERSION 3.18)
project(TorchCodec)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
find_package(PkgConfig REQUIRED)

set(LIBRARY_NAME "torchcodec")

# We want .so file to be named "module.so", not "libmodule.so". This removes
# the "lib" prefix.
set_target_properties(${MOD_NAME} PROPERTIES PREFIX "")

# The install step is invoked within CMakeBuild.build_extension() and just
# copies the built .so files from the temp cmake/setuptools build folder into
# where those .so files are supposed to be within the package, for example in
# `.../src/torchcodec/module.so` if `module` is meant to be imported from
# `torchcodec`.
# That CMAKE_INSTALL_PREFIX variable is set in build_extension(), and utlimately
# originates from the Extension instance's "name" parameter (e.g.
# "torchcodec.module").
# We still need to manually pass "DESTINATION ..." for cmake to copy those files
# in CMAKE_INSTALL_PREFIX instead of CMAKE_INSTALL_PREFIX/lib.
install(TARGETS ${MOD_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX})


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
  src/torchcodec/decoders/core/FFMPEGCommon.h
  src/torchcodec/decoders/core/FFMPEGCommon.cpp
  src/torchcodec/decoders/core/VideoDecoder.h
  src/torchcodec/decoders/core/VideoDecoder.cpp
  src/torchcodec/decoders/core/VideoDecoderOps.h
  src/torchcodec/decoders/core/VideoDecoderOps.cpp
  )

# TODO: remove -fpermissive
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -fpermissive")

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

install(TARGETS ${LIBRARY_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX})
