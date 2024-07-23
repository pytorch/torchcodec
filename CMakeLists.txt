cmake_minimum_required(VERSION 3.18)
project(TorchCodec)

option(ENABLE_CUDA "Enable CUDA decoding using NVDEC" OFF)
option(ENABLE_NVTX "Enable NVTX annotations for profiling" OFF)

add_subdirectory(src/torchcodec/decoders/_core)


option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test/decoders)
endif()
