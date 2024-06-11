cmake_minimum_required(VERSION 3.18)
project(TorchCodec)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)


add_library(ffmpeg4 INTERFACE)
add_library(ffmpeg5 INTERFACE)
add_library(ffmpeg6 INTERFACE)

include(FetchContent)
set(base_url https://pytorch.s3.amazonaws.com/torchaudio/ffmpeg)
FetchContent_Declare(
      f4
      URL ${base_url}/2023-08-14/linux_x86_64/4.4.4.tar.gz
      URL_HASH SHA256=9b87eeba9b6975e25f28ba12163bd713228ed84f4c2b3721bc5ebe92055edb51
      )
    FetchContent_Declare(
      f5
      URL ${base_url}/2023-07-06/linux_x86_64/5.0.3.tar.gz
      URL_HASH SHA256=de3c75c99b9ce33de7efdc144566804ae5880457ce71e185b3f592dc452edce7
      )
    FetchContent_Declare(
      f6
      URL ${base_url}/2023-07-06/linux_x86_64/6.0.tar.gz
      URL_HASH SHA256=04d3916404bab5efadd29f68361b7d13ea71e6242c6473edcb747a41a9fb97a6
      )
FetchContent_MakeAvailable(f4 f5 f6)

# Note: the f?_SOURCE_DIR variables were set by FetchContent_MakeAvailable
target_include_directories(ffmpeg4 INTERFACE ${f4_SOURCE_DIR}/include)
target_include_directories(ffmpeg5 INTERFACE ${f5_SOURCE_DIR}/include)
target_include_directories(ffmpeg6 INTERFACE ${f6_SOURCE_DIR}/include)

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


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

function(make_torchcodec_extension name ffmpeg_target)
  set(
   sources
   FFMPEGCommon.h
   FFMPEGCommon.cpp
   VideoDecoder.h
   VideoDecoder.cpp
   VideoDecoderOps.h
   VideoDecoderOps.cpp
   )
  add_library(${name} SHARED ${sources})
  set_property(TARGET ${name} PROPERTY CXX_STANDARD 17)

  target_include_directories(
    ${name}
    PRIVATE
    ./../../../../
    "${TORCH_INSTALL_PREFIX}/include"
    )

  target_link_libraries(
    ${name}
    ${ffmpeg_target}
    ${TORCH_LIBRARIES}
    )

  install(
    TARGETS ${name}
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
    )

endfunction()

make_torchcodec_extension(torchcodec4 ffmpeg4)
make_torchcodec_extension(torchcodec5 ffmpeg5)
make_torchcodec_extension(torchcodec6 ffmpeg6)
