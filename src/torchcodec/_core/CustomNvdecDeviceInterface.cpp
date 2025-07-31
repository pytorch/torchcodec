// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/types.h>
#include <mutex>

#include "src/torchcodec/_core/CustomNvdecDeviceInterface.h"
#include "src/torchcodec/_core/DeviceInterface.h"
#include "src/torchcodec/_core/FFMPEGCommon.h"

// Include NVIDIA Video Codec SDK headers
#include <cuviddec.h>
#include <nvcuvid.h>

extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {
namespace {

// Register the custom NVDEC device interface with 'custom_nvdec' variant
static bool g_cuda_custom_nvdec = registerDeviceInterface(
    DeviceInterfaceKey(torch::kCUDA, "custom_nvdec"),
    [](const torch::Device& device) {
      return new CustomNvdecDeviceInterface(device);
    });

} // namespace

CustomNvdecDeviceInterface::CustomNvdecDeviceInterface(
    const torch::Device& device)
    : DeviceInterface(device) {
  TORCH_CHECK(
      g_cuda_custom_nvdec, "CustomNvdecDeviceInterface was not registered!");
  TORCH_CHECK(
      device_.type() == torch::kCUDA, "Unsupported device: ", device_.str());
}

CustomNvdecDeviceInterface::~CustomNvdecDeviceInterface() {
  // Clean up NVDEC resources
  if (videoDecoder_) {
    cuvidDestroyDecoder(videoDecoder_);
    videoDecoder_ = nullptr;
  }

  if (videoParser_) {
    cuvidDestroyVideoParser(videoParser_);
    videoParser_ = nullptr;
  }

  isInitialized_ = false;
}

std::optional<const AVCodec*> CustomNvdecDeviceInterface::findCodec(
    const AVCodecID& codecId) {
  // For custom NVDEC, we bypass FFmpeg codec selection entirely
  // We'll handle the codec selection in our own NVDEC initialization
  (void)codecId; // Suppress unused parameter warning
  return std::nullopt;
}

void CustomNvdecDeviceInterface::initializeContext(
    AVCodecContext* codecContext) {
  // Don't set hw_device_ctx - we handle decoding directly with NVDEC SDK
  // Just ensure CUDA context exists for PyTorch tensors
  torch::Tensor dummyTensor = torch::empty(
      {1}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));

  // Initialize our custom NVDEC decoder
  initializeNvdecDecoder(codecContext->codec_id);
}

void CustomNvdecDeviceInterface::initializeNvdecDecoder(AVCodecID codecId) {
  if (isInitialized_) {
    return; // Already initialized
  }

  // Convert AVCodecID to NVDEC codec type
  cudaVideoCodec nvCodec;
  switch (codecId) {
    case AV_CODEC_ID_H264:
      nvCodec = cudaVideoCodec_H264;
      break;
    case AV_CODEC_ID_HEVC:
      nvCodec = cudaVideoCodec_HEVC;
      break;
    case AV_CODEC_ID_AV1:
      nvCodec = cudaVideoCodec_AV1;
      break;
    case AV_CODEC_ID_VP8:
      nvCodec = cudaVideoCodec_VP8;
      break;
    case AV_CODEC_ID_VP9:
      nvCodec = cudaVideoCodec_VP9;
      break;
    default:
      TORCH_CHECK(
          false,
          "Unsupported codec for custom NVDEC: ",
          avcodec_get_name(codecId));
  }

  // Get current CUDA context
  CUresult cuResult = cuCtxGetCurrent(&cudaContext_);
  TORCH_CHECK(
      cuResult == CUDA_SUCCESS,
      "Failed to get current CUDA context: ",
      cuResult);
  TORCH_CHECK(cudaContext_ != nullptr, "No CUDA context available");

  // Initialize video format structure
  memset(&videoFormat_, 0, sizeof(videoFormat_));
  videoFormat_.codec = nvCodec;
  videoFormat_.coded_width = 0; // Will be set when we get the first frame
  videoFormat_.coded_height = 0; // Will be set when we get the first frame
  videoFormat_.chroma_format = cudaVideoChromaFormat_420;
  videoFormat_.bit_depth_luma_minus8 = 0;
  videoFormat_.bit_depth_chroma_minus8 = 0;

  isInitialized_ = true;

  // NVDEC decoder basic initialization complete
  (void)0; // No-op to avoid unused variable warnings
}

UniqueAVFrame CustomNvdecDeviceInterface::decodePacketDirectly(
    ReferenceAVPacket& packet) {
  TORCH_CHECK(isInitialized_, "NVDEC decoder not initialized");

  // Extract compressed data from AVPacket
  uint8_t* compressedData = packet->data;
  int size = packet->size;
  int64_t pts = packet->pts;

  TORCH_CHECK(compressedData != nullptr && size > 0, "Invalid packet data");

  // For now, we need to create the decoder when we get the first frame with
  // dimensions In a full implementation, you would:
  // 1. Parse the compressed data to get video dimensions if not already known
  // 2. Create the CUDA video decoder with proper dimensions
  // 3. Submit the compressed data to the decoder
  // 4. Get the decoded frame data
  // 5. Convert to AVFrame format

  // This is a basic structure - the actual NVDEC API is more complex
  if (videoDecoder_ == nullptr) {
    // Would need to create decoder here with proper dimensions
    // For now, return early to avoid the TORCH_CHECK below
    // NVDEC decoder creation not yet implemented - need video dimensions
    return UniqueAVFrame(nullptr);
  }

  // TODO: Implement actual NVDEC decoding pipeline
  // This would involve:
  // - cuvidDecodePicture() to submit compressed data
  // - cuvidMapVideoFrame() to get decoded frame data
  // - Converting the GPU frame data to AVFrame format
  // - cuvidUnmapVideoFrame() to release the frame

  (void)pts; // Suppress unused parameter warning for now
  TORCH_CHECK(false, "NVDEC decoding pipeline not yet fully implemented");
  return UniqueAVFrame(nullptr);
}

UniqueAVFrame CustomNvdecDeviceInterface::convertNvdecOutputToAVFrame(
    uint8_t* decodedFrame,
    int width,
    int height,
    int64_t pts,
    int64_t duration) {
  TORCH_CHECK(decodedFrame != nullptr, "Invalid decoded frame data");
  TORCH_CHECK(width > 0 && height > 0, "Invalid frame dimensions");

  // Allocate AVFrame
  UniqueAVFrame avFrame(av_frame_alloc());
  TORCH_CHECK(avFrame.get() != nullptr, "Failed to allocate AVFrame");

  // Set frame properties
  avFrame->width = width;
  avFrame->height = height;
  avFrame->format = AV_PIX_FMT_CUDA; // Indicate this is GPU data
  avFrame->pts = pts;
  avFrame->duration = duration;

  // For NVDEC output, we typically get NV12 format (YUV 4:2:0 with interleaved
  // UV) Set up GPU data pointers for NV12 format
  avFrame->data[0] = decodedFrame; // Y plane
  avFrame->data[1] = decodedFrame + (width * height); // UV plane for NV12
  avFrame->data[2] = nullptr;
  avFrame->data[3] = nullptr;

  // Set line sizes
  avFrame->linesize[0] = width; // Y plane stride
  avFrame->linesize[1] = width; // UV plane stride (interleaved U and V)
  avFrame->linesize[2] = 0;
  avFrame->linesize[3] = 0;

  // Successfully converted NVDEC frame to AVFrame

  return avFrame;
}

void CustomNvdecDeviceInterface::convertAVFrameToFrameOutput(
    const VideoStreamOptions& videoStreamOptions,
    const AVRational& timeBase,
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  // For custom NVDEC, the frame should already be on GPU
  // We need to convert from NVDEC's output format (typically NV12) to RGB

  TORCH_CHECK(
      avFrame->format == AV_PIX_FMT_CUDA,
      "Expected CUDA format frame from custom NVDEC decoder");

  // TODO: Implement custom GPU-based color conversion
  /*
  Example implementation using CUDA kernels or NPP:

  auto frameDims = getHeightAndWidthFromOptionsOrAVFrame(videoStreamOptions,
  avFrame); int height = frameDims.height; int width = frameDims.width;

  torch::Tensor& dst = frameOutput.data;
  if (preAllocatedOutputTensor.has_value()) {
    dst = preAllocatedOutputTensor.value();
  } else {
    dst = allocateEmptyHWCTensor(height, width, device_);
  }

  // Convert NV12 to RGB using custom CUDA kernel or NPP
  convertNV12ToRGB_Custom(
      avFrame->data[0],  // Y plane
      avFrame->data[1],  // UV plane
      avFrame->linesize[0],
      static_cast<uint8_t*>(dst.data_ptr()),
      dst.stride(0),
      width,
      height
  );
  */

  // For now, fall back to CPU conversion as placeholder
  auto cpuDevice = torch::Device(torch::kCPU);
  auto cpuInterface = createDeviceInterface(cpuDevice);

  FrameOutput cpuFrameOutput;
  cpuInterface->convertAVFrameToFrameOutput(
      videoStreamOptions,
      timeBase,
      avFrame,
      cpuFrameOutput,
      preAllocatedOutputTensor);

  frameOutput.data = cpuFrameOutput.data.to(device_);
}

} // namespace facebook::torchcodec
