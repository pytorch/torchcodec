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

// TODO: Include NVIDIA Video Codec SDK headers when available
// #include "NvDecoder/NvDecoder.h"
// #include "Utils/NvCodecUtils.h"

extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {
namespace {

// Register the custom NVDEC device interface with 'custom_nvdec' variant
static bool g_cuda_custom_nvdec =
    registerDeviceInterface(
        DeviceInterfaceKey(torch::kCUDA, "custom_nvdec"),
        [](const torch::Device& device) {
          return new CustomNvdecDeviceInterface(device);
        });

} // namespace

CustomNvdecDeviceInterface::CustomNvdecDeviceInterface(const torch::Device& device)
    : DeviceInterface(device) {
  TORCH_CHECK(g_cuda_custom_nvdec, "CustomNvdecDeviceInterface was not registered!");
  TORCH_CHECK(
      device_.type() == torch::kCUDA, "Unsupported device: ", device_.str());
}

CustomNvdecDeviceInterface::~CustomNvdecDeviceInterface() {
  if (nvdecDecoder_) {
    // TODO: Clean up NVDEC decoder
    // delete nvdecDecoder_;
    nvdecDecoder_ = nullptr;
  }
}

std::optional<const AVCodec*> CustomNvdecDeviceInterface::findCodec(
    const AVCodecID& codecId) {
  // For custom NVDEC, we bypass FFmpeg codec selection entirely
  // We'll handle the codec selection in our own NVDEC initialization
  return std::nullopt;
}

void CustomNvdecDeviceInterface::initializeContext(AVCodecContext* codecContext) {
  // Don't set hw_device_ctx - we handle decoding directly with NVDEC SDK
  // Just ensure CUDA context exists for PyTorch tensors
  torch::Tensor dummyTensor = torch::empty(
      {1}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));
  
  // Initialize our custom NVDEC decoder
  initializeNvdecDecoder(codecContext->codec_id);
}

void CustomNvdecDeviceInterface::initializeNvdecDecoder(AVCodecID codecId) {
  // TODO: Initialize NVDEC decoder with custom parameters
  // This is where you would create the NvDecoder instance with your desired settings
  
  /*
  Example implementation with NVIDIA Video Codec SDK:
  
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
    default:
      TORCH_CHECK(false, "Unsupported codec for custom NVDEC: ", codecId);
  }
  
  // Create NVDEC decoder with custom settings
  CUcontext cuContext = nullptr;  // Use current CUDA context
  nvdecDecoder_ = new NvDecoder(
      cuContext,
      false,  // bUseDeviceFrame - we'll handle GPU memory ourselves
      nvCodec,
      nullptr,  // pLockFn
      nullptr,  // pUnlockFn
      nullptr,  // pUserData
      0,        // nWidth - will be set from stream
      0,        // nHeight - will be set from stream
      1000,     // nMaxWidth - adjust as needed
      1000,     // nMaxHeight - adjust as needed
      true      // bDeviceFramePitched
  );
  */
  
  // For now, just mark as initialized
  TORCH_CHECK(codecId != AV_CODEC_ID_NONE, "Invalid codec ID for custom NVDEC");
  
  // Placeholder: In a real implementation, nvdecDecoder_ would be initialized here
  // nvdecDecoder_ = /* create NvDecoder instance */;
}

UniqueAVFrame CustomNvdecDeviceInterface::decodePacket(ReferenceAVPacket& packet) {
  TORCH_CHECK(nvdecDecoder_ != nullptr, "NVDEC decoder not initialized");
  
  // TODO: Implement direct NVDEC decoding
  /*
  Example implementation with NVIDIA Video Codec SDK:
  
  // Extract compressed data from AVPacket
  uint8_t* compressedData = packet->data;
  int size = packet->size;
  int64_t pts = packet->pts;
  int64_t duration = packet->duration;
  
  // Decode with NVDEC
  int numFramesDecoded = nvdecDecoder_->Decode(compressedData, size, 0, pts);
  
  if (numFramesDecoded > 0) {
    // Get decoded frames from NVDEC
    uint8_t** decodedFrames = nvdecDecoder_->GetFrame();
    
    // Convert first decoded frame to AVFrame
    return convertNvdecOutputToAVFrame(
        decodedFrames[0],
        nvdecDecoder_->GetWidth(),
        nvdecDecoder_->GetHeight(),
        pts,
        duration);
  }
  */
  
  // Placeholder implementation - in reality this would decode using NVDEC SDK
  TORCH_CHECK(false, "Custom NVDEC decoding not yet implemented - requires NVIDIA Video Codec SDK");
  return UniqueAVFrame(nullptr);
}

UniqueAVFrame CustomNvdecDeviceInterface::convertNvdecOutputToAVFrame(
    uint8_t* decodedFrame, 
    int width, 
    int height, 
    int64_t pts, 
    int64_t duration) {
  
  // TODO: Convert NVDEC output to AVFrame
  /*
  Example implementation:
  
  UniqueAVFrame avFrame(av_frame_alloc());
  TORCH_CHECK(avFrame.get() != nullptr, "Failed to allocate AVFrame");
  
  // Set frame properties
  avFrame->width = width;
  avFrame->height = height;
  avFrame->format = AV_PIX_FMT_CUDA;  // Indicate this is GPU data
  avFrame->pts = pts;
  avFrame->pkt_duration = duration;
  
  // Set up GPU data pointers
  // The exact implementation depends on NVDEC output format (usually NV12)
  avFrame->data[0] = decodedFrame;  // Y plane
  avFrame->data[1] = decodedFrame + (width * height);  // UV plane for NV12
  avFrame->linesize[0] = width;
  avFrame->linesize[1] = width;  // UV plane has same width for NV12
  
  return avFrame;
  */
  
  // Placeholder
  TORCH_CHECK(false, "NVDEC to AVFrame conversion not yet implemented");
  return UniqueAVFrame(nullptr);
}

void CustomNvdecDeviceInterface::convertAVFrameToFrameOutput(
    const VideoStreamOptions& videoStreamOptions,
    const AVRational& timeBase,
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  
  // For custom NVDEC, the frame should already be on GPU
  // We need to convert from NVDEC's output format (typically NV12) to RGB
  
  TORCH_CHECK(avFrame->format == AV_PIX_FMT_CUDA,
              "Expected CUDA format frame from custom NVDEC decoder");
  
  // TODO: Implement custom GPU-based color conversion
  /*
  Example implementation using CUDA kernels or NPP:
  
  auto frameDims = getHeightAndWidthFromOptionsOrAVFrame(videoStreamOptions, avFrame);
  int height = frameDims.height;
  int width = frameDims.width;
  
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