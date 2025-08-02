// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/types.h>
#include <mutex>
#include <vector>

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

// NVDEC callback functions
static int CUDAAPI
HandleVideoSequence(void* pUserData, CUVIDEOFORMAT* pVideoFormat) {
  // printf("Static HandleVideoSequence called\n");
  CustomNvdecDeviceInterface* decoder =
      static_cast<CustomNvdecDeviceInterface*>(pUserData);
  return decoder->handleVideoSequence(pVideoFormat);
}

static int CUDAAPI
HandlePictureDecode(void* pUserData, CUVIDPICPARAMS* pPicParams) {
  // printf("Static HandlePictureDecode called\n");
  CustomNvdecDeviceInterface* decoder =
      static_cast<CustomNvdecDeviceInterface*>(pUserData);
  return decoder->handlePictureDecode(pPicParams);
}

static int CUDAAPI
HandlePictureDisplay(void* pUserData, CUVIDPARSERDISPINFO* pDispInfo) {
  // printf("Static HandlePictureDisplay called\n");
  CustomNvdecDeviceInterface* decoder =
      static_cast<CustomNvdecDeviceInterface*>(pUserData);
  return decoder->handlePictureDisplay(pDispInfo);
}

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
  // Clean up any remaining frames in the queue
  {
    std::lock_guard<std::mutex> lock(frameQueueMutex_);
    while (!frameQueue_.empty()) {
      FrameData frameData = frameQueue_.front();
      frameQueue_.pop();

      // Unmap the frame if it's still mapped
      if (decoder_ && frameData.framePtr != 0) {
        cuvidUnmapVideoFrame(decoder_, frameData.framePtr);
      }
    }
  }

  // Clean up decoder
  if (decoder_) {
    cuvidDestroyDecoder(decoder_);
    decoder_ = nullptr;
  }

  // Clean up video parser
  if (videoParser_) {
    cuvidDestroyVideoParser(videoParser_);
    videoParser_ = nullptr;
  }

  isInitialized_ = false;
  parserInitialized_ = false;
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

  // Initialize video parser with the codec ID and extradata
  initializeVideoParser(codecContext->codec_id, codecContext->extradata, codecContext->extradata_size);
}

void CustomNvdecDeviceInterface::initializeNvdecDecoder(AVCodecID codecId) {
  if (isInitialized_) {
    return; // Already initialized
  }

  // Store the codec ID for later use
  currentCodecId_ = codecId;

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

  // Initialize video format structure (decoder will be created in
  // handleVideoSequence)
  memset(&videoFormat_, 0, sizeof(videoFormat_));
  videoFormat_.codec = nvCodec;
  videoFormat_.coded_width = 0; // Will be set when we get the first frame
  videoFormat_.coded_height = 0; // Will be set when we get the first frame
  videoFormat_.chroma_format = cudaVideoChromaFormat_420;
  videoFormat_.bit_depth_luma_minus8 = 0;
  videoFormat_.bit_depth_chroma_minus8 = 0;

  isInitialized_ = true;
}

void CustomNvdecDeviceInterface::initializeVideoParser(AVCodecID codecId, uint8_t* extradata, int extradata_size) {
  if (parserInitialized_) {
    return;
  }

  // printf("Initializing NVDEC video parser for codec\n");
  
  // Set up video parser parameters
  CUVIDPARSERPARAMS parserParams = {};
  parserParams.CodecType = videoFormat_.codec;
  parserParams.ulMaxNumDecodeSurfaces = 1;
  parserParams.ulClockRate = 1000;
  parserParams.ulErrorThreshold = 0;
  parserParams.ulMaxDisplayDelay = 1;
  parserParams.pUserData = this;
  parserParams.pfnSequenceCallback = HandleVideoSequence;
  parserParams.pfnDecodePicture = HandlePictureDecode;
  parserParams.pfnDisplayPicture = HandlePictureDisplay;
  
  // printf("Parser params: pUserData=%p, pfnSequenceCallback=%p, pfnDecodePicture=%p, pfnDisplayPicture=%p\n", 
  //        parserParams.pUserData, (void*)parserParams.pfnSequenceCallback, 
  //        (void*)parserParams.pfnDecodePicture, (void*)parserParams.pfnDisplayPicture);

  CUresult result = cuvidCreateVideoParser(&videoParser_, &parserParams);
  TORCH_CHECK(
      result == CUDA_SUCCESS, "Failed to create video parser: ", result);

  parserInitialized_ = true;
}

int CustomNvdecDeviceInterface::handleVideoSequence(
    CUVIDEOFORMAT* pVideoFormat) {
      // printf("In CustomNvdecDeviceInterface::handleVideoSequence\n");
  TORCH_CHECK(pVideoFormat != nullptr, "Invalid video format");

  // Store video format
  videoFormat_ = *pVideoFormat;

  // Get current CUDA context
  CUresult cuResult = cuCtxGetCurrent(&context_);
  if (cuResult != CUDA_SUCCESS || context_ == nullptr) {
    TORCH_CHECK(false, "Failed to get CUDA context for device ", device_.index());
  }

  // Create decoder with the video format
  CUVIDDECODECREATEINFO createInfo = {};
  createInfo.CodecType = pVideoFormat->codec;
  createInfo.ulWidth = pVideoFormat->coded_width;
  createInfo.ulHeight = pVideoFormat->coded_height;
  createInfo.ulNumDecodeSurfaces = 4;
  createInfo.ChromaFormat = pVideoFormat->chroma_format;
  createInfo.OutputFormat = cudaVideoSurfaceFormat_NV12;
  createInfo.bitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;
  createInfo.ulTargetWidth = pVideoFormat->coded_width;
  createInfo.ulTargetHeight = pVideoFormat->coded_height;
  createInfo.ulNumOutputSurfaces = 2;
  createInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
  createInfo.vidLock = nullptr;

  CUresult result = cuvidCreateDecoder(&decoder_, &createInfo);
  if (result != CUDA_SUCCESS) {
    TORCH_CHECK(false, "Failed to create NVDEC decoder: ", result);
  }

  return 1; // Success
}

int CustomNvdecDeviceInterface::handlePictureDecode(
    CUVIDPICPARAMS* pPicParams) {
  TORCH_CHECK(pPicParams != nullptr, "Invalid picture parameters");
  // printf("In CustomNvdecDeviceInterface::handlePictureDecode\n");

  if (!decoder_) {
    return 0; // No decoder available
  }

  CUresult result = cuvidDecodePicture(decoder_, pPicParams);
  return (result == CUDA_SUCCESS) ? 1 : 0;
}

int CustomNvdecDeviceInterface::handlePictureDisplay(
    CUVIDPARSERDISPINFO* pDispInfo) {
  TORCH_CHECK(pDispInfo != nullptr, "Invalid display info");

  // Queue the frame for later retrieval
  std::lock_guard<std::mutex> lock(frameQueueMutex_);

  // Map the decoded frame
  CUdeviceptr framePtr = 0;
  unsigned int pitch = 0;
  CUVIDPROCPARAMS procParams = {};
  procParams.progressive_frame = pDispInfo->progressive_frame;
  procParams.top_field_first = pDispInfo->top_field_first;
  procParams.unpaired_field = pDispInfo->repeat_first_field < 0;

  CUresult result = cuvidMapVideoFrame(
      decoder_,
      pDispInfo->picture_index,
      &framePtr,
      &pitch,
      &procParams);
  if (result == CUDA_SUCCESS) {
    FrameData frameData = {framePtr, pitch, *pDispInfo};
    frameQueue_.push(frameData);
  }

  return 1;
}

UniqueAVFrame CustomNvdecDeviceInterface::decodePacketDirectly(
    ReferenceAVPacket& packet) {
  TORCH_CHECK(isInitialized_, "NVDEC decoder not initialized");

  // Extract compressed data from AVPacket
  uint8_t* compressedData = packet->data;
  int size = packet->size;
  int64_t pts = packet->pts;

  TORCH_CHECK(compressedData != nullptr && size > 0, "Invalid packet data");

  // Video parser should already be initialized from initializeContext
  TORCH_CHECK(parserInitialized_, "Video parser not initialized");

  // Parse the packet data (now already in Annex B format from bitstream filter)
  // printf("About to parse packet: size=%d, pts=%lld\n", size, pts);
  // printf("First 8 bytes: %02x %02x %02x %02x %02x %02x %02x %02x\n", 
  //        compressedData[0], compressedData[1], compressedData[2], compressedData[3],
  //        compressedData[4], compressedData[5], compressedData[6], compressedData[7]);

  CUVIDSOURCEDATAPACKET cudaPacket = {0};  // Initialize all fields to 0
  cudaPacket.payload = compressedData;
  cudaPacket.payload_size = size;
  cudaPacket.flags = CUVID_PKT_TIMESTAMP;
  cudaPacket.timestamp = pts;

  CUresult result = cuvidParseVideoData(videoParser_, &cudaPacket);
  // printf("Parse result: %d\n", result);
  TORCH_CHECK(result == CUDA_SUCCESS, "Failed to parse video data: ", result);

  // Check if we have any decoded frames available
  std::lock_guard<std::mutex> lock(frameQueueMutex_);
  if (frameQueue_.empty()) {
    // No frame ready yet (async decoding)
    return UniqueAVFrame(nullptr);
  }

  // Get the first available frame
  FrameData frameData = frameQueue_.front();
  frameQueue_.pop();

  // Convert the NVDEC frame to AVFrame
  UniqueAVFrame avFrame = convertCudaFrameToAVFrame(frameData.framePtr, frameData.pitch, frameData.dispInfo);

  // Unmap the frame
  cuvidUnmapVideoFrame(decoder_, frameData.framePtr);

  return avFrame;
}


UniqueAVFrame CustomNvdecDeviceInterface::convertCudaFrameToAVFrame(
    CUdeviceptr framePtr,
    unsigned int pitch,
    const CUVIDPARSERDISPINFO& dispInfo) {
  TORCH_CHECK(framePtr != 0, "Invalid CUDA frame pointer");

  // Get frame dimensions from video format
  int width = videoFormat_.coded_width;
  int height = videoFormat_.coded_height;

  TORCH_CHECK(width > 0 && height > 0, "Invalid frame dimensions");
  TORCH_CHECK(pitch >= width, "Pitch must be >= width");

  // printf("Frame conversion: width=%d, height=%d, pitch=%u\n", width, height, pitch);

  // Allocate AVFrame
  UniqueAVFrame avFrame(av_frame_alloc());
  TORCH_CHECK(avFrame.get() != nullptr, "Failed to allocate AVFrame");

  // Set frame properties
  avFrame->width = width;
  avFrame->height = height;
  avFrame->format = AV_PIX_FMT_CUDA; // Indicate this is GPU data
  avFrame->pts = dispInfo.timestamp;
  avFrame->duration = 0; // Will be set by caller if needed

  // For NVDEC output in NV12 format, we need to set up the data pointers
  // The framePtr points to the beginning of the NV12 data
  avFrame->data[0] = reinterpret_cast<uint8_t*>(framePtr); // Y plane
  avFrame->data[1] = reinterpret_cast<uint8_t*>(framePtr + (pitch * height)); // UV plane (using pitch, not width)
  avFrame->data[2] = nullptr;
  avFrame->data[3] = nullptr;

  // Set line sizes for NV12 format using the actual NVDEC pitch
  avFrame->linesize[0] = pitch; // Y plane stride (use actual pitch from NVDEC)
  avFrame->linesize[1] = pitch; // UV plane stride (use actual pitch from NVDEC)
  avFrame->linesize[2] = 0;
  avFrame->linesize[3] = 0;

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

  auto cpuDevice = torch::Device(torch::kCUDA);
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
