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

#include "src/torchcodec/_core/nvcuvid_include/cuviddec.h"
#include "src/torchcodec/_core/nvcuvid_include/nvcuvid.h"

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
  CustomNvdecDeviceInterface* decoder =
      static_cast<CustomNvdecDeviceInterface*>(pUserData);
  return decoder->handleVideoSequence(pVideoFormat);
}

static int CUDAAPI
HandlePictureDecode(void* pUserData, CUVIDPICPARAMS* pPicParams) {
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
  printf("  IN CNI::CustomNvdecDeviceInterface\n");
  fflush(stdout);
  TORCH_CHECK(
      g_cuda_custom_nvdec, "CustomNvdecDeviceInterface was not registered!");
  TORCH_CHECK(
      device_.type() == torch::kCUDA, "Unsupported device: ", device_.str());
}

CustomNvdecDeviceInterface::~CustomNvdecDeviceInterface() {
  printf("  IN CNI::destructor\n");
  fflush(stdout);
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

  parserCreated_ = false;
}

std::optional<const AVCodec*> CustomNvdecDeviceInterface::findCodec(
    const AVCodecID& codecId) {

  // TODONVDEC uhh???
  printf("  IN CNI::findCodec\n");
  fflush(stdout);
  // For custom NVDEC, we bypass FFmpeg codec selection entirely
  // We'll handle the codec selection in our own NVDEC initialization
  (void)codecId; // Suppress unused parameter warning
  return std::nullopt;
}

void CustomNvdecDeviceInterface::initializeContext(
    AVCodecContext* codecContext) {
  printf("  IN CNI::initializeContext\n");
  fflush(stdout);
  // Don't set hw_device_ctx - we handle decoding directly with NVDEC SDK
  // Just ensure CUDA context exists for PyTorch tensors
  torch::Tensor dummyTensor = torch::empty(
      {1}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));

  // Convert FFmpeg codec ID to NVDEC codec enum
  cudaVideoCodec nvCodec;
  switch (codecContext->codec_id) {
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
          avcodec_get_name(codecContext->codec_id));
  }

  // TODONVDEC figure out why this is needed and where videoFormat_ is actually used.
  // Maybe this isn't needed at all since this gets overridden in handleVideoSequence?
  memset(&videoFormat_, 0, sizeof(videoFormat_));
  videoFormat_.codec = nvCodec;
  videoFormat_.coded_width = 0; // Will be set when we get the first frame
  videoFormat_.coded_height = 0; // Will be set when we get the first frame
  videoFormat_.chroma_format = cudaVideoChromaFormat_420;
  videoFormat_.bit_depth_luma_minus8 = 0;
  videoFormat_.bit_depth_chroma_minus8 = 0;

  // TODONVDEC: The nvdec docs clearly state that the CUvideoparser is an
  // optional component, and that we could just rely on FFMpeg instead:
  // https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-video-decoder-api-prog-guide/index.html#video-decoder-pipeline.
  // The tradeoff is unclear to me ATM.
  createVideoParser();
}


void CustomNvdecDeviceInterface::createVideoParser() {
  printf("  IN CNI::createVideoParser\n");
  fflush(stdout);
  if (parserCreated_) {
    // TODONVDEC - is this needed?
    return;
  }
  
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

  CUresult result = cuvidCreateVideoParser(&videoParser_, &parserParams);
  TORCH_CHECK(
      result == CUDA_SUCCESS, "Failed to create video parser: ", result);

  parserCreated_ = true;
}

// This callback is called by the parser within cuvidParseVideoData, either when
// the parser encounters the start of the headers, or when "there is a change in
// the sequence" - I don't know what that means. Maybe a resolution change?
int CustomNvdecDeviceInterface::handleVideoSequence(
    CUVIDEOFORMAT* pVideoFormat) {
  printf("    IN CNI::handleVideoSequence\n");
  fflush(stdout);
  TORCH_CHECK(pVideoFormat != nullptr, "Invalid video format");

  // Store video format
  videoFormat_ = *pVideoFormat;


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
  printf("    IN CNI::handlePictureDecode\n");
  fflush(stdout);
  TORCH_CHECK(pPicParams != nullptr, "Invalid picture parameters");

  if (!decoder_) {
    return 0; // No decoder available
  }

  CUresult result = cuvidDecodePicture(decoder_, pPicParams);
  return (result == CUDA_SUCCESS) ? 1 : 0;
}

int CustomNvdecDeviceInterface::handlePictureDisplay(
    CUVIDPARSERDISPINFO* pDispInfo) {
  printf("    IN CNI::handlePictureDisplay\n");
  fflush(stdout);
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
  printf("  IN CNI::decodePacketDirectly\n");
  fflush(stdout);

  // Extract compressed data from AVPacket
  uint8_t* compressedData = packet->data;
  int size = packet->size;
  int64_t pts = packet->pts;

  TORCH_CHECK(compressedData != nullptr && size > 0, "Invalid packet data");
  TORCH_CHECK(parserCreated_, "Video parser not initialized");

  // TODONVDEC: double check this against pynvvideocodec, especially the flags
  // which should be used to indicate end of stream
  CUVIDSOURCEDATAPACKET cudaPacket = {0};
  cudaPacket.payload = compressedData;
  cudaPacket.payload_size = size;
  cudaPacket.flags = CUVID_PKT_TIMESTAMP;
  cudaPacket.timestamp = pts;

  printf("  In CNI calling cuvidParseVideoData\n");
  fflush(stdout);
  CUresult result = cuvidParseVideoData(videoParser_, &cudaPacket);
  printf("  In CNI after cuvidParseVideoData\n");
  fflush(stdout);
  TORCH_CHECK(result == CUDA_SUCCESS, "Failed to parse video data: ", result);

  std::lock_guard<std::mutex> lock(frameQueueMutex_);
  if (frameQueue_.empty()) {
    printf("  No frame ready after parsing\n");
    fflush(stdout);
    // TODONVDEC: might want return AVERROR(EAGAIN), since this seems morally
    // equivalent.
    return UniqueAVFrame(nullptr);
  }

  FrameData frameData = frameQueue_.front();
  frameQueue_.pop();

  // Convert the NVDEC frame to AVFrame
  UniqueAVFrame avFrame = convertCudaFrameToAVFrame(frameData.framePtr, frameData.pitch, frameData.dispInfo);

  // TODONVDEC: Understand this. This is related to the concept of "output surface". 
  // https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-video-decoder-api-prog-guide/index.html#preparing-the-decoded-frame-for-further-processing
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

  printf("  In CNI convertAVFrameToFrameOutput\n");
  fflush(stdout);

  TORCH_CHECK(
      avFrame->format == AV_PIX_FMT_CUDA,
      "Expected CUDA format frame from custom NVDEC decoder");

  // TODONVDEC: we use the 'default' cuda device interface for color conversion.
  // That's a temporary hack to make things work. *IF* we keep both device
  // interfaces then we should abstract the color conversion stuff separately.
  // If we only keep this device interface, we can just integrate the color
  // conversion code here.
  auto cudaDevice = torch::Device(torch::kCUDA);
  auto cudaInterface = createDeviceInterface(cudaDevice);
  AVCodecContext dummyCodecContext = {};
  cudaInterface->initializeContext(&dummyCodecContext);

  FrameOutput cudaFrameOutput;
  cudaInterface->convertAVFrameToFrameOutput(
      videoStreamOptions,
      timeBase,
      avFrame,
      cudaFrameOutput,
      preAllocatedOutputTensor);

  frameOutput.data = cudaFrameOutput.data.to(device_);
}

} // namespace facebook::torchcodec
