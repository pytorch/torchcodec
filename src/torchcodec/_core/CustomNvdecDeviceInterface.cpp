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

// HandlePictureDisplay callback removed - we now call handlePictureDisplay directly
// from handlePictureDecode like DALI does

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
  
  // Initialize frame buffer for B-frame reordering
  frameBuffer_.resize(MAX_DECODE_SURFACES);
  printf("  Initialized frame buffer with %d slots\n", MAX_DECODE_SURFACES);
  fflush(stdout);
}

CustomNvdecDeviceInterface::~CustomNvdecDeviceInterface() {
  printf("  IN CNI::destructor\n");
  fflush(stdout);
  // Clean up any remaining frames in the buffer
  {
    std::lock_guard<std::mutex> lock(frameBufferMutex_);
    for (auto& frame : frameBuffer_) {
      frame.available = false;
      frame.pts = -1;
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
  parserParams.pfnDisplayPicture = nullptr;  // Like DALI - we handle display manually

  CUresult result = cuvidCreateVideoParser(&videoParser_, &parserParams);
  TORCH_CHECK(
      result == CUDA_SUCCESS, "Failed to create video parser: ", result);

  parserCreated_ = true;
}

// This callback is called by the parser within cuvidParseVideoData, either when
// the parser encounters the start of the headers, or when "there is a change in
// the sequence" - which, I assume means a change in any one of CUVIDEOFORMAT
// fields?
int CustomNvdecDeviceInterface::handleVideoSequence(
    CUVIDEOFORMAT* pVideoFormat) {
  printf("    IN CNI::handleVideoSequence\n");
  fflush(stdout);
  TORCH_CHECK(pVideoFormat != nullptr, "Invalid video format");

  // Store video format
  videoFormat_ = *pVideoFormat;


  // Create decoder with the video format
  CUVIDDECODECREATEINFO createInfo = { 0 };
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

  // TODONVDEC: We are re-recreating a decoder, which I think assumes there is
  // no sequence change and that this is only called at the start of the header
  // (see comment above).
  // We should consider change of sequence, and also look into re-configuring
  // APIs. Also need a CUVideoDecoder cache somewhere.
  CUresult result = cuvidCreateDecoder(&decoder_, &createInfo);
  if (result != CUDA_SUCCESS) {
    TORCH_CHECK(false, "Failed to create NVDEC decoder: ", result);
  }

  return 1;
}

// Parser triggers this callback when bitstream data for one frame is ready
int CustomNvdecDeviceInterface::handlePictureDecode(
    CUVIDPICPARAMS* pPicParams) {
  printf("    IN CNI::handlePictureDecode\n");
  fflush(stdout);

  TORCH_CHECK(pPicParams != nullptr, "Invalid picture parameters");
  TORCH_CHECK(
      decoder_ != nullptr, "Decoder not initialized before picture decode");

  // TODONVDEC: Better understand the CUVIDPICPARAMS object. Confusingly the doc
  // say that it's up to the application to fill it up:
  // https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-video-decoder-api-prog-guide/index.html#decoding-the-framefield
  
  // Doc say that calling cuvidDecodePicture kicks of the hardware decoding of the frame (async!).
  // We know the frame was successfully decoded when cuvidMapVideoFrame returns
  // successfully.
  // https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-video-decoder-api-prog-guide/index.html#preparing-the-decoded-frame-for-further-processing
  CUresult result = cuvidDecodePicture(decoder_, pPicParams);
  
  if (result == CUDA_SUCCESS) {
    // Like DALI: manually create display info and call handlePictureDisplay directly
    CUVIDPARSERDISPINFO dispInfo = {};
    dispInfo.picture_index = pPicParams->CurrPicIdx;
    dispInfo.progressive_frame = !pPicParams->field_pic_flag;
    dispInfo.top_field_first = pPicParams->bottom_field_flag ^ 1;
    dispInfo.repeat_first_field = 0;
    
    // Like DALI: get PTS from queue for proper frame timing
    int64_t framePts = AV_NOPTS_VALUE;
    if (!pipedPts_.empty()) {
      framePts = pipedPts_.front();
      pipedPts_.pop();
    }
    dispInfo.timestamp = framePts;
    
    printf("    Calling handlePictureDisplay directly from decode callback, assigned PTS=%ld (from queue)\n", framePts);
    fflush(stdout);
    handlePictureDisplay(&dispInfo);
    
    return 1;
  } else {
    printf("    cuvidDecodePicture failed with result: %d\n", result);
    fflush(stdout);
    return 0;
  }
}

// Called directly from handlePictureDecode when a frame is ready for display.
// This is no longer triggered by the parser - we control timing manually like DALI.
int CustomNvdecDeviceInterface::handlePictureDisplay(
    CUVIDPARSERDISPINFO* pDispInfo) {
  printf("    IN CNI::handlePictureDisplay, PTS=%ld\n", pDispInfo->timestamp);
  fflush(stdout);
  TORCH_CHECK(pDispInfo != nullptr, "Invalid display info");

  // Buffer frame for B-frame reordering (like DALI)
  std::lock_guard<std::mutex> lock(frameBufferMutex_);
  BufferedFrame* slot = findEmptySlot();
  slot->dispInfo = *pDispInfo;
  slot->pts = pDispInfo->timestamp;
  slot->available = true;
  
  printf("      Buffered frame with PTS=%ld in slot, buffer has %zu frames available\n", 
         slot->pts, std::count_if(frameBuffer_.begin(), frameBuffer_.end(), 
         [](const BufferedFrame& f) { return f.available; }));
  fflush(stdout);

  return 1;
}

int CustomNvdecDeviceInterface::sendPacket(ReferenceAVPacket& packet) {
  printf("  IN CNI::sendPacket\n");
  fflush(stdout);

  if (!parserCreated_) {
    printf("  Parser not created, returning error\n");
    return AVERROR(EINVAL);
  }

  CUVIDSOURCEDATAPACKET cudaPacket = {0};
  
  if (packet.get() && packet->data && packet->size > 0) {
    // Regular packet with data
    cudaPacket.payload = packet->data;
    cudaPacket.payload_size = packet->size;
    cudaPacket.flags = CUVID_PKT_TIMESTAMP;
    cudaPacket.timestamp = packet->pts;
    
    // Like DALI: store PTS in queue to assign to frames as they come out
    pipedPts_.push(packet->pts);
    printf("  Sending packet with size %d, pts %ld (queued for frame assignment)\n", packet->size, packet->pts);
  } else {
    // End of stream packet
    cudaPacket.flags = CUVID_PKT_ENDOFSTREAM;
    eofSent_ = true;
    printf("  Sending end of stream packet\n");
  }

  CUresult result = cuvidParseVideoData(videoParser_, &cudaPacket);
  if (result != CUDA_SUCCESS) {
    printf("  cuvidParseVideoData failed with result: %d\n", result);
    fflush(stdout);
    return AVERROR_EXTERNAL;
  }

  printf("  Packet sent successfully\n");
  fflush(stdout);
  return 0;
}

int CustomNvdecDeviceInterface::receiveFrame(UniqueAVFrame& frame) {
  printf("  IN CNI::receiveFrame\n");
  fflush(stdout);

  std::lock_guard<std::mutex> lock(frameBufferMutex_);
  
  // Find frame with earliest PTS for display order (like DALI)
  BufferedFrame* earliestFrame = findFrameWithEarliestPts();
  
  if (earliestFrame == nullptr) {
    if (eofSent_) {
      printf("  No frames available and EOF sent, returning AVERROR_EOF\n");
      fflush(stdout);
      return AVERROR_EOF;
    } else {
      printf("  No frames available, returning AVERROR(EAGAIN)\n");
      fflush(stdout);
      return AVERROR(EAGAIN);
    }
  }

  CUVIDPARSERDISPINFO dispInfo = earliestFrame->dispInfo;
  int64_t pts = earliestFrame->pts;
  
  // Mark slot as used
  earliestFrame->available = false;
  earliestFrame->pts = -1;

  printf("  Processing frame with PTS=%ld in display order, remaining available frames: %zu\n", 
         pts, std::count_if(frameBuffer_.begin(), frameBuffer_.end(), 
         [](const BufferedFrame& f) { return f.available; }));

  // Now map the frame (this was previously done in handlePictureDisplay)
  CUdeviceptr framePtr = 0;
  unsigned int pitch = 0;
  CUVIDPROCPARAMS procParams = {};
  procParams.progressive_frame = dispInfo.progressive_frame;
  procParams.top_field_first = dispInfo.top_field_first;
  procParams.unpaired_field = dispInfo.repeat_first_field < 0;

  CUresult result = cuvidMapVideoFrame(
      decoder_,
      dispInfo.picture_index,
      &framePtr,
      &pitch,
      &procParams);

  if (result != CUDA_SUCCESS) {
    printf("  cuvidMapVideoFrame failed with result: %d\n", result);
    fflush(stdout);
    return AVERROR_EXTERNAL;
  }

  printf("  Frame mapped successfully, converting to AVFrame\n");
  
  // Convert the NVDEC frame to AVFrame
  frame = convertCudaFrameToAVFrame(framePtr, pitch, dispInfo);

  // Unmap the frame
  cuvidUnmapVideoFrame(decoder_, framePtr);

  printf("  Frame received and converted successfully (PTS=%ld)\n", pts);
  fflush(stdout);
  return 0;
}

void CustomNvdecDeviceInterface::flush() {
  printf("  IN CNI::flush\n");
  fflush(stdout);

  // Send end of stream packet to flush decoder
  if (parserCreated_ && !eofSent_) {
    CUVIDSOURCEDATAPACKET cudaPacket = {0};
    cudaPacket.flags = CUVID_PKT_ENDOFSTREAM;
    cuvidParseVideoData(videoParser_, &cudaPacket);
    eofSent_ = true;
  }

  // Clear any remaining frames in buffer
  std::lock_guard<std::mutex> lock(frameBufferMutex_);
  for (auto& frame : frameBuffer_) {
    frame.available = false;
    frame.pts = -1;
  }

  printf("  Flush completed\n");
  fflush(stdout);
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

// Helper method to find an empty slot in frame buffer (like DALI's FindEmptySlot)
CustomNvdecDeviceInterface::BufferedFrame* 
CustomNvdecDeviceInterface::findEmptySlot() {
  for (auto& frame : frameBuffer_) {
    if (!frame.available) {
      return &frame;
    }
  }
  // If no empty slots, expand buffer like DALI does
  frameBuffer_.emplace_back();
  printf("  Expanded frame buffer to size %zu\n", frameBuffer_.size());
  fflush(stdout);
  return &frameBuffer_.back();
}

// Helper method to find frame with earliest PTS for display order
CustomNvdecDeviceInterface::BufferedFrame* 
CustomNvdecDeviceInterface::findFrameWithEarliestPts() {
  BufferedFrame* earliest = nullptr;
  
  for (auto& frame : frameBuffer_) {
    if (frame.available) {
      if (earliest == nullptr || frame.pts < earliest->pts) {
        earliest = &frame;
      }
    }
  }
  return earliest;
}

} // namespace facebook::torchcodec
