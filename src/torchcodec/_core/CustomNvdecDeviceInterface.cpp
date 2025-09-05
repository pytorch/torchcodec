// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/types.h>
#include <mutex>
#include <vector>
#include <unistd.h>  // For usleep

#include "src/torchcodec/_core/CustomNvdecDeviceInterface.h"
#include "src/torchcodec/_core/DeviceInterface.h"
#include "src/torchcodec/_core/FFMPEGCommon.h"

#include "src/torchcodec/_core/nvcuvid_include/cuviddec.h"
#include "src/torchcodec/_core/nvcuvid_include/nvcuvid.h"
#include <cuda_runtime.h>  // For cudaStreamSynchronize

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
  TORCH_CHECK(
      g_cuda_custom_nvdec, "CustomNvdecDeviceInterface was not registered!");
  TORCH_CHECK(
      device_.type() == torch::kCUDA, "Unsupported device: ", device_.str());
  
  // Initialize frame buffer for B-frame reordering
  frameBuffer_.resize(MAX_DECODE_SURFACES);
  
  // Initialize decode surface tracking (like DALI)
  surfaceInUse_.resize(MAX_DECODE_SURFACES, false);
  
}

CustomNvdecDeviceInterface::~CustomNvdecDeviceInterface() {
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

void CustomNvdecDeviceInterface::setTimeBase(const AVRational& timeBase) {
  timeBase_ = timeBase;
}


void CustomNvdecDeviceInterface::createVideoParser() {
  if (parserCreated_) {
    // TODONVDEC - is this needed?
    return;
  }
  
  // Set up video parser parameters
  CUVIDPARSERPARAMS parserParams = {};
  parserParams.CodecType = videoFormat_.codec;
  // Set to dummy value initially, sequence callback will update this
  // as recommended by NVDEC docs
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
  TORCH_CHECK(pVideoFormat != nullptr, "Invalid video format");

  // Store video format
  videoFormat_ = *pVideoFormat;

  // Use min_num_decode_surfaces from video format for optimal memory allocation
  // as recommended by NVDEC docs and implemented in DALI
  unsigned int numSurfaces = pVideoFormat->min_num_decode_surfaces;
  if (numSurfaces == 0) {
    numSurfaces = 20;  // DALI's fallback value
  }

  // Create decoder with the video format
  CUVIDDECODECREATEINFO createInfo = { 0 };
  createInfo.CodecType = pVideoFormat->codec;
  createInfo.ulWidth = pVideoFormat->coded_width;
  createInfo.ulHeight = pVideoFormat->coded_height;
  createInfo.ulNumDecodeSurfaces = numSurfaces;
  createInfo.ChromaFormat = pVideoFormat->chroma_format;
  createInfo.OutputFormat = cudaVideoSurfaceFormat_NV12;
  createInfo.bitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;
  createInfo.ulTargetWidth = pVideoFormat->display_area.right - pVideoFormat->display_area.left;
  createInfo.ulTargetHeight = pVideoFormat->display_area.bottom - pVideoFormat->display_area.top;
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

  // Return the number of decode surfaces to update parser's ulMaxNumDecodeSurfaces
  // This follows NVDEC docs recommendation and DALI's implementation
  return numSurfaces;
}

// Parser triggers this callback when bitstream data for one frame is ready
int CustomNvdecDeviceInterface::handlePictureDecode(
    CUVIDPICPARAMS* pPicParams) {

  // Like DALI: if we're flushing, don't process new decode operations
  if (flush_) {
    return 0;
  }

  TORCH_CHECK(pPicParams != nullptr, "Invalid picture parameters");
  TORCH_CHECK(
      decoder_ != nullptr, "Decoder not initialized before picture decode");

  // Like DALI: wait for decode surface to become available
  int totalWait = 0;
  constexpr int sleepPeriod = 500; // microseconds
  constexpr int timeoutSec = 20;
  constexpr bool enableTimeout = false;
  
  int surfaceIndex = pPicParams->CurrPicIdx;
  
  while (surfaceIndex < surfaceInUse_.size() && surfaceInUse_[surfaceIndex]) {
    if (enableTimeout && totalWait++ > timeoutSec * 1000000 / sleepPeriod) {
      return 0;
    }
    usleep(sleepPeriod);
  }


  // Doc say that calling cuvidDecodePicture kicks of the hardware decoding of the frame (async!).
  // We know the frame was successfully decoded when cuvidMapVideoFrame returns
  // successfully.
  // https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-video-decoder-api-prog-guide/index.html#preparing-the-decoded-frame-for-further-processing
  CUresult result = cuvidDecodePicture(decoder_, pPicParams);
  
  if (result == CUDA_SUCCESS) {
    // Mark surface as in-use (like DALI)
    if (surfaceIndex < surfaceInUse_.size()) {
      surfaceInUse_[surfaceIndex] = true;
    }
    
    // Like DALI: manually create display info and call handlePictureDisplay directly
    CUVIDPARSERDISPINFO dispInfo = {};
    dispInfo.picture_index = pPicParams->CurrPicIdx;
    dispInfo.progressive_frame = !pPicParams->field_pic_flag;
    dispInfo.top_field_first = pPicParams->bottom_field_flag ^ 1;
    dispInfo.repeat_first_field = 0;
    
    // Like DALI: call handlePictureDisplay directly, PTS will be assigned there
    handlePictureDisplay(&dispInfo);
    
    return 1;
  } else {
    return 0;
  }
}

// Called directly from handlePictureDecode when a frame is ready for display.
// This is no longer triggered by the parser - we control timing manually like DALI.
int CustomNvdecDeviceInterface::handlePictureDisplay(
    CUVIDPARSERDISPINFO* pDispInfo) {
  TORCH_CHECK(pDispInfo != nullptr, "Invalid display info");

  // EXPERIMENT: More robust PTS assignment
  // Instead of simple FIFO, find the smallest unused PTS from queue
  // This handles cases where queue gets out of sync due to B-frame reordering
  if (!pipedPts_.empty()) {
    // Find the smallest PTS in the queue (earliest in time)
    std::vector<int64_t> queueContents;
    std::queue<int64_t> tempQueue = pipedPts_;
    while (!tempQueue.empty()) {
      queueContents.push_back(tempQueue.front());
      tempQueue.pop();
    }
    
    // Find minimum PTS
    auto minIt = std::min_element(queueContents.begin(), queueContents.end());
    currentPts_ = *minIt;
    
    // Rebuild queue without the selected PTS
    std::queue<int64_t> newQueue;
    bool removed = false;
    for (int64_t pts : queueContents) {
      if (pts != currentPts_ || removed) {
        newQueue.push(pts);
      } else {
        removed = true; // Remove only the first instance
      }
    }
    pipedPts_ = newQueue;
    
  } else {
    // Like DALI: handle case where one packet produces multiple frames
    // Reuse the current PTS for unexpected extra frames
  }
  
  int64_t framePts = currentPts_;
  
  // Set the PTS in the display info
  pDispInfo->timestamp = framePts;
  

  // Buffer frame for B-frame reordering (like DALI)
  std::lock_guard<std::mutex> lock(frameBufferMutex_);
  BufferedFrame* slot = findEmptySlot();
  slot->dispInfo = *pDispInfo;
  slot->pts = framePts;  // Use the PTS we just assigned
  slot->available = true;
  

  return 1;
}

int CustomNvdecDeviceInterface::sendPacket(ReferenceAVPacket& packet) {

  if (!parserCreated_) {
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
    
  } else {
    // End of stream packet
    cudaPacket.flags = CUVID_PKT_ENDOFSTREAM;
    eofSent_ = true;
  }

  CUresult result = cuvidParseVideoData(videoParser_, &cudaPacket);
  if (result != CUDA_SUCCESS) {
    return AVERROR_EXTERNAL;
  }

  return 0;
}

int CustomNvdecDeviceInterface::receiveFrame(UniqueAVFrame& frame) {

  std::lock_guard<std::mutex> lock(frameBufferMutex_);
  
  // Find frame with earliest PTS for display order (like DALI)
  BufferedFrame* earliestFrame = findFrameWithEarliestPts();
  
  if (earliestFrame == nullptr) {
    if (eofSent_) {
      return AVERROR_EOF;
    } else {
      return AVERROR(EAGAIN);
    }
  }

  CUVIDPARSERDISPINFO dispInfo = earliestFrame->dispInfo;
  int64_t pts = earliestFrame->pts;
  
  // Mark slot as used
  earliestFrame->available = false;
  earliestFrame->pts = -1;


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
    return AVERROR_EXTERNAL;
  }

  
  // Convert the NVDEC frame to AVFrame, passing the correct PTS
  frame = convertCudaFrameToAVFrame(framePtr, pitch, dispInfo, timeBase_);

  // Unmap the frame
  cuvidUnmapVideoFrame(decoder_, framePtr);

  // Mark surface as free (like DALI does in convert_frame)
  int surfaceIndex = dispInfo.picture_index;
  if (surfaceIndex < surfaceInUse_.size()) {
    surfaceInUse_[surfaceIndex] = false;
  }

  return 0;
}

void CustomNvdecDeviceInterface::flush() {

  // Set flush flag like DALI to prevent new decode operations
  flush_ = true;

  // Send EOS packet to drain decoder like DALI does
  if (parserCreated_ && !eofSent_) {
    CUVIDSOURCEDATAPACKET cudaPacket = {0};
    cudaPacket.flags = CUVID_PKT_ENDOFSTREAM;
    CUresult result = cuvidParseVideoData(videoParser_, &cudaPacket);
    if (result == CUDA_SUCCESS) {
      eofSent_ = true;
    }
  }

  // Clear flush flag like DALI does
  flush_ = false;

  // Clear frame buffer like DALI
  size_t availableFrames = 0;
  {
    std::lock_guard<std::mutex> lock(frameBufferMutex_);
    availableFrames = std::count_if(frameBuffer_.begin(), frameBuffer_.end(), 
        [](const BufferedFrame& f) { return f.available; });
    for (auto& frame : frameBuffer_) {
      frame.available = false;
      frame.pts = -1;
    }
  }

  // Clear PTS queue like DALI
  size_t ptsQueueSize = pipedPts_.size();
  while (!pipedPts_.empty()) {
    pipedPts_.pop();
  }

  // Synchronize CUDA stream to ensure all operations complete
  // TODONVDEC make sure this is syncing the right stream, not necessarily stream 0
  cudaStreamSynchronize(0);

  // Clear decode surface usage tracking
  for (size_t i = 0; i < surfaceInUse_.size(); ++i) {
    surfaceInUse_[i] = false;
  }
  
  // Reset current PTS like DALI does
  currentPts_ = AV_NOPTS_VALUE;

  // Reset EOF flag so we can decode more (like DALI does)
  eofSent_ = false;

}



UniqueAVFrame CustomNvdecDeviceInterface::convertCudaFrameToAVFrame(
    CUdeviceptr framePtr,
    unsigned int pitch,
    const CUVIDPARSERDISPINFO& dispInfo,
    const AVRational& timeBase) {
  TORCH_CHECK(framePtr != 0, "Invalid CUDA frame pointer");

  // Get frame dimensions from video format display area (not coded dimensions)
  // This matches DALI's approach and avoids padding issues
  int width = videoFormat_.display_area.right - videoFormat_.display_area.left;
  int height = videoFormat_.display_area.bottom - videoFormat_.display_area.top;

  TORCH_CHECK(width > 0 && height > 0, "Invalid frame dimensions");
  TORCH_CHECK(pitch >= width, "Pitch must be >= width");


  // Allocate AVFrame
  UniqueAVFrame avFrame(av_frame_alloc());
  TORCH_CHECK(avFrame.get() != nullptr, "Failed to allocate AVFrame");

  // Set frame properties
  avFrame->width = width;
  avFrame->height = height;
  avFrame->format = AV_PIX_FMT_CUDA; // Indicate this is GPU data
  avFrame->pts = dispInfo.timestamp; // This PTS was set correctly by handlePictureDisplay
  
  // Calculate frame duration from NVDEC frame rate and stream timebase
  if (videoFormat_.frame_rate.numerator > 0 && videoFormat_.frame_rate.denominator > 0 &&
      timeBase.num > 0 && timeBase.den > 0) {
    // Duration in seconds = frame_rate.denominator / frame_rate.numerator
    // Duration in timebase units = (duration_seconds * timeBase.den) / timeBase.num
    // = (frame_rate.denominator * timeBase.den) / (frame_rate.numerator * timeBase.num)
    avFrame->duration = (int64_t)((videoFormat_.frame_rate.denominator * timeBase.den) / 
                                  (videoFormat_.frame_rate.numerator * timeBase.num));
  } else {
    avFrame->duration = 0; // Unknown duration
  }
  

  // Set color space and color range from NVDEC video format (like DALI does)
  // This is crucial for proper color conversion!
  
  // Map NVDEC matrix coefficients to FFmpeg color space
  switch (videoFormat_.video_signal_description.matrix_coefficients) {
    case 1:  // ITU-R BT.709
      avFrame->colorspace = AVCOL_SPC_BT709;
      break;
    case 5:  // ITU-R BT.470-2 System B, G (BT.601 PAL)
    case 6:  // ITU-R BT.601-6 NTSC  
      avFrame->colorspace = AVCOL_SPC_SMPTE170M; // BT.601
      break;
    default:
      // Default to BT.709 for unknown coefficients
      avFrame->colorspace = AVCOL_SPC_BT709;
      break;
  }
  
  // Set color range from full range flag
  if (videoFormat_.video_signal_description.video_full_range_flag) {
    avFrame->color_range = AVCOL_RANGE_JPEG;  // Full range (0-255)
  } else {
    avFrame->color_range = AVCOL_RANGE_MPEG;  // Limited range (16-235)
  }
  

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

  // Store timeBase for duration calculations in convertCudaFrameToAVFrame
  timeBase_ = timeBase;


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
  
  if (earliest) {
  }
  
  return earliest;
}

} // namespace facebook::torchcodec
