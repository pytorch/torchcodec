// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/decoders/_core/VideoDecoder.h"
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include "src/torchcodec/decoders/_core/DeviceInterface.h"
#include "torch/types.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

namespace facebook::torchcodec {
namespace {

double ptsToSeconds(int64_t pts, int den) {
  return static_cast<double>(pts) / den;
}

double ptsToSeconds(int64_t pts, const AVRational& timeBase) {
  return ptsToSeconds(pts, timeBase.den);
}

int64_t secondsToClosestPts(double seconds, const AVRational& timeBase) {
  return static_cast<int64_t>(std::round(seconds * timeBase.den));
}

struct AVInput {
  UniqueAVFormatContext formatContext;
  std::unique_ptr<AVIOBytesContext> ioBytesContext;
};

AVInput createAVFormatContextFromFilePath(const std::string& videoFilePath) {
  AVFormatContext* formatContext = nullptr;
  int open_ret = avformat_open_input(
      &formatContext, videoFilePath.c_str(), nullptr, nullptr);
  if (open_ret != 0) {
    throw std::invalid_argument(
        "Could not open input file: " + videoFilePath + " " +
        getFFMPEGErrorStringFromErrorCode(open_ret));
  }
  TORCH_CHECK(formatContext != nullptr);
  AVInput toReturn;
  toReturn.formatContext.reset(formatContext);
  return toReturn;
}

AVInput createAVFormatContextFromBuffer(const void* buffer, size_t length) {
  AVInput toReturn;
  toReturn.formatContext.reset(avformat_alloc_context());
  TORCH_CHECK(
      toReturn.formatContext.get() != nullptr,
      "Unable to alloc avformat context");
  constexpr int kAVIOInternalTemporaryBufferSize = 64 * 1024;
  toReturn.ioBytesContext.reset(
      new AVIOBytesContext(buffer, length, kAVIOInternalTemporaryBufferSize));
  if (!toReturn.ioBytesContext) {
    throw std::runtime_error("Failed to create AVIOBytesContext");
  }
  toReturn.formatContext->pb = toReturn.ioBytesContext->getAVIO();
  AVFormatContext* tempFormatContext = toReturn.formatContext.release();
  int open_ret =
      avformat_open_input(&tempFormatContext, nullptr, nullptr, nullptr);
  toReturn.formatContext.reset(tempFormatContext);
  if (open_ret != 0) {
    throw std::runtime_error(
        std::string("Failed to open input buffer: ") +
        getFFMPEGErrorStringFromErrorCode(open_ret));
  }
  return toReturn;
}

std::vector<std::string> splitStringWithDelimiters(
    const std::string& str,
    const std::string& delims) {
  std::vector<std::string> result;
  if (str.empty()) {
    return result;
  }

  std::string::size_type start = 0, end = 0;
  while ((end = str.find_first_of(delims, start)) != std::string::npos) {
    result.push_back(str.substr(start, end - start));
    start = end + 1;
  }
  result.push_back(str.substr(start));
  return result;
}

VideoDecoder::ColorConversionLibrary getDefaultColorConversionLibrary(
    int width) {
  VideoDecoder::ColorConversionLibrary library =
      VideoDecoder::ColorConversionLibrary::SWSCALE;
  // However, swscale requires widths to be multiples of 32:
  // https://stackoverflow.com/questions/74351955/turn-off-sw-scale-conversion-to-planar-yuv-32-byte-alignment-requirements
  // so we fall back to filtergraph if the width is not a multiple of 32.
  if (width % 32 != 0) {
    library = VideoDecoder::ColorConversionLibrary::FILTERGRAPH;
  }
  return library;
}

} // namespace

// Returns a [N]CHW *view* of a [N]HWC input tensor, if the options require so.
// The [N] leading batch-dimension is optional i.e. the input tensor can be 3D
// or 4D.
// Calling permute() is guaranteed to return a view as per the docs:
// https://pytorch.org/docs/stable/generated/torch.permute.html
torch::Tensor VideoDecoder::maybePermuteHWC2CHW(
    int streamIndex,
    torch::Tensor& hwcTensor) {
  if (streamInfos_[streamIndex].videoStreamOptions.dimensionOrder == "NHWC") {
    return hwcTensor;
  }
  auto numDimensions = hwcTensor.dim();
  auto shape = hwcTensor.sizes();
  if (numDimensions == 3) {
    TORCH_CHECK(shape[2] == 3, "Not a HWC tensor: ", shape);
    return hwcTensor.permute({2, 0, 1});
  } else if (numDimensions == 4) {
    TORCH_CHECK(shape[3] == 3, "Not a NHWC tensor: ", shape);
    return hwcTensor.permute({0, 3, 1, 2});
  } else {
    TORCH_CHECK(
        false, "Expected tensor with 3 or 4 dimensions, got ", numDimensions);
  }
}

VideoDecoder::VideoStreamOptions::VideoStreamOptions(
    const std::string& optionsString) {
  std::vector<std::string> tokens =
      splitStringWithDelimiters(optionsString, ",");
  for (auto token : tokens) {
    std::vector<std::string> pairs = splitStringWithDelimiters(token, "=");
    if (pairs.size() != 2) {
      throw std::runtime_error(
          "Invalid option: " + token +
          ". Options must be in the form 'option=value'.");
    }
    std::string key = pairs[0];
    std::string value = pairs[1];
    if (key == "ffmpeg_thread_count") {
      ffmpegThreadCount = std::stoi(value);
      if (ffmpegThreadCount < 0) {
        throw std::runtime_error(
            "Invalid ffmpeg_thread_count=" + value +
            ". ffmpeg_thread_count must be >= 0.");
      }
    } else if (key == "dimension_order") {
      if (value != "NHWC" && value != "NCHW") {
        throw std::runtime_error(
            "Invalid dimension_order=" + value +
            ". dimensionOrder must be either NHWC or NCHW.");
      }
      dimensionOrder = value;
    } else if (key == "width") {
      width = std::stoi(value);
    } else if (key == "height") {
      height = std::stoi(value);
    } else if (key == "color_conversion_library") {
      if (value == "filtergraph") {
        colorConversionLibrary = ColorConversionLibrary::FILTERGRAPH;
      } else if (value == "swscale") {
        colorConversionLibrary = ColorConversionLibrary::SWSCALE;
      } else {
        throw std::runtime_error(
            "Invalid color_conversion_library=" + value +
            ". color_conversion_library must be either "
            "filtergraph or swscale.");
      }
    } else {
      throw std::runtime_error(
          "Invalid option: " + key +
          ". Valid options are: "
          "ffmpeg_thread_count=<int>,dimension_order=<string>");
    }
  }
}

VideoDecoder::FrameBatchOutput::FrameBatchOutput(
    int64_t numFrames,
    const VideoStreamOptions& videoStreamOptions,
    const StreamMetadata& streamMetadata)
    : ptsSeconds(torch::empty({numFrames}, {torch::kFloat64})),
      durationSeconds(torch::empty({numFrames}, {torch::kFloat64})) {
  auto frameDims = getHeightAndWidthFromOptionsOrMetadata(
      videoStreamOptions, streamMetadata);
  int height = frameDims.height;
  int width = frameDims.width;
  data = allocateEmptyHWCTensor(
      height, width, videoStreamOptions.device, numFrames);
}

bool VideoDecoder::DecodedFrameContext::operator==(
    const VideoDecoder::DecodedFrameContext& other) {
  return decodedWidth == other.decodedWidth &&
      decodedHeight == other.decodedHeight &&
      decodedFormat == other.decodedFormat &&
      expectedWidth == other.expectedWidth &&
      expectedHeight == other.expectedHeight;
}

bool VideoDecoder::DecodedFrameContext::operator!=(
    const VideoDecoder::DecodedFrameContext& other) {
  return !(*this == other);
}

VideoDecoder::VideoDecoder(const std::string& videoFilePath, SeekMode seekMode)
    : seekMode_(seekMode) {
  AVInput input = createAVFormatContextFromFilePath(videoFilePath);
  formatContext_ = std::move(input.formatContext);

  initializeDecoder();
}

VideoDecoder::VideoDecoder(const void* buffer, size_t length, SeekMode seekMode)
    : seekMode_(seekMode) {
  TORCH_CHECK(buffer != nullptr, "Video buffer cannot be nullptr!");

  AVInput input = createAVFormatContextFromBuffer(buffer, length);
  formatContext_ = std::move(input.formatContext);
  ioBytesContext_ = std::move(input.ioBytesContext);

  initializeDecoder();
}

void VideoDecoder::initializeDecoder() {
  TORCH_CHECK(!initialized_, "Attempted double initialization.");

  // In principle, the AVFormatContext should be filled in by the call to
  // avformat_open_input() which reads the header. However, some formats do not
  // store enough info in the header, so we call avformat_find_stream_info()
  // which decodes a few frames to get missing info. For more, see:
  //   https://ffmpeg.org/doxygen/7.0/group__lavf__decoding.html
  int ffmpegStatus = avformat_find_stream_info(formatContext_.get(), nullptr);
  if (ffmpegStatus < 0) {
    throw std::runtime_error(
        "Failed to find stream info: " +
        getFFMPEGErrorStringFromErrorCode(ffmpegStatus));
  }

  for (unsigned int i = 0; i < formatContext_->nb_streams; i++) {
    AVStream* avStream = formatContext_->streams[i];
    StreamMetadata streamMetadata;

    TORCH_CHECK(
        static_cast<int>(i) == avStream->index,
        "Our stream index, " + std::to_string(i) +
            ", does not match AVStream's index, " +
            std::to_string(avStream->index) + ".");
    streamMetadata.streamIndex = i;
    streamMetadata.mediaType = avStream->codecpar->codec_type;
    streamMetadata.codecName = avcodec_get_name(avStream->codecpar->codec_id);
    streamMetadata.bitRate = avStream->codecpar->bit_rate;

    int64_t frameCount = avStream->nb_frames;
    if (frameCount > 0) {
      streamMetadata.numFrames = frameCount;
    }

    if (avStream->duration > 0 && avStream->time_base.den > 0) {
      streamMetadata.durationSeconds =
          av_q2d(avStream->time_base) * avStream->duration;
    }

    double fps = av_q2d(avStream->r_frame_rate);
    if (fps > 0) {
      streamMetadata.averageFps = fps;
    }

    if (avStream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      containerMetadata_.numVideoStreams++;
    } else if (avStream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
      containerMetadata_.numAudioStreams++;
    }

    containerMetadata_.allStreamMetadata.push_back(streamMetadata);
  }

  if (formatContext_->duration > 0) {
    containerMetadata_.durationSeconds =
        ptsToSeconds(formatContext_->duration, AV_TIME_BASE);
  }

  if (formatContext_->bit_rate > 0) {
    containerMetadata_.bitRate = formatContext_->bit_rate;
  }

  int bestVideoStream = getBestStreamIndex(AVMEDIA_TYPE_VIDEO);
  if (bestVideoStream >= 0) {
    containerMetadata_.bestVideoStreamIndex = bestVideoStream;
  }

  int bestAudioStream = getBestStreamIndex(AVMEDIA_TYPE_AUDIO);
  if (bestAudioStream >= 0) {
    containerMetadata_.bestAudioStreamIndex = bestAudioStream;
  }

  if (seekMode_ == SeekMode::exact) {
    scanFileAndUpdateMetadataAndIndex();
  }

  initialized_ = true;
}

void VideoDecoder::createFilterGraph(
    StreamInfo& streamInfo,
    int expectedOutputHeight,
    int expectedOutputWidth) {
  FilterGraphContext& filterGraphContext = streamInfo.filterGraphContext;
  filterGraphContext.filterGraph.reset(avfilter_graph_alloc());
  TORCH_CHECK(filterGraphContext.filterGraph.get() != nullptr);

  if (streamInfo.videoStreamOptions.ffmpegThreadCount.has_value()) {
    filterGraphContext.filterGraph->nb_threads =
        streamInfo.videoStreamOptions.ffmpegThreadCount.value();
  }

  const AVFilter* buffersrc = avfilter_get_by_name("buffer");
  const AVFilter* buffersink = avfilter_get_by_name("buffersink");
  AVCodecContext* codecContext = streamInfo.codecContext.get();

  std::stringstream filterArgs;
  filterArgs << "video_size=" << codecContext->width << "x"
             << codecContext->height;
  filterArgs << ":pix_fmt=" << codecContext->pix_fmt;
  filterArgs << ":time_base=" << streamInfo.stream->time_base.num << "/"
             << streamInfo.stream->time_base.den;
  filterArgs << ":pixel_aspect=" << codecContext->sample_aspect_ratio.num << "/"
             << codecContext->sample_aspect_ratio.den;

  int ffmpegStatus = avfilter_graph_create_filter(
      &filterGraphContext.sourceContext,
      buffersrc,
      "in",
      filterArgs.str().c_str(),
      nullptr,
      filterGraphContext.filterGraph.get());
  if (ffmpegStatus < 0) {
    throw std::runtime_error(
        std::string("Failed to create filter graph: ") + filterArgs.str() +
        ": " + getFFMPEGErrorStringFromErrorCode(ffmpegStatus));
  }

  ffmpegStatus = avfilter_graph_create_filter(
      &filterGraphContext.sinkContext,
      buffersink,
      "out",
      nullptr,
      nullptr,
      filterGraphContext.filterGraph.get());
  if (ffmpegStatus < 0) {
    throw std::runtime_error(
        "Failed to create filter graph: " +
        getFFMPEGErrorStringFromErrorCode(ffmpegStatus));
  }

  enum AVPixelFormat pix_fmts[] = {AV_PIX_FMT_RGB24, AV_PIX_FMT_NONE};

  ffmpegStatus = av_opt_set_int_list(
      filterGraphContext.sinkContext,
      "pix_fmts",
      pix_fmts,
      AV_PIX_FMT_NONE,
      AV_OPT_SEARCH_CHILDREN);
  if (ffmpegStatus < 0) {
    throw std::runtime_error(
        "Failed to set output pixel formats: " +
        getFFMPEGErrorStringFromErrorCode(ffmpegStatus));
  }

  UniqueAVFilterInOut outputs(avfilter_inout_alloc());
  UniqueAVFilterInOut inputs(avfilter_inout_alloc());

  outputs->name = av_strdup("in");
  outputs->filter_ctx = filterGraphContext.sourceContext;
  outputs->pad_idx = 0;
  outputs->next = nullptr;
  inputs->name = av_strdup("out");
  inputs->filter_ctx = filterGraphContext.sinkContext;
  inputs->pad_idx = 0;
  inputs->next = nullptr;

  std::stringstream description;
  description << "scale=" << expectedOutputWidth << ":" << expectedOutputHeight;
  description << ":sws_flags=bilinear";

  AVFilterInOut* outputsTmp = outputs.release();
  AVFilterInOut* inputsTmp = inputs.release();
  ffmpegStatus = avfilter_graph_parse_ptr(
      filterGraphContext.filterGraph.get(),
      description.str().c_str(),
      &inputsTmp,
      &outputsTmp,
      nullptr);
  outputs.reset(outputsTmp);
  inputs.reset(inputsTmp);
  if (ffmpegStatus < 0) {
    throw std::runtime_error(
        "Failed to parse filter description: " +
        getFFMPEGErrorStringFromErrorCode(ffmpegStatus));
  }

  ffmpegStatus =
      avfilter_graph_config(filterGraphContext.filterGraph.get(), nullptr);
  if (ffmpegStatus < 0) {
    throw std::runtime_error(
        "Failed to configure filter graph: " +
        getFFMPEGErrorStringFromErrorCode(ffmpegStatus));
  }
}

int VideoDecoder::getBestStreamIndex(AVMediaType mediaType) {
  AVCodecOnlyUseForCallingAVFindBestStream avCodec = nullptr;
  int streamIndex =
      av_find_best_stream(formatContext_.get(), mediaType, -1, -1, &avCodec, 0);
  return streamIndex;
}

void VideoDecoder::addVideoStreamDecoder(
    int preferredStreamIndex,
    const VideoStreamOptions& videoStreamOptions) {
  TORCH_CHECK(
      activeStreamIndex_ == NO_ACTIVE_STREAM,
      "Can only add one single stream.");
  TORCH_CHECK(formatContext_.get() != nullptr);

  AVCodecOnlyUseForCallingAVFindBestStream avCodec = nullptr;
  int streamIndex = av_find_best_stream(
      formatContext_.get(),
      AVMEDIA_TYPE_VIDEO,
      preferredStreamIndex,
      -1,
      &avCodec,
      0);
  if (streamIndex < 0) {
    throw std::invalid_argument("No valid stream found in input file.");
  }
  TORCH_CHECK(avCodec != nullptr);

  StreamInfo& streamInfo = streamInfos_[streamIndex];
  streamInfo.streamIndex = streamIndex;
  streamInfo.timeBase = formatContext_->streams[streamIndex]->time_base;
  streamInfo.stream = formatContext_->streams[streamIndex];

  if (streamInfo.stream->codecpar->codec_type != AVMEDIA_TYPE_VIDEO) {
    throw std::invalid_argument(
        "Stream with index " + std::to_string(streamIndex) +
        " is not a video stream.");
  }

  if (videoStreamOptions.device.type() == torch::kCUDA) {
    avCodec = makeAVCodecOnlyUseForCallingAVFindBestStream(
        findCudaCodec(
            videoStreamOptions.device, streamInfo.stream->codecpar->codec_id)
            .value_or(avCodec));
  }

  StreamMetadata& streamMetadata =
      containerMetadata_.allStreamMetadata[streamIndex];
  if (seekMode_ == SeekMode::approximate &&
      !streamMetadata.averageFps.has_value()) {
    throw std::runtime_error(
        "Seek mode is approximate, but stream " + std::to_string(streamIndex) +
        " does not have an average fps in its metadata.");
  }

  AVCodecContext* codecContext = avcodec_alloc_context3(avCodec);
  TORCH_CHECK(codecContext != nullptr);
  codecContext->thread_count = videoStreamOptions.ffmpegThreadCount.value_or(0);
  streamInfo.codecContext.reset(codecContext);

  int retVal = avcodec_parameters_to_context(
      streamInfo.codecContext.get(), streamInfo.stream->codecpar);
  TORCH_CHECK_EQ(retVal, AVSUCCESS);

  if (videoStreamOptions.device.type() == torch::kCPU) {
    // No more initialization needed for CPU.
  } else if (videoStreamOptions.device.type() == torch::kCUDA) {
    initializeContextOnCuda(videoStreamOptions.device, codecContext);
  } else {
    TORCH_CHECK(
        false, "Invalid device type: " + videoStreamOptions.device.str());
  }

  retVal = avcodec_open2(streamInfo.codecContext.get(), avCodec, nullptr);
  if (retVal < AVSUCCESS) {
    throw std::invalid_argument(getFFMPEGErrorStringFromErrorCode(retVal));
  }

  codecContext->time_base = streamInfo.stream->time_base;
  activeStreamIndex_ = streamIndex;
  updateMetadataWithCodecContext(streamInfo.streamIndex, codecContext);
  streamInfo.videoStreamOptions = videoStreamOptions;

  // By default, we want to use swscale for color conversion because it is
  // faster. However, it has width requirements, so we may need to fall back
  // to filtergraph. We also need to respect what was requested from the
  // options; we respect the options unconditionally, so it's possible for
  // swscale's width requirements to be violated. We don't expose the ability to
  // choose color conversion library publicly; we only use this ability
  // internally.
  int width = videoStreamOptions.width.value_or(codecContext->width);
  auto defaultLibrary = getDefaultColorConversionLibrary(width);
  streamInfo.colorConversionLibrary =
      videoStreamOptions.colorConversionLibrary.value_or(defaultLibrary);
}

void VideoDecoder::updateMetadataWithCodecContext(
    int streamIndex,
    AVCodecContext* codecContext) {
  containerMetadata_.allStreamMetadata[streamIndex].width = codecContext->width;
  containerMetadata_.allStreamMetadata[streamIndex].height =
      codecContext->height;
  auto codedId = codecContext->codec_id;
  containerMetadata_.allStreamMetadata[streamIndex].codecName =
      std::string(avcodec_get_name(codedId));
}

VideoDecoder::ContainerMetadata VideoDecoder::getContainerMetadata() const {
  return containerMetadata_;
}

torch::Tensor VideoDecoder::getKeyFrameIndices(int streamIndex) {
  validateUserProvidedStreamIndex(streamIndex);
  validateScannedAllStreams("getKeyFrameIndices");

  const std::vector<FrameInfo>& keyFrames = streamInfos_[streamIndex].keyFrames;
  torch::Tensor keyFrameIndices =
      torch::empty({static_cast<int64_t>(keyFrames.size())}, {torch::kInt64});
  for (size_t i = 0; i < keyFrames.size(); ++i) {
    keyFrameIndices[i] = keyFrames[i].frameIndex;
  }

  return keyFrameIndices;
}

int VideoDecoder::getKeyFrameIndexForPtsUsingScannedIndex(
    const std::vector<VideoDecoder::FrameInfo>& keyFrames,
    int64_t pts) const {
  auto upperBound = std::upper_bound(
      keyFrames.begin(),
      keyFrames.end(),
      pts,
      [](int64_t pts, const VideoDecoder::FrameInfo& frameInfo) {
        return pts < frameInfo.pts;
      });
  if (upperBound == keyFrames.begin()) {
    return -1;
  }
  return upperBound - 1 - keyFrames.begin();
}

void VideoDecoder::scanFileAndUpdateMetadataAndIndex() {
  if (scannedAllStreams_) {
    return;
  }

  AutoAVPacket autoAVPacket;
  while (true) {
    ReferenceAVPacket packet(autoAVPacket);

    // av_read_frame is a misleading name: it gets the next **packet**.
    int ffmpegStatus = av_read_frame(formatContext_.get(), packet.get());

    if (ffmpegStatus == AVERROR_EOF) {
      break;
    }

    if (ffmpegStatus != AVSUCCESS) {
      throw std::runtime_error(
          "Failed to read frame from input file: " +
          getFFMPEGErrorStringFromErrorCode(ffmpegStatus));
    }

    if (packet->flags & AV_PKT_FLAG_DISCARD) {
      continue;
    }

    // We got a valid packet. Let's figure out what stream it belongs to and
    // record its relevant metadata.
    int streamIndex = packet->stream_index;
    auto& streamMetadata = containerMetadata_.allStreamMetadata[streamIndex];
    streamMetadata.minPtsFromScan = std::min(
        streamMetadata.minPtsFromScan.value_or(INT64_MAX), packet->pts);
    streamMetadata.maxPtsFromScan = std::max(
        streamMetadata.maxPtsFromScan.value_or(INT64_MIN),
        packet->pts + packet->duration);
    streamMetadata.numFramesFromScan =
        streamMetadata.numFramesFromScan.value_or(0) + 1;

    // Note that we set the other value in this struct, nextPts, only after
    // we have scanned all packets and sorted by pts.
    FrameInfo frameInfo = {packet->pts};
    if (packet->flags & AV_PKT_FLAG_KEY) {
      frameInfo.isKeyFrame = true;
      streamInfos_[streamIndex].keyFrames.push_back(frameInfo);
    }
    streamInfos_[streamIndex].allFrames.push_back(frameInfo);
  }

  // Set all per-stream metadata that requires knowing the content of all
  // packets.
  for (size_t streamIndex = 0;
       streamIndex < containerMetadata_.allStreamMetadata.size();
       ++streamIndex) {
    auto& streamMetadata = containerMetadata_.allStreamMetadata[streamIndex];
    auto avStream = formatContext_->streams[streamIndex];

    streamMetadata.numFramesFromScan =
        streamInfos_[streamIndex].allFrames.size();

    if (streamMetadata.minPtsFromScan.has_value()) {
      streamMetadata.minPtsSecondsFromScan =
          *streamMetadata.minPtsFromScan * av_q2d(avStream->time_base);
    }
    if (streamMetadata.maxPtsFromScan.has_value()) {
      streamMetadata.maxPtsSecondsFromScan =
          *streamMetadata.maxPtsFromScan * av_q2d(avStream->time_base);
    }
  }

  // Reset the seek-cursor back to the beginning.
  int ffmepgStatus =
      avformat_seek_file(formatContext_.get(), 0, INT64_MIN, 0, 0, 0);
  if (ffmepgStatus < 0) {
    throw std::runtime_error(
        "Could not seek file to pts=0: " +
        getFFMPEGErrorStringFromErrorCode(ffmepgStatus));
  }

  // Sort all frames by their pts.
  for (auto& [streamIndex, streamInfo] : streamInfos_) {
    std::sort(
        streamInfo.keyFrames.begin(),
        streamInfo.keyFrames.end(),
        [](const FrameInfo& frameInfo1, const FrameInfo& frameInfo2) {
          return frameInfo1.pts < frameInfo2.pts;
        });
    std::sort(
        streamInfo.allFrames.begin(),
        streamInfo.allFrames.end(),
        [](const FrameInfo& frameInfo1, const FrameInfo& frameInfo2) {
          return frameInfo1.pts < frameInfo2.pts;
        });

    size_t keyFrameIndex = 0;
    for (size_t i = 0; i < streamInfo.allFrames.size(); ++i) {
      streamInfo.allFrames[i].frameIndex = i;
      if (streamInfo.allFrames[i].isKeyFrame) {
        TORCH_CHECK(
            keyFrameIndex < streamInfo.keyFrames.size(),
            "The allFrames vec claims it has MORE keyFrames than the keyFrames vec. There's a bug in torchcodec.");
        streamInfo.keyFrames[keyFrameIndex].frameIndex = i;
        ++keyFrameIndex;
      }
      if (i + 1 < streamInfo.allFrames.size()) {
        streamInfo.allFrames[i].nextPts = streamInfo.allFrames[i + 1].pts;
      }
    }
    TORCH_CHECK(
        keyFrameIndex == streamInfo.keyFrames.size(),
        "The allFrames vec claims it has LESS keyFrames than the keyFrames vec. There's a bug in torchcodec.");
  }

  scannedAllStreams_ = true;
}

int VideoDecoder::getKeyFrameIndexForPts(
    const StreamInfo& streamInfo,
    int64_t pts) const {
  if (streamInfo.keyFrames.empty()) {
    return av_index_search_timestamp(
        streamInfo.stream, pts, AVSEEK_FLAG_BACKWARD);
  } else {
    return getKeyFrameIndexForPtsUsingScannedIndex(streamInfo.keyFrames, pts);
  }
}

/*
Videos have I frames and non-I frames (P and B frames). Non-I frames need data
from the previous I frame to be decoded.

Imagine the cursor is at a random frame with PTS=x and we wish to seek to a
user-specified PTS=y.

If y < x, we don't have a choice but to seek backwards to the highest I frame
before y.

If y > x, we have two choices:

1. We could keep decoding forward until we hit y. Illustrated below:

I    P     P    P    I    P    P    P    I    P    P    I    P    P    I    P
                          x         y

2. We could try to jump to an I frame between x and y (indicated by j below).
And then start decoding until we encounter y. Illustrated below:

I    P     P    P    I    P    P    P    I    P    P    I    P    P    I    P
                          x              j         y

(2) is more efficient than (1) if there is an I frame between x and y.
*/
bool VideoDecoder::canWeAvoidSeekingForStream(
    const StreamInfo& streamInfo,
    int64_t currentPts,
    int64_t targetPts) const {
  if (targetPts < currentPts) {
    // We can never skip a seek if we are seeking backwards.
    return false;
  }
  if (currentPts == targetPts) {
    // We are seeking to the exact same frame as we are currently at. Without
    // caching we have to rewind back and decode the frame again.
    // TODO: https://github.com/pytorch-labs/torchcodec/issues/84 we could
    // implement caching.
    return false;
  }
  // We are seeking forwards.
  // We can only skip a seek if both currentPts and targetPts share the same
  // keyframe.
  int currentKeyFrameIndex = getKeyFrameIndexForPts(streamInfo, currentPts);
  int targetKeyFrameIndex = getKeyFrameIndexForPts(streamInfo, targetPts);
  return currentKeyFrameIndex >= 0 && targetKeyFrameIndex >= 0 &&
      currentKeyFrameIndex == targetKeyFrameIndex;
}

// This method looks at currentPts and desiredPts and seeks in the
// AVFormatContext if it is needed. We can skip seeking in certain cases. See
// the comment of canWeAvoidSeeking() for details.
void VideoDecoder::maybeSeekToBeforeDesiredPts() {
  if (activeStreamIndex_ == NO_ACTIVE_STREAM) {
    return;
  }
  StreamInfo& streamInfo = streamInfos_[activeStreamIndex_];
  streamInfo.discardFramesBeforePts =
      secondsToClosestPts(*desiredPtsSeconds_, streamInfo.timeBase);

  decodeStats_.numSeeksAttempted++;

  int64_t desiredPtsForStream = *desiredPtsSeconds_ * streamInfo.timeBase.den;
  if (canWeAvoidSeekingForStream(
          streamInfo, streamInfo.currentPts, desiredPtsForStream)) {
    decodeStats_.numSeeksSkipped++;
    return;
  }
  int64_t desiredPts =
      secondsToClosestPts(*desiredPtsSeconds_, streamInfo.timeBase);

  // For some encodings like H265, FFMPEG sometimes seeks past the point we
  // set as the max_ts. So we use our own index to give it the exact pts of
  // the key frame that we want to seek to.
  // See https://github.com/pytorch/torchcodec/issues/179 for more details.
  // See https://trac.ffmpeg.org/ticket/11137 for the underlying ffmpeg bug.
  if (!streamInfo.keyFrames.empty()) {
    int desiredKeyFrameIndex = getKeyFrameIndexForPtsUsingScannedIndex(
        streamInfo.keyFrames, desiredPts);
    desiredKeyFrameIndex = std::max(desiredKeyFrameIndex, 0);
    desiredPts = streamInfo.keyFrames[desiredKeyFrameIndex].pts;
  }

  int ffmepgStatus = avformat_seek_file(
      formatContext_.get(),
      streamInfo.streamIndex,
      INT64_MIN,
      desiredPts,
      desiredPts,
      0);
  if (ffmepgStatus < 0) {
    throw std::runtime_error(
        "Could not seek file to pts=" + std::to_string(desiredPts) + ": " +
        getFFMPEGErrorStringFromErrorCode(ffmepgStatus));
  }
  decodeStats_.numFlushes++;
  avcodec_flush_buffers(streamInfo.codecContext.get());
}

VideoDecoder::AVFrameStream VideoDecoder::decodeAVFrame(
    std::function<bool(AVFrame*)> filterFunction) {
  if (activeStreamIndex_ == NO_ACTIVE_STREAM) {
    throw std::runtime_error("No active streams configured.");
  }

  resetDecodeStats();

  // Seek if needed.
  if (desiredPtsSeconds_.has_value()) {
    maybeSeekToBeforeDesiredPts();
    desiredPtsSeconds_ = std::nullopt;
  }

  StreamInfo& streamInfo = streamInfos_[activeStreamIndex_];

  // Need to get the next frame or error from PopFrame.
  UniqueAVFrame avFrame(av_frame_alloc());
  AutoAVPacket autoAVPacket;
  int ffmpegStatus = AVSUCCESS;
  bool reachedEOF = false;
  while (true) {
    ffmpegStatus =
        avcodec_receive_frame(streamInfo.codecContext.get(), avFrame.get());

    if (ffmpegStatus != AVSUCCESS && ffmpegStatus != AVERROR(EAGAIN)) {
      // Non-retriable error
      break;
    }

    decodeStats_.numFramesReceivedByDecoder++;
    // Is this the kind of frame we're looking for?
    if (ffmpegStatus == AVSUCCESS && filterFunction(avFrame.get())) {
      // Yes, this is the frame we'll return; break out of the decoding loop.
      break;
    } else if (ffmpegStatus == AVSUCCESS) {
      // No, but we received a valid frame - just not the kind we're looking
      // for. The logic below will read packets and send them to the decoder.
      // But since we did just receive a frame, we should skip reading more
      // packets and sending them to the decoder and just try to receive more
      // frames from the decoder.
      continue;
    }

    if (reachedEOF) {
      // We don't have any more packets to send to the decoder. So keep on
      // pulling frames from its internal buffers.
      continue;
    }

    // We still haven't found the frame we're looking for. So let's read more
    // packets and send them to the decoder.
    ReferenceAVPacket packet(autoAVPacket);
    ffmpegStatus = av_read_frame(formatContext_.get(), packet.get());
    decodeStats_.numPacketsRead++;

    if (ffmpegStatus == AVERROR_EOF) {
      // End of file reached. We must drain the codec by sending a nullptr
      // packet.
      ffmpegStatus = avcodec_send_packet(
          streamInfo.codecContext.get(),
          /*avpkt=*/nullptr);
      if (ffmpegStatus < AVSUCCESS) {
        throw std::runtime_error(
            "Could not flush decoder: " +
            getFFMPEGErrorStringFromErrorCode(ffmpegStatus));
      }

      // We've reached the end of file so we can't read any more packets from
      // it, but the decoder may still have frames to read in its buffer.
      // Continue iterating to try reading frames.
      reachedEOF = true;
      continue;
    }

    if (ffmpegStatus < AVSUCCESS) {
      throw std::runtime_error(
          "Could not read frame from input file: " +
          getFFMPEGErrorStringFromErrorCode(ffmpegStatus));
    }

    if (packet->stream_index != activeStreamIndex_) {
      continue;
    }

    // We got a valid packet. Send it to the decoder, and we'll receive it in
    // the next iteration.
    ffmpegStatus =
        avcodec_send_packet(streamInfo.codecContext.get(), packet.get());
    if (ffmpegStatus < AVSUCCESS) {
      throw std::runtime_error(
          "Could not push packet to decoder: " +
          getFFMPEGErrorStringFromErrorCode(ffmpegStatus));
    }

    decodeStats_.numPacketsSentToDecoder++;
  }

  if (ffmpegStatus < AVSUCCESS) {
    if (reachedEOF || ffmpegStatus == AVERROR_EOF) {
      throw VideoDecoder::EndOfFileException(
          "Requested next frame while there are no more frames left to "
          "decode.");
    }
    throw std::runtime_error(
        "Could not receive frame from decoder: " +
        getFFMPEGErrorStringFromErrorCode(ffmpegStatus));
  }

  // Note that we don't flush the decoder when we reach EOF (even though that's
  // mentioned in https://ffmpeg.org/doxygen/trunk/group__lavc__encdec.html).
  // This is because we may have packets internally in the decoder that we
  // haven't received as frames. Eventually we will either hit AVERROR_EOF from
  // av_receive_frame() or the user will have seeked to a different location in
  // the file and that will flush the decoder.
  streamInfo.currentPts = avFrame->pts;
  streamInfo.currentDuration = getDuration(avFrame);

  return AVFrameStream(std::move(avFrame), activeStreamIndex_);
}

VideoDecoder::FrameOutput VideoDecoder::convertAVFrameToFrameOutput(
    VideoDecoder::AVFrameStream& avFrameStream,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  // Convert the frame to tensor.
  FrameOutput frameOutput;
  int streamIndex = avFrameStream.streamIndex;
  AVFrame* avFrame = avFrameStream.avFrame.get();
  frameOutput.streamIndex = streamIndex;
  auto& streamInfo = streamInfos_[streamIndex];
  TORCH_CHECK(streamInfo.stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO);
  frameOutput.ptsSeconds = ptsToSeconds(
      avFrame->pts, formatContext_->streams[streamIndex]->time_base);
  frameOutput.durationSeconds = ptsToSeconds(
      getDuration(avFrame), formatContext_->streams[streamIndex]->time_base);
  // TODO: we should fold preAllocatedOutputTensor into AVFrameStream.
  if (streamInfo.videoStreamOptions.device.type() == torch::kCPU) {
    convertAVFrameToFrameOutputOnCPU(
        avFrameStream, frameOutput, preAllocatedOutputTensor);
  } else if (streamInfo.videoStreamOptions.device.type() == torch::kCUDA) {
    convertAVFrameToFrameOutputOnCuda(
        streamInfo.videoStreamOptions.device,
        streamInfo.videoStreamOptions,
        avFrameStream,
        frameOutput,
        preAllocatedOutputTensor);
  } else {
    TORCH_CHECK(
        false,
        "Invalid device type: " + streamInfo.videoStreamOptions.device.str());
  }
  return frameOutput;
}

// Note [preAllocatedOutputTensor with swscale and filtergraph]:
// Callers may pass a pre-allocated tensor, where the output.data tensor will
// be stored. This parameter is honored in any case, but it only leads to a
// speed-up when swscale is used. With swscale, we can tell ffmpeg to place the
// decoded frame directly into `preAllocatedtensor.data_ptr()`. We haven't yet
// found a way to do that with filtegraph.
// TODO: Figure out whether that's possible!
// Dimension order of the preAllocatedOutputTensor must be HWC, regardless of
// `dimension_order` parameter. It's up to callers to re-shape it if needed.
void VideoDecoder::convertAVFrameToFrameOutputOnCPU(
    VideoDecoder::AVFrameStream& avFrameStream,
    FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  int streamIndex = avFrameStream.streamIndex;
  AVFrame* avFrame = avFrameStream.avFrame.get();
  auto& streamInfo = streamInfos_[streamIndex];

  auto frameDims = getHeightAndWidthFromOptionsOrAVFrame(
      streamInfo.videoStreamOptions, *avFrame);
  int expectedOutputHeight = frameDims.height;
  int expectedOutputWidth = frameDims.width;

  if (preAllocatedOutputTensor.has_value()) {
    auto shape = preAllocatedOutputTensor.value().sizes();
    TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == expectedOutputHeight) &&
            (shape[1] == expectedOutputWidth) && (shape[2] == 3),
        "Expected pre-allocated tensor of shape ",
        expectedOutputHeight,
        "x",
        expectedOutputWidth,
        "x3, got ",
        shape);
  }

  torch::Tensor outputTensor;
  // We need to compare the current frame context with our previous frame
  // context. If they are different, then we need to re-create our colorspace
  // conversion objects. We create our colorspace conversion objects late so
  // that we don't have to depend on the unreliable metadata in the header.
  // And we sometimes re-create them because it's possible for frame
  // resolution to change mid-stream. Finally, we want to reuse the colorspace
  // conversion objects as much as possible for performance reasons.
  enum AVPixelFormat frameFormat =
      static_cast<enum AVPixelFormat>(avFrame->format);
  auto frameContext = DecodedFrameContext{
      avFrame->width,
      avFrame->height,
      frameFormat,
      expectedOutputWidth,
      expectedOutputHeight};

  if (streamInfo.colorConversionLibrary == ColorConversionLibrary::SWSCALE) {
    outputTensor = preAllocatedOutputTensor.value_or(allocateEmptyHWCTensor(
        expectedOutputHeight, expectedOutputWidth, torch::kCPU));

    if (!streamInfo.swsContext || streamInfo.prevFrameContext != frameContext) {
      createSwsContext(streamInfo, frameContext, avFrame->colorspace);
      streamInfo.prevFrameContext = frameContext;
    }
    int resultHeight =
        convertAVFrameToTensorUsingSwsScale(streamIndex, avFrame, outputTensor);
    // If this check failed, it would mean that the frame wasn't reshaped to
    // the expected height.
    // TODO: Can we do the same check for width?
    TORCH_CHECK(
        resultHeight == expectedOutputHeight,
        "resultHeight != expectedOutputHeight: ",
        resultHeight,
        " != ",
        expectedOutputHeight);

    frameOutput.data = outputTensor;
  } else if (
      streamInfo.colorConversionLibrary ==
      ColorConversionLibrary::FILTERGRAPH) {
    if (!streamInfo.filterGraphContext.filterGraph ||
        streamInfo.prevFrameContext != frameContext) {
      createFilterGraph(streamInfo, expectedOutputHeight, expectedOutputWidth);
      streamInfo.prevFrameContext = frameContext;
    }
    outputTensor = convertAVFrameToTensorUsingFilterGraph(streamIndex, avFrame);

    // Similarly to above, if this check fails it means the frame wasn't
    // reshaped to its expected dimensions by filtergraph.
    auto shape = outputTensor.sizes();
    TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == expectedOutputHeight) &&
            (shape[1] == expectedOutputWidth) && (shape[2] == 3),
        "Expected output tensor of shape ",
        expectedOutputHeight,
        "x",
        expectedOutputWidth,
        "x3, got ",
        shape);

    if (preAllocatedOutputTensor.has_value()) {
      // We have already validated that preAllocatedOutputTensor and
      // outputTensor have the same shape.
      preAllocatedOutputTensor.value().copy_(outputTensor);
      frameOutput.data = preAllocatedOutputTensor.value();
    } else {
      frameOutput.data = outputTensor;
    }
  } else {
    throw std::runtime_error(
        "Invalid color conversion library: " +
        std::to_string(static_cast<int>(streamInfo.colorConversionLibrary)));
  }
}

VideoDecoder::FrameOutput VideoDecoder::getFramePlayedAtNoDemux(
    double seconds) {
  for (auto& [streamIndex, streamInfo] : streamInfos_) {
    double frameStartTime =
        ptsToSeconds(streamInfo.currentPts, streamInfo.timeBase);
    double frameEndTime = ptsToSeconds(
        streamInfo.currentPts + streamInfo.currentDuration,
        streamInfo.timeBase);
    if (seconds >= frameStartTime && seconds < frameEndTime) {
      // We are in the same frame as the one we just returned. However, since we
      // don't cache it locally, we have to rewind back.
      seconds = frameStartTime;
      break;
    }
  }

  setCursorPtsInSeconds(seconds);
  AVFrameStream avFrameStream =
      decodeAVFrame([seconds, this](AVFrame* avFrame) {
        StreamInfo& streamInfo = streamInfos_[activeStreamIndex_];
        double frameStartTime = ptsToSeconds(avFrame->pts, streamInfo.timeBase);
        double frameEndTime = ptsToSeconds(
            avFrame->pts + getDuration(avFrame), streamInfo.timeBase);
        if (frameStartTime > seconds) {
          // FFMPEG seeked past the frame we are looking for even though we
          // set max_ts to be our needed timestamp in avformat_seek_file()
          // in maybeSeekToBeforeDesiredPts().
          // This could be a bug in FFMPEG: https://trac.ffmpeg.org/ticket/11137
          // In this case we return the very next frame instead of throwing an
          // exception.
          // TODO: Maybe log to stderr for Debug builds?
          return true;
        }
        return seconds >= frameStartTime && seconds < frameEndTime;
      });

  // Convert the frame to tensor.
  FrameOutput frameOutput = convertAVFrameToFrameOutput(avFrameStream);
  frameOutput.data =
      maybePermuteHWC2CHW(frameOutput.streamIndex, frameOutput.data);
  return frameOutput;
}

void VideoDecoder::validateUserProvidedStreamIndex(int streamIndex) {
  int streamsSize =
      static_cast<int>(containerMetadata_.allStreamMetadata.size());
  TORCH_CHECK(
      streamIndex >= 0 && streamIndex < streamsSize,
      "Invalid stream index=" + std::to_string(streamIndex) +
          "; valid indices are in the range [0, " +
          std::to_string(streamsSize) + ").");
  TORCH_CHECK(
      streamInfos_.count(streamIndex) > 0,
      "Provided stream index=" + std::to_string(streamIndex) +
          " was not previously added.");
}

void VideoDecoder::validateScannedAllStreams(const std::string& msg) {
  if (!scannedAllStreams_) {
    throw std::runtime_error(
        "Must scan all streams to update metadata before calling " + msg);
  }
}

void VideoDecoder::validateFrameIndex(
    const StreamMetadata& streamMetadata,
    int64_t frameIndex) {
  int64_t numFrames = getNumFrames(streamMetadata);
  TORCH_CHECK(
      frameIndex >= 0 && frameIndex < numFrames,
      "Invalid frame index=" + std::to_string(frameIndex) +
          " for streamIndex=" + std::to_string(streamMetadata.streamIndex) +
          " numFrames=" + std::to_string(numFrames));
}

VideoDecoder::FrameOutput VideoDecoder::getFrameAtIndex(
    int streamIndex,
    int64_t frameIndex) {
  auto frameOutput = getFrameAtIndexInternal(streamIndex, frameIndex);
  frameOutput.data = maybePermuteHWC2CHW(streamIndex, frameOutput.data);
  return frameOutput;
}

int64_t VideoDecoder::getPts(
    const StreamInfo& streamInfo,
    const StreamMetadata& streamMetadata,
    int64_t frameIndex) {
  switch (seekMode_) {
    case SeekMode::exact:
      return streamInfo.allFrames[frameIndex].pts;
    case SeekMode::approximate:
      return secondsToClosestPts(
          frameIndex / streamMetadata.averageFps.value(), streamInfo.timeBase);
    default:
      throw std::runtime_error("Unknown SeekMode");
  }
}

int64_t VideoDecoder::getNumFrames(const StreamMetadata& streamMetadata) {
  switch (seekMode_) {
    case SeekMode::exact:
      return streamMetadata.numFramesFromScan.value();
    case SeekMode::approximate:
      return streamMetadata.numFrames.value();
    default:
      throw std::runtime_error("Unknown SeekMode");
  }
}

double VideoDecoder::getMinSeconds(const StreamMetadata& streamMetadata) {
  switch (seekMode_) {
    case SeekMode::exact:
      return streamMetadata.minPtsSecondsFromScan.value();
    case SeekMode::approximate:
      return 0;
    default:
      throw std::runtime_error("Unknown SeekMode");
  }
}

double VideoDecoder::getMaxSeconds(const StreamMetadata& streamMetadata) {
  switch (seekMode_) {
    case SeekMode::exact:
      return streamMetadata.maxPtsSecondsFromScan.value();
    case SeekMode::approximate:
      return streamMetadata.durationSeconds.value();
    default:
      throw std::runtime_error("Unknown SeekMode");
  }
}

int64_t VideoDecoder::secondsToIndexLowerBound(
    double seconds,
    const StreamInfo& streamInfo,
    const StreamMetadata& streamMetadata) {
  switch (seekMode_) {
    case SeekMode::exact: {
      auto frame = std::lower_bound(
          streamInfo.allFrames.begin(),
          streamInfo.allFrames.end(),
          seconds,
          [&streamInfo](const FrameInfo& info, double start) {
            return ptsToSeconds(info.nextPts, streamInfo.timeBase) <= start;
          });

      return frame - streamInfo.allFrames.begin();
    }
    case SeekMode::approximate:
      return std::floor(seconds * streamMetadata.averageFps.value());
    default:
      throw std::runtime_error("Unknown SeekMode");
  }
}

int64_t VideoDecoder::secondsToIndexUpperBound(
    double seconds,
    const StreamInfo& streamInfo,
    const StreamMetadata& streamMetadata) {
  switch (seekMode_) {
    case SeekMode::exact: {
      auto frame = std::upper_bound(
          streamInfo.allFrames.begin(),
          streamInfo.allFrames.end(),
          seconds,
          [&streamInfo](double stop, const FrameInfo& info) {
            return stop <= ptsToSeconds(info.pts, streamInfo.timeBase);
          });

      return frame - streamInfo.allFrames.begin();
    }
    case SeekMode::approximate:
      return std::ceil(seconds * streamMetadata.averageFps.value());
    default:
      throw std::runtime_error("Unknown SeekMode");
  }
}

VideoDecoder::FrameOutput VideoDecoder::getFrameAtIndexInternal(
    int streamIndex,
    int64_t frameIndex,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  validateUserProvidedStreamIndex(streamIndex);

  const auto& streamInfo = streamInfos_[streamIndex];
  const auto& streamMetadata =
      containerMetadata_.allStreamMetadata[streamIndex];
  validateFrameIndex(streamMetadata, frameIndex);

  int64_t pts = getPts(streamInfo, streamMetadata, frameIndex);
  setCursorPtsInSeconds(ptsToSeconds(pts, streamInfo.timeBase));
  return getNextFrameNoDemuxInternal(preAllocatedOutputTensor);
}

VideoDecoder::FrameBatchOutput VideoDecoder::getFramesAtIndices(
    int streamIndex,
    const std::vector<int64_t>& frameIndices) {
  validateUserProvidedStreamIndex(streamIndex);

  auto indicesAreSorted =
      std::is_sorted(frameIndices.begin(), frameIndices.end());

  std::vector<size_t> argsort;
  if (!indicesAreSorted) {
    // if frameIndices is [13, 10, 12, 11]
    // when sorted, it's  [10, 11, 12, 13] <-- this is the sorted order we want
    //                                         to use to decode the frames
    // and argsort is     [ 1,  3,  2,  0]
    argsort.resize(frameIndices.size());
    for (size_t i = 0; i < argsort.size(); ++i) {
      argsort[i] = i;
    }
    std::sort(
        argsort.begin(), argsort.end(), [&frameIndices](size_t a, size_t b) {
          return frameIndices[a] < frameIndices[b];
        });
  }

  const auto& streamMetadata =
      containerMetadata_.allStreamMetadata[streamIndex];
  const auto& streamInfo = streamInfos_[streamIndex];
  const auto& videoStreamOptions = streamInfo.videoStreamOptions;
  FrameBatchOutput frameBatchOutput(
      frameIndices.size(), videoStreamOptions, streamMetadata);

  auto previousIndexInVideo = -1;
  for (size_t f = 0; f < frameIndices.size(); ++f) {
    auto indexInOutput = indicesAreSorted ? f : argsort[f];
    auto indexInVideo = frameIndices[indexInOutput];

    validateFrameIndex(streamMetadata, indexInVideo);

    if ((f > 0) && (indexInVideo == previousIndexInVideo)) {
      // Avoid decoding the same frame twice
      auto previousIndexInOutput = indicesAreSorted ? f - 1 : argsort[f - 1];
      frameBatchOutput.data[indexInOutput].copy_(
          frameBatchOutput.data[previousIndexInOutput]);
      frameBatchOutput.ptsSeconds[indexInOutput] =
          frameBatchOutput.ptsSeconds[previousIndexInOutput];
      frameBatchOutput.durationSeconds[indexInOutput] =
          frameBatchOutput.durationSeconds[previousIndexInOutput];
    } else {
      FrameOutput frameOutput = getFrameAtIndexInternal(
          streamIndex, indexInVideo, frameBatchOutput.data[indexInOutput]);
      frameBatchOutput.ptsSeconds[indexInOutput] = frameOutput.ptsSeconds;
      frameBatchOutput.durationSeconds[indexInOutput] =
          frameOutput.durationSeconds;
    }
    previousIndexInVideo = indexInVideo;
  }
  frameBatchOutput.data =
      maybePermuteHWC2CHW(streamIndex, frameBatchOutput.data);
  return frameBatchOutput;
}

VideoDecoder::FrameBatchOutput VideoDecoder::getFramesPlayedAt(
    int streamIndex,
    const std::vector<double>& timestamps) {
  validateUserProvidedStreamIndex(streamIndex);

  const auto& streamMetadata =
      containerMetadata_.allStreamMetadata[streamIndex];
  const auto& streamInfo = streamInfos_[streamIndex];

  double minSeconds = getMinSeconds(streamMetadata);
  double maxSeconds = getMaxSeconds(streamMetadata);

  // The frame played at timestamp t and the one played at timestamp `t +
  // eps` are probably the same frame, with the same index. The easiest way to
  // avoid decoding that unique frame twice is to convert the input timestamps
  // to indices, and leverage the de-duplication logic of getFramesAtIndices.

  std::vector<int64_t> frameIndices(timestamps.size());
  for (size_t i = 0; i < timestamps.size(); ++i) {
    auto frameSeconds = timestamps[i];
    TORCH_CHECK(
        frameSeconds >= minSeconds && frameSeconds < maxSeconds,
        "frame pts is " + std::to_string(frameSeconds) +
            "; must be in range [" + std::to_string(minSeconds) + ", " +
            std::to_string(maxSeconds) + ").");

    frameIndices[i] =
        secondsToIndexLowerBound(frameSeconds, streamInfo, streamMetadata);
  }

  return getFramesAtIndices(streamIndex, frameIndices);
}

VideoDecoder::FrameBatchOutput VideoDecoder::getFramesInRange(
    int streamIndex,
    int64_t start,
    int64_t stop,
    int64_t step) {
  validateUserProvidedStreamIndex(streamIndex);

  const auto& streamMetadata =
      containerMetadata_.allStreamMetadata[streamIndex];
  const auto& streamInfo = streamInfos_[streamIndex];
  int64_t numFrames = getNumFrames(streamMetadata);
  TORCH_CHECK(
      start >= 0, "Range start, " + std::to_string(start) + " is less than 0.");
  TORCH_CHECK(
      stop <= numFrames,
      "Range stop, " + std::to_string(stop) +
          ", is more than the number of frames, " + std::to_string(numFrames));
  TORCH_CHECK(
      step > 0, "Step must be greater than 0; is " + std::to_string(step));

  int64_t numOutputFrames = std::ceil((stop - start) / double(step));
  const auto& videoStreamOptions = streamInfo.videoStreamOptions;
  FrameBatchOutput frameBatchOutput(
      numOutputFrames, videoStreamOptions, streamMetadata);

  for (int64_t i = start, f = 0; i < stop; i += step, ++f) {
    FrameOutput frameOutput =
        getFrameAtIndexInternal(streamIndex, i, frameBatchOutput.data[f]);
    frameBatchOutput.ptsSeconds[f] = frameOutput.ptsSeconds;
    frameBatchOutput.durationSeconds[f] = frameOutput.durationSeconds;
  }
  frameBatchOutput.data =
      maybePermuteHWC2CHW(streamIndex, frameBatchOutput.data);
  return frameBatchOutput;
}

VideoDecoder::FrameBatchOutput VideoDecoder::getFramesPlayedInRange(
    int streamIndex,
    double startSeconds,
    double stopSeconds) {
  validateUserProvidedStreamIndex(streamIndex);

  const auto& streamMetadata =
      containerMetadata_.allStreamMetadata[streamIndex];
  TORCH_CHECK(
      startSeconds <= stopSeconds,
      "Start seconds (" + std::to_string(startSeconds) +
          ") must be less than or equal to stop seconds (" +
          std::to_string(stopSeconds) + ".");

  const auto& streamInfo = streamInfos_[streamIndex];
  const auto& videoStreamOptions = streamInfo.videoStreamOptions;

  // Special case needed to implement a half-open range. At first glance, this
  // may seem unnecessary, as our search for stopFrame can return the end, and
  // we don't include stopFramIndex in our output. However, consider the
  // following scenario:
  //
  //   frame=0, pts=0.0
  //   frame=1, pts=0.3
  //
  //   interval A: [0.2, 0.2)
  //   interval B: [0.2, 0.15)
  //
  // Both intervals take place between the pts values for frame 0 and frame 1,
  // which by our abstract player, means that both intervals map to frame 0. By
  // the definition of a half open interval, interval A should return no frames.
  // Interval B should return frame 0. However, for both A and B, the individual
  // values of the intervals will map to the same frame indices below. Hence, we
  // need this special case below.
  if (startSeconds == stopSeconds) {
    FrameBatchOutput frameBatchOutput(0, videoStreamOptions, streamMetadata);
    frameBatchOutput.data =
        maybePermuteHWC2CHW(streamIndex, frameBatchOutput.data);
    return frameBatchOutput;
  }

  double minSeconds = getMinSeconds(streamMetadata);
  double maxSeconds = getMaxSeconds(streamMetadata);
  TORCH_CHECK(
      startSeconds >= minSeconds && startSeconds < maxSeconds,
      "Start seconds is " + std::to_string(startSeconds) +
          "; must be in range [" + std::to_string(minSeconds) + ", " +
          std::to_string(maxSeconds) + ").");
  TORCH_CHECK(
      stopSeconds <= maxSeconds,
      "Stop seconds (" + std::to_string(stopSeconds) +
          "; must be less than or equal to " + std::to_string(maxSeconds) +
          ").");

  // Note that we look at nextPts for a frame, and not its pts or duration.
  // Our abstract player displays frames starting at the pts for that frame
  // until the pts for the next frame. There are two consequences:
  //
  //   1. We ignore the duration for a frame. A frame is played until the
  //   next frame replaces it. This model is robust to durations being 0 or
  //   incorrect; our source of truth is the pts for frames. If duration is
  //   accurate, the nextPts for a frame would be equivalent to pts +
  //   duration.
  //   2. In order to establish if the start of an interval maps to a
  //   particular frame, we need to figure out if it is ordered after the
  //   frame's pts, but before the next frames's pts.

  int64_t startFrameIndex =
      secondsToIndexLowerBound(startSeconds, streamInfo, streamMetadata);
  int64_t stopFrameIndex =
      secondsToIndexUpperBound(stopSeconds, streamInfo, streamMetadata);
  int64_t numFrames = stopFrameIndex - startFrameIndex;

  FrameBatchOutput frameBatchOutput(
      numFrames, videoStreamOptions, streamMetadata);
  for (int64_t i = startFrameIndex, f = 0; i < stopFrameIndex; ++i, ++f) {
    FrameOutput frameOutput =
        getFrameAtIndexInternal(streamIndex, i, frameBatchOutput.data[f]);
    frameBatchOutput.ptsSeconds[f] = frameOutput.ptsSeconds;
    frameBatchOutput.durationSeconds[f] = frameOutput.durationSeconds;
  }
  frameBatchOutput.data =
      maybePermuteHWC2CHW(streamIndex, frameBatchOutput.data);

  return frameBatchOutput;
}

VideoDecoder::FrameOutput VideoDecoder::getNextFrameNoDemux() {
  auto output = getNextFrameNoDemuxInternal();
  output.data = maybePermuteHWC2CHW(output.streamIndex, output.data);
  return output;
}

VideoDecoder::FrameOutput VideoDecoder::getNextFrameNoDemuxInternal(
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  AVFrameStream avFrameStream = decodeAVFrame([this](AVFrame* avFrame) {
    StreamInfo& activeStreamInfo = streamInfos_[activeStreamIndex_];
    return avFrame->pts >= activeStreamInfo.discardFramesBeforePts;
  });
  return convertAVFrameToFrameOutput(avFrameStream, preAllocatedOutputTensor);
}

void VideoDecoder::setCursorPtsInSeconds(double seconds) {
  desiredPtsSeconds_ = seconds;
}

VideoDecoder::DecodeStats VideoDecoder::getDecodeStats() const {
  return decodeStats_;
}

void VideoDecoder::resetDecodeStats() {
  decodeStats_ = DecodeStats{};
}

double VideoDecoder::getPtsSecondsForFrame(
    int streamIndex,
    int64_t frameIndex) {
  validateUserProvidedStreamIndex(streamIndex);
  validateScannedAllStreams("getPtsSecondsForFrame");

  const auto& streamInfo = streamInfos_[streamIndex];
  const auto& streamMetadata =
      containerMetadata_.allStreamMetadata[streamIndex];
  validateFrameIndex(streamMetadata, frameIndex);

  return ptsToSeconds(
      streamInfo.allFrames[frameIndex].pts, streamInfo.timeBase);
}

void VideoDecoder::createSwsContext(
    StreamInfo& streamInfo,
    const DecodedFrameContext& frameContext,
    const enum AVColorSpace colorspace) {
  SwsContext* swsContext = sws_getContext(
      frameContext.decodedWidth,
      frameContext.decodedHeight,
      frameContext.decodedFormat,
      frameContext.expectedWidth,
      frameContext.expectedHeight,
      AV_PIX_FMT_RGB24,
      SWS_BILINEAR,
      nullptr,
      nullptr,
      nullptr);
  TORCH_CHECK(swsContext, "sws_getContext() returned nullptr");

  int* invTable = nullptr;
  int* table = nullptr;
  int srcRange, dstRange, brightness, contrast, saturation;
  int ret = sws_getColorspaceDetails(
      swsContext,
      &invTable,
      &srcRange,
      &table,
      &dstRange,
      &brightness,
      &contrast,
      &saturation);
  TORCH_CHECK(ret != -1, "sws_getColorspaceDetails returned -1");

  const int* colorspaceTable = sws_getCoefficients(colorspace);
  ret = sws_setColorspaceDetails(
      swsContext,
      colorspaceTable,
      srcRange,
      colorspaceTable,
      dstRange,
      brightness,
      contrast,
      saturation);
  TORCH_CHECK(ret != -1, "sws_setColorspaceDetails returned -1");

  streamInfo.swsContext.reset(swsContext);
}

int VideoDecoder::convertAVFrameToTensorUsingSwsScale(
    int streamIndex,
    const AVFrame* avFrame,
    torch::Tensor& outputTensor) {
  StreamInfo& activeStreamInfo = streamInfos_[streamIndex];
  SwsContext* swsContext = activeStreamInfo.swsContext.get();
  uint8_t* pointers[4] = {
      outputTensor.data_ptr<uint8_t>(), nullptr, nullptr, nullptr};
  int expectedOutputWidth = outputTensor.sizes()[1];
  int linesizes[4] = {expectedOutputWidth * 3, 0, 0, 0};
  int resultHeight = sws_scale(
      swsContext,
      avFrame->data,
      avFrame->linesize,
      0,
      avFrame->height,
      pointers,
      linesizes);
  return resultHeight;
}

torch::Tensor VideoDecoder::convertAVFrameToTensorUsingFilterGraph(
    int streamIndex,
    const AVFrame* avFrame) {
  FilterGraphContext& filterGraphContext =
      streamInfos_[streamIndex].filterGraphContext;
  int ffmpegStatus =
      av_buffersrc_write_frame(filterGraphContext.sourceContext, avFrame);
  if (ffmpegStatus < AVSUCCESS) {
    throw std::runtime_error("Failed to add frame to buffer source context");
  }

  UniqueAVFrame filteredAVFrame(av_frame_alloc());
  ffmpegStatus = av_buffersink_get_frame(
      filterGraphContext.sinkContext, filteredAVFrame.get());
  TORCH_CHECK_EQ(filteredAVFrame->format, AV_PIX_FMT_RGB24);

  auto frameDims = getHeightAndWidthFromResizedAVFrame(*filteredAVFrame.get());
  int height = frameDims.height;
  int width = frameDims.width;
  std::vector<int64_t> shape = {height, width, 3};
  std::vector<int64_t> strides = {filteredAVFrame->linesize[0], 3, 1};
  AVFrame* filteredAVFramePtr = filteredAVFrame.release();
  auto deleter = [filteredAVFramePtr](void*) {
    UniqueAVFrame avFrameToDelete(filteredAVFramePtr);
  };
  return torch::from_blob(
      filteredAVFramePtr->data[0], shape, strides, deleter, {torch::kUInt8});
}

VideoDecoder::~VideoDecoder() {
  for (auto& [streamIndex, streamInfo] : streamInfos_) {
    auto& device = streamInfo.videoStreamOptions.device;
    if (device.type() == torch::kCPU) {
    } else if (device.type() == torch::kCUDA) {
      releaseContextOnCuda(device, streamInfo.codecContext.get());
    } else {
      TORCH_CHECK(false, "Invalid device type: " + device.str());
    }
  }
}

FrameDims getHeightAndWidthFromResizedAVFrame(const AVFrame& resizedAVFrame) {
  return FrameDims(resizedAVFrame.height, resizedAVFrame.width);
}

FrameDims getHeightAndWidthFromOptionsOrMetadata(
    const VideoDecoder::VideoStreamOptions& videoStreamOptions,
    const VideoDecoder::StreamMetadata& streamMetadata) {
  return FrameDims(
      videoStreamOptions.height.value_or(*streamMetadata.height),
      videoStreamOptions.width.value_or(*streamMetadata.width));
}

FrameDims getHeightAndWidthFromOptionsOrAVFrame(
    const VideoDecoder::VideoStreamOptions& videoStreamOptions,
    const AVFrame& avFrame) {
  return FrameDims(
      videoStreamOptions.height.value_or(avFrame.height),
      videoStreamOptions.width.value_or(avFrame.width));
}

torch::Tensor allocateEmptyHWCTensor(
    int height,
    int width,
    torch::Device device,
    std::optional<int> numFrames) {
  auto tensorOptions = torch::TensorOptions()
                           .dtype(torch::kUInt8)
                           .layout(torch::kStrided)
                           .device(device);
  TORCH_CHECK(height > 0, "height must be > 0, got: ", height);
  TORCH_CHECK(width > 0, "width must be > 0, got: ", width);
  if (numFrames.has_value()) {
    auto numFramesValue = numFrames.value();
    TORCH_CHECK(
        numFramesValue >= 0, "numFrames must be >= 0, got: ", numFramesValue);
    return torch::empty({numFramesValue, height, width, 3}, tensorOptions);
  } else {
    return torch::empty({height, width, 3}, tensorOptions);
  }
}

std::ostream& operator<<(
    std::ostream& os,
    const VideoDecoder::DecodeStats& stats) {
  os << "DecodeStats{"
     << "numFramesReceivedByDecoder=" << stats.numFramesReceivedByDecoder
     << ", numPacketsRead=" << stats.numPacketsRead
     << ", numPacketsSentToDecoder=" << stats.numPacketsSentToDecoder
     << ", numSeeksAttempted=" << stats.numSeeksAttempted
     << ", numSeeksSkipped=" << stats.numSeeksSkipped
     << ", numFlushes=" << stats.numFlushes << "}";

  return os;
}

} // namespace facebook::torchcodec
