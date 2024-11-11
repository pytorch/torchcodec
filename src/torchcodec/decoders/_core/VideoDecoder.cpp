// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/decoders/_core/VideoDecoder.h"
#include <cstdint>
#include <cstdio>
#include <iostream>
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

VideoDecoder::ColorConversionLibrary getDefaultColorConversionLibraryForWidth(
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
torch::Tensor VideoDecoder::MaybePermuteHWC2CHW(
    int streamIndex,
    torch::Tensor& hwcTensor) {
  if (streams_[streamIndex].options.dimensionOrder == "NHWC") {
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

VideoDecoder::VideoStreamDecoderOptions::VideoStreamDecoderOptions(
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

VideoDecoder::BatchDecodedOutput::BatchDecodedOutput(
    int64_t numFrames,
    const VideoStreamDecoderOptions& options,
    const StreamMetadata& metadata)
    : ptsSeconds(torch::empty({numFrames}, {torch::kFloat64})),
      durationSeconds(torch::empty({numFrames}, {torch::kFloat64})) {
  auto frameDims = getHeightAndWidthFromOptionsOrMetadata(options, metadata);
  int height = frameDims.height;
  int width = frameDims.width;
  frames = allocateEmptyHWCTensor(height, width, options.device, numFrames);
}

bool VideoDecoder::SwsContextKey::operator==(
    const VideoDecoder::SwsContextKey& other) {
  return decodedWidth == other.decodedWidth && decodedHeight == decodedHeight &&
      decodedFormat == other.decodedFormat &&
      outputWidth == other.outputWidth && outputHeight == other.outputHeight;
}

bool VideoDecoder::SwsContextKey::operator!=(
    const VideoDecoder::SwsContextKey& other) {
  return !(*this == other);
}

VideoDecoder::VideoDecoder() {}

void VideoDecoder::initializeDecoder() {
  // Some formats don't store enough info in the header so we read/decode a few
  // frames to grab that. This is needed for the filter graph. Note: If this
  // takes a long time, consider initializing the filter graph after the first
  // frame decode.
  int ffmpegStatus = avformat_find_stream_info(formatContext_.get(), nullptr);
  if (ffmpegStatus < 0) {
    throw std::runtime_error(
        "Failed to find stream info: " +
        getFFMPEGErrorStringFromErrorCode(ffmpegStatus));
  }
  containerMetadata_.streams.resize(0);
  for (int i = 0; i < formatContext_->nb_streams; i++) {
    AVStream* stream = formatContext_->streams[i];
    containerMetadata_.streams.resize(containerMetadata_.streams.size() + 1);
    auto& curr = containerMetadata_.streams.back();
    curr.streamIndex = i;
    curr.mediaType = stream->codecpar->codec_type;
    curr.codecName = avcodec_get_name(stream->codecpar->codec_id);
    curr.bitRate = stream->codecpar->bit_rate;

    int64_t frameCount = stream->nb_frames;
    if (frameCount > 0) {
      curr.numFrames = frameCount;
    }
    if (stream->duration > 0 && stream->time_base.den > 0) {
      curr.durationSeconds = av_q2d(stream->time_base) * stream->duration;
    }
    double fps = av_q2d(stream->r_frame_rate);
    if (fps > 0) {
      curr.averageFps = fps;
    }

    if (stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      containerMetadata_.numVideoStreams++;
    } else if (stream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
      containerMetadata_.numAudioStreams++;
    }
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
}

std::unique_ptr<VideoDecoder> VideoDecoder::createFromFilePath(
    const std::string& videoFilePath,
    const VideoDecoder::DecoderOptions& options) {
  AVInput input = createAVFormatContextFromFilePath(videoFilePath);
  std::unique_ptr<VideoDecoder> decoder(new VideoDecoder());
  decoder->formatContext_ = std::move(input.formatContext);
  decoder->options_ = options;
  decoder->initializeDecoder();
  return decoder;
}

std::unique_ptr<VideoDecoder> VideoDecoder::createFromBuffer(
    const void* buffer,
    size_t length,
    const VideoDecoder::DecoderOptions& options) {
  TORCH_CHECK(buffer != nullptr, "Video buffer cannot be nullptr!");
  AVInput input = createAVFormatContextFromBuffer(buffer, length);
  std::unique_ptr<VideoDecoder> decoder(new VideoDecoder());
  decoder->formatContext_ = std::move(input.formatContext);
  decoder->ioBytesContext_ = std::move(input.ioBytesContext);
  decoder->options_ = options;
  decoder->initializeDecoder();
  return decoder;
}

void VideoDecoder::initializeFilterGraphForStream(
    int streamIndex,
    const VideoStreamDecoderOptions& options) {
  FilterState& filterState = streams_[streamIndex].filterState;
  if (filterState.filterGraph) {
    return;
  }
  filterState.filterGraph.reset(avfilter_graph_alloc());
  TORCH_CHECK(filterState.filterGraph.get() != nullptr);
  if (options.ffmpegThreadCount.has_value()) {
    filterState.filterGraph->nb_threads = options.ffmpegThreadCount.value();
  }
  const AVFilter* buffersrc = avfilter_get_by_name("buffer");
  const AVFilter* buffersink = avfilter_get_by_name("buffersink");
  enum AVPixelFormat pix_fmts[] = {AV_PIX_FMT_RGB24, AV_PIX_FMT_NONE};
  const StreamInfo& activeStream = streams_[streamIndex];

  AVCodecContext* codecContext = activeStream.codecContext.get();
  char args[512];
  snprintf(
      args,
      sizeof(args),
      "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
      codecContext->width,
      codecContext->height,
      codecContext->pix_fmt,
      activeStream.stream->time_base.num,
      activeStream.stream->time_base.den,
      codecContext->sample_aspect_ratio.num,
      codecContext->sample_aspect_ratio.den);

  int ffmpegStatus = avfilter_graph_create_filter(
      &filterState.sourceContext,
      buffersrc,
      "in",
      args,
      nullptr,
      filterState.filterGraph.get());
  if (ffmpegStatus < 0) {
    throw std::runtime_error(
        std::string("Failed to create filter graph: ") + args + ": " +
        getFFMPEGErrorStringFromErrorCode(ffmpegStatus));
  }
  ffmpegStatus = avfilter_graph_create_filter(
      &filterState.sinkContext,
      buffersink,
      "out",
      nullptr,
      nullptr,
      filterState.filterGraph.get());
  if (ffmpegStatus < 0) {
    throw std::runtime_error(
        "Failed to create filter graph: " +
        getFFMPEGErrorStringFromErrorCode(ffmpegStatus));
  }
  ffmpegStatus = av_opt_set_int_list(
      filterState.sinkContext,
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
  outputs->filter_ctx = filterState.sourceContext;
  outputs->pad_idx = 0;
  outputs->next = nullptr;
  inputs->name = av_strdup("out");
  inputs->filter_ctx = filterState.sinkContext;
  inputs->pad_idx = 0;
  inputs->next = nullptr;
  char description[512];
  auto frameDims = getHeightAndWidthFromOptionsOrMetadata(
      options, containerMetadata_.streams[streamIndex]);
  int height = frameDims.height;
  int width = frameDims.width;

  std::snprintf(
      description,
      sizeof(description),
      "scale=%d:%d:sws_flags=bilinear",
      width,
      height);
  AVFilterInOut* outputsTmp = outputs.release();
  AVFilterInOut* inputsTmp = inputs.release();
  ffmpegStatus = avfilter_graph_parse_ptr(
      filterState.filterGraph.get(),
      description,
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
  ffmpegStatus = avfilter_graph_config(filterState.filterGraph.get(), nullptr);
  if (ffmpegStatus < 0) {
    throw std::runtime_error(
        "Failed to configure filter graph: " +
        getFFMPEGErrorStringFromErrorCode(ffmpegStatus));
  }
}

int VideoDecoder::getBestStreamIndex(AVMediaType mediaType) {
  AVCodecPtr codec = nullptr;
  int streamNumber =
      av_find_best_stream(formatContext_.get(), mediaType, -1, -1, &codec, 0);
  return streamNumber;
}

void VideoDecoder::addVideoStreamDecoder(
    int preferredStreamNumber,
    const VideoStreamDecoderOptions& options) {
  if (activeStreamIndices_.count(preferredStreamNumber) > 0) {
    throw std::invalid_argument(
        "Stream with index " + std::to_string(preferredStreamNumber) +
        " is already active.");
  }
  TORCH_CHECK(formatContext_.get() != nullptr);
  AVCodecPtr codec = nullptr;
  int streamNumber = av_find_best_stream(
      formatContext_.get(),
      AVMEDIA_TYPE_VIDEO,
      preferredStreamNumber,
      -1,
      &codec,
      0);
  if (streamNumber < 0) {
    throw std::invalid_argument("No valid stream found in input file.");
  }
  TORCH_CHECK(codec != nullptr);
  StreamInfo& streamInfo = streams_[streamNumber];
  streamInfo.streamIndex = streamNumber;
  streamInfo.timeBase = formatContext_->streams[streamNumber]->time_base;
  streamInfo.stream = formatContext_->streams[streamNumber];
  if (streamInfo.stream->codecpar->codec_type != AVMEDIA_TYPE_VIDEO) {
    throw std::invalid_argument(
        "Stream with index " + std::to_string(streamNumber) +
        " is not a video stream.");
  }
  AVCodecContext* codecContext = avcodec_alloc_context3(codec);
  codecContext->thread_count = options.ffmpegThreadCount.value_or(0);
  TORCH_CHECK(codecContext != nullptr);
  streamInfo.codecContext.reset(codecContext);
  int retVal = avcodec_parameters_to_context(
      streamInfo.codecContext.get(), streamInfo.stream->codecpar);
  if (options.device.type() == torch::kCPU) {
    // No more initialization needed for CPU.
  } else if (options.device.type() == torch::kCUDA) {
    initializeContextOnCuda(options.device, codecContext);
  } else {
    TORCH_CHECK(false, "Invalid device type: " + options.device.str());
  }
  TORCH_CHECK_EQ(retVal, AVSUCCESS);
  retVal = avcodec_open2(streamInfo.codecContext.get(), codec, nullptr);
  if (retVal < AVSUCCESS) {
    throw std::invalid_argument(getFFMPEGErrorStringFromErrorCode(retVal));
  }
  codecContext->time_base = streamInfo.stream->time_base;
  activeStreamIndices_.insert(streamNumber);
  updateMetadataWithCodecContext(streamInfo.streamIndex, codecContext);
  streamInfo.options = options;
  int width = options.width.value_or(codecContext->width);

  // Use swscale for color conversion by default because it is faster.
  VideoDecoder::ColorConversionLibrary defaultColorConversionLibrary =
      getDefaultColorConversionLibraryForWidth(width);
  // If the user specifies the color conversion library (example in
  // benchmarks), we use that instead.
  auto colorConversionLibrary =
      options.colorConversionLibrary.value_or(defaultColorConversionLibrary);

  if (colorConversionLibrary == ColorConversionLibrary::FILTERGRAPH) {
    initializeFilterGraphForStream(streamNumber, options);
    streamInfo.colorConversionLibrary = ColorConversionLibrary::FILTERGRAPH;
  } else if (colorConversionLibrary == ColorConversionLibrary::SWSCALE) {
    streamInfo.colorConversionLibrary = ColorConversionLibrary::SWSCALE;
  } else {
    throw std::invalid_argument(
        "Invalid colorConversionLibrary=" +
        std::to_string(static_cast<int>(colorConversionLibrary)) +
        ". colorConversionLibrary must be either "
        "filtergraph or swscale.");
  }
}

void VideoDecoder::updateMetadataWithCodecContext(
    int streamIndex,
    AVCodecContext* codecContext) {
  containerMetadata_.streams[streamIndex].width = codecContext->width;
  containerMetadata_.streams[streamIndex].height = codecContext->height;
  auto codedId = codecContext->codec_id;
  containerMetadata_.streams[streamIndex].codecName =
      std::string(avcodec_get_name(codedId));
}

VideoDecoder::ContainerMetadata VideoDecoder::getContainerMetadata() const {
  return containerMetadata_;
}

int VideoDecoder::getKeyFrameIndexForPtsUsingEncoderIndex(
    AVStream* stream,
    int64_t pts) const {
  int currentKeyFrameIndex =
      av_index_search_timestamp(stream, pts, AVSEEK_FLAG_BACKWARD);
  return currentKeyFrameIndex;
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
  if (scanned_all_streams_) {
    return;
  }
  while (true) {
    UniqueAVPacket packet(av_packet_alloc());
    int ffmpegStatus = av_read_frame(formatContext_.get(), packet.get());
    if (ffmpegStatus == AVERROR_EOF) {
      break;
    }
    if (ffmpegStatus != AVSUCCESS) {
      throw std::runtime_error(
          "Failed to read frame from input file: " +
          getFFMPEGErrorStringFromErrorCode(ffmpegStatus));
    }
    int streamIndex = packet->stream_index;

    if (packet->flags & AV_PKT_FLAG_DISCARD) {
      continue;
    }
    auto& stream = containerMetadata_.streams[streamIndex];
    stream.minPtsFromScan =
        std::min(stream.minPtsFromScan.value_or(INT64_MAX), packet->pts);
    stream.maxPtsFromScan = std::max(
        stream.maxPtsFromScan.value_or(INT64_MIN),
        packet->pts + packet->duration);
    stream.numFramesFromScan = stream.numFramesFromScan.value_or(0) + 1;

    FrameInfo frameInfo;
    frameInfo.pts = packet->pts;

    if (packet->flags & AV_PKT_FLAG_KEY) {
      streams_[streamIndex].keyFrames.push_back(frameInfo);
    }
    streams_[streamIndex].allFrames.push_back(frameInfo);
  }
  for (int i = 0; i < containerMetadata_.streams.size(); ++i) {
    auto& streamMetadata = containerMetadata_.streams[i];
    auto stream = formatContext_->streams[i];
    if (streamMetadata.minPtsFromScan.has_value()) {
      streamMetadata.minPtsSecondsFromScan =
          *streamMetadata.minPtsFromScan * av_q2d(stream->time_base);
    }
    if (streamMetadata.maxPtsFromScan.has_value()) {
      streamMetadata.maxPtsSecondsFromScan =
          *streamMetadata.maxPtsFromScan * av_q2d(stream->time_base);
    }
  }
  int ffmepgStatus =
      avformat_seek_file(formatContext_.get(), 0, INT64_MIN, 0, 0, 0);
  if (ffmepgStatus < 0) {
    throw std::runtime_error(
        "Could not seek file to pts=0: " +
        getFFMPEGErrorStringFromErrorCode(ffmepgStatus));
  }
  for (auto& [streamIndex, stream] : streams_) {
    std::sort(
        stream.keyFrames.begin(),
        stream.keyFrames.end(),
        [](const FrameInfo& frameInfo1, const FrameInfo& frameInfo2) {
          return frameInfo1.pts < frameInfo2.pts;
        });
    std::sort(
        stream.allFrames.begin(),
        stream.allFrames.end(),
        [](const FrameInfo& frameInfo1, const FrameInfo& frameInfo2) {
          return frameInfo1.pts < frameInfo2.pts;
        });

    for (int i = 0; i < stream.allFrames.size(); ++i) {
      if (i + 1 < stream.allFrames.size()) {
        stream.allFrames[i].nextPts = stream.allFrames[i + 1].pts;
      }
    }
  }
  scanned_all_streams_ = true;
}

int VideoDecoder::getKeyFrameIndexForPts(
    const StreamInfo& streamInfo,
    int64_t pts) const {
  if (streamInfo.keyFrames.empty()) {
    return getKeyFrameIndexForPtsUsingEncoderIndex(streamInfo.stream, pts);
  }
  return getKeyFrameIndexForPtsUsingScannedIndex(streamInfo.keyFrames, pts);
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

We use av_index_search_timestamp to see if there is an I frame between x and y.
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
  if (activeStreamIndices_.size() == 0) {
    return;
  }
  for (int streamIndex : activeStreamIndices_) {
    StreamInfo& streamInfo = streams_[streamIndex];
    // clang-format off: clang format clashes
    streamInfo.discardFramesBeforePts = secondsToClosestPts(*maybeDesiredPts_, streamInfo.timeBase);
    // clang-format on
  }

  decodeStats_.numSeeksAttempted++;
  // See comment for canWeAvoidSeeking() for details on why this optimization
  // works.
  bool mustSeek = false;
  for (int streamIndex : activeStreamIndices_) {
    StreamInfo& streamInfo = streams_[streamIndex];
    int64_t desiredPtsForStream = *maybeDesiredPts_ * streamInfo.timeBase.den;
    if (!canWeAvoidSeekingForStream(
            streamInfo, streamInfo.currentPts, desiredPtsForStream)) {
      mustSeek = true;
      break;
    }
  }
  if (!mustSeek) {
    decodeStats_.numSeeksSkipped++;
    return;
  }
  int firstActiveStreamIndex = *activeStreamIndices_.begin();
  const auto& firstStreamInfo = streams_[firstActiveStreamIndex];
  int64_t desiredPts =
      secondsToClosestPts(*maybeDesiredPts_, firstStreamInfo.timeBase);

  // For some encodings like H265, FFMPEG sometimes seeks past the point we
  // set as the max_ts. So we use our own index to give it the exact pts of
  // the key frame that we want to seek to.
  // See https://github.com/pytorch/torchcodec/issues/179 for more details.
  // See https://trac.ffmpeg.org/ticket/11137 for the underlying ffmpeg bug.
  if (!firstStreamInfo.keyFrames.empty()) {
    int desiredKeyFrameIndex = getKeyFrameIndexForPtsUsingScannedIndex(
        firstStreamInfo.keyFrames, desiredPts);
    desiredKeyFrameIndex = std::max(desiredKeyFrameIndex, 0);
    desiredPts = firstStreamInfo.keyFrames[desiredKeyFrameIndex].pts;
  }

  int ffmepgStatus = avformat_seek_file(
      formatContext_.get(),
      firstStreamInfo.streamIndex,
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
  for (int streamIndex : activeStreamIndices_) {
    StreamInfo& streamInfo = streams_[streamIndex];
    avcodec_flush_buffers(streamInfo.codecContext.get());
  }
}

VideoDecoder::RawDecodedOutput VideoDecoder::getDecodedOutputWithFilter(
    std::function<bool(int, AVFrame*)> filterFunction) {
  auto start = std::chrono::high_resolution_clock::now();
  if (activeStreamIndices_.size() == 0) {
    throw std::runtime_error("No active streams configured.");
  }
  resetDecodeStats();
  if (maybeDesiredPts_.has_value()) {
    maybeSeekToBeforeDesiredPts();
    maybeDesiredPts_ = std::nullopt;
  }
  auto seekDone = std::chrono::high_resolution_clock::now();
  // Need to get the next frame or error from PopFrame.
  UniqueAVFrame frame(av_frame_alloc());
  int ffmpegStatus = AVSUCCESS;
  bool reachedEOF = false;
  int frameStreamIndex = -1;
  while (true) {
    frameStreamIndex = -1;
    bool gotPermanentErrorOnAnyActiveStream = false;
    for (int streamIndex : activeStreamIndices_) {
      StreamInfo& streamInfo = streams_[streamIndex];
      ffmpegStatus =
          avcodec_receive_frame(streamInfo.codecContext.get(), frame.get());
      bool gotNonRetriableError =
          ffmpegStatus != AVSUCCESS && ffmpegStatus != AVERROR(EAGAIN);
      if (gotNonRetriableError) {
        gotPermanentErrorOnAnyActiveStream = true;
        break;
      }
      if (ffmpegStatus == AVSUCCESS) {
        frameStreamIndex = streamIndex;
        break;
      }
    }
    if (gotPermanentErrorOnAnyActiveStream) {
      break;
    }
    decodeStats_.numFramesReceivedByDecoder++;
    bool gotNeededFrame = ffmpegStatus == AVSUCCESS &&
        filterFunction(frameStreamIndex, frame.get());
    if (gotNeededFrame) {
      break;
    } else if (ffmpegStatus == AVSUCCESS) {
      // No need to send more packets here as the decoder may have frames in
      // its buffer.
      continue;
    }
    if (reachedEOF) {
      // We don't have any more packets to send to the decoder. So keep on
      // pulling frames from its internal buffers.
      continue;
    }
    UniqueAVPacket packet(av_packet_alloc());
    ffmpegStatus = av_read_frame(formatContext_.get(), packet.get());
    decodeStats_.numPacketsRead++;
    if (ffmpegStatus == AVERROR_EOF) {
      // End of file reached. We must drain all codecs by sending a nullptr
      // packet.
      for (int streamIndex : activeStreamIndices_) {
        StreamInfo& streamInfo = streams_[streamIndex];
        ffmpegStatus = avcodec_send_packet(
            streamInfo.codecContext.get(),
            /*avpkt=*/nullptr);
        if (ffmpegStatus < AVSUCCESS) {
          throw std::runtime_error(
              "Could not flush decoder: " +
              getFFMPEGErrorStringFromErrorCode(ffmpegStatus));
        }
      }
      reachedEOF = true;
      continue;
    }
    if (ffmpegStatus < AVSUCCESS) {
      throw std::runtime_error(
          "Could not read frame from input file: " +
          getFFMPEGErrorStringFromErrorCode(ffmpegStatus));
    }
    if (activeStreamIndices_.count(packet->stream_index) == 0) {
      // This packet is not for any of the active streams.
      continue;
    }
    ffmpegStatus = avcodec_send_packet(
        streams_[packet->stream_index].codecContext.get(), packet.get());
    decodeStats_.numPacketsSentToDecoder++;
    if (ffmpegStatus < AVSUCCESS) {
      throw std::runtime_error(
          "Could not push packet to decoder: " +
          getFFMPEGErrorStringFromErrorCode(ffmpegStatus));
    }
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
  auto decodeDone = std::chrono::high_resolution_clock::now();
  // Note that we don't flush the decoder when we reach EOF (even though that's
  // mentioned in https://ffmpeg.org/doxygen/trunk/group__lavc__encdec.html).
  // This is because we may have packets internally in the decoder that we
  // haven't received as frames. Eventually we will either hit AVERROR_EOF from
  // av_receive_frame() or the user will have seeked to a different location in
  // the file and that will flush the decoder.
  StreamInfo& activeStream = streams_[frameStreamIndex];
  activeStream.currentPts = frame->pts;
  activeStream.currentDuration = getDuration(frame);
  RawDecodedOutput rawOutput;
  rawOutput.streamIndex = frameStreamIndex;
  rawOutput.frame = std::move(frame);
  return rawOutput;
}

VideoDecoder::DecodedOutput VideoDecoder::convertAVFrameToDecodedOutput(
    VideoDecoder::RawDecodedOutput& rawOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  // Convert the frame to tensor.
  DecodedOutput output;
  int streamIndex = rawOutput.streamIndex;
  AVFrame* frame = rawOutput.frame.get();
  output.streamIndex = streamIndex;
  auto& streamInfo = streams_[streamIndex];
  output.streamType = streams_[streamIndex].stream->codecpar->codec_type;
  output.pts = frame->pts;
  output.ptsSeconds =
      ptsToSeconds(frame->pts, formatContext_->streams[streamIndex]->time_base);
  output.duration = getDuration(frame);
  output.durationSeconds = ptsToSeconds(
      getDuration(frame), formatContext_->streams[streamIndex]->time_base);
  // TODO: we should fold preAllocatedOutputTensor into RawDecodedOutput.
  if (streamInfo.options.device.type() == torch::kCPU) {
    convertAVFrameToDecodedOutputOnCPU(
        rawOutput, output, preAllocatedOutputTensor);
  } else if (streamInfo.options.device.type() == torch::kCUDA) {
    convertAVFrameToDecodedOutputOnCuda(
        streamInfo.options.device,
        streamInfo.options,
        rawOutput,
        output,
        preAllocatedOutputTensor);
  } else {
    TORCH_CHECK(
        false, "Invalid device type: " + streamInfo.options.device.str());
  }
  return output;
}

// Note [preAllocatedOutputTensor with swscale and filtergraph]:
// Callers may pass a pre-allocated tensor, where the output frame tensor will
// be stored. This parameter is honored in any case, but it only leads to a
// speed-up when swscale is used. With swscale, we can tell ffmpeg to place the
// decoded frame directly into `preAllocatedtensor.data_ptr()`. We haven't yet
// found a way to do that with filtegraph.
// TODO: Figure out whether that's possible!
// Dimension order of the preAllocatedOutputTensor must be HWC, regardless of
// `dimension_order` parameter. It's up to callers to re-shape it if needed.
void VideoDecoder::convertAVFrameToDecodedOutputOnCPU(
    VideoDecoder::RawDecodedOutput& rawOutput,
    DecodedOutput& output,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  int streamIndex = rawOutput.streamIndex;
  AVFrame* frame = rawOutput.frame.get();
  auto& streamInfo = streams_[streamIndex];

  auto frameDims =
      getHeightAndWidthFromOptionsOrAVFrame(streamInfo.options, *frame);
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
  if (output.streamType == AVMEDIA_TYPE_VIDEO) {
    if (streamInfo.colorConversionLibrary == ColorConversionLibrary::SWSCALE) {
      outputTensor = preAllocatedOutputTensor.value_or(allocateEmptyHWCTensor(
          expectedOutputHeight, expectedOutputWidth, torch::kCPU));

      int resultHeight =
          convertFrameToBufferUsingSwsScale(streamIndex, frame, outputTensor);
      // If this check failed, it would mean that the frame wasn't reshaped to
      // the expected height.
      // TODO: Can we do the same check for width?
      TORCH_CHECK(
          resultHeight == expectedOutputHeight,
          "resultHeight != expectedOutputHeight: ",
          resultHeight,
          " != ",
          expectedOutputHeight);

      output.frame = outputTensor;
    } else if (
        streamInfo.colorConversionLibrary ==
        ColorConversionLibrary::FILTERGRAPH) {
      outputTensor = convertFrameToTensorUsingFilterGraph(streamIndex, frame);

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
        output.frame = preAllocatedOutputTensor.value();
      } else {
        output.frame = outputTensor;
      }
    } else {
      throw std::runtime_error(
          "Invalid color conversion library: " +
          std::to_string(static_cast<int>(streamInfo.colorConversionLibrary)));
    }

  } else if (output.streamType == AVMEDIA_TYPE_AUDIO) {
    // TODO: https://github.com/pytorch-labs/torchcodec/issues/85 implement
    // audio decoding.
    throw std::runtime_error("Audio is not supported yet.");
  }
}

VideoDecoder::DecodedOutput VideoDecoder::getFramePlayedAtTimestampNoDemux(
    double seconds) {
  for (auto& [streamIndex, stream] : streams_) {
    double frameStartTime = ptsToSeconds(stream.currentPts, stream.timeBase);
    double frameEndTime = ptsToSeconds(
        stream.currentPts + stream.currentDuration, stream.timeBase);
    if (seconds >= frameStartTime && seconds < frameEndTime) {
      // We are in the same frame as the one we just returned. However, since we
      // don't cache it locally, we have to rewind back.
      seconds = frameStartTime;
      break;
    }
  }
  setCursorPtsInSeconds(seconds);
  RawDecodedOutput rawOutput = getDecodedOutputWithFilter(
      [seconds, this](int frameStreamIndex, AVFrame* frame) {
        StreamInfo& stream = streams_[frameStreamIndex];
        double frameStartTime = ptsToSeconds(frame->pts, stream.timeBase);
        double frameEndTime =
            ptsToSeconds(frame->pts + getDuration(frame), stream.timeBase);
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
  auto output = convertAVFrameToDecodedOutput(rawOutput);
  output.frame = MaybePermuteHWC2CHW(output.streamIndex, output.frame);
  return output;
}

void VideoDecoder::validateUserProvidedStreamIndex(uint64_t streamIndex) {
  size_t streamsSize = containerMetadata_.streams.size();
  TORCH_CHECK(
      streamIndex >= 0 && streamIndex < streamsSize,
      "Invalid stream index=" + std::to_string(streamIndex) +
          "; valid indices are in the range [0, " +
          std::to_string(streamsSize) + ").");
  TORCH_CHECK(
      streams_.count(streamIndex) > 0,
      "Provided stream index=" + std::to_string(streamIndex) +
          " was not previously added.");
}

void VideoDecoder::validateScannedAllStreams(const std::string& msg) {
  if (!scanned_all_streams_) {
    throw std::runtime_error(
        "Must scan all streams to update metadata before calling " + msg);
  }
}

void VideoDecoder::validateFrameIndex(
    const StreamInfo& stream,
    int64_t frameIndex) {
  TORCH_CHECK(
      frameIndex >= 0 && frameIndex < stream.allFrames.size(),
      "Invalid frame index=" + std::to_string(frameIndex) +
          " for streamIndex=" + std::to_string(stream.streamIndex) +
          " numFrames=" + std::to_string(stream.allFrames.size()));
}

VideoDecoder::DecodedOutput VideoDecoder::getFrameAtIndex(
    int streamIndex,
    int64_t frameIndex) {
  auto output = getFrameAtIndexInternal(streamIndex, frameIndex);
  output.frame = MaybePermuteHWC2CHW(streamIndex, output.frame);
  return output;
}

VideoDecoder::DecodedOutput VideoDecoder::getFrameAtIndexInternal(
    int streamIndex,
    int64_t frameIndex,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  validateUserProvidedStreamIndex(streamIndex);
  validateScannedAllStreams("getFrameAtIndex");

  const auto& stream = streams_[streamIndex];
  validateFrameIndex(stream, frameIndex);

  int64_t pts = stream.allFrames[frameIndex].pts;
  setCursorPtsInSeconds(ptsToSeconds(pts, stream.timeBase));
  return getNextFrameOutputNoDemuxInternal(preAllocatedOutputTensor);
}

VideoDecoder::BatchDecodedOutput VideoDecoder::getFramesAtIndices(
    int streamIndex,
    const std::vector<int64_t>& frameIndices) {
  validateUserProvidedStreamIndex(streamIndex);
  validateScannedAllStreams("getFramesAtIndices");

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

  const auto& streamMetadata = containerMetadata_.streams[streamIndex];
  const auto& stream = streams_[streamIndex];
  const auto& options = stream.options;
  BatchDecodedOutput output(frameIndices.size(), options, streamMetadata);

  auto previousIndexInVideo = -1;
  for (auto f = 0; f < frameIndices.size(); ++f) {
    auto indexInOutput = indicesAreSorted ? f : argsort[f];
    auto indexInVideo = frameIndices[indexInOutput];
    if (indexInVideo < 0 || indexInVideo >= stream.allFrames.size()) {
      throw std::runtime_error(
          "Invalid frame index=" + std::to_string(indexInVideo));
    }
    if ((f > 0) && (indexInVideo == previousIndexInVideo)) {
      // Avoid decoding the same frame twice
      auto previousIndexInOutput = indicesAreSorted ? f - 1 : argsort[f - 1];
      output.frames[indexInOutput].copy_(output.frames[previousIndexInOutput]);
      output.ptsSeconds[indexInOutput] =
          output.ptsSeconds[previousIndexInOutput];
      output.durationSeconds[indexInOutput] =
          output.durationSeconds[previousIndexInOutput];
    } else {
      DecodedOutput singleOut = getFrameAtIndexInternal(
          streamIndex, indexInVideo, output.frames[indexInOutput]);
      output.ptsSeconds[indexInOutput] = singleOut.ptsSeconds;
      output.durationSeconds[indexInOutput] = singleOut.durationSeconds;
    }
    previousIndexInVideo = indexInVideo;
  }
  output.frames = MaybePermuteHWC2CHW(streamIndex, output.frames);
  return output;
}

VideoDecoder::BatchDecodedOutput VideoDecoder::getFramesPlayedByTimestamps(
    int streamIndex,
    const std::vector<double>& timestamps) {
  validateUserProvidedStreamIndex(streamIndex);
  validateScannedAllStreams("getFramesPlayedByTimestamps");

  // The frame played at timestamp t and the one played at timestamp `t +
  // eps` are probably the same frame, with the same index. The easiest way to
  // avoid decoding that unique frame twice is to convert the input timestamps
  // to indices, and leverage the de-duplication logic of getFramesAtIndices.
  // This means this function requires a scan.
  // TODO: longer term, we should implement this without requiring a scan

  const auto& streamMetadata = containerMetadata_.streams[streamIndex];
  const auto& stream = streams_[streamIndex];
  double minSeconds = streamMetadata.minPtsSecondsFromScan.value();
  double maxSeconds = streamMetadata.maxPtsSecondsFromScan.value();

  std::vector<int64_t> frameIndices(timestamps.size());
  for (auto i = 0; i < timestamps.size(); ++i) {
    auto framePts = timestamps[i];
    TORCH_CHECK(
        framePts >= minSeconds && framePts < maxSeconds,
        "frame pts is " + std::to_string(framePts) + "; must be in range [" +
            std::to_string(minSeconds) + ", " + std::to_string(maxSeconds) +
            ").");

    auto it = std::lower_bound(
        stream.allFrames.begin(),
        stream.allFrames.end(),
        framePts,
        [&stream](const FrameInfo& info, double framePts) {
          return ptsToSeconds(info.nextPts, stream.timeBase) <= framePts;
        });
    int64_t frameIndex = it - stream.allFrames.begin();
    frameIndices[i] = frameIndex;
  }

  return getFramesAtIndices(streamIndex, frameIndices);
}

VideoDecoder::BatchDecodedOutput VideoDecoder::getFramesInRange(
    int streamIndex,
    int64_t start,
    int64_t stop,
    int64_t step) {
  validateUserProvidedStreamIndex(streamIndex);
  validateScannedAllStreams("getFramesInRange");

  const auto& streamMetadata = containerMetadata_.streams[streamIndex];
  const auto& stream = streams_[streamIndex];
  TORCH_CHECK(
      start >= 0, "Range start, " + std::to_string(start) + " is less than 0.");
  TORCH_CHECK(
      stop <= stream.allFrames.size(),
      "Range stop, " + std::to_string(stop) +
          ", is more than the number of frames, " +
          std::to_string(stream.allFrames.size()));
  TORCH_CHECK(
      step > 0, "Step must be greater than 0; is " + std::to_string(step));

  int64_t numOutputFrames = std::ceil((stop - start) / double(step));
  const auto& options = stream.options;
  BatchDecodedOutput output(numOutputFrames, options, streamMetadata);

  for (int64_t i = start, f = 0; i < stop; i += step, ++f) {
    DecodedOutput singleOut =
        getFrameAtIndexInternal(streamIndex, i, output.frames[f]);
    output.ptsSeconds[f] = singleOut.ptsSeconds;
    output.durationSeconds[f] = singleOut.durationSeconds;
  }
  output.frames = MaybePermuteHWC2CHW(streamIndex, output.frames);
  return output;
}

VideoDecoder::BatchDecodedOutput
VideoDecoder::getFramesPlayedByTimestampInRange(
    int streamIndex,
    double startSeconds,
    double stopSeconds) {
  validateUserProvidedStreamIndex(streamIndex);
  validateScannedAllStreams("getFramesPlayedByTimestampInRange");

  const auto& streamMetadata = containerMetadata_.streams[streamIndex];
  double minSeconds = streamMetadata.minPtsSecondsFromScan.value();
  double maxSeconds = streamMetadata.maxPtsSecondsFromScan.value();
  TORCH_CHECK(
      startSeconds <= stopSeconds,
      "Start seconds (" + std::to_string(startSeconds) +
          ") must be less than or equal to stop seconds (" +
          std::to_string(stopSeconds) + ".");
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

  const auto& stream = streams_[streamIndex];
  const auto& options = stream.options;

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
    BatchDecodedOutput output(0, options, streamMetadata);
    output.frames = MaybePermuteHWC2CHW(streamIndex, output.frames);
    return output;
  }

  // Note that we look at nextPts for a frame, and not its pts or duration. Our
  // abstract player displays frames starting at the pts for that frame until
  // the pts for the next frame. There are two consequences:
  //
  //   1. We ignore the duration for a frame. A frame is played until the
  //   next frame replaces it. This model is robust to durations being 0 or
  //   incorrect; our source of truth is the pts for frames. If duration is
  //   accurate, the nextPts for a frame would be equivalent to pts + duration.
  //   2. In order to establish if the start of an interval maps to a particular
  //   frame, we need to figure out if it is ordered after the frame's pts, but
  //   before the next frames's pts.
  auto startFrame = std::lower_bound(
      stream.allFrames.begin(),
      stream.allFrames.end(),
      startSeconds,
      [&stream](const FrameInfo& info, double start) {
        return ptsToSeconds(info.nextPts, stream.timeBase) <= start;
      });

  auto stopFrame = std::upper_bound(
      stream.allFrames.begin(),
      stream.allFrames.end(),
      stopSeconds,
      [&stream](double stop, const FrameInfo& info) {
        return stop <= ptsToSeconds(info.pts, stream.timeBase);
      });

  int64_t startFrameIndex = startFrame - stream.allFrames.begin();
  int64_t stopFrameIndex = stopFrame - stream.allFrames.begin();
  int64_t numFrames = stopFrameIndex - startFrameIndex;
  BatchDecodedOutput output(numFrames, options, streamMetadata);
  for (int64_t i = startFrameIndex, f = 0; i < stopFrameIndex; ++i, ++f) {
    DecodedOutput singleOut =
        getFrameAtIndexInternal(streamIndex, i, output.frames[f]);
    output.ptsSeconds[f] = singleOut.ptsSeconds;
    output.durationSeconds[f] = singleOut.durationSeconds;
  }
  output.frames = MaybePermuteHWC2CHW(streamIndex, output.frames);

  return output;
}

VideoDecoder::RawDecodedOutput VideoDecoder::getNextRawDecodedOutputNoDemux() {
  auto rawOutput =
      getDecodedOutputWithFilter([this](int frameStreamIndex, AVFrame* frame) {
        StreamInfo& activeStream = streams_[frameStreamIndex];
        return frame->pts >= activeStream.discardFramesBeforePts;
      });
  return rawOutput;
}

VideoDecoder::DecodedOutput VideoDecoder::getNextFrameNoDemux() {
  auto output = getNextFrameOutputNoDemuxInternal();
  output.frame = MaybePermuteHWC2CHW(output.streamIndex, output.frame);
  return output;
}

VideoDecoder::DecodedOutput VideoDecoder::getNextFrameOutputNoDemuxInternal(
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  auto rawOutput = getNextRawDecodedOutputNoDemux();
  return convertAVFrameToDecodedOutput(rawOutput, preAllocatedOutputTensor);
}

void VideoDecoder::setCursorPtsInSeconds(double seconds) {
  maybeDesiredPts_ = seconds;
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

  const auto& stream = streams_[streamIndex];
  validateFrameIndex(stream, frameIndex);

  return ptsToSeconds(stream.allFrames[frameIndex].pts, stream.timeBase);
}

int VideoDecoder::convertFrameToBufferUsingSwsScale(
    int streamIndex,
    const AVFrame* frame,
    torch::Tensor& outputTensor) {
  enum AVPixelFormat frameFormat =
      static_cast<enum AVPixelFormat>(frame->format);
  StreamInfo& activeStream = streams_[streamIndex];

  int expectedOutputHeight = outputTensor.sizes()[0];
  int expectedOutputWidth = outputTensor.sizes()[1];
  auto curFrameSwsContextKey = SwsContextKey{
      frame->width,
      frame->height,
      frameFormat,
      expectedOutputWidth,
      expectedOutputHeight};
  if (activeStream.swsContext.get() == nullptr ||
      activeStream.swsContextKey != curFrameSwsContextKey) {
    SwsContext* swsContext = sws_getContext(
        frame->width,
        frame->height,
        frameFormat,
        expectedOutputWidth,
        expectedOutputHeight,
        AV_PIX_FMT_RGB24,
        SWS_BILINEAR,
        nullptr,
        nullptr,
        nullptr);
    int* invTable = nullptr;
    int* table = nullptr;
    int srcRange, dstRange, brightness, contrast, saturation;
    sws_getColorspaceDetails(
        swsContext,
        &invTable,
        &srcRange,
        &table,
        &dstRange,
        &brightness,
        &contrast,
        &saturation);
    const int* colorspaceTable = sws_getCoefficients(frame->colorspace);
    sws_setColorspaceDetails(
        swsContext,
        colorspaceTable,
        srcRange,
        colorspaceTable,
        dstRange,
        brightness,
        contrast,
        saturation);
    activeStream.swsContextKey = curFrameSwsContextKey;
    activeStream.swsContext.reset(swsContext);
  }
  SwsContext* swsContext = activeStream.swsContext.get();
  uint8_t* pointers[4] = {
      outputTensor.data_ptr<uint8_t>(), nullptr, nullptr, nullptr};
  int linesizes[4] = {expectedOutputWidth * 3, 0, 0, 0};
  int resultHeight = sws_scale(
      swsContext,
      frame->data,
      frame->linesize,
      0,
      frame->height,
      pointers,
      linesizes);
  return resultHeight;
}

torch::Tensor VideoDecoder::convertFrameToTensorUsingFilterGraph(
    int streamIndex,
    const AVFrame* frame) {
  FilterState& filterState = streams_[streamIndex].filterState;
  int ffmpegStatus = av_buffersrc_write_frame(filterState.sourceContext, frame);
  if (ffmpegStatus < AVSUCCESS) {
    throw std::runtime_error("Failed to add frame to buffer source context");
  }
  UniqueAVFrame filteredFrame(av_frame_alloc());
  ffmpegStatus =
      av_buffersink_get_frame(filterState.sinkContext, filteredFrame.get());
  TORCH_CHECK_EQ(filteredFrame->format, AV_PIX_FMT_RGB24);
  auto frameDims = getHeightAndWidthFromResizedAVFrame(*filteredFrame.get());
  int height = frameDims.height;
  int width = frameDims.width;
  std::vector<int64_t> shape = {height, width, 3};
  std::vector<int64_t> strides = {filteredFrame->linesize[0], 3, 1};
  AVFrame* filteredFramePtr = filteredFrame.release();
  auto deleter = [filteredFramePtr](void*) {
    UniqueAVFrame frameToDelete(filteredFramePtr);
  };
  torch::Tensor tensor = torch::from_blob(
      filteredFramePtr->data[0], shape, strides, deleter, {torch::kUInt8});
  return tensor;
}

VideoDecoder::~VideoDecoder() {
  for (auto& [streamIndex, stream] : streams_) {
    auto& device = stream.options.device;
    if (device.type() == torch::kCPU) {
    } else if (device.type() == torch::kCUDA) {
      releaseContextOnCuda(device, stream.codecContext.get());
    } else {
      TORCH_CHECK(false, "Invalid device type: " + device.str());
    }
  }
}

FrameDims getHeightAndWidthFromResizedAVFrame(const AVFrame& resizedAVFrame) {
  return FrameDims(resizedAVFrame.height, resizedAVFrame.width);
}

FrameDims getHeightAndWidthFromOptionsOrMetadata(
    const VideoDecoder::VideoStreamDecoderOptions& options,
    const VideoDecoder::StreamMetadata& metadata) {
  return FrameDims(
      options.height.value_or(*metadata.height),
      options.width.value_or(*metadata.width));
}

FrameDims getHeightAndWidthFromOptionsOrAVFrame(
    const VideoDecoder::VideoStreamDecoderOptions& options,
    const AVFrame& avFrame) {
  return FrameDims(
      options.height.value_or(avFrame.height),
      options.width.value_or(avFrame.width));
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
