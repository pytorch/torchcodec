// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/decoders/_core/VideoDecoder.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
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
#include <libavutil/log.h>
#include <libavutil/pixdesc.h>
#include <libswresample/swresample.h>
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

} // namespace

// --------------------------------------------------------------------------
// CONSTRUCTORS, INITIALIZATION, DESTRUCTORS
// --------------------------------------------------------------------------

VideoDecoder::VideoDecoder(const std::string& videoFilePath, SeekMode seekMode)
    : seekMode_(seekMode) {
  setFFmpegLogLevel();

  AVFormatContext* rawContext = nullptr;
  int status =
      avformat_open_input(&rawContext, videoFilePath.c_str(), nullptr, nullptr);
  TORCH_CHECK(
      status == 0,
      "Could not open input file: " + videoFilePath + " " +
          getFFMPEGErrorStringFromErrorCode(status));
  TORCH_CHECK(rawContext != nullptr);
  formatContext_.reset(rawContext);

  initializeDecoder();
}

VideoDecoder::VideoDecoder(
    std::unique_ptr<AVIOContextHolder> context,
    SeekMode seekMode)
    : seekMode_(seekMode), avioContextHolder_(std::move(context)) {
  setFFmpegLogLevel();

  TORCH_CHECK(avioContextHolder_, "Context holder cannot be null");

  // Because FFmpeg requires a reference to a pointer in the call to open, we
  // can't use a unique pointer here. Note that means we must call free if open
  // fails.
  AVFormatContext* rawContext = avformat_alloc_context();
  TORCH_CHECK(rawContext != nullptr, "Unable to alloc avformat context");

  rawContext->pb = avioContextHolder_->getAVIOContext();
  int status = avformat_open_input(&rawContext, nullptr, nullptr, nullptr);
  if (status != 0) {
    avformat_free_context(rawContext);
    TORCH_CHECK(
        false,
        "Failed to open input buffer: " +
            getFFMPEGErrorStringFromErrorCode(status));
  }

  formatContext_.reset(rawContext);

  initializeDecoder();
}

VideoDecoder::~VideoDecoder() {
  for (auto& [streamIndex, streamInfo] : streamInfos_) {
    auto& device = streamInfo.videoStreamOptions.device;
    if (device) {
      device->releaseContext(streamInfo.codecContext.get());
    }
  }
}

void VideoDecoder::initializeDecoder() {
  TORCH_CHECK(!initialized_, "Attempted double initialization.");

  // In principle, the AVFormatContext should be filled in by the call to
  // avformat_open_input() which reads the header. However, some formats do not
  // store enough info in the header, so we call avformat_find_stream_info()
  // which decodes a few frames to get missing info. For more, see:
  //   https://ffmpeg.org/doxygen/7.0/group__lavf__decoding.html
  int status = avformat_find_stream_info(formatContext_.get(), nullptr);
  if (status < 0) {
    throw std::runtime_error(
        "Failed to find stream info: " +
        getFFMPEGErrorStringFromErrorCode(status));
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
    if (avStream->start_time != AV_NOPTS_VALUE) {
      streamMetadata.beginStreamFromHeader =
          av_q2d(avStream->time_base) * avStream->start_time;
    }

    if (avStream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      double fps = av_q2d(avStream->r_frame_rate);
      if (fps > 0) {
        streamMetadata.averageFps = fps;
      }
      containerMetadata_.numVideoStreams++;
    } else if (avStream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
      AVSampleFormat format =
          static_cast<AVSampleFormat>(avStream->codecpar->format);

      // If the AVSampleFormat is not recognized, we get back nullptr. We have
      // to make sure we don't initialize a std::string with nullptr. There's
      // nothing to do on the else branch because we're already using an
      // optional; it'll just remain empty.
      const char* rawSampleFormat = av_get_sample_fmt_name(format);
      if (rawSampleFormat != nullptr) {
        streamMetadata.sampleFormat = std::string(rawSampleFormat);
      }
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

void VideoDecoder::setFFmpegLogLevel() {
  auto logLevel = AV_LOG_QUIET;
  const char* logLevelEnv = std::getenv("TORCHCODEC_FFMPEG_LOG_LEVEL");
  if (logLevelEnv != nullptr) {
    if (std::strcmp(logLevelEnv, "QUIET") == 0) {
      logLevel = AV_LOG_QUIET;
    } else if (std::strcmp(logLevelEnv, "PANIC") == 0) {
      logLevel = AV_LOG_PANIC;
    } else if (std::strcmp(logLevelEnv, "FATAL") == 0) {
      logLevel = AV_LOG_FATAL;
    } else if (std::strcmp(logLevelEnv, "ERROR") == 0) {
      logLevel = AV_LOG_ERROR;
    } else if (std::strcmp(logLevelEnv, "WARNING") == 0) {
      logLevel = AV_LOG_WARNING;
    } else if (std::strcmp(logLevelEnv, "INFO") == 0) {
      logLevel = AV_LOG_INFO;
    } else if (std::strcmp(logLevelEnv, "VERBOSE") == 0) {
      logLevel = AV_LOG_VERBOSE;
    } else if (std::strcmp(logLevelEnv, "DEBUG") == 0) {
      logLevel = AV_LOG_DEBUG;
    } else if (std::strcmp(logLevelEnv, "TRACE") == 0) {
      logLevel = AV_LOG_TRACE;
    } else {
      TORCH_CHECK(
          false,
          "Invalid TORCHCODEC_FFMPEG_LOG_LEVEL: ",
          logLevelEnv,
          ". Use e.g. 'QUIET', 'PANIC', 'VERBOSE', etc.");
    }
  }
  av_log_set_level(logLevel);
}

int VideoDecoder::getBestStreamIndex(AVMediaType mediaType) {
  AVCodecOnlyUseForCallingAVFindBestStream avCodec = nullptr;
  int streamIndex =
      av_find_best_stream(formatContext_.get(), mediaType, -1, -1, &avCodec, 0);
  return streamIndex;
}

// --------------------------------------------------------------------------
// VIDEO METADATA QUERY API
// --------------------------------------------------------------------------

void VideoDecoder::scanFileAndUpdateMetadataAndIndex() {
  if (scannedAllStreams_) {
    return;
  }

  for (unsigned int i = 0; i < formatContext_->nb_streams; ++i) {
    // We want to scan and update the metadata of all streams.
    TORCH_CHECK(
        formatContext_->streams[i]->discard != AVDISCARD_ALL,
        "Did you add a stream before you called for a scan?");
  }

  AutoAVPacket autoAVPacket;
  while (true) {
    ReferenceAVPacket packet(autoAVPacket);

    // av_read_frame is a misleading name: it gets the next **packet**.
    int status = av_read_frame(formatContext_.get(), packet.get());

    if (status == AVERROR_EOF) {
      break;
    }

    if (status != AVSUCCESS) {
      throw std::runtime_error(
          "Failed to read frame from input file: " +
          getFFMPEGErrorStringFromErrorCode(status));
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
  int status = avformat_seek_file(formatContext_.get(), 0, INT64_MIN, 0, 0, 0);
  if (status < 0) {
    throw std::runtime_error(
        "Could not seek file to pts=0: " +
        getFFMPEGErrorStringFromErrorCode(status));
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

VideoDecoder::ContainerMetadata VideoDecoder::getContainerMetadata() const {
  return containerMetadata_;
}

torch::Tensor VideoDecoder::getKeyFrameIndices() {
  validateActiveStream(AVMEDIA_TYPE_VIDEO);
  validateScannedAllStreams("getKeyFrameIndices");

  const std::vector<FrameInfo>& keyFrames =
      streamInfos_[activeStreamIndex_].keyFrames;
  torch::Tensor keyFrameIndices =
      torch::empty({static_cast<int64_t>(keyFrames.size())}, {torch::kInt64});
  for (size_t i = 0; i < keyFrames.size(); ++i) {
    keyFrameIndices[i] = keyFrames[i].frameIndex;
  }

  return keyFrameIndices;
}

// --------------------------------------------------------------------------
// ADDING STREAMS API
// --------------------------------------------------------------------------

void VideoDecoder::addStream(
    int streamIndex,
    AVMediaType mediaType,
    DeviceInterface* device,
    std::optional<int> ffmpegThreadCount) {
  TORCH_CHECK(
      activeStreamIndex_ == NO_ACTIVE_STREAM,
      "Can only add one single stream.");
  TORCH_CHECK(
      mediaType == AVMEDIA_TYPE_VIDEO || mediaType == AVMEDIA_TYPE_AUDIO,
      "Can only add video or audio streams.");
  TORCH_CHECK(formatContext_.get() != nullptr);

  AVCodecOnlyUseForCallingAVFindBestStream avCodec = nullptr;

  activeStreamIndex_ = av_find_best_stream(
      formatContext_.get(), mediaType, streamIndex, -1, &avCodec, 0);

  if (activeStreamIndex_ < 0) {
    throw std::invalid_argument(
        "No valid stream found in input file. Is " +
        std::to_string(streamIndex) + " of the desired media type?");
  }

  TORCH_CHECK(avCodec != nullptr);

  StreamInfo& streamInfo = streamInfos_[activeStreamIndex_];
  streamInfo.streamIndex = activeStreamIndex_;
  streamInfo.timeBase = formatContext_->streams[activeStreamIndex_]->time_base;
  streamInfo.stream = formatContext_->streams[activeStreamIndex_];
  streamInfo.avMediaType = mediaType;

  // This should never happen, checking just to be safe.
  TORCH_CHECK(
      streamInfo.stream->codecpar->codec_type == mediaType,
      "FFmpeg found stream with index ",
      activeStreamIndex_,
      " which is of the wrong media type.");

  // TODO_CODE_QUALITY it's pretty meh to have a video-specific logic within
  // addStream() which is supposed to be generic
  if (mediaType == AVMEDIA_TYPE_VIDEO) {
    if (device) {
      avCodec = makeAVCodecOnlyUseForCallingAVFindBestStream(
          device->findCodec(streamInfo.stream->codecpar->codec_id)
              .value_or(avCodec));
    }
  }

  AVCodecContext* codecContext = avcodec_alloc_context3(avCodec);
  TORCH_CHECK(codecContext != nullptr);
  streamInfo.codecContext.reset(codecContext);

  int retVal = avcodec_parameters_to_context(
      streamInfo.codecContext.get(), streamInfo.stream->codecpar);
  TORCH_CHECK_EQ(retVal, AVSUCCESS);

  streamInfo.codecContext->thread_count = ffmpegThreadCount.value_or(0);
  streamInfo.codecContext->pkt_timebase = streamInfo.stream->time_base;

  // TODO_CODE_QUALITY same as above.
  if (mediaType == AVMEDIA_TYPE_VIDEO) {
    if (device) {
      device->initializeContext(codecContext);
    }
  }

  retVal = avcodec_open2(streamInfo.codecContext.get(), avCodec, nullptr);
  if (retVal < AVSUCCESS) {
    throw std::invalid_argument(getFFMPEGErrorStringFromErrorCode(retVal));
  }

  codecContext->time_base = streamInfo.stream->time_base;
  containerMetadata_.allStreamMetadata[activeStreamIndex_].codecName =
      std::string(avcodec_get_name(codecContext->codec_id));

  // We will only need packets from the active stream, so we tell FFmpeg to
  // discard packets from the other streams. Note that av_read_frame() may still
  // return some of those un-desired packet under some conditions, so it's still
  // important to discard/demux correctly in the inner decoding loop.
  for (unsigned int i = 0; i < formatContext_->nb_streams; ++i) {
    if (i != static_cast<unsigned int>(activeStreamIndex_)) {
      formatContext_->streams[i]->discard = AVDISCARD_ALL;
    }
  }
}

void VideoDecoder::addVideoStream(
    int streamIndex,
    const VideoStreamOptions& videoStreamOptions) {
  addStream(
      streamIndex,
      AVMEDIA_TYPE_VIDEO,
      videoStreamOptions.device.get(),
      videoStreamOptions.ffmpegThreadCount);

  auto& streamMetadata =
      containerMetadata_.allStreamMetadata[activeStreamIndex_];

  if (seekMode_ == SeekMode::approximate &&
      !streamMetadata.averageFps.has_value()) {
    throw std::runtime_error(
        "Seek mode is approximate, but stream " +
        std::to_string(activeStreamIndex_) +
        " does not have an average fps in its metadata.");
  }

  auto& streamInfo = streamInfos_[activeStreamIndex_];
  streamInfo.videoStreamOptions = videoStreamOptions;

  streamMetadata.width = streamInfo.codecContext->width;
  streamMetadata.height = streamInfo.codecContext->height;

  // By default, we want to use swscale for color conversion because it is
  // faster. However, it has width requirements, so we may need to fall back
  // to filtergraph. We also need to respect what was requested from the
  // options; we respect the options unconditionally, so it's possible for
  // swscale's width requirements to be violated. We don't expose the ability to
  // choose color conversion library publicly; we only use this ability
  // internally.
  int width = videoStreamOptions.width.value_or(streamInfo.codecContext->width);

  // swscale requires widths to be multiples of 32:
  // https://stackoverflow.com/questions/74351955/turn-off-sw-scale-conversion-to-planar-yuv-32-byte-alignment-requirements
  // so we fall back to filtergraph if the width is not a multiple of 32.
  auto defaultLibrary = (width % 32 == 0)
      ? VideoDecoder::ColorConversionLibrary::SWSCALE
      : VideoDecoder::ColorConversionLibrary::FILTERGRAPH;

  streamInfo.colorConversionLibrary =
      videoStreamOptions.colorConversionLibrary.value_or(defaultLibrary);
}

void VideoDecoder::addAudioStream(
    int streamIndex,
    const AudioStreamOptions& audioStreamOptions) {
  TORCH_CHECK(
      seekMode_ == SeekMode::approximate,
      "seek_mode must be 'approximate' for audio streams.");

  addStream(streamIndex, AVMEDIA_TYPE_AUDIO);

  auto& streamInfo = streamInfos_[activeStreamIndex_];
  streamInfo.audioStreamOptions = audioStreamOptions;

  auto& streamMetadata =
      containerMetadata_.allStreamMetadata[activeStreamIndex_];
  streamMetadata.sampleRate =
      static_cast<int64_t>(streamInfo.codecContext->sample_rate);
  streamMetadata.numChannels =
      static_cast<int64_t>(getNumChannels(streamInfo.codecContext));

  // FFmpeg docs say that the decoder will try to decode natively in this
  // format, if it can. Docs don't say what the decoder does when it doesn't
  // support that format, but it looks like it does nothing, so this probably
  // doesn't hurt.
  streamInfo.codecContext->request_sample_fmt = AV_SAMPLE_FMT_FLTP;
}

// --------------------------------------------------------------------------
// HIGH-LEVEL DECODING ENTRY-POINTS
// --------------------------------------------------------------------------

VideoDecoder::FrameOutput VideoDecoder::getNextFrame() {
  auto output = getNextFrameInternal();
  if (streamInfos_[activeStreamIndex_].avMediaType == AVMEDIA_TYPE_VIDEO) {
    output.data = maybePermuteHWC2CHW(output.data);
  }
  return output;
}

VideoDecoder::FrameOutput VideoDecoder::getNextFrameInternal(
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  validateActiveStream();
  UniqueAVFrame avFrame = decodeAVFrame(
      [this](const UniqueAVFrame& avFrame) { return avFrame->pts >= cursor_; });
  return convertAVFrameToFrameOutput(avFrame, preAllocatedOutputTensor);
}

VideoDecoder::FrameOutput VideoDecoder::getFrameAtIndex(int64_t frameIndex) {
  auto frameOutput = getFrameAtIndexInternal(frameIndex);
  frameOutput.data = maybePermuteHWC2CHW(frameOutput.data);
  return frameOutput;
}

VideoDecoder::FrameOutput VideoDecoder::getFrameAtIndexInternal(
    int64_t frameIndex,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  validateActiveStream(AVMEDIA_TYPE_VIDEO);

  const auto& streamInfo = streamInfos_[activeStreamIndex_];
  const auto& streamMetadata =
      containerMetadata_.allStreamMetadata[activeStreamIndex_];
  validateFrameIndex(streamMetadata, frameIndex);

  int64_t pts = getPts(frameIndex);
  setCursorPtsInSeconds(ptsToSeconds(pts, streamInfo.timeBase));
  return getNextFrameInternal(preAllocatedOutputTensor);
}

VideoDecoder::FrameBatchOutput VideoDecoder::getFramesAtIndices(
    const std::vector<int64_t>& frameIndices) {
  validateActiveStream(AVMEDIA_TYPE_VIDEO);

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
      containerMetadata_.allStreamMetadata[activeStreamIndex_];
  const auto& streamInfo = streamInfos_[activeStreamIndex_];
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
          indexInVideo, frameBatchOutput.data[indexInOutput]);
      frameBatchOutput.ptsSeconds[indexInOutput] = frameOutput.ptsSeconds;
      frameBatchOutput.durationSeconds[indexInOutput] =
          frameOutput.durationSeconds;
    }
    previousIndexInVideo = indexInVideo;
  }
  frameBatchOutput.data = maybePermuteHWC2CHW(frameBatchOutput.data);
  return frameBatchOutput;
}

VideoDecoder::FrameBatchOutput
VideoDecoder::getFramesInRange(int64_t start, int64_t stop, int64_t step) {
  validateActiveStream(AVMEDIA_TYPE_VIDEO);

  const auto& streamMetadata =
      containerMetadata_.allStreamMetadata[activeStreamIndex_];
  const auto& streamInfo = streamInfos_[activeStreamIndex_];
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
        getFrameAtIndexInternal(i, frameBatchOutput.data[f]);
    frameBatchOutput.ptsSeconds[f] = frameOutput.ptsSeconds;
    frameBatchOutput.durationSeconds[f] = frameOutput.durationSeconds;
  }
  frameBatchOutput.data = maybePermuteHWC2CHW(frameBatchOutput.data);
  return frameBatchOutput;
}

VideoDecoder::FrameOutput VideoDecoder::getFramePlayedAt(double seconds) {
  validateActiveStream(AVMEDIA_TYPE_VIDEO);
  StreamInfo& streamInfo = streamInfos_[activeStreamIndex_];
  double frameStartTime =
      ptsToSeconds(streamInfo.lastDecodedAvFramePts, streamInfo.timeBase);
  double frameEndTime = ptsToSeconds(
      streamInfo.lastDecodedAvFramePts + streamInfo.lastDecodedAvFrameDuration,
      streamInfo.timeBase);
  if (seconds >= frameStartTime && seconds < frameEndTime) {
    // We are in the same frame as the one we just returned. However, since we
    // don't cache it locally, we have to rewind back.
    seconds = frameStartTime;
  }

  setCursorPtsInSeconds(seconds);
  UniqueAVFrame avFrame =
      decodeAVFrame([seconds, this](const UniqueAVFrame& avFrame) {
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
  FrameOutput frameOutput = convertAVFrameToFrameOutput(avFrame);
  frameOutput.data = maybePermuteHWC2CHW(frameOutput.data);
  return frameOutput;
}

VideoDecoder::FrameBatchOutput VideoDecoder::getFramesPlayedAt(
    const std::vector<double>& timestamps) {
  validateActiveStream(AVMEDIA_TYPE_VIDEO);

  const auto& streamMetadata =
      containerMetadata_.allStreamMetadata[activeStreamIndex_];

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

    frameIndices[i] = secondsToIndexLowerBound(frameSeconds);
  }

  return getFramesAtIndices(frameIndices);
}

VideoDecoder::FrameBatchOutput VideoDecoder::getFramesPlayedInRange(
    double startSeconds,
    double stopSeconds) {
  validateActiveStream(AVMEDIA_TYPE_VIDEO);
  const auto& streamMetadata =
      containerMetadata_.allStreamMetadata[activeStreamIndex_];
  TORCH_CHECK(
      startSeconds <= stopSeconds,
      "Start seconds (" + std::to_string(startSeconds) +
          ") must be less than or equal to stop seconds (" +
          std::to_string(stopSeconds) + ".");

  const auto& streamInfo = streamInfos_[activeStreamIndex_];
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
    frameBatchOutput.data = maybePermuteHWC2CHW(frameBatchOutput.data);
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

  int64_t startFrameIndex = secondsToIndexLowerBound(startSeconds);
  int64_t stopFrameIndex = secondsToIndexUpperBound(stopSeconds);
  int64_t numFrames = stopFrameIndex - startFrameIndex;

  FrameBatchOutput frameBatchOutput(
      numFrames, videoStreamOptions, streamMetadata);
  for (int64_t i = startFrameIndex, f = 0; i < stopFrameIndex; ++i, ++f) {
    FrameOutput frameOutput =
        getFrameAtIndexInternal(i, frameBatchOutput.data[f]);
    frameBatchOutput.ptsSeconds[f] = frameOutput.ptsSeconds;
    frameBatchOutput.durationSeconds[f] = frameOutput.durationSeconds;
  }
  frameBatchOutput.data = maybePermuteHWC2CHW(frameBatchOutput.data);

  return frameBatchOutput;
}

// Note [Audio Decoding Design]
// This note explains why audio decoding is implemented the way it is, and why
// it inherently differs from video decoding.
//
// Like for video, FFmpeg exposes the concept of a frame for audio streams. An
// audio frame is a contiguous sequence of samples, where a sample consists of
// `numChannels` values. An audio frame, or a sequence thereof, is always
// converted into a tensor of shape `(numChannels, numSamplesPerChannel)`.
//
// The notion of 'frame' in audio isn't what users want to interact with. Users
// want to interact with samples. The C++ and core APIs return frames, because
// we want those to be close to FFmpeg concepts, but the higher-level public
// APIs expose samples. As a result:
// - We don't expose index-based APIs for audio, because that would mean
//   exposing the concept of audio frame. For now, we think exposing time-based
//   APIs is more natural.
// - We never perform a scan for audio streams. We don't need to, since we won't
//   be converting timestamps to indices. That's why we enforce the seek_mode
//   to be "approximate" (which is slightly misleading, because technically the
//   output samples will be at their exact positions. But this incongruence is
//   only exposed at the C++/core private levels).
//
// Audio frames are of variable dimensions: in the same stream, a frame can
// contain 1024 samples and the next one may contain 512 [1]. This makes it
// impossible to stack audio frames in the same way we can stack video frames.
// This is one of the main reasons we cannot reuse the same pre-allocation logic
// we have for videos in getFramesPlayedInRange(): pre-allocating a batch
// requires constant (and known) frame dimensions. That's also why
// *concatenated* along the samples dimension, not stacked.
//
// [IMPORTANT!] There is one key invariant that we must respect when decoding
// audio frames:
//
// BEFORE DECODING FRAME i, WE MUST DECODE ALL FRAMES j < i.
//
// Always. Why? We don't know. What we know is that if we don't, we get clipped,
// incorrect audio as output [2]. All other (correct) libraries like TorchAudio
// or Decord do something similar, whether it was intended or not. This has a
// few implications:
// - The **only** place we're allowed to seek to in an audio stream is the
//   stream's beginning. This ensures that if we need a frame, we'll have
//   decoded all previous frames.
// - Because of that, we don't allow the public APIs to seek. Public APIs can
//   call next() and `getFramesPlayedInRangeAudio()`, but they cannot manually
//   seek.
// - We try not to seek, when we can avoid it. Typically if the next frame we
//   need is in the future, we don't seek back to the beginning, we just decode
//   all the frames in-between.
//
// [2] If you're brave and curious, you can read the long "Seek offset for
// audio" note in https://github.com/pytorch/torchcodec/pull/507/files, which
// sums up past (and failed) attemps at working around this issue.
VideoDecoder::AudioFramesOutput VideoDecoder::getFramesPlayedInRangeAudio(
    double startSeconds,
    std::optional<double> stopSecondsOptional) {
  validateActiveStream(AVMEDIA_TYPE_AUDIO);

  if (stopSecondsOptional.has_value()) {
    TORCH_CHECK(
        startSeconds <= *stopSecondsOptional,
        "Start seconds (" + std::to_string(startSeconds) +
            ") must be less than or equal to stop seconds (" +
            std::to_string(*stopSecondsOptional) + ").");
  }

  if (stopSecondsOptional.has_value() && startSeconds == *stopSecondsOptional) {
    // For consistency with video
    return AudioFramesOutput{torch::empty({0, 0}), 0.0};
  }

  StreamInfo& streamInfo = streamInfos_[activeStreamIndex_];

  auto startPts = secondsToClosestPts(startSeconds, streamInfo.timeBase);
  if (startPts < streamInfo.lastDecodedAvFramePts +
          streamInfo.lastDecodedAvFrameDuration) {
    // If we need to seek backwards, then we have to seek back to the beginning
    // of the stream.
    // See [Audio Decoding Design].
    setCursor(INT64_MIN);
  }

  // TODO-AUDIO Pre-allocate a long-enough tensor instead of creating a vec +
  // cat(). This would save a copy. We know the duration of the output and the
  // sample rate, so in theory we know the number of output samples.
  std::vector<torch::Tensor> frames;

  std::optional<double> firstFramePtsSeconds = std::nullopt;
  auto stopPts = stopSecondsOptional.has_value()
      ? secondsToClosestPts(*stopSecondsOptional, streamInfo.timeBase)
      : INT64_MAX;
  auto finished = false;
  while (!finished) {
    try {
      UniqueAVFrame avFrame =
          decodeAVFrame([startPts](const UniqueAVFrame& avFrame) {
            return startPts < avFrame->pts + getDuration(avFrame);
          });
      auto frameOutput = convertAVFrameToFrameOutput(avFrame);
      if (!firstFramePtsSeconds.has_value()) {
        firstFramePtsSeconds = frameOutput.ptsSeconds;
      }
      frames.push_back(frameOutput.data);
    } catch (const EndOfFileException& e) {
      finished = true;
    }

    // If stopSeconds is in [begin, end] of the last decoded frame, we should
    // stop decoding more frames. Note that if we were to use [begin, end),
    // which may seem more natural, then we would decode the frame starting at
    // stopSeconds, which isn't what we want!
    auto lastDecodedAvFrameEnd = streamInfo.lastDecodedAvFramePts +
        streamInfo.lastDecodedAvFrameDuration;
    finished |= (streamInfo.lastDecodedAvFramePts) <= stopPts &&
        (stopPts <= lastDecodedAvFrameEnd);
  }

  auto lastSamples = maybeFlushSwrBuffers();
  if (lastSamples.has_value()) {
    frames.push_back(*lastSamples);
  }

  TORCH_CHECK(
      frames.size() > 0 && firstFramePtsSeconds.has_value(),
      "No audio frames were decoded. ",
      "This is probably because start_seconds is too high? ",
      "Current value is ",
      startSeconds);

  return AudioFramesOutput{torch::cat(frames, 1), *firstFramePtsSeconds};
}

// --------------------------------------------------------------------------
// SEEKING APIs
// --------------------------------------------------------------------------

void VideoDecoder::setCursorPtsInSeconds(double seconds) {
  // We don't allow public audio decoding APIs to seek, see [Audio Decoding
  // Design]
  validateActiveStream(AVMEDIA_TYPE_VIDEO);
  setCursor(
      secondsToClosestPts(seconds, streamInfos_[activeStreamIndex_].timeBase));
}

void VideoDecoder::setCursor(int64_t pts) {
  cursorWasJustSet_ = true;
  cursor_ = pts;
}

/*
Videos have I frames and non-I frames (P and B frames). Non-I frames need data
from the previous I frame to be decoded.

Imagine the cursor is at a random frame with PTS=lastDecodedAvFramePts (x for
brevity) and we wish to seek to a user-specified PTS=y.

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
bool VideoDecoder::canWeAvoidSeeking() const {
  const StreamInfo& streamInfo = streamInfos_.at(activeStreamIndex_);
  if (streamInfo.avMediaType == AVMEDIA_TYPE_AUDIO) {
    // For audio, we only need to seek if a backwards seek was requested within
    // getFramesPlayedInRangeAudio(), when setCursorPtsInSeconds() was called.
    // For more context, see [Audio Decoding Design]
    return !cursorWasJustSet_;
  }
  int64_t lastDecodedAvFramePts =
      streamInfos_.at(activeStreamIndex_).lastDecodedAvFramePts;
  if (cursor_ < lastDecodedAvFramePts) {
    // We can never skip a seek if we are seeking backwards.
    return false;
  }
  if (lastDecodedAvFramePts == cursor_) {
    // We are seeking to the exact same frame as we are currently at. Without
    // caching we have to rewind back and decode the frame again.
    // TODO: https://github.com/pytorch-labs/torchcodec/issues/84 we could
    // implement caching.
    return false;
  }
  // We are seeking forwards.
  // We can only skip a seek if both lastDecodedAvFramePts and
  // cursor_ share the same keyframe.
  int lastDecodedAvFrameIndex = getKeyFrameIndexForPts(lastDecodedAvFramePts);
  int targetKeyFrameIndex = getKeyFrameIndexForPts(cursor_);
  return lastDecodedAvFrameIndex >= 0 && targetKeyFrameIndex >= 0 &&
      lastDecodedAvFrameIndex == targetKeyFrameIndex;
}

// This method looks at currentPts and desiredPts and seeks in the
// AVFormatContext if it is needed. We can skip seeking in certain cases. See
// the comment of canWeAvoidSeeking() for details.
void VideoDecoder::maybeSeekToBeforeDesiredPts() {
  validateActiveStream();
  StreamInfo& streamInfo = streamInfos_[activeStreamIndex_];

  decodeStats_.numSeeksAttempted++;
  if (canWeAvoidSeeking()) {
    decodeStats_.numSeeksSkipped++;
    return;
  }

  int64_t desiredPts = cursor_;

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

  int status = avformat_seek_file(
      formatContext_.get(),
      streamInfo.streamIndex,
      INT64_MIN,
      desiredPts,
      desiredPts,
      0);
  if (status < 0) {
    throw std::runtime_error(
        "Could not seek file to pts=" + std::to_string(desiredPts) + ": " +
        getFFMPEGErrorStringFromErrorCode(status));
  }
  decodeStats_.numFlushes++;
  avcodec_flush_buffers(streamInfo.codecContext.get());
}

// --------------------------------------------------------------------------
// LOW-LEVEL DECODING
// --------------------------------------------------------------------------

UniqueAVFrame VideoDecoder::decodeAVFrame(
    std::function<bool(const UniqueAVFrame&)> filterFunction) {
  validateActiveStream();

  resetDecodeStats();

  if (cursorWasJustSet_) {
    maybeSeekToBeforeDesiredPts();
    cursorWasJustSet_ = false;
  }

  StreamInfo& streamInfo = streamInfos_[activeStreamIndex_];

  // Need to get the next frame or error from PopFrame.
  UniqueAVFrame avFrame(av_frame_alloc());
  AutoAVPacket autoAVPacket;
  int status = AVSUCCESS;
  bool reachedEOF = false;
  while (true) {
    status =
        avcodec_receive_frame(streamInfo.codecContext.get(), avFrame.get());

    if (status != AVSUCCESS && status != AVERROR(EAGAIN)) {
      // Non-retriable error
      break;
    }

    decodeStats_.numFramesReceivedByDecoder++;
    // Is this the kind of frame we're looking for?
    if (status == AVSUCCESS && filterFunction(avFrame)) {
      // Yes, this is the frame we'll return; break out of the decoding loop.
      break;
    } else if (status == AVSUCCESS) {
      // No, but we received a valid frame - just not the kind we're looking
      // for. The logic below will read packets and send them to the decoder.
      // But since we did just receive a frame, we should skip reading more
      // packets and sending them to the decoder and just try to receive more
      // frames from the decoder.
      continue;
    }

    if (reachedEOF) {
      // We don't have any more packets to receive. So keep on pulling frames
      // from its internal buffers.
      continue;
    }

    // We still haven't found the frame we're looking for. So let's read more
    // packets and send them to the decoder.
    ReferenceAVPacket packet(autoAVPacket);
    do {
      status = av_read_frame(formatContext_.get(), packet.get());
      decodeStats_.numPacketsRead++;

      if (status == AVERROR_EOF) {
        // End of file reached. We must drain the codec by sending a nullptr
        // packet.
        status = avcodec_send_packet(
            streamInfo.codecContext.get(),
            /*avpkt=*/nullptr);
        if (status < AVSUCCESS) {
          throw std::runtime_error(
              "Could not flush decoder: " +
              getFFMPEGErrorStringFromErrorCode(status));
        }

        reachedEOF = true;
        break;
      }

      if (status < AVSUCCESS) {
        throw std::runtime_error(
            "Could not read frame from input file: " +
            getFFMPEGErrorStringFromErrorCode(status));
      }
    } while (packet->stream_index != activeStreamIndex_);

    if (reachedEOF) {
      // We don't have any more packets to send to the decoder. So keep on
      // pulling frames from its internal buffers.
      continue;
    }

    // We got a valid packet. Send it to the decoder, and we'll receive it in
    // the next iteration.
    status = avcodec_send_packet(streamInfo.codecContext.get(), packet.get());
    if (status < AVSUCCESS) {
      throw std::runtime_error(
          "Could not push packet to decoder: " +
          getFFMPEGErrorStringFromErrorCode(status));
    }

    decodeStats_.numPacketsSentToDecoder++;
  }

  if (status < AVSUCCESS) {
    if (reachedEOF || status == AVERROR_EOF) {
      throw VideoDecoder::EndOfFileException(
          "Requested next frame while there are no more frames left to "
          "decode.");
    }
    throw std::runtime_error(
        "Could not receive frame from decoder: " +
        getFFMPEGErrorStringFromErrorCode(status));
  }

  // Note that we don't flush the decoder when we reach EOF (even though that's
  // mentioned in https://ffmpeg.org/doxygen/trunk/group__lavc__encdec.html).
  // This is because we may have packets internally in the decoder that we
  // haven't received as frames. Eventually we will either hit AVERROR_EOF from
  // av_receive_frame() or the user will have seeked to a different location in
  // the file and that will flush the decoder.
  streamInfo.lastDecodedAvFramePts = avFrame->pts;
  streamInfo.lastDecodedAvFrameDuration = getDuration(avFrame);

  return avFrame;
}

// --------------------------------------------------------------------------
// AVFRAME <-> FRAME OUTPUT CONVERSION
// --------------------------------------------------------------------------

VideoDecoder::FrameOutput VideoDecoder::convertAVFrameToFrameOutput(
    UniqueAVFrame& avFrame,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  // Convert the frame to tensor.
  FrameOutput frameOutput;
  auto& streamInfo = streamInfos_[activeStreamIndex_];
  frameOutput.ptsSeconds = ptsToSeconds(
      avFrame->pts, formatContext_->streams[activeStreamIndex_]->time_base);
  frameOutput.durationSeconds = ptsToSeconds(
      getDuration(avFrame),
      formatContext_->streams[activeStreamIndex_]->time_base);
  if (streamInfo.avMediaType == AVMEDIA_TYPE_AUDIO) {
    convertAudioAVFrameToFrameOutputOnCPU(avFrame, frameOutput);
  } else if (!streamInfo.videoStreamOptions.device) {
    convertAVFrameToFrameOutputOnCPU(
        avFrame, frameOutput, preAllocatedOutputTensor);
  } else if (streamInfo.videoStreamOptions.device) {
    streamInfo.videoStreamOptions.device->convertAVFrameToFrameOutput(
        streamInfo.videoStreamOptions,
        avFrame,
        frameOutput,
        preAllocatedOutputTensor);
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
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  auto& streamInfo = streamInfos_[activeStreamIndex_];

  auto frameDims = getHeightAndWidthFromOptionsOrAVFrame(
      streamInfo.videoStreamOptions, avFrame);
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
        convertAVFrameToTensorUsingSwsScale(avFrame, outputTensor);
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
    outputTensor = convertAVFrameToTensorUsingFilterGraph(avFrame);

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

int VideoDecoder::convertAVFrameToTensorUsingSwsScale(
    const UniqueAVFrame& avFrame,
    torch::Tensor& outputTensor) {
  StreamInfo& activeStreamInfo = streamInfos_[activeStreamIndex_];
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
    const UniqueAVFrame& avFrame) {
  FilterGraphContext& filterGraphContext =
      streamInfos_[activeStreamIndex_].filterGraphContext;
  int status =
      av_buffersrc_write_frame(filterGraphContext.sourceContext, avFrame.get());
  if (status < AVSUCCESS) {
    throw std::runtime_error("Failed to add frame to buffer source context");
  }

  UniqueAVFrame filteredAVFrame(av_frame_alloc());
  status = av_buffersink_get_frame(
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

void VideoDecoder::convertAudioAVFrameToFrameOutputOnCPU(
    UniqueAVFrame& srcAVFrame,
    FrameOutput& frameOutput) {
  AVSampleFormat sourceSampleFormat =
      static_cast<AVSampleFormat>(srcAVFrame->format);
  AVSampleFormat desiredSampleFormat = AV_SAMPLE_FMT_FLTP;

  int sourceSampleRate = srcAVFrame->sample_rate;
  int desiredSampleRate =
      streamInfos_[activeStreamIndex_].audioStreamOptions.sampleRate.value_or(
          sourceSampleRate);

  bool mustConvert =
      (sourceSampleFormat != desiredSampleFormat ||
       sourceSampleRate != desiredSampleRate);

  UniqueAVFrame convertedAVFrame;
  if (mustConvert) {
    convertedAVFrame = convertAudioAVFrameSampleFormatAndSampleRate(
        srcAVFrame,
        sourceSampleFormat,
        desiredSampleFormat,
        sourceSampleRate,
        desiredSampleRate);
  }
  const UniqueAVFrame& avFrame = mustConvert ? convertedAVFrame : srcAVFrame;

  AVSampleFormat format = static_cast<AVSampleFormat>(avFrame->format);
  TORCH_CHECK(
      format == desiredSampleFormat,
      "Something went wrong, the frame didn't get converted to the desired format. ",
      "Desired format = ",
      av_get_sample_fmt_name(desiredSampleFormat),
      "source format = ",
      av_get_sample_fmt_name(format));

  auto numSamples = avFrame->nb_samples; // per channel
  auto numChannels = getNumChannels(avFrame);

  frameOutput.data = torch::empty({numChannels, numSamples}, torch::kFloat32);

  if (numSamples > 0) {
    uint8_t* outputChannelData =
        static_cast<uint8_t*>(frameOutput.data.data_ptr());
    auto numBytesPerChannel = numSamples * av_get_bytes_per_sample(format);
    for (auto channel = 0; channel < numChannels;
         ++channel, outputChannelData += numBytesPerChannel) {
      memcpy(
          outputChannelData,
          avFrame->extended_data[channel],
          numBytesPerChannel);
    }
  }
}

UniqueAVFrame VideoDecoder::convertAudioAVFrameSampleFormatAndSampleRate(
    const UniqueAVFrame& srcAVFrame,
    AVSampleFormat sourceSampleFormat,
    AVSampleFormat desiredSampleFormat,
    int sourceSampleRate,
    int desiredSampleRate) {
  auto& streamInfo = streamInfos_[activeStreamIndex_];

  if (!streamInfo.swrContext) {
    createSwrContext(
        streamInfo,
        sourceSampleFormat,
        desiredSampleFormat,
        sourceSampleRate,
        desiredSampleRate);
  }

  UniqueAVFrame convertedAVFrame(av_frame_alloc());
  TORCH_CHECK(
      convertedAVFrame,
      "Could not allocate frame for sample format conversion.");

  setChannelLayout(convertedAVFrame, srcAVFrame);
  convertedAVFrame->format = static_cast<int>(desiredSampleFormat);
  convertedAVFrame->sample_rate = desiredSampleRate;
  if (sourceSampleRate != desiredSampleRate) {
    // Note that this is an upper bound on the number of output samples.
    // `swr_convert()` will likely not fill convertedAVFrame with that many
    // samples if sample rate conversion is needed. It will buffer the last few
    // ones because those require future samples. That's also why we reset
    // nb_samples after the call to `swr_convert()`.
    // We could also use `swr_get_out_samples()` to determine the number of
    // output samples, but empirically `av_rescale_rnd()` seems to provide a
    // tighter bound.
    convertedAVFrame->nb_samples = av_rescale_rnd(
        swr_get_delay(streamInfo.swrContext.get(), sourceSampleRate) +
            srcAVFrame->nb_samples,
        desiredSampleRate,
        sourceSampleRate,
        AV_ROUND_UP);
  } else {
    convertedAVFrame->nb_samples = srcAVFrame->nb_samples;
  }

  auto status = av_frame_get_buffer(convertedAVFrame.get(), 0);
  TORCH_CHECK(
      status == AVSUCCESS,
      "Could not allocate frame buffers for sample format conversion: ",
      getFFMPEGErrorStringFromErrorCode(status));

  auto numConvertedSamples = swr_convert(
      streamInfo.swrContext.get(),
      convertedAVFrame->data,
      convertedAVFrame->nb_samples,
      static_cast<const uint8_t**>(
          const_cast<const uint8_t**>(srcAVFrame->data)),
      srcAVFrame->nb_samples);
  // numConvertedSamples can be 0 if we're downsampling by a great factor and
  // the first frame doesn't contain a lot of samples. It should be handled
  // properly by the caller.
  TORCH_CHECK(
      numConvertedSamples >= 0,
      "Error in swr_convert: ",
      getFFMPEGErrorStringFromErrorCode(numConvertedSamples));

  // See comment above about nb_samples
  convertedAVFrame->nb_samples = numConvertedSamples;

  return convertedAVFrame;
}

std::optional<torch::Tensor> VideoDecoder::maybeFlushSwrBuffers() {
  // When sample rate conversion is involved, swresample buffers some of the
  // samples in-between calls to swr_convert (see the libswresample docs).
  // That's because the last few samples in a given frame require future samples
  // from the next frame to be properly converted. This function flushes out the
  // samples that are stored in swresample's buffers.
  auto& streamInfo = streamInfos_[activeStreamIndex_];
  if (!streamInfo.swrContext) {
    return std::nullopt;
  }
  auto numRemainingSamples = // this is an upper bound
      swr_get_out_samples(streamInfo.swrContext.get(), 0);

  if (numRemainingSamples == 0) {
    return std::nullopt;
  }

  auto numChannels = getNumChannels(streamInfo.codecContext);
  torch::Tensor lastSamples =
      torch::empty({numChannels, numRemainingSamples}, torch::kFloat32);

  std::vector<uint8_t*> outputBuffers(numChannels);
  for (auto i = 0; i < numChannels; i++) {
    outputBuffers[i] = static_cast<uint8_t*>(lastSamples[i].data_ptr());
  }

  auto actualNumRemainingSamples = swr_convert(
      streamInfo.swrContext.get(),
      outputBuffers.data(),
      numRemainingSamples,
      nullptr,
      0);

  return lastSamples.narrow(
      /*dim=*/1, /*start=*/0, /*length=*/actualNumRemainingSamples);
}

// --------------------------------------------------------------------------
// OUTPUT ALLOCATION AND SHAPE CONVERSION
// --------------------------------------------------------------------------

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
  torch::Device device = (videoStreamOptions.device)
      ? videoStreamOptions.device->device()
      : torch::kCPU;
  data = allocateEmptyHWCTensor(height, width, device, numFrames);
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

// Returns a [N]CHW *view* of a [N]HWC input tensor, if the options require so.
// The [N] leading batch-dimension is optional i.e. the input tensor can be 3D
// or 4D.
// Calling permute() is guaranteed to return a view as per the docs:
// https://pytorch.org/docs/stable/generated/torch.permute.html
torch::Tensor VideoDecoder::maybePermuteHWC2CHW(torch::Tensor& hwcTensor) {
  if (streamInfos_[activeStreamIndex_].videoStreamOptions.dimensionOrder ==
      "NHWC") {
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

// --------------------------------------------------------------------------
// COLOR CONVERSION UTILS AND INITIALIZERS
// --------------------------------------------------------------------------

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

  int status = avfilter_graph_create_filter(
      &filterGraphContext.sourceContext,
      buffersrc,
      "in",
      filterArgs.str().c_str(),
      nullptr,
      filterGraphContext.filterGraph.get());
  if (status < 0) {
    throw std::runtime_error(
        std::string("Failed to create filter graph: ") + filterArgs.str() +
        ": " + getFFMPEGErrorStringFromErrorCode(status));
  }

  status = avfilter_graph_create_filter(
      &filterGraphContext.sinkContext,
      buffersink,
      "out",
      nullptr,
      nullptr,
      filterGraphContext.filterGraph.get());
  if (status < 0) {
    throw std::runtime_error(
        "Failed to create filter graph: " +
        getFFMPEGErrorStringFromErrorCode(status));
  }

  enum AVPixelFormat pix_fmts[] = {AV_PIX_FMT_RGB24, AV_PIX_FMT_NONE};

  status = av_opt_set_int_list(
      filterGraphContext.sinkContext,
      "pix_fmts",
      pix_fmts,
      AV_PIX_FMT_NONE,
      AV_OPT_SEARCH_CHILDREN);
  if (status < 0) {
    throw std::runtime_error(
        "Failed to set output pixel formats: " +
        getFFMPEGErrorStringFromErrorCode(status));
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
  status = avfilter_graph_parse_ptr(
      filterGraphContext.filterGraph.get(),
      description.str().c_str(),
      &inputsTmp,
      &outputsTmp,
      nullptr);
  outputs.reset(outputsTmp);
  inputs.reset(inputsTmp);
  if (status < 0) {
    throw std::runtime_error(
        "Failed to parse filter description: " +
        getFFMPEGErrorStringFromErrorCode(status));
  }

  status = avfilter_graph_config(filterGraphContext.filterGraph.get(), nullptr);
  if (status < 0) {
    throw std::runtime_error(
        "Failed to configure filter graph: " +
        getFFMPEGErrorStringFromErrorCode(status));
  }
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

void VideoDecoder::createSwrContext(
    StreamInfo& streamInfo,
    AVSampleFormat sourceSampleFormat,
    AVSampleFormat desiredSampleFormat,
    int sourceSampleRate,
    int desiredSampleRate) {
  auto swrContext = allocateSwrContext(
      streamInfo.codecContext,
      sourceSampleFormat,
      desiredSampleFormat,
      sourceSampleRate,
      desiredSampleRate);

  auto status = swr_init(swrContext);
  TORCH_CHECK(
      status == AVSUCCESS,
      "Couldn't initialize SwrContext: ",
      getFFMPEGErrorStringFromErrorCode(status),
      ". If the error says 'Invalid argument', it's likely that you are using "
      "a buggy FFmpeg version. FFmpeg4 is known to fail here in some "
      "valid scenarios. Try to upgrade FFmpeg?");
  streamInfo.swrContext.reset(swrContext);
}

// --------------------------------------------------------------------------
// PTS <-> INDEX CONVERSIONS
// --------------------------------------------------------------------------

int VideoDecoder::getKeyFrameIndexForPts(int64_t pts) const {
  const StreamInfo& streamInfo = streamInfos_.at(activeStreamIndex_);
  if (streamInfo.keyFrames.empty()) {
    return av_index_search_timestamp(
        streamInfo.stream, pts, AVSEEK_FLAG_BACKWARD);
  } else {
    return getKeyFrameIndexForPtsUsingScannedIndex(streamInfo.keyFrames, pts);
  }
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

int64_t VideoDecoder::secondsToIndexLowerBound(double seconds) {
  auto& streamInfo = streamInfos_[activeStreamIndex_];
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
    case SeekMode::approximate: {
      auto& streamMetadata =
          containerMetadata_.allStreamMetadata[activeStreamIndex_];
      TORCH_CHECK(
          streamMetadata.averageFps.has_value(),
          "Cannot use approximate mode since we couldn't find the average fps from the metadata.");
      return std::floor(seconds * streamMetadata.averageFps.value());
    }
    default:
      throw std::runtime_error("Unknown SeekMode");
  }
}

int64_t VideoDecoder::secondsToIndexUpperBound(double seconds) {
  auto& streamInfo = streamInfos_[activeStreamIndex_];
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
    case SeekMode::approximate: {
      auto& streamMetadata =
          containerMetadata_.allStreamMetadata[activeStreamIndex_];
      TORCH_CHECK(
          streamMetadata.averageFps.has_value(),
          "Cannot use approximate mode since we couldn't find the average fps from the metadata.");
      return std::ceil(seconds * streamMetadata.averageFps.value());
    }
    default:
      throw std::runtime_error("Unknown SeekMode");
  }
}

int64_t VideoDecoder::getPts(int64_t frameIndex) {
  auto& streamInfo = streamInfos_[activeStreamIndex_];
  switch (seekMode_) {
    case SeekMode::exact:
      return streamInfo.allFrames[frameIndex].pts;
    case SeekMode::approximate: {
      auto& streamMetadata =
          containerMetadata_.allStreamMetadata[activeStreamIndex_];
      TORCH_CHECK(
          streamMetadata.averageFps.has_value(),
          "Cannot use approximate mode since we couldn't find the average fps from the metadata.");
      return secondsToClosestPts(
          frameIndex / streamMetadata.averageFps.value(), streamInfo.timeBase);
    }
    default:
      throw std::runtime_error("Unknown SeekMode");
  }
}

// --------------------------------------------------------------------------
// STREAM AND METADATA APIS
// --------------------------------------------------------------------------

int64_t VideoDecoder::getNumFrames(const StreamMetadata& streamMetadata) {
  switch (seekMode_) {
    case SeekMode::exact:
      return streamMetadata.numFramesFromScan.value();
    case SeekMode::approximate: {
      TORCH_CHECK(
          streamMetadata.numFrames.has_value(),
          "Cannot use approximate mode since we couldn't find the number of frames from the metadata.");
      return streamMetadata.numFrames.value();
    }
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
    case SeekMode::approximate: {
      TORCH_CHECK(
          streamMetadata.durationSeconds.has_value(),
          "Cannot use approximate mode since we couldn't find the duration from the metadata.");
      return streamMetadata.durationSeconds.value();
    }
    default:
      throw std::runtime_error("Unknown SeekMode");
  }
}

// --------------------------------------------------------------------------
// VALIDATION UTILS
// --------------------------------------------------------------------------

void VideoDecoder::validateActiveStream(
    std::optional<AVMediaType> avMediaType) {
  auto errorMsg =
      "Provided stream index=" + std::to_string(activeStreamIndex_) +
      " was not previously added.";
  TORCH_CHECK(activeStreamIndex_ != NO_ACTIVE_STREAM, errorMsg);
  TORCH_CHECK(streamInfos_.count(activeStreamIndex_) > 0, errorMsg);

  int allStreamMetadataSize =
      static_cast<int>(containerMetadata_.allStreamMetadata.size());
  TORCH_CHECK(
      activeStreamIndex_ >= 0 && activeStreamIndex_ < allStreamMetadataSize,
      "Invalid stream index=" + std::to_string(activeStreamIndex_) +
          "; valid indices are in the range [0, " +
          std::to_string(allStreamMetadataSize) + ").");

  if (avMediaType.has_value()) {
    TORCH_CHECK(
        streamInfos_[activeStreamIndex_].avMediaType == avMediaType.value(),
        "The method you called isn't supported. ",
        "If you're seeing this error, you are probably trying to call an ",
        "unsupported method on an audio stream.");
  }
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

// --------------------------------------------------------------------------
// MORALLY PRIVATE UTILS
// --------------------------------------------------------------------------

VideoDecoder::DecodeStats VideoDecoder::getDecodeStats() const {
  return decodeStats_;
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

void VideoDecoder::resetDecodeStats() {
  decodeStats_ = DecodeStats{};
}

double VideoDecoder::getPtsSecondsForFrame(int64_t frameIndex) {
  validateActiveStream(AVMEDIA_TYPE_VIDEO);
  validateScannedAllStreams("getPtsSecondsForFrame");

  const auto& streamInfo = streamInfos_[activeStreamIndex_];
  const auto& streamMetadata =
      containerMetadata_.allStreamMetadata[activeStreamIndex_];
  validateFrameIndex(streamMetadata, frameIndex);

  return ptsToSeconds(
      streamInfo.allFrames[frameIndex].pts, streamInfo.timeBase);
}

// --------------------------------------------------------------------------
// FrameDims APIs
// --------------------------------------------------------------------------

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
    const UniqueAVFrame& avFrame) {
  return FrameDims(
      videoStreamOptions.height.value_or(avFrame->height),
      videoStreamOptions.width.value_or(avFrame->width));
}

VideoDecoder::SeekMode seekModeFromString(std::string_view seekMode) {
  if (seekMode == "exact") {
    return VideoDecoder::SeekMode::exact;
  } else if (seekMode == "approximate") {
    return VideoDecoder::SeekMode::approximate;
  } else {
    TORCH_CHECK(false, "Invalid seek mode: " + std::string(seekMode));
  }
}

} // namespace facebook::torchcodec
