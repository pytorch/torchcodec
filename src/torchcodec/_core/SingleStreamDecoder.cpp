// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/SingleStreamDecoder.h"
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include "torch/types.h"

namespace facebook::torchcodec {
namespace {

double ptsToSeconds(int64_t pts, const AVRational& timeBase) {
  // To perform the multiplication before the division, av_q2d is not used
  return static_cast<double>(pts) * timeBase.num / timeBase.den;
}

int64_t secondsToClosestPts(double seconds, const AVRational& timeBase) {
  return static_cast<int64_t>(
      std::round(seconds * timeBase.den / timeBase.num));
}

// Some videos aren't properly encoded and do not specify pts values for
// packets, and thus for frames. Unset values correspond to INT64_MIN. When that
// happens, we fallback to the dts value which hopefully exists and is correct.
// Accessing AVFrames and AVPackets's pts values should **always** go through
// the helpers below. Then, the "pts" fields in our structs like FrameInfo.pts
// should be interpreted as "pts if it exists, dts otherwise".
int64_t getPtsOrDts(ReferenceAVPacket& packet) {
  return packet->pts == INT64_MIN ? packet->dts : packet->pts;
}

int64_t getPtsOrDts(const UniqueAVFrame& avFrame) {
  return avFrame->pts == INT64_MIN ? avFrame->pkt_dts : avFrame->pts;
}

} // namespace

// --------------------------------------------------------------------------
// CONSTRUCTORS, INITIALIZATION, DESTRUCTORS
// --------------------------------------------------------------------------

SingleStreamDecoder::SingleStreamDecoder(
    const std::string& videoFilePath,
    SeekMode seekMode)
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

SingleStreamDecoder::SingleStreamDecoder(
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

void SingleStreamDecoder::initializeDecoder() {
  TORCH_CHECK(!initialized_, "Attempted double initialization.");

  // In principle, the AVFormatContext should be filled in by the call to
  // avformat_open_input() which reads the header. However, some formats do not
  // store enough info in the header, so we call avformat_find_stream_info()
  // which decodes a few frames to get missing info. For more, see:
  //   https://ffmpeg.org/doxygen/7.0/group__lavf__decoding.html
  int status = avformat_find_stream_info(formatContext_.get(), nullptr);
  TORCH_CHECK(
      status >= 0,
      "Failed to find stream info: ",
      getFFMPEGErrorStringFromErrorCode(status));

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
      streamMetadata.numFramesFromHeader = frameCount;
    }

    if (avStream->duration > 0 && avStream->time_base.den > 0) {
      streamMetadata.durationSecondsFromHeader =
          ptsToSeconds(avStream->duration, avStream->time_base);
    }
    if (avStream->start_time != AV_NOPTS_VALUE) {
      streamMetadata.beginStreamSecondsFromHeader =
          ptsToSeconds(avStream->start_time, avStream->time_base);
    }

    if (avStream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      double fps = av_q2d(avStream->r_frame_rate);
      if (fps > 0) {
        streamMetadata.averageFpsFromHeader = fps;
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
    AVRational defaultTimeBase{1, AV_TIME_BASE};
    containerMetadata_.durationSecondsFromHeader =
        ptsToSeconds(formatContext_->duration, defaultTimeBase);
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

int SingleStreamDecoder::getBestStreamIndex(AVMediaType mediaType) {
  AVCodecOnlyUseForCallingAVFindBestStream avCodec = nullptr;
  int streamIndex =
      av_find_best_stream(formatContext_.get(), mediaType, -1, -1, &avCodec, 0);
  return streamIndex;
}

// --------------------------------------------------------------------------
// VIDEO METADATA QUERY API
// --------------------------------------------------------------------------

void SingleStreamDecoder::sortAllFrames() {
  // Sort the allFrames and keyFrames vecs in each stream, and also sets
  // additional fields of the FrameInfo entries like nextPts and frameIndex
  // This is called at the end of a scan, or when setting a user-defined frame
  // mapping.
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
}

void SingleStreamDecoder::scanFileAndUpdateMetadataAndIndex() {
  if (scannedAllStreams_) {
    return;
  }

  AutoAVPacket autoAVPacket;
  while (true) {
    ReferenceAVPacket packet(autoAVPacket);

    // av_read_frame is a misleading name: it gets the next **packet**.
    int status = av_read_frame(formatContext_.get(), packet.get());

    if (status == AVERROR_EOF) {
      break;
    }

    TORCH_CHECK(
        status == AVSUCCESS,
        "Failed to read frame from input file: ",
        getFFMPEGErrorStringFromErrorCode(status));

    if (packet->flags & AV_PKT_FLAG_DISCARD) {
      continue;
    }

    // We got a valid packet. Let's figure out what stream it belongs to and
    // record its relevant metadata.
    int streamIndex = packet->stream_index;
    auto& streamMetadata = containerMetadata_.allStreamMetadata[streamIndex];
    streamMetadata.beginStreamPtsFromContent = std::min(
        streamMetadata.beginStreamPtsFromContent.value_or(INT64_MAX),
        getPtsOrDts(packet));
    streamMetadata.endStreamPtsFromContent = std::max(
        streamMetadata.endStreamPtsFromContent.value_or(INT64_MIN),
        getPtsOrDts(packet) + packet->duration);
    streamMetadata.numFramesFromContent =
        streamMetadata.numFramesFromContent.value_or(0) + 1;

    // Note that we set the other value in this struct, nextPts, only after
    // we have scanned all packets and sorted by pts.
    FrameInfo frameInfo = {getPtsOrDts(packet)};
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

    streamMetadata.numFramesFromContent =
        streamInfos_[streamIndex].allFrames.size();

    if (streamMetadata.beginStreamPtsFromContent.has_value()) {
      streamMetadata.beginStreamPtsSecondsFromContent = ptsToSeconds(
          *streamMetadata.beginStreamPtsFromContent, avStream->time_base);
    }
    if (streamMetadata.endStreamPtsFromContent.has_value()) {
      streamMetadata.endStreamPtsSecondsFromContent = ptsToSeconds(
          *streamMetadata.endStreamPtsFromContent, avStream->time_base);
    }
  }

  // Reset the seek-cursor back to the beginning.
  int status = avformat_seek_file(formatContext_.get(), 0, INT64_MIN, 0, 0, 0);
  TORCH_CHECK(
      status >= 0,
      "Could not seek file to pts=0: ",
      getFFMPEGErrorStringFromErrorCode(status));

  // Sort all frames by their pts.
  sortAllFrames();
  scannedAllStreams_ = true;
}

void SingleStreamDecoder::readCustomFrameMappingsUpdateMetadataAndIndex(
    int streamIndex,
    FrameMappings customFrameMappings) {
  auto& all_frames = customFrameMappings.all_frames;
  auto& is_key_frame = customFrameMappings.is_key_frame;
  auto& duration = customFrameMappings.duration;
  TORCH_CHECK(
      all_frames.size(0) == is_key_frame.size(0) &&
          is_key_frame.size(0) == duration.size(0),
      "all_frames, is_key_frame, and duration from custom_frame_mappings were not same size.");

  auto& streamMetadata = containerMetadata_.allStreamMetadata[streamIndex];

  streamMetadata.beginStreamPtsFromContent = all_frames[0].item<int64_t>();
  streamMetadata.endStreamPtsFromContent =
      all_frames[-1].item<int64_t>() + duration[-1].item<int64_t>();

  auto avStream = formatContext_->streams[streamIndex];
  streamMetadata.beginStreamPtsSecondsFromContent = ptsToSeconds(
      *streamMetadata.beginStreamPtsFromContent, avStream->time_base);

  streamMetadata.endStreamPtsSecondsFromContent = ptsToSeconds(
      *streamMetadata.endStreamPtsFromContent, avStream->time_base);

  streamMetadata.numFramesFromContent = all_frames.size(0);
  for (int64_t i = 0; i < all_frames.size(0); ++i) {
    FrameInfo frameInfo;
    frameInfo.pts = all_frames[i].item<int64_t>();
    frameInfo.isKeyFrame = is_key_frame[i].item<bool>();
    streamInfos_[streamIndex].allFrames.push_back(frameInfo);
    if (frameInfo.isKeyFrame) {
      streamInfos_[streamIndex].keyFrames.push_back(frameInfo);
    }
  }
  // Sort all frames by their pts
  sortAllFrames();
}

ContainerMetadata SingleStreamDecoder::getContainerMetadata() const {
  return containerMetadata_;
}

torch::Tensor SingleStreamDecoder::getKeyFrameIndices() {
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

void SingleStreamDecoder::addStream(
    int streamIndex,
    AVMediaType mediaType,
    const torch::Device& device,
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

  deviceInterface_ = createDeviceInterface(device);

  // This should never happen, checking just to be safe.
  TORCH_CHECK(
      streamInfo.stream->codecpar->codec_type == mediaType,
      "FFmpeg found stream with index ",
      activeStreamIndex_,
      " which is of the wrong media type.");

  // TODO_CODE_QUALITY it's pretty meh to have a video-specific logic within
  // addStream() which is supposed to be generic
  if (mediaType == AVMEDIA_TYPE_VIDEO) {
    if (deviceInterface_) {
      avCodec = makeAVCodecOnlyUseForCallingAVFindBestStream(
          deviceInterface_->findCodec(streamInfo.stream->codecpar->codec_id)
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
    if (deviceInterface_) {
      deviceInterface_->initializeContext(codecContext);
    }
  }

  retVal = avcodec_open2(streamInfo.codecContext.get(), avCodec, nullptr);
  TORCH_CHECK(retVal >= AVSUCCESS, getFFMPEGErrorStringFromErrorCode(retVal));

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

void SingleStreamDecoder::addVideoStream(
    int streamIndex,
    const VideoStreamOptions& videoStreamOptions,
    std::optional<FrameMappings> customFrameMappings) {
  addStream(
      streamIndex,
      AVMEDIA_TYPE_VIDEO,
      videoStreamOptions.device,
      videoStreamOptions.ffmpegThreadCount);

  auto& streamMetadata =
      containerMetadata_.allStreamMetadata[activeStreamIndex_];

  if (seekMode_ == SeekMode::approximate) {
    TORCH_CHECK(
        streamMetadata.averageFpsFromHeader.has_value(),
        "Seek mode is approximate, but stream ",
        std::to_string(activeStreamIndex_),
        " does not have an average fps in its metadata.");
  }

  auto& streamInfo = streamInfos_[activeStreamIndex_];
  streamInfo.videoStreamOptions = videoStreamOptions;

  streamMetadata.width = streamInfo.codecContext->width;
  streamMetadata.height = streamInfo.codecContext->height;
  streamMetadata.sampleAspectRatio =
      streamInfo.codecContext->sample_aspect_ratio;

  if (seekMode_ == SeekMode::custom_frame_mappings) {
    TORCH_CHECK(
        customFrameMappings.has_value(),
        "Missing frame mappings when custom_frame_mappings seek mode is set.");
    readCustomFrameMappingsUpdateMetadataAndIndex(
        streamIndex, customFrameMappings.value());
  }
}

void SingleStreamDecoder::addAudioStream(
    int streamIndex,
    const AudioStreamOptions& audioStreamOptions) {
  TORCH_CHECK(
      seekMode_ == SeekMode::approximate,
      "seek_mode must be 'approximate' for audio streams.");
  if (audioStreamOptions.numChannels.has_value()) {
    TORCH_CHECK(
        *audioStreamOptions.numChannels > 0 &&
            *audioStreamOptions.numChannels <= AV_NUM_DATA_POINTERS,
        "num_channels must be > 0 and <= AV_NUM_DATA_POINTERS (usually 8). Got: ",
        *audioStreamOptions.numChannels);
  }

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

FrameOutput SingleStreamDecoder::getNextFrame() {
  auto output = getNextFrameInternal();
  if (streamInfos_[activeStreamIndex_].avMediaType == AVMEDIA_TYPE_VIDEO) {
    output.data = maybePermuteHWC2CHW(output.data);
  }
  return output;
}

FrameOutput SingleStreamDecoder::getNextFrameInternal(
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  validateActiveStream();
  UniqueAVFrame avFrame = decodeAVFrame([this](const UniqueAVFrame& avFrame) {
    return getPtsOrDts(avFrame) >= cursor_;
  });
  return convertAVFrameToFrameOutput(avFrame, preAllocatedOutputTensor);
}

FrameOutput SingleStreamDecoder::getFrameAtIndex(int64_t frameIndex) {
  auto frameOutput = getFrameAtIndexInternal(frameIndex);
  frameOutput.data = maybePermuteHWC2CHW(frameOutput.data);
  return frameOutput;
}

FrameOutput SingleStreamDecoder::getFrameAtIndexInternal(
    int64_t frameIndex,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  validateActiveStream(AVMEDIA_TYPE_VIDEO);

  const auto& streamInfo = streamInfos_[activeStreamIndex_];
  const auto& streamMetadata =
      containerMetadata_.allStreamMetadata[activeStreamIndex_];

  std::optional<int64_t> numFrames = getNumFrames(streamMetadata);
  if (numFrames.has_value()) {
    // If the frameIndex is negative, we convert it to a positive index
    frameIndex = frameIndex >= 0 ? frameIndex : frameIndex + numFrames.value();
  }
  validateFrameIndex(streamMetadata, frameIndex);

  int64_t pts = getPts(frameIndex);
  setCursorPtsInSeconds(ptsToSeconds(pts, streamInfo.timeBase));
  return getNextFrameInternal(preAllocatedOutputTensor);
}

FrameBatchOutput SingleStreamDecoder::getFramesAtIndices(
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

FrameBatchOutput SingleStreamDecoder::getFramesInRange(
    int64_t start,
    int64_t stop,
    int64_t step) {
  validateActiveStream(AVMEDIA_TYPE_VIDEO);

  const auto& streamMetadata =
      containerMetadata_.allStreamMetadata[activeStreamIndex_];
  const auto& streamInfo = streamInfos_[activeStreamIndex_];
  TORCH_CHECK(
      start >= 0, "Range start, " + std::to_string(start) + " is less than 0.");
  TORCH_CHECK(
      step > 0, "Step must be greater than 0; is " + std::to_string(step));

  // Note that if we do not have the number of frames available in our metadata,
  // then we assume that the upper part of the range is valid.
  std::optional<int64_t> numFrames = getNumFrames(streamMetadata);
  if (numFrames.has_value()) {
    TORCH_CHECK(
        stop <= numFrames.value(),
        "Range stop, " + std::to_string(stop) +
            ", is more than the number of frames, " +
            std::to_string(numFrames.value()));
  }

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

FrameOutput SingleStreamDecoder::getFramePlayedAt(double seconds) {
  validateActiveStream(AVMEDIA_TYPE_VIDEO);
  StreamInfo& streamInfo = streamInfos_[activeStreamIndex_];
  double lastDecodedStartTime =
      ptsToSeconds(streamInfo.lastDecodedAvFramePts, streamInfo.timeBase);
  double lastDecodedEndTime = ptsToSeconds(
      streamInfo.lastDecodedAvFramePts + streamInfo.lastDecodedAvFrameDuration,
      streamInfo.timeBase);
  if (seconds >= lastDecodedStartTime && seconds < lastDecodedEndTime) {
    // We are in the same frame as the one we just returned. However, since we
    // don't cache it locally, we have to rewind back.
    seconds = lastDecodedStartTime;
  }

  setCursorPtsInSeconds(seconds);
  UniqueAVFrame avFrame =
      decodeAVFrame([seconds, this](const UniqueAVFrame& avFrame) {
        StreamInfo& streamInfo = streamInfos_[activeStreamIndex_];
        double frameStartTime =
            ptsToSeconds(getPtsOrDts(avFrame), streamInfo.timeBase);
        double frameEndTime = ptsToSeconds(
            getPtsOrDts(avFrame) + getDuration(avFrame), streamInfo.timeBase);
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

FrameBatchOutput SingleStreamDecoder::getFramesPlayedAt(
    const std::vector<double>& timestamps) {
  validateActiveStream(AVMEDIA_TYPE_VIDEO);

  const auto& streamMetadata =
      containerMetadata_.allStreamMetadata[activeStreamIndex_];

  double minSeconds = getMinSeconds(streamMetadata);
  std::optional<double> maxSeconds = getMaxSeconds(streamMetadata);

  // The frame played at timestamp t and the one played at timestamp `t +
  // eps` are probably the same frame, with the same index. The easiest way to
  // avoid decoding that unique frame twice is to convert the input timestamps
  // to indices, and leverage the de-duplication logic of getFramesAtIndices.

  std::vector<int64_t> frameIndices(timestamps.size());
  for (size_t i = 0; i < timestamps.size(); ++i) {
    auto frameSeconds = timestamps[i];
    TORCH_CHECK(
        frameSeconds >= minSeconds,
        "frame pts is " + std::to_string(frameSeconds) +
            "; must be greater than or equal to " + std::to_string(minSeconds) +
            ".");

    // Note that if we can't determine the maximum number of seconds from the
    // metadata, then we assume the frame's pts is valid.
    if (maxSeconds.has_value()) {
      TORCH_CHECK(
          frameSeconds < maxSeconds.value(),
          "frame pts is " + std::to_string(frameSeconds) +
              "; must be less than " + std::to_string(maxSeconds.value()) +
              ".");
    }

    frameIndices[i] = secondsToIndexLowerBound(frameSeconds);
  }

  return getFramesAtIndices(frameIndices);
}

FrameBatchOutput SingleStreamDecoder::getFramesPlayedInRange(
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
  TORCH_CHECK(
      startSeconds >= minSeconds,
      "Start seconds is " + std::to_string(startSeconds) +
          "; must be greater than or equal to " + std::to_string(minSeconds) +
          ".");

  // Note that if we can't determine the maximum seconds from the metadata, then
  // we assume upper range is valid.
  std::optional<double> maxSeconds = getMaxSeconds(streamMetadata);
  if (maxSeconds.has_value()) {
    TORCH_CHECK(
        startSeconds < maxSeconds.value(),
        "Start seconds is " + std::to_string(startSeconds) +
            "; must be less than " + std::to_string(maxSeconds.value()) + ".");
    TORCH_CHECK(
        stopSeconds <= maxSeconds.value(),
        "Stop seconds (" + std::to_string(stopSeconds) +
            "; must be less than or equal to " +
            std::to_string(maxSeconds.value()) + ").");
  }

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
AudioFramesOutput SingleStreamDecoder::getFramesPlayedInRangeAudio(
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

  StreamInfo& streamInfo = streamInfos_[activeStreamIndex_];

  if (stopSecondsOptional.has_value() && startSeconds == *stopSecondsOptional) {
    // For consistency with video
    int numChannels = getNumChannels(streamInfo.codecContext);
    return AudioFramesOutput{torch::empty({numChannels, 0}), 0.0};
  }

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
          decodeAVFrame([startPts, stopPts](const UniqueAVFrame& avFrame) {
            return startPts < getPtsOrDts(avFrame) + getDuration(avFrame) &&
                stopPts > getPtsOrDts(avFrame);
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
      "This is probably because start_seconds is too high(",
      startSeconds,
      "),",
      "or because stop_seconds(",
      stopSecondsOptional,
      ") is too low.");

  return AudioFramesOutput{torch::cat(frames, 1), *firstFramePtsSeconds};
}

// --------------------------------------------------------------------------
// SEEKING APIs
// --------------------------------------------------------------------------

void SingleStreamDecoder::setCursorPtsInSeconds(double seconds) {
  // We don't allow public audio decoding APIs to seek, see [Audio Decoding
  // Design]
  validateActiveStream(AVMEDIA_TYPE_VIDEO);
  setCursor(
      secondsToClosestPts(seconds, streamInfos_[activeStreamIndex_].timeBase));
}

void SingleStreamDecoder::setCursor(int64_t pts) {
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
bool SingleStreamDecoder::canWeAvoidSeeking() const {
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
    // TODO: https://github.com/pytorch/torchcodec/issues/84 we could
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
void SingleStreamDecoder::maybeSeekToBeforeDesiredPts() {
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
  TORCH_CHECK(
      status >= 0,
      "Could not seek file to pts=",
      std::to_string(desiredPts),
      ": ",
      getFFMPEGErrorStringFromErrorCode(status));

  decodeStats_.numFlushes++;
  avcodec_flush_buffers(streamInfo.codecContext.get());
}

// --------------------------------------------------------------------------
// LOW-LEVEL DECODING
// --------------------------------------------------------------------------

UniqueAVFrame SingleStreamDecoder::decodeAVFrame(
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
        TORCH_CHECK(
            status >= AVSUCCESS,
            "Could not flush decoder: ",
            getFFMPEGErrorStringFromErrorCode(status));

        reachedEOF = true;
        break;
      }

      TORCH_CHECK(
          status >= AVSUCCESS,
          "Could not read frame from input file: ",
          getFFMPEGErrorStringFromErrorCode(status));

    } while (packet->stream_index != activeStreamIndex_);

    if (reachedEOF) {
      // We don't have any more packets to send to the decoder. So keep on
      // pulling frames from its internal buffers.
      continue;
    }

    // We got a valid packet. Send it to the decoder, and we'll receive it in
    // the next iteration.
    status = avcodec_send_packet(streamInfo.codecContext.get(), packet.get());
    TORCH_CHECK(
        status >= AVSUCCESS,
        "Could not push packet to decoder: ",
        getFFMPEGErrorStringFromErrorCode(status));

    decodeStats_.numPacketsSentToDecoder++;
  }

  if (status < AVSUCCESS) {
    if (reachedEOF || status == AVERROR_EOF) {
      throw SingleStreamDecoder::EndOfFileException(
          "Requested next frame while there are no more frames left to "
          "decode.");
    }
    TORCH_CHECK(
        false,
        "Could not receive frame from decoder: ",
        getFFMPEGErrorStringFromErrorCode(status));
  }

  // Note that we don't flush the decoder when we reach EOF (even though that's
  // mentioned in https://ffmpeg.org/doxygen/trunk/group__lavc__encdec.html).
  // This is because we may have packets internally in the decoder that we
  // haven't received as frames. Eventually we will either hit AVERROR_EOF from
  // av_receive_frame() or the user will have seeked to a different location in
  // the file and that will flush the decoder.
  streamInfo.lastDecodedAvFramePts = getPtsOrDts(avFrame);
  streamInfo.lastDecodedAvFrameDuration = getDuration(avFrame);

  return avFrame;
}

// --------------------------------------------------------------------------
// AVFRAME <-> FRAME OUTPUT CONVERSION
// --------------------------------------------------------------------------

FrameOutput SingleStreamDecoder::convertAVFrameToFrameOutput(
    UniqueAVFrame& avFrame,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  // Convert the frame to tensor.
  FrameOutput frameOutput;
  auto& streamInfo = streamInfos_[activeStreamIndex_];
  frameOutput.ptsSeconds = ptsToSeconds(
      getPtsOrDts(avFrame),
      formatContext_->streams[activeStreamIndex_]->time_base);
  frameOutput.durationSeconds = ptsToSeconds(
      getDuration(avFrame),
      formatContext_->streams[activeStreamIndex_]->time_base);
  if (streamInfo.avMediaType == AVMEDIA_TYPE_AUDIO) {
    convertAudioAVFrameToFrameOutputOnCPU(avFrame, frameOutput);
  } else if (deviceInterface_) {
    deviceInterface_->convertAVFrameToFrameOutput(
        streamInfo.videoStreamOptions,
        streamInfo.timeBase,
        avFrame,
        frameOutput,
        preAllocatedOutputTensor);
  }
  return frameOutput;
}

void SingleStreamDecoder::convertAudioAVFrameToFrameOutputOnCPU(
    UniqueAVFrame& srcAVFrame,
    FrameOutput& frameOutput) {
  AVSampleFormat srcSampleFormat =
      static_cast<AVSampleFormat>(srcAVFrame->format);
  AVSampleFormat outSampleFormat = AV_SAMPLE_FMT_FLTP;

  StreamInfo& streamInfo = streamInfos_[activeStreamIndex_];
  int srcSampleRate = srcAVFrame->sample_rate;
  int outSampleRate =
      streamInfo.audioStreamOptions.sampleRate.value_or(srcSampleRate);

  int srcNumChannels = getNumChannels(streamInfo.codecContext);
  TORCH_CHECK(
      srcNumChannels == getNumChannels(srcAVFrame),
      "The frame has ",
      getNumChannels(srcAVFrame),
      " channels, expected ",
      srcNumChannels,
      ". If you are hitting this, it may be because you are using "
      "a buggy FFmpeg version. FFmpeg4 is known to fail here in some "
      "valid scenarios. Try to upgrade FFmpeg?");
  int outNumChannels =
      streamInfo.audioStreamOptions.numChannels.value_or(srcNumChannels);

  bool mustConvert =
      (srcSampleFormat != outSampleFormat || srcSampleRate != outSampleRate ||
       srcNumChannels != outNumChannels);

  UniqueAVFrame convertedAVFrame;
  if (mustConvert) {
    if (!streamInfo.swrContext) {
      streamInfo.swrContext.reset(createSwrContext(
          srcSampleFormat,
          outSampleFormat,
          srcSampleRate,
          outSampleRate,
          srcAVFrame,
          outNumChannels));
    }

    convertedAVFrame = convertAudioAVFrameSamples(
        streamInfo.swrContext,
        srcAVFrame,
        outSampleFormat,
        outSampleRate,
        outNumChannels);
  }
  const UniqueAVFrame& avFrame = mustConvert ? convertedAVFrame : srcAVFrame;

  AVSampleFormat format = static_cast<AVSampleFormat>(avFrame->format);
  TORCH_CHECK(
      format == outSampleFormat,
      "Something went wrong, the frame didn't get converted to the desired format. ",
      "Desired format = ",
      av_get_sample_fmt_name(outSampleFormat),
      "source format = ",
      av_get_sample_fmt_name(format));

  int numChannels = getNumChannels(avFrame);
  TORCH_CHECK(
      numChannels == outNumChannels,
      "Something went wrong, the frame didn't get converted to the desired ",
      "number of channels = ",
      outNumChannels,
      ". Got ",
      numChannels,
      " instead.");

  auto numSamples = avFrame->nb_samples; // per channel

  frameOutput.data = torch::empty({numChannels, numSamples}, torch::kFloat32);

  if (numSamples > 0) {
    uint8_t* outputChannelData =
        static_cast<uint8_t*>(frameOutput.data.data_ptr());
    auto numBytesPerChannel = numSamples * av_get_bytes_per_sample(format);
    for (auto channel = 0; channel < numChannels;
         ++channel, outputChannelData += numBytesPerChannel) {
      std::memcpy(
          outputChannelData,
          avFrame->extended_data[channel],
          numBytesPerChannel);
    }
  }
}

std::optional<torch::Tensor> SingleStreamDecoder::maybeFlushSwrBuffers() {
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

  int numChannels = streamInfo.audioStreamOptions.numChannels.value_or(
      getNumChannels(streamInfo.codecContext));
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

FrameBatchOutput::FrameBatchOutput(
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

// Returns a [N]CHW *view* of a [N]HWC input tensor, if the options require so.
// The [N] leading batch-dimension is optional i.e. the input tensor can be 3D
// or 4D.
// Calling permute() is guaranteed to return a view as per the docs:
// https://pytorch.org/docs/stable/generated/torch.permute.html
torch::Tensor SingleStreamDecoder::maybePermuteHWC2CHW(
    torch::Tensor& hwcTensor) {
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
// PTS <-> INDEX CONVERSIONS
// --------------------------------------------------------------------------

int SingleStreamDecoder::getKeyFrameIndexForPts(int64_t pts) const {
  const StreamInfo& streamInfo = streamInfos_.at(activeStreamIndex_);
  if (streamInfo.keyFrames.empty()) {
    return av_index_search_timestamp(
        streamInfo.stream, pts, AVSEEK_FLAG_BACKWARD);
  } else {
    return getKeyFrameIndexForPtsUsingScannedIndex(streamInfo.keyFrames, pts);
  }
}

int SingleStreamDecoder::getKeyFrameIndexForPtsUsingScannedIndex(
    const std::vector<SingleStreamDecoder::FrameInfo>& keyFrames,
    int64_t pts) const {
  auto upperBound = std::upper_bound(
      keyFrames.begin(),
      keyFrames.end(),
      pts,
      [](int64_t pts, const SingleStreamDecoder::FrameInfo& frameInfo) {
        return pts < frameInfo.pts;
      });
  if (upperBound == keyFrames.begin()) {
    return -1;
  }
  return upperBound - 1 - keyFrames.begin();
}

int64_t SingleStreamDecoder::secondsToIndexLowerBound(double seconds) {
  auto& streamInfo = streamInfos_[activeStreamIndex_];
  switch (seekMode_) {
    case SeekMode::custom_frame_mappings:
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
          streamMetadata.averageFpsFromHeader.has_value(),
          "Cannot use approximate mode since we couldn't find the average fps from the metadata.");
      return std::floor(seconds * streamMetadata.averageFpsFromHeader.value());
    }
    default:
      TORCH_CHECK(false, "Unknown SeekMode");
  }
}

int64_t SingleStreamDecoder::secondsToIndexUpperBound(double seconds) {
  auto& streamInfo = streamInfos_[activeStreamIndex_];
  switch (seekMode_) {
    case SeekMode::custom_frame_mappings:
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
          streamMetadata.averageFpsFromHeader.has_value(),
          "Cannot use approximate mode since we couldn't find the average fps from the metadata.");
      return std::ceil(seconds * streamMetadata.averageFpsFromHeader.value());
    }
    default:
      TORCH_CHECK(false, "Unknown SeekMode");
  }
}

int64_t SingleStreamDecoder::getPts(int64_t frameIndex) {
  auto& streamInfo = streamInfos_[activeStreamIndex_];
  switch (seekMode_) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      return streamInfo.allFrames[frameIndex].pts;
    case SeekMode::approximate: {
      auto& streamMetadata =
          containerMetadata_.allStreamMetadata[activeStreamIndex_];
      TORCH_CHECK(
          streamMetadata.averageFpsFromHeader.has_value(),
          "Cannot use approximate mode since we couldn't find the average fps from the metadata.");
      return secondsToClosestPts(
          frameIndex / streamMetadata.averageFpsFromHeader.value(),
          streamInfo.timeBase);
    }
    default:
      TORCH_CHECK(false, "Unknown SeekMode");
  }
}

// --------------------------------------------------------------------------
// STREAM AND METADATA APIS
// --------------------------------------------------------------------------

std::optional<int64_t> SingleStreamDecoder::getNumFrames(
    const StreamMetadata& streamMetadata) {
  switch (seekMode_) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      return streamMetadata.numFramesFromContent.value();
    case SeekMode::approximate: {
      return streamMetadata.numFramesFromHeader;
    }
    default:
      TORCH_CHECK(false, "Unknown SeekMode");
  }
}

double SingleStreamDecoder::getMinSeconds(
    const StreamMetadata& streamMetadata) {
  switch (seekMode_) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      return streamMetadata.beginStreamPtsSecondsFromContent.value();
    case SeekMode::approximate:
      return 0;
    default:
      TORCH_CHECK(false, "Unknown SeekMode");
  }
}

std::optional<double> SingleStreamDecoder::getMaxSeconds(
    const StreamMetadata& streamMetadata) {
  switch (seekMode_) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      return streamMetadata.endStreamPtsSecondsFromContent.value();
    case SeekMode::approximate: {
      return streamMetadata.durationSecondsFromHeader;
    }
    default:
      TORCH_CHECK(false, "Unknown SeekMode");
  }
}

// --------------------------------------------------------------------------
// VALIDATION UTILS
// --------------------------------------------------------------------------

void SingleStreamDecoder::validateActiveStream(
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

void SingleStreamDecoder::validateScannedAllStreams(const std::string& msg) {
  TORCH_CHECK(
      scannedAllStreams_,
      "Must scan all streams to update metadata before calling ",
      msg);
}

void SingleStreamDecoder::validateFrameIndex(
    const StreamMetadata& streamMetadata,
    int64_t frameIndex) {
  if (frameIndex < 0) {
    throw std::out_of_range(
        "Invalid frame index=" + std::to_string(frameIndex) +
        " for streamIndex=" + std::to_string(streamMetadata.streamIndex) +
        "; negative indices must have an absolute value less than the number of frames, "
        "and the number of frames must be known.");
  }

  // Note that if we do not have the number of frames available in our metadata,
  // then we assume that the frameIndex is valid.
  std::optional<int64_t> numFrames = getNumFrames(streamMetadata);
  if (numFrames.has_value()) {
    if (frameIndex >= numFrames.value()) {
      throw std::out_of_range(
          "Invalid frame index=" + std::to_string(frameIndex) +
          " for streamIndex=" + std::to_string(streamMetadata.streamIndex) +
          "; must be less than " + std::to_string(numFrames.value()));
    }
  }
}

// --------------------------------------------------------------------------
// MORALLY PRIVATE UTILS
// --------------------------------------------------------------------------

SingleStreamDecoder::DecodeStats SingleStreamDecoder::getDecodeStats() const {
  return decodeStats_;
}

std::ostream& operator<<(
    std::ostream& os,
    const SingleStreamDecoder::DecodeStats& stats) {
  os << "DecodeStats{"
     << "numFramesReceivedByDecoder=" << stats.numFramesReceivedByDecoder
     << ", numPacketsRead=" << stats.numPacketsRead
     << ", numPacketsSentToDecoder=" << stats.numPacketsSentToDecoder
     << ", numSeeksAttempted=" << stats.numSeeksAttempted
     << ", numSeeksSkipped=" << stats.numSeeksSkipped
     << ", numFlushes=" << stats.numFlushes << "}";

  return os;
}

void SingleStreamDecoder::resetDecodeStats() {
  decodeStats_ = DecodeStats{};
}

double SingleStreamDecoder::getPtsSecondsForFrame(int64_t frameIndex) {
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
    const VideoStreamOptions& videoStreamOptions,
    const StreamMetadata& streamMetadata) {
  return FrameDims(
      videoStreamOptions.height.value_or(*streamMetadata.height),
      videoStreamOptions.width.value_or(*streamMetadata.width));
}

FrameDims getHeightAndWidthFromOptionsOrAVFrame(
    const VideoStreamOptions& videoStreamOptions,
    const UniqueAVFrame& avFrame) {
  return FrameDims(
      videoStreamOptions.height.value_or(avFrame->height),
      videoStreamOptions.width.value_or(avFrame->width));
}

SingleStreamDecoder::SeekMode seekModeFromString(std::string_view seekMode) {
  if (seekMode == "exact") {
    return SingleStreamDecoder::SeekMode::exact;
  } else if (seekMode == "approximate") {
    return SingleStreamDecoder::SeekMode::approximate;
  } else if (seekMode == "custom_frame_mappings") {
    return SingleStreamDecoder::SeekMode::custom_frame_mappings;
  } else {
    TORCH_CHECK(false, "Invalid seek mode: " + std::string(seekMode));
  }
}

} // namespace facebook::torchcodec
