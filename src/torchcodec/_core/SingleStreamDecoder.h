// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/types.h>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string_view>

#include "src/torchcodec/_core/AVIOContextHolder.h"
#include "src/torchcodec/_core/DeviceInterface.h"
#include "src/torchcodec/_core/FFMPEGCommon.h"
#include "src/torchcodec/_core/Frame.h"
#include "src/torchcodec/_core/StreamOptions.h"

namespace facebook::torchcodec {

// The SingleStreamDecoder class can be used to decode video frames to Tensors.
// Note that SingleStreamDecoder is not thread-safe.
// Do not call non-const APIs concurrently on the same object.
class SingleStreamDecoder {
 public:
  // --------------------------------------------------------------------------
  // CONSTRUCTION API
  // --------------------------------------------------------------------------

  enum class SeekMode { exact, approximate, custom_frame_mappings };

  // Creates a SingleStreamDecoder from the video at videoFilePath.
  explicit SingleStreamDecoder(
      const std::string& videoFilePath,
      SeekMode seekMode = SeekMode::exact);

  // Creates a SingleStreamDecoder using the provided AVIOContext inside the
  // AVIOContextHolder. The AVIOContextHolder is the base class, and the
  // derived class will have specialized how the custom read, seek and writes
  // work.
  explicit SingleStreamDecoder(
      std::unique_ptr<AVIOContextHolder> context,
      SeekMode seekMode = SeekMode::exact);

  // --------------------------------------------------------------------------
  // VIDEO METADATA QUERY API
  // --------------------------------------------------------------------------

  // Updates the metadata of the video to accurate values obtained by scanning
  // the contents of the video file. Also updates each StreamInfo's index, i.e.
  // the allFrames and keyFrames vectors.
  void scanFileAndUpdateMetadataAndIndex();

  // Sorts the keyFrames and allFrames vectors in each StreamInfo by pts.
  void sortAllFrames();

  // Returns the metadata for the container.
  ContainerMetadata getContainerMetadata() const;

  // Returns the key frame indices as a tensor. The tensor is 1D and contains
  // int64 values, where each value is the frame index for a key frame.
  torch::Tensor getKeyFrameIndices();

  // FrameMappings is used for the custom_frame_mappings seek mode to store
  // metadata of frames in a stream. The size of all tensors in this struct must
  // match.

  // --------------------------------------------------------------------------
  // ADDING STREAMS API
  // --------------------------------------------------------------------------
  struct FrameMappings {
    // 1D tensor of int64, each value is the PTS of a frame in timebase units.
    torch::Tensor all_frames;
    // 1D tensor of bool, each value indicates if the corresponding frame in
    // all_frames is a key frame.
    torch::Tensor is_key_frame;
    // 1D tensor of int64, each value is the duration of the corresponding frame
    // in all_frames in timebase units.
    torch::Tensor duration;
  };

  void addVideoStream(
      int streamIndex,
      const VideoStreamOptions& videoStreamOptions = VideoStreamOptions(),
      std::optional<FrameMappings> customFrameMappings = std::nullopt);
  void addAudioStream(
      int streamIndex,
      const AudioStreamOptions& audioStreamOptions = AudioStreamOptions());

  // --------------------------------------------------------------------------
  // DECODING AND SEEKING APIs
  // --------------------------------------------------------------------------

  // Places the cursor at the first frame on or after the position in seconds.
  // Calling getNextFrame() will return the first frame at
  // or after this position.
  void setCursorPtsInSeconds(double seconds);

  // Decodes the frame where the current cursor position is. It also advances
  // the cursor to the next frame.
  FrameOutput getNextFrame();

  FrameOutput getFrameAtIndex(int64_t frameIndex);

  // Returns frames at the given indices for a given stream as a single stacked
  // Tensor.
  FrameBatchOutput getFramesAtIndices(const std::vector<int64_t>& frameIndices);

  // Returns frames within a given range. The range is defined by [start, stop).
  // The values retrieved from the range are: [start, start+step,
  // start+(2*step), start+(3*step), ..., stop). The default for step is 1.
  FrameBatchOutput getFramesInRange(int64_t start, int64_t stop, int64_t step);

  // Decodes the first frame in any added stream that is visible at a given
  // timestamp. Frames in the video have a presentation timestamp and a
  // duration. For example, if a frame has presentation timestamp of 5.0s and a
  // duration of 1.0s, it will be visible in the timestamp range [5.0, 6.0).
  // i.e. it will be returned when this function is called with seconds=5.0 or
  // seconds=5.999, etc.
  FrameOutput getFramePlayedAt(double seconds);

  FrameBatchOutput getFramesPlayedAt(const std::vector<double>& timestamps);

  // Returns frames within a given pts range. The range is defined by
  // [startSeconds, stopSeconds) with respect to the pts values for frames. The
  // returned frames are in pts order.
  //
  // Note that while stopSeconds is excluded in the half open range, this really
  // only makes a difference when stopSeconds is exactly the pts value for a
  // frame. Otherwise, the moment in time immediately before stopSeconds is in
  // the range, and that time maps to the same frame as stopSeconds.
  //
  // The frames returned are the frames that would be played by our abstract
  // player. Our abstract player displays frames based on pts only. It displays
  // frame i starting at the pts for frame i, and stops at the pts for frame
  // i+1. This model ignores a frame's reported duration.
  //
  // Valid values for startSeconds and stopSeconds are:
  //
  //   [beginStreamPtsSecondsFromContent, endStreamPtsSecondsFromContent)
  FrameBatchOutput getFramesPlayedInRange(
      double startSeconds,
      double stopSeconds);

  AudioFramesOutput getFramesPlayedInRangeAudio(
      double startSeconds,
      std::optional<double> stopSecondsOptional = std::nullopt);

  class EndOfFileException : public std::runtime_error {
   public:
    explicit EndOfFileException(const std::string& msg)
        : std::runtime_error(msg) {}
  };

  // --------------------------------------------------------------------------
  // MORALLY PRIVATE APIS
  // --------------------------------------------------------------------------
  // These are APIs that should be private, but that are effectively exposed for
  // practical reasons, typically for testing purposes.

  // Once getFrameAtIndex supports the preAllocatedOutputTensor parameter, we
  // can move it back to private.
  FrameOutput getFrameAtIndexInternal(
      int64_t frameIndex,
      std::optional<torch::Tensor> preAllocatedOutputTensor = std::nullopt);

  // Exposed for _test_frame_pts_equality, which is used to test non-regression
  // of pts resolution (64 to 32 bit floats)
  double getPtsSecondsForFrame(int64_t frameIndex);

  // Exposed for performance testing.
  struct DecodeStats {
    int64_t numSeeksAttempted = 0;
    int64_t numSeeksDone = 0;
    int64_t numSeeksSkipped = 0;
    int64_t numPacketsRead = 0;
    int64_t numPacketsSentToDecoder = 0;
    int64_t numFramesReceivedByDecoder = 0;
    int64_t numFlushes = 0;
  };

  DecodeStats getDecodeStats() const;
  void resetDecodeStats();

 private:
  // --------------------------------------------------------------------------
  // STREAMINFO AND ASSOCIATED STRUCTS
  // --------------------------------------------------------------------------

  struct FrameInfo {
    int64_t pts = 0;

    // The value of the nextPts default is important: the last frame's nextPts
    // will be INT64_MAX, which ensures that the allFrames vec contains
    // FrameInfo structs with *increasing* nextPts values. That's a necessary
    // condition for the binary searches on those values to work properly (as
    // typically done during pts -> index conversions).
    // TODO: This field is unset (left to the default) for entries in the
    // keyFrames vec!
    int64_t nextPts = INT64_MAX;

    // Note that frameIndex is ALWAYS the index into all of the frames in that
    // stream, even when the FrameInfo is part of the key frame index. Given a
    // FrameInfo for a key frame, the frameIndex allows us to know which frame
    // that is in the stream.
    int64_t frameIndex = 0;

    // Indicates whether a frame is a key frame. It may appear redundant as it's
    // only true for FrameInfos in the keyFrames index, but it is needed to
    // correctly map frames between allFrames and keyFrames during the scan.
    bool isKeyFrame = false;
  };

  struct StreamInfo {
    int streamIndex = -1;
    AVStream* stream = nullptr;
    AVMediaType avMediaType = AVMEDIA_TYPE_UNKNOWN;

    AVRational timeBase = {};
    UniqueAVCodecContext codecContext;

    // The FrameInfo indices we built when scanFileAndUpdateMetadataAndIndex was
    // called.
    std::vector<FrameInfo> keyFrames;
    std::vector<FrameInfo> allFrames;

    // TODO since the decoder is single-stream, these should be decoder fields,
    // not streamInfo fields. And they should be defined right next to
    // `cursor_`, with joint documentation.
    int64_t lastDecodedAvFramePts = 0;
    int64_t lastDecodedAvFrameDuration = 0;
    VideoStreamOptions videoStreamOptions;
    AudioStreamOptions audioStreamOptions;

    // color-conversion fields. Only one of FilterGraphContext and
    // UniqueSwsContext should be non-null.
    UniqueSwrContext swrContext;
  };

  // --------------------------------------------------------------------------
  // INITIALIZERS
  // --------------------------------------------------------------------------

  void initializeDecoder();

  // Reads the user provided frame index and updates each StreamInfo's index,
  // i.e. the allFrames and keyFrames vectors, and
  // endStreamPtsSecondsFromContent
  void readCustomFrameMappingsUpdateMetadataAndIndex(
      int streamIndex,
      FrameMappings customFrameMappings);
  // --------------------------------------------------------------------------
  // DECODING APIS AND RELATED UTILS
  // --------------------------------------------------------------------------

  void setCursor(int64_t pts);
  void setCursor(double) = delete; // prevent calls with doubles and floats
  bool canWeAvoidSeeking() const;

  void maybeSeekToBeforeDesiredPts();

  UniqueAVFrame decodeAVFrame(
      std::function<bool(const UniqueAVFrame&)> filterFunction);

  FrameOutput getNextFrameInternal(
      std::optional<torch::Tensor> preAllocatedOutputTensor = std::nullopt);

  torch::Tensor maybePermuteHWC2CHW(torch::Tensor& hwcTensor);

  FrameOutput convertAVFrameToFrameOutput(
      UniqueAVFrame& avFrame,
      std::optional<torch::Tensor> preAllocatedOutputTensor = std::nullopt);

  void convertAVFrameToFrameOutputOnCPU(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor = std::nullopt);

  void convertAudioAVFrameToFrameOutputOnCPU(
      UniqueAVFrame& srcAVFrame,
      FrameOutput& frameOutput);

  torch::Tensor convertAVFrameToTensorUsingFilterGraph(
      const UniqueAVFrame& avFrame);

  int convertAVFrameToTensorUsingSwsScale(
      const UniqueAVFrame& avFrame,
      torch::Tensor& outputTensor);

  std::optional<torch::Tensor> maybeFlushSwrBuffers();

  // --------------------------------------------------------------------------
  // PTS <-> INDEX CONVERSIONS
  // --------------------------------------------------------------------------

  int getKeyFrameIndexForPts(int64_t pts) const;

  // Returns the key frame index of the presentation timestamp using our index.
  // We build this index by scanning the file in
  // scanFileAndUpdateMetadataAndIndex
  int getKeyFrameIndexForPtsUsingScannedIndex(
      const std::vector<SingleStreamDecoder::FrameInfo>& keyFrames,
      int64_t pts) const;

  int64_t secondsToIndexLowerBound(double seconds);

  int64_t secondsToIndexUpperBound(double seconds);

  int64_t getPts(int64_t frameIndex);

  // --------------------------------------------------------------------------
  // STREAM AND METADATA APIS
  // --------------------------------------------------------------------------

  void addStream(
      int streamIndex,
      AVMediaType mediaType,
      const torch::Device& device = torch::kCPU,
      const std::string_view deviceVariant = "default",
      std::optional<int> ffmpegThreadCount = std::nullopt);

  // Returns the "best" stream index for a given media type. The "best" is
  // determined by various heuristics in FFMPEG.
  // See
  // https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga757780d38f482deb4d809c6c521fbcc2
  // for more details about the heuristics.
  // Returns the key frame index of the presentation timestamp using FFMPEG's
  // index. Note that this index may be truncated for some files.
  int getBestStreamIndex(AVMediaType mediaType);

  std::optional<int64_t> getNumFrames(const StreamMetadata& streamMetadata);
  double getMinSeconds(const StreamMetadata& streamMetadata);
  std::optional<double> getMaxSeconds(const StreamMetadata& streamMetadata);

  // --------------------------------------------------------------------------
  // VALIDATION UTILS
  // --------------------------------------------------------------------------

  void validateActiveStream(
      std::optional<AVMediaType> avMediaType = std::nullopt);
  void validateScannedAllStreams(const std::string& msg);
  void validateFrameIndex(
      const StreamMetadata& streamMetadata,
      int64_t frameIndex);

  // --------------------------------------------------------------------------
  // ATTRIBUTES
  // --------------------------------------------------------------------------

  SeekMode seekMode_;
  ContainerMetadata containerMetadata_;
  UniqueDecodingAVFormatContext formatContext_;
  std::unique_ptr<DeviceInterface> deviceInterface_;
  std::map<int, StreamInfo> streamInfos_;
  const int NO_ACTIVE_STREAM = -2;
  int activeStreamIndex_ = NO_ACTIVE_STREAM;

  bool cursorWasJustSet_ = false;
  // The desired position of the cursor in the stream. We send frames >= this
  // pts to the user when they request a frame.
  int64_t cursor_ = INT64_MIN;
  // Stores various internal decoding stats.
  DecodeStats decodeStats_;
  // Stores the AVIOContext for the input buffer.
  std::unique_ptr<AVIOContextHolder> avioContextHolder_;
  // Whether or not we have already scanned all streams to update the metadata.
  bool scannedAllStreams_ = false;
  // Tracks that we've already been initialized.
  bool initialized_ = false;
};

// Prints the SingleStreamDecoder::DecodeStats to the ostream.
std::ostream& operator<<(
    std::ostream& os,
    const SingleStreamDecoder::DecodeStats& stats);

} // namespace facebook::torchcodec
