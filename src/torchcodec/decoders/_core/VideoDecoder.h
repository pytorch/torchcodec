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

#include "src/torchcodec/decoders/_core/AVIOContextHolder.h"
#include "src/torchcodec/decoders/_core/FFMPEGCommon.h"

namespace facebook::torchcodec {

class DeviceInterface;

// The VideoDecoder class can be used to decode video frames to Tensors.
// Note that VideoDecoder is not thread-safe.
// Do not call non-const APIs concurrently on the same object.
class VideoDecoder {
 public:
  ~VideoDecoder();

  // --------------------------------------------------------------------------
  // CONSTRUCTION API
  // --------------------------------------------------------------------------

  enum class SeekMode { exact, approximate };

  // Creates a VideoDecoder from the video at videoFilePath.
  explicit VideoDecoder(
      const std::string& videoFilePath,
      SeekMode seekMode = SeekMode::exact);

  // Creates a VideoDecoder using the provided AVIOContext inside the
  // AVIOContextHolder. The AVIOContextHolder is the base class, and the
  // derived class will have specialized how the custom read, seek and writes
  // work.
  explicit VideoDecoder(
      std::unique_ptr<AVIOContextHolder> context,
      SeekMode seekMode = SeekMode::exact);

  // --------------------------------------------------------------------------
  // VIDEO METADATA QUERY API
  // --------------------------------------------------------------------------

  // Updates the metadata of the video to accurate values obtained by scanning
  // the contents of the video file. Also updates each StreamInfo's index, i.e.
  // the allFrames and keyFrames vectors.
  void scanFileAndUpdateMetadataAndIndex();

  struct StreamMetadata {
    // Common (video and audio) fields derived from the AVStream.
    int streamIndex;
    // See this link for what various values are available:
    // https://ffmpeg.org/doxygen/trunk/group__lavu__misc.html#ga9a84bba4713dfced21a1a56163be1f48
    AVMediaType mediaType;
    std::optional<AVCodecID> codecId;
    std::optional<std::string> codecName;
    std::optional<double> durationSeconds;
    std::optional<double> beginStreamFromHeader;
    std::optional<int64_t> numFrames;
    std::optional<int64_t> numKeyFrames;
    std::optional<double> averageFps;
    std::optional<double> bitRate;

    // More accurate duration, obtained by scanning the file.
    // These presentation timestamps are in time base.
    std::optional<int64_t> minPtsFromScan;
    std::optional<int64_t> maxPtsFromScan;
    // These presentation timestamps are in seconds.
    std::optional<double> minPtsSecondsFromScan;
    std::optional<double> maxPtsSecondsFromScan;
    // This can be useful for index-based seeking.
    std::optional<int64_t> numFramesFromScan;

    // Video-only fields derived from the AVCodecContext.
    std::optional<int64_t> width;
    std::optional<int64_t> height;

    // Audio-only fields
    std::optional<int64_t> sampleRate;
    std::optional<int64_t> numChannels;
    std::optional<std::string> sampleFormat;
  };

  struct ContainerMetadata {
    std::vector<StreamMetadata> allStreamMetadata;
    int numAudioStreams = 0;
    int numVideoStreams = 0;
    // Note that this is the container-level duration, which is usually the max
    // of all stream durations available in the container.
    std::optional<double> durationSeconds;
    // Total BitRate level information at the container level in bit/s
    std::optional<double> bitRate;
    // If set, this is the index to the default audio stream.
    std::optional<int> bestAudioStreamIndex;
    // If set, this is the index to the default video stream.
    std::optional<int> bestVideoStreamIndex;
  };

  // Returns the metadata for the container.
  ContainerMetadata getContainerMetadata() const;

  // Returns the key frame indices as a tensor. The tensor is 1D and contains
  // int64 values, where each value is the frame index for a key frame.
  torch::Tensor getKeyFrameIndices();

  // --------------------------------------------------------------------------
  // ADDING STREAMS API
  // --------------------------------------------------------------------------

  enum ColorConversionLibrary {
    // TODO: Add an AUTO option later.
    // Use the libavfilter library for color conversion.
    FILTERGRAPH,
    // Use the libswscale library for color conversion.
    SWSCALE
  };

  struct VideoStreamOptions {
    VideoStreamOptions() {}

    // Number of threads we pass to FFMPEG for decoding.
    // 0 means FFMPEG will choose the number of threads automatically to fully
    // utilize all cores. If not set, it will be the default FFMPEG behavior for
    // the given codec.
    std::optional<int> ffmpegThreadCount;
    // Currently the dimension order can be either NHWC or NCHW.
    // H=height, W=width, C=channel.
    std::string dimensionOrder = "NCHW";
    // The output height and width of the frame. If not specified, the output
    // is the same as the original video.
    std::optional<int> width;
    std::optional<int> height;
    std::optional<ColorConversionLibrary> colorConversionLibrary;
    // By default we use CPU for decoding for both C++ and python users.
    std::shared_ptr<DeviceInterface> device;
  };

  struct AudioStreamOptions {
    AudioStreamOptions() {}

    std::optional<int> sampleRate;
  };

  void addVideoStream(
      int streamIndex,
      const VideoStreamOptions& videoStreamOptions = VideoStreamOptions());
  void addAudioStream(
      int streamIndex,
      const AudioStreamOptions& audioStreamOptions = AudioStreamOptions());

  // --------------------------------------------------------------------------
  // DECODING AND SEEKING APIs
  // --------------------------------------------------------------------------

  // All public video decoding entry points return either a FrameOutput or a
  // FrameBatchOutput.
  // They are the equivalent of the user-facing Frame and FrameBatch classes in
  // Python. They contain RGB decoded frames along with some associated data
  // like PTS and duration.
  // FrameOutput is also relevant for audio decoding, typically as the output of
  // getNextFrame(), or as a temporary output variable.
  struct FrameOutput {
    // data shape is:
    // - 3D (C, H, W) or (H, W, C) for videos
    // - 2D (numChannels, numSamples) for audio
    torch::Tensor data;
    double ptsSeconds;
    double durationSeconds;
  };

  struct FrameBatchOutput {
    torch::Tensor data; // 4D: of shape NCHW or NHWC.
    torch::Tensor ptsSeconds; // 1D of shape (N,)
    torch::Tensor durationSeconds; // 1D of shape (N,)

    explicit FrameBatchOutput(
        int64_t numFrames,
        const VideoStreamOptions& videoStreamOptions,
        const StreamMetadata& streamMetadata);
  };

  struct AudioFramesOutput {
    torch::Tensor data; // shape is (numChannels, numSamples)
    double ptsSeconds;
  };

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
  //   [minPtsSecondsFromScan, maxPtsSecondsFromScan)
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

  struct FilterGraphContext {
    UniqueAVFilterGraph filterGraph;
    AVFilterContext* sourceContext = nullptr;
    AVFilterContext* sinkContext = nullptr;
  };

  struct DecodedFrameContext {
    int decodedWidth;
    int decodedHeight;
    AVPixelFormat decodedFormat;
    int expectedWidth;
    int expectedHeight;
    bool operator==(const DecodedFrameContext&);
    bool operator!=(const DecodedFrameContext&);
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
    FilterGraphContext filterGraphContext;
    ColorConversionLibrary colorConversionLibrary = FILTERGRAPH;
    UniqueSwsContext swsContext;
    UniqueSwrContext swrContext;

    // Used to know whether a new FilterGraphContext or UniqueSwsContext should
    // be created before decoding a new frame.
    DecodedFrameContext prevFrameContext;
  };

  // --------------------------------------------------------------------------
  // INITIALIZERS
  // --------------------------------------------------------------------------

  void initializeDecoder();
  void setFFmpegLogLevel();
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

  UniqueAVFrame convertAudioAVFrameSampleFormatAndSampleRate(
      const UniqueAVFrame& srcAVFrame,
      AVSampleFormat sourceSampleFormat,
      AVSampleFormat desiredSampleFormat,
      int sourceSampleRate,
      int desiredSampleRate);

  std::optional<torch::Tensor> maybeFlushSwrBuffers();

  // --------------------------------------------------------------------------
  // COLOR CONVERSION LIBRARIES HANDLERS CREATION
  // --------------------------------------------------------------------------

  void createFilterGraph(
      StreamInfo& streamInfo,
      int expectedOutputHeight,
      int expectedOutputWidth);

  void createSwsContext(
      StreamInfo& streamInfo,
      const DecodedFrameContext& frameContext,
      const enum AVColorSpace colorspace);

  void createSwrContext(
      StreamInfo& streamInfo,
      AVSampleFormat sourceSampleFormat,
      AVSampleFormat desiredSampleFormat,
      int sourceSampleRate,
      int desiredSampleRate);

  // --------------------------------------------------------------------------
  // PTS <-> INDEX CONVERSIONS
  // --------------------------------------------------------------------------

  int getKeyFrameIndexForPts(int64_t pts) const;

  // Returns the key frame index of the presentation timestamp using our index.
  // We build this index by scanning the file in
  // scanFileAndUpdateMetadataAndIndex
  int getKeyFrameIndexForPtsUsingScannedIndex(
      const std::vector<VideoDecoder::FrameInfo>& keyFrames,
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
      DeviceInterface* device = nullptr,
      std::optional<int> ffmpegThreadCount = std::nullopt);

  // Returns the "best" stream index for a given media type. The "best" is
  // determined by various heuristics in FFMPEG.
  // See
  // https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga757780d38f482deb4d809c6c521fbcc2
  // for more details about the heuristics.
  // Returns the key frame index of the presentation timestamp using FFMPEG's
  // index. Note that this index may be truncated for some files.
  int getBestStreamIndex(AVMediaType mediaType);

  int64_t getNumFrames(const StreamMetadata& streamMetadata);
  double getMinSeconds(const StreamMetadata& streamMetadata);
  double getMaxSeconds(const StreamMetadata& streamMetadata);

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
  UniqueAVFormatContext formatContext_;
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

// --------------------------------------------------------------------------
// FRAME TENSOR ALLOCATION APIs
// --------------------------------------------------------------------------

// Note [Frame Tensor allocation and height and width]
//
// We always allocate [N]HWC tensors. The low-level decoding functions all
// assume HWC tensors, since this is what FFmpeg natively handles. It's up to
// the high-level decoding entry-points to permute that back to CHW, by calling
// maybePermuteHWC2CHW().
//
// Also, importantly, the way we figure out the the height and width of the
// output frame tensor varies, and depends on the decoding entry-point. In
// *decreasing order of accuracy*, we use the following sources for determining
// height and width:
// - getHeightAndWidthFromResizedAVFrame(). This is the height and width of the
//   AVframe, *post*-resizing. This is only used for single-frame decoding APIs,
//   on CPU, with filtergraph.
// - getHeightAndWidthFromOptionsOrAVFrame(). This is the height and width from
//   the user-specified options if they exist, or the height and width of the
//   AVFrame *before* it is resized. In theory, i.e. if there are no bugs within
//   our code or within FFmpeg code, this should be exactly the same as
//   getHeightAndWidthFromResizedAVFrame(). This is used by single-frame
//   decoding APIs, on CPU with swscale, and on GPU.
// - getHeightAndWidthFromOptionsOrMetadata(). This is the height and width from
//   the user-specified options if they exist, or the height and width form the
//   stream metadata, which itself got its value from the CodecContext, when the
//   stream was added. This is used by batch decoding APIs, for both GPU and
//   CPU.
//
// The source of truth for height and width really is the (resized) AVFrame: it
// comes from the decoded ouptut of FFmpeg. The info from the metadata (i.e.
// from the CodecContext) may not be as accurate. However, the AVFrame is only
// available late in the call stack, when the frame is decoded, while the
// CodecContext is available early when a stream is added. This is why we use
// the CodecContext for pre-allocating batched output tensors (we could
// pre-allocate those only once we decode the first frame to get the info frame
// the AVFrame, but that's a more complex logic).
//
// Because the sources for height and width may disagree, we may end up with
// conflicts: e.g. if we pre-allocate a batch output tensor based on the
// metadata info, but the decoded AVFrame has a different height and width.
// it is very important to check the height and width assumptions where the
// tensors memory is used/filled in order to avoid segfaults.

struct FrameDims {
  int height;
  int width;

  FrameDims(int h, int w) : height(h), width(w) {}
};

// There's nothing preventing you from calling this on a non-resized frame, but
// please don't.
FrameDims getHeightAndWidthFromResizedAVFrame(const AVFrame& resizedAVFrame);

FrameDims getHeightAndWidthFromOptionsOrMetadata(
    const VideoDecoder::VideoStreamOptions& videoStreamOptions,
    const VideoDecoder::StreamMetadata& streamMetadata);

FrameDims getHeightAndWidthFromOptionsOrAVFrame(
    const VideoDecoder::VideoStreamOptions& videoStreamOptions,
    const UniqueAVFrame& avFrame);

torch::Tensor allocateEmptyHWCTensor(
    int height,
    int width,
    torch::Device device,
    std::optional<int> numFrames = std::nullopt);

// Prints the VideoDecoder::DecodeStats to the ostream.
std::ostream& operator<<(
    std::ostream& os,
    const VideoDecoder::DecodeStats& stats);

VideoDecoder::SeekMode seekModeFromString(std::string_view seekMode);

} // namespace facebook::torchcodec
