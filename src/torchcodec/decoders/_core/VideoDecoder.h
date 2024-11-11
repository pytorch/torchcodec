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

#include "src/torchcodec/decoders/_core/FFMPEGCommon.h"

namespace facebook::torchcodec {

/*
The VideoDecoder class can be used to decode video frames to Tensors.

Example usage of this class:
std::string video_file_path = "/path/to/video.mp4";
VideoDecoder video_decoder = VideoDecoder::createFromFilePath(video_file_path);

// After creating the decoder, we can query the metadata:
auto metadata = video_decoder.getContainerMetadata();

// We can also add streams to the decoder:
// -1 sets the default stream.
video_decoder.addVideoStreamDecoder(-1);

// API for seeking and frame extraction:
// Let's extract the first frame at or after pts=5.0 seconds.
video_decoder.setCursorPtsInSeconds(5.0);
auto output = video_decoder->getNextDecodedOutput();
torch::Tensor frame = output.frame;
double presentation_timestamp = output.ptsSeconds;
// Note that presentation_timestamp can be any timestamp at 5.0 or above
// because the frame time may not align exactly with the seek time.
CHECK_GE(presentation_timestamp, 5.0);
*/
// Note that VideoDecoder is not thread-safe.
// Do not call non-const APIs concurrently on the same object.
class VideoDecoder {
 public:
  ~VideoDecoder();

  struct DecoderOptions {
    DecoderOptions() {}
    // TODO: Add options for the entire decoder here, or remove if not needed.
  };

  // --------------------------------------------------------------------------
  // CONSTRUCTION API
  // --------------------------------------------------------------------------

  // Creates a VideoDecoder with the given options for the video in file
  // `videoFilePath`. If it fails, returns an error status.
  static std::unique_ptr<VideoDecoder> createFromFilePath(
      const std::string& videoFilePath,
      const DecoderOptions& options = DecoderOptions());

  // Creates a VideoDecoder from a given buffer. Note that the buffer is not
  // owned by the VideoDecoder.
  static std::unique_ptr<VideoDecoder> createFromBuffer(
      const void* buffer,
      size_t length,
      const DecoderOptions& options = DecoderOptions());

  // --------------------------------------------------------------------------
  // VIDEO METADATA QUERY API
  // --------------------------------------------------------------------------
  // Updates the metadata of the video to accurate values obtained by scanning
  // the contents of the video file.
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
    std::optional<int64_t> numFrames;
    std::optional<int64_t> numKeyFrames;
    std::optional<double> averageFps;
    std::optional<double> bitRate;
    std::optional<std::vector<int64_t>> keyFrames;

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
  };
  struct ContainerMetadata {
    std::vector<StreamMetadata> streams;
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
  struct VideoStreamDecoderOptions {
    VideoStreamDecoderOptions() {}
    explicit VideoStreamDecoderOptions(const std::string& optionsString);
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
    torch::Device device = torch::kCPU;
  };
  struct AudioStreamDecoderOptions {};
  void addVideoStreamDecoder(
      int streamIndex,
      const VideoStreamDecoderOptions& options = VideoStreamDecoderOptions());
  void addAudioStreamDecoder(
      int streamIndex,
      const AudioStreamDecoderOptions& options = AudioStreamDecoderOptions());

  torch::Tensor MaybePermuteHWC2CHW(int streamIndex, torch::Tensor& hwcTensor);

  // ---- SINGLE FRAME SEEK AND DECODING API ----
  // Places the cursor at the first frame on or after the position in seconds.
  // Calling getNextFrameOutputNoDemuxInternal() will return the first frame at
  // or after this position.
  void setCursorPtsInSeconds(double seconds);
  // This is an internal structure that is used to store the decoded output
  // from decoding a frame through color conversion. Example usage is:
  //
  // RawDecodedOutput rawOutput = getDecodedOutputWithFilter();
  // // Now allocate a single tensor or a batch tensor.
  // torch::Tensor userOutput = torch::empty(...);
  // // Now fill in `data` and `size`.
  // rawOutput.data = userOutput.data_ptr();
  // // Now run the color conversion.
  // convertFrameToBufferUsingSwsScale(rawOutput);
  //
  // This structure ensures we always keep the streamIndex and frame together
  // with the data output. Note that AVFrame itself doesn't retain the
  // streamIndex.
  struct RawDecodedOutput {
    // The actual decoded output as a unique pointer to an AVFrame.
    UniqueAVFrame frame;
    // The stream index of the decoded frame.
    int streamIndex;
    // This is an unowned pointer that we copy the frame data to after color
    // conversion.
    // For a single tensor this points to the start of data_ptr. For a batch
    // tensor it may point to the middle of the allocated batch tensor.
    void* data = nullptr;
    // We carry around the size to ensure we don't stomp on memory while doing
    // color conversion.
    size_t size = 0;
  };
  struct DecodedOutput {
    // The actual decoded output as a Tensor.
    torch::Tensor frame;
    // Could be AVMEDIA_TYPE_VIDEO or AVMEDIA_TYPE_AUDIO.
    AVMediaType streamType;
    // The stream index of the decoded frame. Used to distinguish
    // between streams that are of the same type.
    int streamIndex;
    // The presentation timestamp of the decoded frame in time base.
    int64_t pts;
    // The presentation timestamp of the decoded frame in seconds.
    double ptsSeconds;
    // The duration of the decoded frame in time base.
    int64_t duration;
    // The duration of the decoded frame in seconds.
    double durationSeconds;
  };
  class EndOfFileException : public std::runtime_error {
   public:
    explicit EndOfFileException(const std::string& msg)
        : std::runtime_error(msg) {}
  };
  // Decodes the frame where the current cursor position is. It also advances
  // the cursor to the next frame.
  DecodedOutput getNextFrameNoDemux();
  // Decodes the first frame in any added stream that is visible at a given
  // timestamp. Frames in the video have a presentation timestamp and a
  // duration. For example, if a frame has presentation timestamp of 5.0s and a
  // duration of 1.0s, it will be visible in the timestamp range [5.0, 6.0).
  // i.e. it will be returned when this function is called with seconds=5.0 or
  // seconds=5.999, etc.
  DecodedOutput getFramePlayedAtTimestampNoDemux(double seconds);

  DecodedOutput getFrameAtIndex(int streamIndex, int64_t frameIndex);
  // This is morally private but needs to be exposed for C++ tests. Once
  // getFrameAtIndex supports the preAllocatedOutputTensor parameter, we can
  // move it back to private.
  DecodedOutput getFrameAtIndexInternal(
      int streamIndex,
      int64_t frameIndex,
      std::optional<torch::Tensor> preAllocatedOutputTensor = std::nullopt);
  struct BatchDecodedOutput {
    torch::Tensor frames;
    torch::Tensor ptsSeconds;
    torch::Tensor durationSeconds;

    explicit BatchDecodedOutput(
        int64_t numFrames,
        const VideoStreamDecoderOptions& options,
        const StreamMetadata& metadata);
  };

  // Returns frames at the given indices for a given stream as a single stacked
  // Tensor.
  BatchDecodedOutput getFramesAtIndices(
      int streamIndex,
      const std::vector<int64_t>& frameIndices);

  BatchDecodedOutput getFramesPlayedByTimestamps(
      int streamIndex,
      const std::vector<double>& timestamps);

  // Returns frames within a given range for a given stream as a single stacked
  // Tensor. The range is defined by [start, stop). The values retrieved from
  // the range are:
  //    [start, start+step, start+(2*step), start+(3*step), ..., stop)
  // The default for step is 1.
  BatchDecodedOutput
  getFramesInRange(int streamIndex, int64_t start, int64_t stop, int64_t step);

  // Returns frames within a given pts range for a given stream as a single
  // stacked tensor. The range is defined by [startSeconds, stopSeconds) with
  // respect to the pts values for frames. The returned frames are in pts order.
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
  BatchDecodedOutput getFramesPlayedByTimestampInRange(
      int streamIndex,
      double startSeconds,
      double stopSeconds);
  // --------------------------------------------------------------------------
  // DECODER PERFORMANCE STATISTICS API
  // --------------------------------------------------------------------------

  // Only exposed for performance testing.
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

  double getPtsSecondsForFrame(int streamIndex, int64_t frameIndex);

 private:
  struct FrameInfo {
    int64_t pts = 0;
    // The value of this default is important: the last frame's nextPts will be
    // INT64_MAX, which ensures that the allFrames vec contains FrameInfo
    // structs with *increasing* nextPts values. That's a necessary condition
    // for the binary searches on those values to work properly (as typically
    // done during pts -> index conversions.)
    int64_t nextPts = INT64_MAX;
  };
  struct FilterState {
    UniqueAVFilterGraph filterGraph;
    AVFilterContext* sourceContext = nullptr;
    AVFilterContext* sinkContext = nullptr;
  };
  struct SwsContextKey {
    int decodedWidth;
    int decodedHeight;
    AVPixelFormat decodedFormat;
    int outputWidth;
    int outputHeight;
    bool operator==(const SwsContextKey&);
    bool operator!=(const SwsContextKey&);
  };
  // Stores information for each stream.
  struct StreamInfo {
    int streamIndex = -1;
    AVStream* stream = nullptr;
    AVRational timeBase = {};
    UniqueAVCodecContext codecContext;
    // The current position of the cursor in the stream.
    int64_t currentPts = 0;
    int64_t currentDuration = 0;
    // The desired position of the cursor in the stream. We send frames >=
    // this pts to the user when they request a frame.
    // We update this field if the user requested a seek.
    int64_t discardFramesBeforePts = INT64_MIN;
    VideoStreamDecoderOptions options;
    // The filter state associated with this stream (for video streams). The
    // actual graph will be nullptr for inactive streams.
    FilterState filterState;
    ColorConversionLibrary colorConversionLibrary = FILTERGRAPH;
    std::vector<FrameInfo> keyFrames;
    std::vector<FrameInfo> allFrames;
    SwsContextKey swsContextKey;
    UniqueSwsContext swsContext;
  };
  VideoDecoder();
  // Returns the key frame index of the presentation timestamp using FFMPEG's
  // index. Note that this index may be truncated for some files.
  int getKeyFrameIndexForPtsUsingEncoderIndex(AVStream* stream, int64_t pts)
      const;
  // Returns the key frame index of the presentation timestamp using our index.
  // We build this index by scanning the file in buildKeyFrameIndex().
  int getKeyFrameIndexForPtsUsingScannedIndex(
      const std::vector<VideoDecoder::FrameInfo>& keyFrames,
      int64_t pts) const;
  int getKeyFrameIndexForPts(const StreamInfo& stream, int64_t pts) const;
  bool canWeAvoidSeekingForStream(
      const StreamInfo& stream,
      int64_t currentPts,
      int64_t targetPts) const;
  // Returns the "best" stream index for a given media type. The "best" is
  // determined by various heuristics in FFMPEG.
  // See
  // https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga757780d38f482deb4d809c6c521fbcc2
  // for more details about the heuristics.
  int getBestStreamIndex(AVMediaType mediaType);
  void initializeDecoder();
  void validateUserProvidedStreamIndex(uint64_t streamIndex);
  void validateScannedAllStreams(const std::string& msg);
  void validateFrameIndex(const StreamInfo& stream, int64_t frameIndex);
  // Creates and initializes a filter graph for a stream. The filter graph can
  // do rescaling and color conversion.
  void initializeFilterGraphForStream(
      int streamIndex,
      const VideoStreamDecoderOptions& options);
  void maybeSeekToBeforeDesiredPts();
  RawDecodedOutput getDecodedOutputWithFilter(
      std::function<bool(int, AVFrame*)>);
  RawDecodedOutput getNextRawDecodedOutputNoDemux();
  // Once we create a decoder can update the metadata with the codec context.
  // For example, for video streams, we can add the height and width of the
  // decoded stream.
  void updateMetadataWithCodecContext(
      int streamIndex,
      AVCodecContext* codecContext);
  void populateVideoMetadataFromStreamIndex(int streamIndex);
  torch::Tensor convertFrameToTensorUsingFilterGraph(
      int streamIndex,
      const AVFrame* frame);
  int convertFrameToBufferUsingSwsScale(
      int streamIndex,
      const AVFrame* frame,
      torch::Tensor& outputTensor);
  DecodedOutput convertAVFrameToDecodedOutput(
      RawDecodedOutput& rawOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor = std::nullopt);
  void convertAVFrameToDecodedOutputOnCPU(
      RawDecodedOutput& rawOutput,
      DecodedOutput& output,
      std::optional<torch::Tensor> preAllocatedOutputTensor = std::nullopt);

  DecodedOutput getNextFrameOutputNoDemuxInternal(
      std::optional<torch::Tensor> preAllocatedOutputTensor = std::nullopt);

  DecoderOptions options_;
  ContainerMetadata containerMetadata_;
  UniqueAVFormatContext formatContext_;
  std::map<int, StreamInfo> streams_;
  // Stores the stream indices of the active streams, i.e. the streams we are
  // decoding and returning to the user.
  std::set<int> activeStreamIndices_;
  // Set when the user wants to seek and stores the desired pts that the user
  // wants to seek to.
  std::optional<double> maybeDesiredPts_;

  // Stores various internal decoding stats.
  DecodeStats decodeStats_;
  // Stores the AVIOContext for the input buffer.
  std::unique_ptr<AVIOBytesContext> ioBytesContext_;
  // Whether or not we have already scanned all streams to update the metadata.
  bool scanned_all_streams_ = false;
};

// --------------------------------------------------------------------------
// FRAME TENSOR ALLOCATION APIs
// --------------------------------------------------------------------------

// Note [Frame Tensor allocation and height and width]
//
// We always allocate [N]HWC tensors. The low-level decoding functions all
// assume HWC tensors, since this is what FFmpeg natively handles. It's up to
// the high-level decoding entry-points to permute that back to CHW, by calling
// MaybePermuteHWC2CHW().
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
    const VideoDecoder::VideoStreamDecoderOptions& options,
    const VideoDecoder::StreamMetadata& metadata);

FrameDims getHeightAndWidthFromOptionsOrAVFrame(
    const VideoDecoder::VideoStreamDecoderOptions& options,
    const AVFrame& avFrame);

torch::Tensor allocateEmptyHWCTensor(
    int height,
    int width,
    torch::Device device,
    std::optional<int> numFrames = std::nullopt);

// Prints the VideoDecoder::DecodeStats to the ostream.
std::ostream& operator<<(
    std::ostream& os,
    const VideoDecoder::DecodeStats& stats);

} // namespace facebook::torchcodec
