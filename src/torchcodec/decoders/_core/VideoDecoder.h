// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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
auto output = video_decoder->getNextDecodedFrame();
torch::Tensor frame = output.frame;
double presentation_timestamp = output.ptsSeconds;
// Note that presentation_timestamp can be any timestamp at 5.0 or above
// because the frame time may not align exactly with the seek time.
CHECK_GE(presentation_timestamp, 5.0);
*/
// Note that VideoDecoder is not thread-safe.
// Do not call non-const APIs concurrently on the same object.
// TODO: Rename this to be VideoReader.
class VideoDecoder {
 public:
  ~VideoDecoder() = default;

  struct DecoderOptions {
    DecoderOptions() {}
    // TODO: Add options for the entire decoder here.
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

  struct VideoStreamDecoderOptions {
    VideoStreamDecoderOptions() {}
    explicit VideoStreamDecoderOptions(const std::string& optionsString);
    // Number of threads we pass to FFMPEG for decoding.
    // 0 means FFMPEG will choose the number of threads automatically to fully
    // utilize all cores. If not set, it will be the default FFMPEG behavior for
    // the given codec.
    std::optional<int> ffmpegThreadCount;
    // Currently the shape can be either NHWC or NCHW.
    // H=height, W=width, C=channel.
    std::string shape = "NHWC";
    // The output height and width of the frame. If not specified, the output
    // is the same as the original video.
    std::optional<int> width;
    std::optional<int> height;
  };
  struct AudioStreamDecoderOptions {
    // TODO: Add channels, shape, sample options, etc.
  };
  void addVideoStreamDecoder(
      int streamIndex,
      const VideoStreamDecoderOptions& options = VideoStreamDecoderOptions());
  void addAudioStreamDecoder(
      int streamIndex,
      const AudioStreamDecoderOptions& options = AudioStreamDecoderOptions());

  // ---- SINGLE FRAME SEEK AND DECODING API ----
  // Places the cursor at the first frame on or after the position in seconds.
  // Calling getNextFrameAsTensor() will return the first frame at or after this
  // position.
  void setCursorPtsInSeconds(double seconds);
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
  };
  // Decodes the frame where the current cursor position is. It also advances
  // the cursor to the next frame.
  DecodedOutput getNextDecodedOutput();
  // Decodes the frame that is visible at a given timestamp. Frames in the video
  // have a presentation timestamp and a duration. For example, if a frame has
  // presentation timestamp of 5.0s and a duration of 1.0s, it will be visible
  // in the timestamp range [5.0, 6.0). i.e. it will be returned when this
  // function is called with seconds=5.0 or seconds=5.999, etc.
  DecodedOutput getFrameDisplayedAtTimestamp(double seconds);
  DecodedOutput getFrameAtIndex(int streamIndex, int64_t frameIndex);
  struct BatchDecodedOutput {
    torch::Tensor frames;
  };
  // Returns frames at the given indexes for a given stream as a single stacked
  // Tensor.
  BatchDecodedOutput getFramesAtIndexes(
      int streamIndex,
      const std::vector<int64_t>& frameIndexes);

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

 private:
  struct FrameInfo {
    int64_t pts = 0;
    // TODO: Add duration and dts, etc. as need be.
  };
  struct FilterState {
    UniqueAVFilterGraph filterGraph;
    AVFilterContext* sourceContext = nullptr;
    AVFilterContext* sinkContext = nullptr;
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
    // We set this field if the user requested a seek.
    std::optional<int64_t> discardFramesBeforePts = 0;
    VideoStreamDecoderOptions options;
    // The filter state associated with this stream (for video streams). The
    // actual graph will be nullptr for inactive streams.
    FilterState filterState;
    std::vector<FrameInfo> keyFrames;
    std::vector<FrameInfo> allFrames;
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
  // Creates and initializes a filter graph for a stream. The filter graph can
  // do rescaling and color conversion.
  void initializeFilterGraphForStream(
      int streamIndex,
      const VideoStreamDecoderOptions& options);
  void maybeSeekToBeforeDesiredPts();
  DecodedOutput getDecodedOutputWithFilter(std::function<bool(int, AVFrame*)>);
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
  DecodedOutput convertAVFrameToDecodedOutput(
      int streamIndex,
      UniqueAVFrame frame);

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
};

// Prints the VideoDecoder::DecodeStats to the ostream.
std::ostream& operator<<(
    std::ostream& os,
    const VideoDecoder::DecodeStats& stats);

} // namespace facebook::torchcodec
