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
#include "src/torchcodec/_core/FFMPEGCommon.h"

namespace facebook::torchcodec {

// The SingleStreamDecoder class can be used to decode video frames to Tensors.
// Note that SingleStreamDecoder is not thread-safe.
// Do not call non-const APIs concurrently on the same object.
class SingleStreamDecoder {
 public:
  ~SingleStreamDecoder();

  // --------------------------------------------------------------------------
  // CONSTRUCTION API
  // --------------------------------------------------------------------------

  enum class SeekMode { exact, approximate };

  // Creates a SingleStreamDecoder from the video at videoFilePath.
  explicit SingleStreamDecoder(
      const std::string& video_file_path,
      SeekMode seek_mode = SeekMode::exact);

  // Creates a SingleStreamDecoder using the provided AVIOContext inside the
  // AVIOContextHolder. The AVIOContextHolder is the base class, and the
  // derived class will have specialized how the custom read, seek and writes
  // work.
  explicit SingleStreamDecoder(
      std::unique_ptr<AVIOContext_holder> context,
      SeekMode seek_mode = SeekMode::exact);

  // --------------------------------------------------------------------------
  // VIDEO METADATA QUERY API
  // --------------------------------------------------------------------------

  // Updates the metadata of the video to accurate values obtained by scanning
  // the contents of the video file. Also updates each StreamInfo's index, i.e.
  // the allFrames and keyFrames vectors.
  void scan_file_and_update_metadata_and_index();

  struct StreamMetadata {
    // Common (video and audio) fields derived from the AVStream.
    int stream_index;
    // See this link for what various values are available:
    // https://ffmpeg.org/doxygen/trunk/group__lavu__misc.html#ga9a84bba4713dfced21a1a56163be1f48
    AVMediaType media_type;
    std::optional<_avcodec_i_d> codec_id;
    std::optional<std::string> codec_name;
    std::optional<double> duration_seconds;
    std::optional<double> begin_stream_from_header;
    std::optional<int64_t> num_frames;
    std::optional<int64_t> num_key_frames;
    std::optional<double> average_fps;
    std::optional<double> bit_rate;

    // More accurate duration, obtained by scanning the file.
    // These presentation timestamps are in time base.
    std::optional<int64_t> min_pts_from_scan;
    std::optional<int64_t> max_pts_from_scan;
    // These presentation timestamps are in seconds.
    std::optional<double> min_pts_seconds_from_scan;
    std::optional<double> max_pts_seconds_from_scan;
    // This can be useful for index-based seeking.
    std::optional<int64_t> num_frames_from_scan;

    // Video-only fields derived from the AVCodecContext.
    std::optional<int64_t> width;
    std::optional<int64_t> height;

    // Audio-only fields
    std::optional<int64_t> sample_rate;
    std::optional<int64_t> num_channels;
    std::optional<std::string> sample_format;
  };

  struct ContainerMetadata {
    std::vector<_stream_metadata> all_stream_metadata;
    int num_audio_streams = 0;
    int num_video_streams = 0;
    // Note that this is the container-level duration, which is usually the max
    // of all stream durations available in the container.
    std::optional<double> duration_seconds;
    // Total BitRate level information at the container level in bit/s
    std::optional<double> bit_rate;
    // If set, this is the index to the default audio stream.
    std::optional<int> best_audio_stream_index;
    // If set, this is the index to the default video stream.
    std::optional<int> best_video_stream_index;
  };

  // Returns the metadata for the container.
  ContainerMetadata get_container_metadata() const;

  // Returns the key frame indices as a tensor. The tensor is 1D and contains
  // int64 values, where each value is the frame index for a key frame.
  torch::Tensor get_key_frame_indices();

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
    std::optional<int> ffmpeg_thread_count;
    // Currently the dimension order can be either NHWC or NCHW.
    // H=height, W=width, C=channel.
    std::string dimension_order = "NCHW";
    // The output height and width of the frame. If not specified, the output
    // is the same as the original video.
    std::optional<int> width;
    std::optional<int> height;
    std::optional<_color_conversion_library> color_conversion_library;
    // By default we use CPU for decoding for both C++ and python users.
    torch::Device device = torch::kCPU;
  };

  struct AudioStreamOptions {
    AudioStreamOptions() {}

    std::optional<int> sample_rate;
  };

  void add_video_stream(
      int stream_index,
      const VideoStreamOptions& video_stream_options = VideoStreamOptions());
  void add_audio_stream(
      int stream_index,
      const AudioStreamOptions& audio_stream_options = AudioStreamOptions());

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
    double pts_seconds;
    double duration_seconds;
  };

  struct FrameBatchOutput {
    torch::Tensor data; // 4D: of shape NCHW or NHWC.
    torch::Tensor pts_seconds; // 1D of shape (N,)
    torch::Tensor duration_seconds; // 1D of shape (N,)

    explicit FrameBatchOutput(
        int64_t num_frames,
        const VideoStreamOptions& video_stream_options,
        const StreamMetadata& stream_metadata);
  };

  struct AudioFramesOutput {
    torch::Tensor data; // shape is (numChannels, num_samples)
    double pts_seconds;
  };

  // Places the cursor at the first frame on or after the position in seconds.
  // Calling getNextFrame() will return the first frame at
  // or after this position.
  void set_cursor_pts_in_seconds(double seconds);

  // Decodes the frame where the current cursor position is. It also advances
  // the cursor to the next frame.
  FrameOutput get_next_frame();

  FrameOutput get_frame_at_index(int64_t frame_index);

  // Returns frames at the given indices for a given stream as a single stacked
  // Tensor.
  FrameBatchOutput get_frames_at_indices(
      const std::vector<int64_t>& frame_indices);

  // Returns frames within a given range. The range is defined by [start, stop).
  // The values retrieved from the range are: [start, start+step,
  // start+(2*step), start+(3*step), ..., stop). The default for step is 1.
  FrameBatchOutput
  get_frames_in_range(int64_t start, int64_t stop, int64_t step);

  // Decodes the first frame in any added stream that is visible at a given
  // timestamp. Frames in the video have a presentation timestamp and a
  // duration. For example, if a frame has presentation timestamp of 5.0s and a
  // duration of 1.0s, it will be visible in the timestamp range [5.0, 6.0).
  // i.e. it will be returned when this function is called with seconds=5.0 or
  // seconds=5.999, etc.
  FrameOutput get_frame_played_at(double seconds);

  FrameBatchOutput get_frames_played_at(const std::vector<double>& timestamps);

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
  // [minPtsSecondsFromScan, maxPtsSecondsFromScan)
  FrameBatchOutput get_frames_played_in_range(
      double start_seconds,
      double stop_seconds);

  AudioFramesOutput get_frames_played_in_range_audio(
      double start_seconds,
      std::optional<double> stop_seconds_optional = std::nullopt);

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
  FrameOutput get_frame_at_index_internal(
      int64_t frame_index,
      std::optional<torch::Tensor> pre_allocated_output_tensor = std::nullopt);

  // Exposed for _test_frame_pts_equality, which is used to test non-regression
  // of pts resolution (64 to 32 bit floats)
  double get_pts_seconds_for_frame(int64_t frame_index);

  // Exposed for performance testing.
  struct DecodeStats {
    int64_t num_seeks_attempted = 0;
    int64_t num_seeks_done = 0;
    int64_t num_seeks_skipped = 0;
    int64_t num_packets_read = 0;
    int64_t num_packets_sent_to_decoder = 0;
    int64_t num_frames_received_by_decoder = 0;
    int64_t num_flushes = 0;
  };

  DecodeStats get_decode_stats() const;
  void reset_decode_stats();

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
    int64_t next_pts = INT64_MAX;

    // Note that frameIndex is ALWAYS the index into all of the frames in that
    // stream, even when the FrameInfo is part of the key frame index. Given a
    // FrameInfo for a key frame, the frameIndex allows us to know which frame
    // that is in the stream.
    int64_t frame_index = 0;

    // Indicates whether a frame is a key frame. It may appear redundant as it's
    // only true for FrameInfos in the keyFrames index, but it is needed to
    // correctly map frames between allFrames and keyFrames during the scan.
    bool is_key_frame = false;
  };

  struct FilterGraphContext {
    UniqueAVFilterGraph filter_graph;
    AVFilterContext* source_context = nullptr;
    AVFilterContext* sink_context = nullptr;
  };

  struct DecodedFrameContext {
    int decoded_width;
    int decoded_height;
    AVPixelFormat decoded_format;
    int expected_width;
    int expected_height;
    bool operator==(const DecodedFrameContext&);
    bool operator!=(const DecodedFrameContext&);
  };

  struct StreamInfo {
    int stream_index = -1;
    AVStream* stream = nullptr;
    AVMediaType av_media_type = AVMEDIA_TYPE_UNKNOWN;

    AVRational time_base = {};
    UniqueAVCodecContext codec_context;

    // The FrameInfo indices we built when scanFileAndUpdateMetadataAndIndex was
    // called.
    std::vector<_frame_info> key_frames;
    std::vector<_frame_info> all_frames;

    // TODO since the decoder is single-stream, these should be decoder fields,
    // not streamInfo fields. And they should be defined right next to
    // `cursor_`, with joint documentation.
    int64_t last_decoded_avframe_pts = 0;
    int64_t last_decoded_avframe_duration = 0;
    VideoStreamOptions video_stream_options;
    AudioStreamOptions audio_stream_options;

    // color-conversion fields. Only one of FilterGraphContext and
    // UniqueSwsContext should be non-null.
    FilterGraphContext filter_graph_context;
    ColorConversionLibrary color_conversion_library = FILTERGRAPH;
    UniqueSwsContext sws_context;
    UniqueSwrContext swr_context;

    // Used to know whether a new FilterGraphContext or UniqueSwsContext should
    // be created before decoding a new frame.
    DecodedFrameContext prev_frame_context;
  };

  // --------------------------------------------------------------------------
  // INITIALIZERS
  // --------------------------------------------------------------------------

  void initialize_decoder();
  void set_ffmpeg_log_level();
  // --------------------------------------------------------------------------
  // DECODING APIS AND RELATED UTILS
  // --------------------------------------------------------------------------

  void set_cursor(int64_t pts);
  void set_cursor(double) = delete; // prevent calls with doubles and floats
  bool can_we_avoid_seeking() const;

  void maybe_seek_to_before_desired_pts();

  UniqueAVFrame decode_avframe(
      std::function<bool(const UniqueAVFrame&)> filter_function);

  FrameOutput get_next_frame_internal(
      std::optional<torch::Tensor> pre_allocated_output_tensor = std::nullopt);

  torch::Tensor maybePermuteHWC2CHW(torch::Tensor& hwc_tensor);

  FrameOutput convert_avframe_to_frame_output(
      UniqueAVFrame& avframe,
      std::optional<torch::Tensor> pre_allocated_output_tensor = std::nullopt);

  void convert_avframe_to_frame_output_on_c_p_u(
      UniqueAVFrame& avframe,
      FrameOutput& frame_output,
      std::optional<torch::Tensor> pre_allocated_output_tensor = std::nullopt);

  void convert_audio_avframe_to_frame_output_on_c_p_u(
      UniqueAVFrame& src_avframe,
      FrameOutput& frame_output);

  torch::Tensor convert_avframe_to_tensor_using_filter_graph(
      const UniqueAVFrame& avframe);

  int convert_avframe_to_tensor_using_sws_scale(
      const UniqueAVFrame& avframe,
      torch::Tensor& output_tensor);

  UniqueAVFrame convert_audio_avframe_sample_format_and_sample_rate(
      const UniqueAVFrame& src_avframe,
      AVSampleFormat source_sample_format,
      AVSampleFormat desired_sample_format,
      int source_sample_rate,
      int desired_sample_rate);

  std::optional<torch::Tensor> maybe_flush_swr_buffers();

  // --------------------------------------------------------------------------
  // COLOR CONVERSION LIBRARIES HANDLERS CREATION
  // --------------------------------------------------------------------------

  void create_filter_graph(
      StreamInfo& stream_info,
      int expected_output_height,
      int expected_output_width);

  void create_sws_context(
      StreamInfo& stream_info,
      const DecodedFrameContext& frame_context,
      const enum AVColorSpace colorspace);

  void create_swr_context(
      StreamInfo& stream_info,
      AVSampleFormat source_sample_format,
      AVSampleFormat desired_sample_format,
      int source_sample_rate,
      int desired_sample_rate);

  // --------------------------------------------------------------------------
  // PTS <-> INDEX CONVERSIONS
  // --------------------------------------------------------------------------

  int get_key_frame_index_for_pts(int64_t pts) const;

  // Returns the key frame index of the presentation timestamp using our index.
  // We build this index by scanning the file in
  // scanFileAndUpdateMetadataAndIndex
  int get_key_frame_index_for_pts_using_scanned_index(
      const std::vector<_single_stream_decoder::_frame_info>& key_frames,
      int64_t pts) const;

  int64_t seconds_to_index_lower_bound(double seconds);

  int64_t seconds_to_index_upper_bound(double seconds);

  int64_t get_pts(int64_t frame_index);

  // --------------------------------------------------------------------------
  // STREAM AND METADATA APIS
  // --------------------------------------------------------------------------

  void add_stream(
      int stream_index,
      AVMediaType media_type,
      const torch::Device& device = torch::kCPU,
      std::optional<int> ffmpeg_thread_count = std::nullopt);

  // Returns the "best" stream index for a given media type. The "best" is
  // determined by various heuristics in FFMPEG.
  // See
  // https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga757780d38f482deb4d809c6c521fbcc2
  // for more details about the heuristics.
  // Returns the key frame index of the presentation timestamp using FFMPEG's
  // index. Note that this index may be truncated for some files.
  int get_best_stream_index(_avmedia_type media_type);

  int64_t get_num_frames(const StreamMetadata& stream_metadata);
  double get_min_seconds(const StreamMetadata& stream_metadata);
  double get_max_seconds(const StreamMetadata& stream_metadata);

  // --------------------------------------------------------------------------
  // VALIDATION UTILS
  // --------------------------------------------------------------------------

  void validate_active_stream(
      std::optional<_avmedia_type> av_media_type = std::nullopt);
  void validate_scanned_all_streams(const std::string& msg);
  void validate_frame_index(
      const StreamMetadata& stream_metadata,
      int64_t frame_index);

  // --------------------------------------------------------------------------
  // ATTRIBUTES
  // --------------------------------------------------------------------------

  SeekMode seek_mode_;
  ContainerMetadata container_metadata_;
  UniqueAVFormatContext format_context_;
  std::map<int, StreamInfo> stream_infos_;
  const int NO_ACTIVE_STREAM = -2;
  int active_stream_index_ = NO_ACTIVE_STREAM;

  bool cursor_was_just_set_ = false;
  // The desired position of the cursor in the stream. We send frames >= this
  // pts to the user when they request a frame.
  int64_t cursor_ = INT64_MIN;
  // Stores various internal decoding stats.
  DecodeStats decode_stats_;
  // Stores the AVIOContext for the input buffer.
  std::unique_ptr<AVIOContext_holder> avio_context_holder_;
  // Whether or not we have already scanned all streams to update the metadata.
  bool scanned_all_streams_ = false;
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
// AVframe, *post*-resizing. This is only used for single-frame decoding APIs,
// on CPU, with filtergraph.
// - getHeightAndWidthFromOptionsOrAVFrame(). This is the height and width from
// the user-specified options if they exist, or the height and width of the
// AVFrame *before* it is resized. In theory, i.e. if there are no bugs within
// our code or within FFmpeg code, this should be exactly the same as
// getHeightAndWidthFromResizedAVFrame(). This is used by single-frame
// decoding APIs, on CPU with swscale, and on GPU.
// - getHeightAndWidthFromOptionsOrMetadata(). This is the height and width from
// the user-specified options if they exist, or the height and width form the
// stream metadata, which itself got its value from the CodecContext, when the
// stream was added. This is used by batch decoding APIs, for both GPU and
// CPU.
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
FrameDims get_height_and_width_from_resized_avframe(
    const AVFrame& resized_avframe);

FrameDims get_height_and_width_from_options_or_metadata(
    const SingleStreamDecoder::VideoStreamOptions& video_stream_options,
    const SingleStreamDecoder::StreamMetadata& stream_metadata);

FrameDims get_height_and_width_from_options_or_avframe(
    const SingleStreamDecoder::VideoStreamOptions& video_stream_options,
    const UniqueAVFrame& avframe);

torch::Tensor allocate_empty_h_w_c_tensor(
    int height,
    int width,
    torch::Device device,
    std::optional<int> num_frames = std::nullopt);

// Prints the SingleStreamDecoder::DecodeStats to the ostream.
std::ostream& operator<<(
    std::ostream& os,
    const SingleStreamDecoder::DecodeStats& stats);

SingleStreamDecoder::SeekMode seek_mode_from_string(std::string_view seek_mode);

} // namespace facebook::torchcodec
