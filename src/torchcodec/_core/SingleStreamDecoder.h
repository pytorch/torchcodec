// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <string_view>

#include "AVIOContextHolder.h"
#include "DeviceInterface.h"
#include "FFMPEGCommon.h"
#include "Frame.h"
#include "Metadata.h"
#include "StableABICompat.h"
#include "StreamOptions.h"
#include "Transform.h"

namespace facebook::torchcodec {

// The SingleStreamDecoder class can be used to decode video frames to Tensors.
// Note that SingleStreamDecoder is not thread-safe.
// Do not call non-const APIs concurrently on the same object.
class FORCE_PUBLIC_VISIBILITY SingleStreamDecoder {
 public:
  // --------------------------------------------------------------------------
  // CONSTRUCTION API
  // --------------------------------------------------------------------------

  // Creates a SingleStreamDecoder from the video at videoFilePath.
  explicit SingleStreamDecoder(
      const std::string& video_file_path,
      SeekMode seek_mode = SeekMode::exact);

  // Creates a SingleStreamDecoder using the provided AVIOContext inside the
  // AVIOContextHolder. The AVIOContextHolder wraps an IOInterface that
  // specializes how the custom read, seek and writes work (in-memory tensor,
  // Python file-like, ...).
  explicit SingleStreamDecoder(
      std::unique_ptr<AVIOContextHolder> context,
      SeekMode seek_mode = SeekMode::exact);

  // --------------------------------------------------------------------------
  // VIDEO METADATA QUERY API
  // --------------------------------------------------------------------------

  // Updates the metadata of the video to accurate values obtained by scanning
  // the contents of the video file. Also updates each StreamInfo's index, i.e.
  // the allFrames and keyFrames vectors.
  void scan_file_and_update_metadata_and_index();

  // Sorts the keyFrames and allFrames vectors in each StreamInfo by pts.
  void sort_all_frames();

  // Returns the metadata for the container.
  ContainerMetadata get_container_metadata() const;

  // Returns the seek mode of this decoder.
  SeekMode get_seek_mode() const;

  // Returns the active stream index. Returns -2 if no stream is active.
  int get_active_stream_index() const;

  // Returns the key frame indices as a tensor. The tensor is 1D and contains
  // int64 values, where each value is the frame index for a key frame.
  torch::stable::Tensor get_key_frame_indices();

  // FrameMappings is used for the custom_frame_mappings seek mode to store
  // metadata of frames in a stream. The size of all tensors in this struct must
  // match.

  // --------------------------------------------------------------------------
  // ADDING STREAMS API
  // --------------------------------------------------------------------------
  struct FrameMappings {
    // 1D tensor of int64, each value is the PTS of a frame in timebase units.
    torch::stable::Tensor all_frames;
    // 1D tensor of bool, each value indicates if the corresponding frame in
    // all_frames is a key frame.
    torch::stable::Tensor is_key_frame;
    // 1D tensor of int64, each value is the duration of the corresponding frame
    // in all_frames in timebase units.
    torch::stable::Tensor duration;
  };

  void add_video_stream(
      int stream_index,
      std::vector<Transform*>& transforms,
      const VideoStreamOptions& video_stream_options = VideoStreamOptions(),
      std::optional<FrameMappings> custom_frame_mappings = std::nullopt);
  void add_audio_stream(
      int stream_index,
      const AudioStreamOptions& audio_stream_options = AudioStreamOptions());

  // --------------------------------------------------------------------------
  // DECODING AND SEEKING APIs
  // --------------------------------------------------------------------------

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
      const torch::stable::Tensor& frame_indices);

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

  FrameBatchOutput get_frames_played_at(
      const torch::stable::Tensor& timestamps);

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
  //
  // If fps is specified, frames are resampled to match the target frame
  // rate by duplicating or dropping frames as necessary.
  FrameBatchOutput get_frames_played_in_range(
      double start_seconds,
      double stop_seconds,
      std::optional<double> fps = std::nullopt);

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
      std::optional<torch::stable::Tensor> pre_allocated_output_tensor =
          std::nullopt);

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

  std::string get_device_interface_details() const;

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

  struct StreamInfo {
    int stream_index = -1;
    AVStream* stream = nullptr;
    AVMediaType av_media_type = AVMEDIA_TYPE_UNKNOWN;

    AVRational time_base = {};
    SharedAVCodecContext codec_context;

    // The FrameInfo indices we built when scanFileAndUpdateMetadataAndIndex was
    // called.
    std::vector<FrameInfo> key_frames;
    std::vector<FrameInfo> all_frames;

    VideoStreamOptions video_stream_options;
    AudioStreamOptions audio_stream_options;
  };

  // --------------------------------------------------------------------------
  // INITIALIZERS
  // --------------------------------------------------------------------------

  void initialize_decoder();

  // Reads the user provided frame index and updates each StreamInfo's index,
  // i.e. the allFrames and keyFrames vectors, and
  // endStreamPtsSecondsFromContent
  void read_custom_frame_mappings_update_metadata_and_index(
      int stream_index,
      FrameMappings custom_frame_mappings);
  // --------------------------------------------------------------------------
  // DECODING APIS AND RELATED UTILS
  // --------------------------------------------------------------------------

  void set_cursor(int64_t pts);
  void set_cursor(double) = delete; // prevent calls with doubles and floats
  bool can_we_avoid_seeking() const;

  bool maybe_seek_to_before_desired_pts();

  UniqueAVFrame decode_av_frame(
      std::function<bool(const AVFrame&)> filter_function);

  FrameOutput get_next_frame_internal(
      std::optional<torch::stable::Tensor> pre_allocated_output_tensor =
          std::nullopt);

  // Permutes HWC to CHW if needed, then converts to float32 and normalizes to
  // [0, 1] if the active stream's outputDtype is FLOAT32.
  torch::stable::Tensor maybe_permute_and_convert_dtype(
      torch::stable::Tensor& tensor);

  FrameOutput convert_av_frame_to_frame_output(
      const AVFrame& av_frame,
      std::optional<torch::stable::Tensor> pre_allocated_output_tensor =
          std::nullopt);

  // --------------------------------------------------------------------------
  // PTS <-> INDEX CONVERSIONS
  // --------------------------------------------------------------------------

  int get_key_frame_identifier(int64_t pts) const;

  // Returns the key frame index of the presentation timestamp using our index.
  // We build this index by scanning the file in
  // scanFileAndUpdateMetadataAndIndex
  int get_key_frame_index_for_pts_using_scanned_index(
      const std::vector<SingleStreamDecoder::FrameInfo>& key_frames,
      int64_t pts) const;

  int64_t seconds_to_index_lower_bound(double seconds) const;

  int64_t seconds_to_index_upper_bound(double seconds);

  int64_t get_pts(int64_t frame_index);

  // Returns the output frame dimensions for video frames.
  // If resizedOutputDims_ is set (via resize, crop, or rotation transforms),
  // returns that. Otherwise, returns preRotationDims_.
  //
  // Note: if resizedOutputDims_ is null, there is no rotation (the
  // rotation transform would have set it), so preRotationDims_ ==
  // postRotationDims_. This makes it safe to use preRotationDims_ as the
  // fallback.
  FrameDims get_output_dims() const;

  // --------------------------------------------------------------------------
  // STREAM AND METADATA APIS
  // --------------------------------------------------------------------------

  void add_stream(
      int stream_index,
      AVMediaType media_type,
      const StableDevice& device = StableDevice(kStableCPU),
      const std::string_view device_variant = "default",
      std::optional<int> ffmpeg_thread_count = std::nullopt);

  // Returns the "best" stream index for a given media type. The "best" is
  // determined by various heuristics in FFMPEG.
  // See
  // https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga757780d38f482deb4d809c6c521fbcc2
  // for more details about the heuristics.
  // Returns the key frame index of the presentation timestamp using FFMPEG's
  // index. Note that this index may be truncated for some files.

  // --------------------------------------------------------------------------
  // VALIDATION UTILS
  // --------------------------------------------------------------------------

  void validate_active_stream(
      std::optional<AVMediaType> av_media_type = std::nullopt);
  void validate_scanned_all_streams(const std::string& msg);
  void validate_frame_index(
      const StreamMetadata& stream_metadata,
      int64_t frame_index);

  // --------------------------------------------------------------------------
  // ATTRIBUTES
  // --------------------------------------------------------------------------

  SeekMode seek_mode_;
  ContainerMetadata container_metadata_;
  UniqueDecodingAVFormatContext format_context_;
  std::unique_ptr<DeviceInterface> device_interface_;
  std::map<int, StreamInfo> stream_infos_;
  const int no_active_stream_ = -2;
  int active_stream_index_ = no_active_stream_;

  // The desired position of the cursor in the stream. We send frames >= this
  // pts to the user when they request a frame.
  int64_t cursor_ = INT64_MIN;
  bool cursor_was_just_set_ = false;
  // Whether the container is an MPEG program stream, whose demuxer needs
  // special care after a seek. See decode_av_frame().
  bool is_mpeg_ps_ = false;
  // Initialized to INT64_MIN instead of 0. With 0, canWeAvoidSeeking() could
  // incorrectly skip a seek when the internal FFmpeg frame index (used by
  // av_index_search_timestamp() in approximate mode) had not yet been built,
  // as some formats (mkv, webm) delay building it until the first seek.
  // With INT64_MIN, we always seek when retrieving the first frame. This
  // means we correctly seek when the first requested frame is far into the
  // video, at the cost of an unnecessary (likely cheap) seek when the first
  // requested frame is near the start.
  // See: https://github.com/meta-pytorch/torchcodec/pull/1259
  int64_t last_decoded_av_frame_pts_ = INT64_MIN;
  int64_t last_decoded_av_frame_duration_ = 0;
  int64_t last_decoded_frame_index_ = INT64_MIN;

  // Stores various internal decoding stats.
  DecodeStats decode_stats_;

  // Stores the AVIOContext for the input buffer.
  std::unique_ptr<AVIOContextHolder> avio_context_holder_;

  // We will receive a vector of transforms upon adding a stream and store it
  // here. However, we need to know if any of those operations change the
  // dimensions of the output frame. If they do, we need to figure out what are
  // the final dimensions of the output frame after ALL transformations. We
  // figure this out as soon as we receive the transforms. If any of the
  // transforms change the final output frame dimensions, we store that in
  // resizedOutputDims_. If resizedOutputDims_ has no value, that means there
  // are no transforms that change the output frame dimensions.
  //
  // The priority order for output frame dimensions is:
  //
  // 1. resizedOutputDims_; the resize requested by the user (or rotation)
  //    always takes priority.
  // 2. The dimensions of the actual decoded AVFrame. This can change
  //    per-decoded frame, and is unknown in SingleStreamDecoder. Only the
  //    DeviceInterface learns it immediately after decoding a raw frame but
  //    before the color conversion.
  // 3. preRotationDims_; the raw encoded dimensions from FFmpeg metadata
  //    (before any rotation is applied). Used as fallback for tensor
  //    allocation when resizedOutputDims_ is not set, which only happens
  //    when no rotation is needed, so preRotationDims_ is the correct value.
  std::vector<std::unique_ptr<Transform>> transforms_;
  std::optional<FrameDims> resized_output_dims_;
  FrameDims pre_rotation_dims_;

  // Whether or not we have already scanned all streams to update the metadata.
  bool scanned_all_streams_ = false;

  // Tracks that we've already been initialized.
  bool initialized_ = false;
};

// Prints the SingleStreamDecoder::DecodeStats to the ostream.
std::ostream& operator<<(
    std::ostream& os,
    const SingleStreamDecoder::DecodeStats& stats);

} // namespace facebook::torchcodec
