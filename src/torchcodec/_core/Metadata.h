// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
}

namespace facebook::torchcodec {

enum class SeekMode { exact, approximate, custom_frame_mappings };

struct StreamMetadata {
  // Common (video and audio) fields derived from the AVStream.
  int stream_index = -1;

  // See this link for what various values are available:
  // https://ffmpeg.org/doxygen/trunk/group__lavu__misc.html#ga9a84bba4713dfced21a1a56163be1f48
  AVMediaType media_type = AVMEDIA_TYPE_UNKNOWN;

  std::optional<AVCodecID> codec_id;
  std::optional<std::string> codec_name;
  std::optional<double> duration_seconds_from_header;
  std::optional<double> begin_stream_seconds_from_header;
  std::optional<int64_t> num_frames_from_header;
  std::optional<int64_t> num_key_frames;
  std::optional<double> average_fps_from_header;
  std::optional<double> bit_rate;

  // Used as fallback in approximate mode when stream duration is unavailable.
  std::optional<double> duration_seconds_from_container;

  // More accurate duration, obtained by scanning the file.
  // These presentation timestamps are in time base.
  std::optional<int64_t> begin_stream_pts_from_content;
  std::optional<int64_t> end_stream_pts_from_content;

  // These presentation timestamps are in seconds.
  std::optional<double> begin_stream_pts_seconds_from_content;
  std::optional<double> end_stream_pts_seconds_from_content;

  // This can be useful for index-based seeking.
  std::optional<int64_t> num_frames_from_content;

  // Video-only fields
  // Post-rotation dimensions
  std::optional<int> post_rotation_width;
  std::optional<int> post_rotation_height;
  std::optional<AVRational> sample_aspect_ratio;
  // Rotation angle in degrees from display matrix, in the range [-180, 180].
  std::optional<double> rotation;
  std::optional<AVColorPrimaries> color_primaries;
  std::optional<AVColorSpace> color_space;
  std::optional<AVColorTransferCharacteristic> color_transfer_characteristic;
  // The pixel format of the encoded video, e.g. "yuv420p".
  std::optional<std::string> pixel_format;

  // Audio-only fields
  std::optional<int64_t> sample_rate;
  std::optional<int64_t> num_channels;
  std::optional<std::string> sample_format;

  // Computed methods with fallback logic
  std::optional<double> get_duration_seconds(SeekMode seek_mode) const;
  double get_begin_stream_seconds(SeekMode seek_mode) const;
  std::optional<double> get_end_stream_seconds(SeekMode seek_mode) const;
  std::optional<int64_t> get_num_frames(SeekMode seek_mode) const;
  std::optional<double> get_average_fps(SeekMode seek_mode) const;

  // Color metadata name accessors. These return nullopt if the field is unset
  // or if FFmpeg returns NULL for the name.
  std::optional<std::string> get_color_primaries_name() const;
  std::optional<std::string> get_color_space_name() const;
  std::optional<std::string> get_color_transfer_characteristic_name() const;
};

struct ContainerMetadata {
  std::vector<StreamMetadata> all_stream_metadata;
  int num_audio_streams = 0;
  int num_video_streams = 0;

  // Note that this is the container-level duration, which is usually the max
  // of all stream durations available in the container.
  std::optional<double> duration_seconds_from_header;

  // Total BitRate level information at the container level in bit/s
  std::optional<double> bit_rate;

  // If set, this is the index to the default audio stream.
  std::optional<int> best_audio_stream_index;

  // If set, this is the index to the default video stream.
  std::optional<int> best_video_stream_index;
};

// Builds the container and per-stream metadata that the header describes, from
// an already-opened and probed AVFormatContext.
ContainerMetadata get_container_metadata_from_format_context(
    AVFormatContext* format_context);

} // namespace facebook::torchcodec
