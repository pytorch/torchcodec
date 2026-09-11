// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "Metadata.h"

#include <algorithm>

#include "FFMPEGCommon.h"
#include "StableABICompat.h"
#include "Transform.h"

extern "C" {
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {

namespace {
int get_best_stream_index(AVFormatContext* format_context, AVMediaType type) {
  AVCodecOnlyUseForCallingAVFindBestStream av_codec = nullptr;
  return av_find_best_stream(format_context, type, -1, -1, &av_codec, 0);
}
} // namespace

// Builds the container and per-stream metadata that the header describes.
// Shared by SingleStreamDecoder and by the Demuxer building block, so that both
// report exactly the same thing.
ContainerMetadata get_container_metadata_from_format_context(
    AVFormatContext* format_context) {
  ContainerMetadata metadata;

  if (format_context->duration > 0) {
    AVRational default_time_base{1, AV_TIME_BASE};
    metadata.duration_seconds_from_header =
        pts_to_seconds(format_context->duration, default_time_base);
  }

  if (format_context->bit_rate > 0) {
    metadata.bit_rate = format_context->bit_rate;
  }

  int best_video_stream =
      get_best_stream_index(format_context, AVMEDIA_TYPE_VIDEO);
  if (best_video_stream >= 0) {
    metadata.best_video_stream_index = best_video_stream;
  }

  int best_audio_stream =
      get_best_stream_index(format_context, AVMEDIA_TYPE_AUDIO);
  if (best_audio_stream >= 0) {
    metadata.best_audio_stream_index = best_audio_stream;
  }

  for (unsigned int i = 0; i < format_context->nb_streams; i++) {
    AVStream* av_stream = format_context->streams[i];
    StreamMetadata stream_metadata;

    STD_TORCH_CHECK(
        static_cast<int>(i) == av_stream->index,
        "Our stream index, " + std::to_string(i) +
            ", does not match AVStream's index, " +
            std::to_string(av_stream->index) + ".");
    stream_metadata.stream_index = i;
    stream_metadata.codec_name =
        avcodec_get_name(av_stream->codecpar->codec_id);
    stream_metadata.media_type = av_stream->codecpar->codec_type;
    stream_metadata.bit_rate = av_stream->codecpar->bit_rate;

    int64_t frame_count = av_stream->nb_frames;
    if (frame_count > 0) {
      stream_metadata.num_frames_from_header = frame_count;
    }

    if (av_stream->duration > 0 && av_stream->time_base.den > 0) {
      stream_metadata.duration_seconds_from_header =
          pts_to_seconds(av_stream->duration, av_stream->time_base);
    }
    if (av_stream->start_time != AV_NOPTS_VALUE) {
      stream_metadata.begin_stream_seconds_from_header =
          pts_to_seconds(av_stream->start_time, av_stream->time_base);
    }

    if (av_stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      double fps = av_q2d(av_stream->r_frame_rate);
      if (fps > 0) {
        stream_metadata.average_fps_from_header = fps;
      }
      stream_metadata.rotation = get_rotation_from_stream(av_stream);

      // Report post-rotation dimensions: swap width/height for 90 or -90
      // degree rotations so metadata matches what the decoder returns.
      int width = av_stream->codecpar->width;
      int height = av_stream->codecpar->height;
      Rotation rotation = rotation_from_degrees(stream_metadata.rotation);
      // 90° rotations swap dimensions
      if (rotation == Rotation::CCW90 || rotation == Rotation::CW90) {
        std::swap(width, height);
      }
      stream_metadata.post_rotation_width = width;
      stream_metadata.post_rotation_height = height;

      stream_metadata.sample_aspect_ratio =
          av_stream->codecpar->sample_aspect_ratio;

      if (av_stream->codecpar->color_primaries != AVCOL_PRI_UNSPECIFIED) {
        stream_metadata.color_primaries = av_stream->codecpar->color_primaries;
      }
      if (av_stream->codecpar->color_space != AVCOL_SPC_UNSPECIFIED) {
        stream_metadata.color_space = av_stream->codecpar->color_space;
      }
      if (av_stream->codecpar->color_trc != AVCOL_TRC_UNSPECIFIED) {
        stream_metadata.color_transfer_characteristic =
            av_stream->codecpar->color_trc;
      }
      AVPixelFormat pixel_format =
          static_cast<AVPixelFormat>(av_stream->codecpar->format);
      // If the AVPixelFormat is not recognized, we get back nullptr. We have
      // to make sure we don't initialize a std::string with nullptr. There's
      // nothing to do on the else branch because we're already using an
      // optional; it'll just remain empty.
      const char* raw_pixel_format = av_get_pix_fmt_name(pixel_format);
      if (raw_pixel_format != nullptr) {
        stream_metadata.pixel_format = std::string(raw_pixel_format);
      }

      metadata.num_video_streams++;
    } else if (av_stream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
      AVSampleFormat format =
          static_cast<AVSampleFormat>(av_stream->codecpar->format);
      stream_metadata.sample_rate =
          static_cast<int64_t>(av_stream->codecpar->sample_rate);
      stream_metadata.num_channels =
          static_cast<int64_t>(get_num_channels(av_stream->codecpar));

      // If the AVSampleFormat is not recognized, we get back nullptr. We have
      // to make sure we don't initialize a std::string with nullptr. There's
      // nothing to do on the else branch because we're already using an
      // optional; it'll just remain empty.
      const char* raw_sample_format = av_get_sample_fmt_name(format);
      if (raw_sample_format != nullptr) {
        stream_metadata.sample_format = std::string(raw_sample_format);
      }
      metadata.num_audio_streams++;
    }

    stream_metadata.duration_seconds_from_container =
        metadata.duration_seconds_from_header;

    metadata.all_stream_metadata.push_back(stream_metadata);
  }

  return metadata;
}

std::optional<double> StreamMetadata::get_duration_seconds(
    SeekMode seek_mode) const {
  switch (seek_mode) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      STD_TORCH_CHECK(
          end_stream_pts_seconds_from_content.has_value() &&
              begin_stream_pts_seconds_from_content.has_value(),
          "Missing beginStreamPtsSecondsFromContent or endStreamPtsSecondsFromContent");
      return end_stream_pts_seconds_from_content.value() -
          begin_stream_pts_seconds_from_content.value();
    case SeekMode::approximate:
      if (duration_seconds_from_header.has_value()) {
        return duration_seconds_from_header.value();
      }
      if (num_frames_from_header.has_value() &&
          average_fps_from_header.has_value() &&
          average_fps_from_header.value() != 0.0) {
        return static_cast<double>(num_frames_from_header.value()) /
            average_fps_from_header.value();
      }
      if (duration_seconds_from_container.has_value()) {
        return duration_seconds_from_container.value();
      }
      return std::nullopt;
    default:
      STD_TORCH_CHECK(false, "Unknown SeekMode");
  }
}

double StreamMetadata::get_begin_stream_seconds(SeekMode seek_mode) const {
  switch (seek_mode) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      STD_TORCH_CHECK(
          begin_stream_pts_seconds_from_content.has_value(),
          "Missing beginStreamPtsSecondsFromContent");
      return begin_stream_pts_seconds_from_content.value();
    case SeekMode::approximate:
      if (begin_stream_seconds_from_header.has_value()) {
        return begin_stream_seconds_from_header.value();
      }
      return 0.0;
    default:
      STD_TORCH_CHECK(false, "Unknown SeekMode");
  }
}

std::optional<double> StreamMetadata::get_end_stream_seconds(
    SeekMode seek_mode) const {
  switch (seek_mode) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      STD_TORCH_CHECK(
          end_stream_pts_seconds_from_content.has_value(),
          "Missing endStreamPtsSecondsFromContent");
      return end_stream_pts_seconds_from_content.value();
    case SeekMode::approximate: {
      auto dur = get_duration_seconds(seek_mode);
      if (dur.has_value()) {
        return get_begin_stream_seconds(seek_mode) + dur.value();
      }
      return std::nullopt;
    }
    default:
      STD_TORCH_CHECK(false, "Unknown SeekMode");
  }
}

std::optional<int64_t> StreamMetadata::get_num_frames(
    SeekMode seek_mode) const {
  switch (seek_mode) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      STD_TORCH_CHECK(
          num_frames_from_content.has_value(), "Missing numFramesFromContent");
      return num_frames_from_content.value();
    case SeekMode::approximate: {
      auto duration_seconds = get_duration_seconds(seek_mode);
      if (num_frames_from_header.has_value()) {
        return num_frames_from_header.value();
      }
      if (average_fps_from_header.has_value() && duration_seconds.has_value()) {
        return static_cast<int64_t>(
            average_fps_from_header.value() * duration_seconds.value());
      }
      return std::nullopt;
    }
    default:
      STD_TORCH_CHECK(false, "Unknown SeekMode");
  }
}

std::optional<double> StreamMetadata::get_average_fps(
    SeekMode seek_mode) const {
  switch (seek_mode) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact: {
      auto num_frames = get_num_frames(seek_mode);
      if (num_frames.has_value() &&
          begin_stream_pts_seconds_from_content.has_value() &&
          end_stream_pts_seconds_from_content.has_value()) {
        double duration = end_stream_pts_seconds_from_content.value() -
            begin_stream_pts_seconds_from_content.value();
        if (duration != 0.0) {
          return static_cast<double>(num_frames.value()) / duration;
        }
      }
      return average_fps_from_header;
    }
    case SeekMode::approximate:
      return average_fps_from_header;
    default:
      STD_TORCH_CHECK(false, "Unknown SeekMode");
  }
}

std::optional<std::string> StreamMetadata::get_color_primaries_name() const {
  if (!color_primaries.has_value()) {
    return std::nullopt;
  }
  const char* name = av_color_primaries_name(*color_primaries);
  if (name == nullptr) {
    return std::nullopt;
  }
  return std::string(name);
}

std::optional<std::string> StreamMetadata::get_color_space_name() const {
  if (!color_space.has_value()) {
    return std::nullopt;
  }
  const char* name = av_color_space_name(*color_space);
  if (name == nullptr) {
    return std::nullopt;
  }
  return std::string(name);
}

std::optional<std::string>
StreamMetadata::get_color_transfer_characteristic_name() const {
  if (!color_transfer_characteristic.has_value()) {
    return std::nullopt;
  }
  const char* name = av_color_transfer_name(*color_transfer_characteristic);
  if (name == nullptr) {
    return std::nullopt;
  }
  return std::string(name);
}

} // namespace facebook::torchcodec
