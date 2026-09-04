// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "SingleStreamDecoder.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string_view>
#include "Demuxer.h"
#include "Metadata.h"
#include "PacketDecoder.h"
#include "StableABICompat.h"

extern "C" {
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {

// --------------------------------------------------------------------------
// CONSTRUCTORS, INITIALIZATION, DESTRUCTORS
// --------------------------------------------------------------------------

SingleStreamDecoder::SingleStreamDecoder(
    const std::string& video_file_path,
    SeekMode seek_mode)
    : seek_mode_(seek_mode) {
  set_ffmpeg_log_level();

  AVFormatContext* raw_context = nullptr;
  int status = avformat_open_input(
      &raw_context, video_file_path.c_str(), nullptr, nullptr);
  STD_TORCH_CHECK(
      status == 0,
      "Could not open input file: " + video_file_path + " " +
          get_ffmpeg_error_string_from_error_code(status));
  STD_TORCH_CHECK(raw_context != nullptr, "Failed to allocate AVFormatContext");
  format_context_.reset(raw_context);

  initialize_decoder();
}

SingleStreamDecoder::SingleStreamDecoder(
    std::unique_ptr<AVIOContextHolder> context,
    SeekMode seek_mode)
    : seek_mode_(seek_mode), avio_context_holder_(std::move(context)) {
  set_ffmpeg_log_level();

  STD_TORCH_CHECK(avio_context_holder_, "Context holder cannot be null");

  // Because FFmpeg requires a reference to a pointer in the call to open, we
  // can't use a unique pointer here. Note that means we must call free if open
  // fails.
  AVFormatContext* raw_context = avformat_alloc_context();
  STD_TORCH_CHECK(raw_context != nullptr, "Unable to alloc avformat context");

  raw_context->pb = avio_context_holder_->get_avio_context();
  int status = avformat_open_input(&raw_context, nullptr, nullptr, nullptr);
  if (status != 0) {
    avformat_free_context(raw_context);
    STD_TORCH_CHECK(
        false,
        "Failed to open input buffer: " +
            get_ffmpeg_error_string_from_error_code(status));
  }

  format_context_.reset(raw_context);

  initialize_decoder();
}

void SingleStreamDecoder::initialize_decoder() {
  STD_TORCH_CHECK(!initialized_, "Attempted double initialization.");

  is_mpeg_ps_ = std::string_view(format_context_->iformat->name) == "mpeg";

  // In principle, the AVFormatContext should be filled in by the call to
  // avformat_open_input() which reads the header. However, some formats do not
  // store enough info in the header, so we call avformat_find_stream_info()
  // which decodes a few frames to get missing info. For more, see:
  //   https://ffmpeg.org/doxygen/7.0/group__lavf__decoding.html
  int status = avformat_find_stream_info(format_context_.get(), nullptr);
  STD_TORCH_CHECK(
      status >= 0,
      "Failed to find stream info: ",
      get_ffmpeg_error_string_from_error_code(status));

  container_metadata_ =
      get_container_metadata_from_format_context(format_context_.get());

  if (seek_mode_ == SeekMode::exact) {
    scan_file_and_update_metadata_and_index();
  }

  initialized_ = true;
}

// --------------------------------------------------------------------------
// VIDEO METADATA QUERY API
// --------------------------------------------------------------------------

void SingleStreamDecoder::sort_all_frames() {
  // Sort the allFrames and keyFrames vecs in each stream, and also sets
  // additional fields of the FrameInfo entries like nextPts and frameIndex
  // This is called at the end of a scan, or when setting a user-defined frame
  // mapping.
  for (auto& [stream_index, stream_info] : stream_infos_) {
    std::sort(
        stream_info.key_frames.begin(),
        stream_info.key_frames.end(),
        [](const FrameInfo& frame_info1, const FrameInfo& frame_info2) {
          return frame_info1.pts < frame_info2.pts;
        });
    std::sort(
        stream_info.all_frames.begin(),
        stream_info.all_frames.end(),
        [](const FrameInfo& frame_info1, const FrameInfo& frame_info2) {
          return frame_info1.pts < frame_info2.pts;
        });

    size_t key_frame_index = 0;
    for (size_t i = 0; i < stream_info.all_frames.size(); ++i) {
      stream_info.all_frames[i].frame_index = i;
      if (stream_info.all_frames[i].is_key_frame) {
        STD_TORCH_CHECK(
            key_frame_index < stream_info.key_frames.size(),
            "The allFrames vec claims it has MORE keyFrames than the keyFrames vec. There's a bug in torchcodec.");
        stream_info.key_frames[key_frame_index].frame_index = i;
        ++key_frame_index;
      }
      if (i + 1 < stream_info.all_frames.size()) {
        stream_info.all_frames[i].next_pts = stream_info.all_frames[i + 1].pts;
      }
    }
    STD_TORCH_CHECK(
        key_frame_index == stream_info.key_frames.size(),
        "The allFrames vec claims it has LESS keyFrames than the keyFrames vec. There's a bug in torchcodec.");
  }
}

void SingleStreamDecoder::scan_file_and_update_metadata_and_index() {
  if (scanned_all_streams_) {
    return;
  }

  AutoAVPacket auto_av_packet;
  while (true) {
    ReferenceAVPacket packet(auto_av_packet);

    // av_read_frame is a misleading name: it gets the next **packet**.
    int status = av_read_frame(format_context_.get(), packet.get());

    if (status == AVERROR_EOF) {
      break;
    }

    STD_TORCH_CHECK(
        status == AVSUCCESS,
        "Failed to read frame from input file: ",
        get_ffmpeg_error_string_from_error_code(status));

    if (packet->flags & AV_PKT_FLAG_DISCARD) {
      continue;
    }

    // We got a valid packet. Let's figure out what stream it belongs to and
    // record its relevant metadata.
    int stream_index = packet->stream_index;

    auto& stream_metadata =
        container_metadata_.all_stream_metadata[stream_index];
    stream_metadata.begin_stream_pts_from_content = std::min(
        stream_metadata.begin_stream_pts_from_content.value_or(INT64_MAX),
        get_pts_or_dts(packet));
    stream_metadata.end_stream_pts_from_content = std::max(
        stream_metadata.end_stream_pts_from_content.value_or(INT64_MIN),
        get_pts_or_dts(packet) + packet->duration);
    stream_metadata.num_frames_from_content =
        stream_metadata.num_frames_from_content.value_or(0) + 1;

    // Note that we set the other value in this struct, nextPts, only after
    // we have scanned all packets and sorted by pts.
    FrameInfo frame_info = {get_pts_or_dts(packet)};
    if (packet->flags & AV_PKT_FLAG_KEY) {
      frame_info.is_key_frame = true;
      stream_infos_[stream_index].key_frames.push_back(frame_info);
    }
    stream_infos_[stream_index].all_frames.push_back(frame_info);
  }

  // Set all per-stream metadata that requires knowing the content of all
  // packets.
  for (size_t stream_index = 0;
       stream_index < container_metadata_.all_stream_metadata.size();
       ++stream_index) {
    auto& stream_metadata =
        container_metadata_.all_stream_metadata[stream_index];
    auto av_stream = format_context_->streams[stream_index];

    stream_metadata.num_frames_from_content =
        stream_infos_[stream_index].all_frames.size();

    // This ensures that we are robust in handling cases where
    // we are decoding in exact mode and numFrames is 0. The current metadata
    // validation logic assumes that these values should not be None
    if (stream_metadata.num_frames_from_content.value() == 0) {
      stream_metadata.begin_stream_pts_from_content = 0;
      stream_metadata.end_stream_pts_from_content = 0;
    }

    if (stream_metadata.begin_stream_pts_from_content.has_value()) {
      stream_metadata.begin_stream_pts_seconds_from_content = pts_to_seconds(
          *stream_metadata.begin_stream_pts_from_content, av_stream->time_base);
    }
    if (stream_metadata.end_stream_pts_from_content.has_value()) {
      stream_metadata.end_stream_pts_seconds_from_content = pts_to_seconds(
          *stream_metadata.end_stream_pts_from_content, av_stream->time_base);
    }
  }

  // Reset the seek-cursor back to the beginning.
  int status = avformat_seek_file(format_context_.get(), 0, INT64_MIN, 0, 0, 0);
  STD_TORCH_CHECK(
      status >= 0, get_seek_error_message(format_context_.get(), 0, status));

  // Sort all frames by their pts.
  sort_all_frames();
  scanned_all_streams_ = true;
}

void SingleStreamDecoder::read_custom_frame_mappings_update_metadata_and_index(
    int stream_index,
    FrameMappings custom_frame_mappings) {
  STD_TORCH_CHECK(
      custom_frame_mappings.all_frames.scalar_type() == kStableInt64 &&
          custom_frame_mappings.is_key_frame.scalar_type() == kStableBool &&
          custom_frame_mappings.duration.scalar_type() == kStableInt64,
      "all_frames and duration tensors must be int64 dtype, and is_key_frame tensor must be a bool dtype.");
  const torch::stable::Tensor& all_frames = custom_frame_mappings.all_frames;
  const torch::stable::Tensor& is_key_frame =
      custom_frame_mappings.is_key_frame;
  const torch::stable::Tensor& duration = custom_frame_mappings.duration;
  STD_TORCH_CHECK(
      all_frames.sizes()[0] == is_key_frame.sizes()[0] &&
          is_key_frame.sizes()[0] == duration.sizes()[0],
      "all_frames, is_key_frame, and duration from custom_frame_mappings were not same size.");

  // Allocate vectors using num frames to reduce reallocations
  int64_t num_frames = all_frames.sizes()[0];
  stream_infos_[stream_index].all_frames.reserve(num_frames);
  stream_infos_[stream_index].key_frames.reserve(num_frames);
  auto pts_data = const_accessor<int64_t, 1>(all_frames);
  auto is_key_frame_data = const_accessor<bool, 1>(is_key_frame);
  auto duration_data = const_accessor<int64_t, 1>(duration);

  auto& stream_metadata = container_metadata_.all_stream_metadata[stream_index];

  stream_metadata.begin_stream_pts_from_content = pts_data[0];
  stream_metadata.end_stream_pts_from_content =
      pts_data[num_frames - 1] + duration_data[num_frames - 1];

  auto av_stream = format_context_->streams[stream_index];
  stream_metadata.begin_stream_pts_seconds_from_content = pts_to_seconds(
      *stream_metadata.begin_stream_pts_from_content, av_stream->time_base);

  stream_metadata.end_stream_pts_seconds_from_content = pts_to_seconds(
      *stream_metadata.end_stream_pts_from_content, av_stream->time_base);

  stream_metadata.num_frames_from_content = num_frames;
  for (int64_t i = 0; i < num_frames; ++i) {
    FrameInfo frame_info;
    frame_info.pts = pts_data[i];
    frame_info.is_key_frame = is_key_frame_data[i];
    stream_infos_[stream_index].all_frames.push_back(frame_info);
    if (frame_info.is_key_frame) {
      stream_infos_[stream_index].key_frames.push_back(frame_info);
    }
  }
  sort_all_frames();
}

ContainerMetadata SingleStreamDecoder::get_container_metadata() const {
  return container_metadata_;
}

SeekMode SingleStreamDecoder::get_seek_mode() const {
  return seek_mode_;
}

int SingleStreamDecoder::get_active_stream_index() const {
  return active_stream_index_;
}

torch::stable::Tensor SingleStreamDecoder::get_key_frame_indices() {
  validate_active_stream(AVMEDIA_TYPE_VIDEO);
  validate_scanned_all_streams("getKeyFrameIndices");

  const std::vector<FrameInfo>& key_frames =
      stream_infos_[active_stream_index_].key_frames;
  torch::stable::Tensor key_frame_indices = torch::stable::empty(
      {static_cast<int64_t>(key_frames.size())}, kStableInt64);
  auto key_frame_indices_accessor =
      mutable_accessor<int64_t, 1>(key_frame_indices);
  for (size_t i = 0; i < key_frames.size(); ++i) {
    key_frame_indices_accessor[i] = key_frames[i].frame_index;
  }

  return key_frame_indices;
}

// --------------------------------------------------------------------------
// ADDING STREAMS API
// --------------------------------------------------------------------------

void SingleStreamDecoder::add_stream(
    int stream_index,
    AVMediaType media_type,
    const StableDevice& device,
    const std::string_view device_variant,
    std::optional<int> ffmpeg_thread_count) {
  STD_TORCH_CHECK(
      active_stream_index_ == no_active_stream_,
      "Can only add one single stream.");
  STD_TORCH_CHECK(
      media_type == AVMEDIA_TYPE_VIDEO || media_type == AVMEDIA_TYPE_AUDIO,
      "Can only add video or audio streams.");
  STD_TORCH_CHECK(format_context_.get() != nullptr, "Format context is null");

  AVCodecOnlyUseForCallingAVFindBestStream av_codec = nullptr;

  active_stream_index_ = av_find_best_stream(
      format_context_.get(), media_type, stream_index, -1, &av_codec, 0);

  STD_TORCH_CHECK(
      active_stream_index_ >= 0,
      "No valid stream found in input file. Is ",
      stream_index,
      " of the desired media type?");

  STD_TORCH_CHECK(av_codec != nullptr, "Codec not found");

  StreamInfo& stream_info = stream_infos_[active_stream_index_];
  stream_info.stream_index = active_stream_index_;
  stream_info.time_base =
      format_context_->streams[active_stream_index_]->time_base;
  stream_info.stream = format_context_->streams[active_stream_index_];
  stream_info.av_media_type = media_type;

  // This should never happen, checking just to be safe.
  STD_TORCH_CHECK(
      stream_info.stream->codecpar->codec_type == media_type,
      "FFmpeg found stream with index ",
      active_stream_index_,
      " which is of the wrong media type.");

  device_interface_ = create_device_interface(device, device_variant);
  STD_TORCH_CHECK(
      device_interface_ != nullptr,
      "Failed to create device interface. This should never happen, please report.");

  // TODO_CODE_QUALITY it's pretty meh to have a video-specific logic within
  // addStream() which is supposed to be generic
  if (media_type == AVMEDIA_TYPE_VIDEO) {
    av_codec = make_av_codec_only_use_for_calling_av_find_best_stream(
        device_interface_->find_codec(stream_info.stream->codecpar->codec_id)
            .value_or(av_codec));
  }

  // Create + configure + open the codec context (shared with PacketDecoder).
  stream_info.codec_context = create_and_open_codec_context(
      stream_info.stream,
      av_codec,
      device_interface_.get(),
      ffmpeg_thread_count);

  // Initialize the device interface with the codec context
  device_interface_->initialize(stream_info.codec_context);

  container_metadata_.all_stream_metadata[active_stream_index_].codec_name =
      std::string(avcodec_get_name(stream_info.codec_context->codec_id));

  // We will only need packets from the active stream, so we tell FFmpeg to
  // discard packets from the other streams. Note that av_read_frame() may still
  // return some of those un-desired packet under some conditions, so it's still
  // important to discard/demux correctly in the inner decoding loop.
  for (unsigned int i = 0; i < format_context_->nb_streams; ++i) {
    if (i != static_cast<unsigned int>(active_stream_index_)) {
      format_context_->streams[i]->discard = AVDISCARD_ALL;
    }
  }
}

void SingleStreamDecoder::add_video_stream(
    int stream_index,
    std::vector<Transform*>& transforms,
    const VideoStreamOptions& video_stream_options,
    std::optional<FrameMappings> custom_frame_mappings) {
  STD_TORCH_CHECK(
      transforms.empty() || video_stream_options.device == kStableCPU,
      " Transforms are only supported for CPU devices.");

  add_stream(
      stream_index,
      AVMEDIA_TYPE_VIDEO,
      video_stream_options.device,
      video_stream_options.device_variant,
      video_stream_options.ffmpeg_thread_count);

  auto& stream_metadata =
      container_metadata_.all_stream_metadata[active_stream_index_];

  if (seek_mode_ == SeekMode::approximate) {
    STD_TORCH_CHECK(
        stream_metadata.average_fps_from_header.has_value(),
        "Seek mode is approximate, but stream ",
        std::to_string(active_stream_index_),
        " does not have an average fps in its metadata.");
  }

  auto& stream_info = stream_infos_[active_stream_index_];
  stream_info.video_stream_options = video_stream_options;

  if (seek_mode_ == SeekMode::custom_frame_mappings) {
    STD_TORCH_CHECK(
        custom_frame_mappings.has_value(),
        "Missing frame mappings when custom_frame_mappings seek mode is set.");
    read_custom_frame_mappings_update_metadata_and_index(
        active_stream_index_, custom_frame_mappings.value());
  }

  stream_info.video_stream_options.output_dtype = resolve_output_dtype(
      stream_info.video_stream_options.output_dtype_config,
      static_cast<AVPixelFormat>(stream_info.stream->codecpar->format));

  // Set preRotationDims_ for the active stream. These are the raw encoded
  // dimensions from FFmpeg, used as a fallback for tensor pre-allocation when
  // no resize/rotation transforms are applied.
  pre_rotation_dims_ = FrameDims(
      stream_info.stream->codecpar->height,
      stream_info.stream->codecpar->width);

  FrameDims curr_input_dims = pre_rotation_dims_;

  // If there's rotation, prepend a RotationTransform to handle it in the
  // filter graph. This way user transforms (resize, crop) operate in
  // post-rotation coordinate space, preserving x/y coordinates for crops.
  //
  // It is critical to apply the rotation *before* any user-supplied
  // transforms. By design, we want:
  //   A: VideoDecoder(..., transforms=tv_transforms)[i]
  // to be equivalent to:
  //   B: tv_transforms(VideoDecoder(...)[i])
  // In B, rotation is applied before transforms, so A must behave the same.
  //
  // TODO: benchmark the performance of doing this additional filtergraph
  // transform
  Rotation rotation = rotation_from_degrees(stream_metadata.rotation);
  if (rotation != Rotation::NONE) {
    auto rotation_transform =
        std::make_unique<RotationTransform>(rotation, curr_input_dims);
    curr_input_dims = rotation_transform->get_output_frame_dims().value();
    resized_output_dims_ = curr_input_dims;
    transforms_.push_back(std::move(rotation_transform));
  }

  // Note that we are claiming ownership of the transform objects passed in to
  // us.
  // Validate and add user transforms
  for (auto& transform : transforms) {
    STD_TORCH_CHECK(
        transform != nullptr, "Transforms should never be nullptr!");
    transform->validate(curr_input_dims);
    if (transform->get_output_frame_dims().has_value()) {
      resized_output_dims_ = transform->get_output_frame_dims().value();
      curr_input_dims = resized_output_dims_.value();
    }
    transforms_.push_back(std::unique_ptr<Transform>(transform));
  }

  // Pass the resolved options (AUTO -> UINT8/FLOAT32) so the device interface
  // sees a definite OutputDtype.
  device_interface_->initialize_video(
      stream_info.stream,
      format_context_,
      stream_info.video_stream_options,
      transforms_,
      resized_output_dims_);
}

void SingleStreamDecoder::add_audio_stream(
    int stream_index,
    const AudioStreamOptions& audio_stream_options) {
  STD_TORCH_CHECK(
      seek_mode_ == SeekMode::approximate,
      "seek_mode must be 'approximate' for audio streams.");
  if (audio_stream_options.num_channels.has_value()) {
    STD_TORCH_CHECK(
        *audio_stream_options.num_channels > 0,
        "num_channels must be > 0. Got: ",
        *audio_stream_options.num_channels);
  }

  // We hardcode ffmpegThreadCount=1 for audio, see
  // https://github.com/pytorch/torchcodec/issues/1253 and
  // https://github.com/pytorch/torchcodec/pull/1254
  add_stream(
      stream_index, AVMEDIA_TYPE_AUDIO, StableDevice(kStableCPU), "default", 1);

  auto& stream_info = stream_infos_[active_stream_index_];
  stream_info.audio_stream_options = audio_stream_options;

  // FFmpeg docs say that the decoder will try to decode natively in this
  // format, if it can. Docs don't say what the decoder does when it doesn't
  // support that format, but it looks like it does nothing, so this probably
  // doesn't hurt.
  stream_info.codec_context->request_sample_fmt = AV_SAMPLE_FMT_FLTP;

  // Initialize device interface for audio
  double stream_start_seconds = stream_info.stream->start_time != AV_NOPTS_VALUE
      ? pts_to_seconds(stream_info.stream->start_time, stream_info.time_base)
      : 0.0;
  device_interface_->initialize_audio(
      audio_stream_options, stream_start_seconds);
}

// --------------------------------------------------------------------------
// HIGH-LEVEL DECODING ENTRY-POINTS
// --------------------------------------------------------------------------

FrameOutput SingleStreamDecoder::get_next_frame() {
  auto output = get_next_frame_internal();
  if (stream_infos_[active_stream_index_].av_media_type == AVMEDIA_TYPE_VIDEO) {
    output.data = maybe_permute_and_convert_dtype(output.data);
  }
  return output;
}

FrameOutput SingleStreamDecoder::get_next_frame_internal(
    std::optional<torch::stable::Tensor> pre_allocated_output_tensor) {
  validate_active_stream();
  UniqueAVFrame av_frame = decode_av_frame([this](const AVFrame& av_frame) {
    return get_pts_or_dts(av_frame) >= cursor_;
  });
  return convert_av_frame_to_frame_output(
      *av_frame, pre_allocated_output_tensor);
}

FrameOutput SingleStreamDecoder::get_frame_at_index(int64_t frame_index) {
  auto frame_output = get_frame_at_index_internal(frame_index);
  frame_output.data = maybe_permute_and_convert_dtype(frame_output.data);
  return frame_output;
}

FrameOutput SingleStreamDecoder::get_frame_at_index_internal(
    int64_t frame_index,
    std::optional<torch::stable::Tensor> pre_allocated_output_tensor) {
  validate_active_stream(AVMEDIA_TYPE_VIDEO);

  const auto& stream_info = stream_infos_[active_stream_index_];
  const auto& stream_metadata =
      container_metadata_.all_stream_metadata[active_stream_index_];

  std::optional<int64_t> num_frames =
      stream_metadata.get_num_frames(seek_mode_);
  if (num_frames.has_value()) {
    // If the frameIndex is negative, we convert it to a positive index
    frame_index =
        frame_index >= 0 ? frame_index : frame_index + num_frames.value();
  }
  validate_frame_index(stream_metadata, frame_index);

  // Only set cursor if we're not decoding sequentially: when decoding
  // sequentially, we don't need to seek anywhere, so by *not* setting the
  // cursor we allow canWeAvoidSeeking() to return true early.
  if (frame_index != last_decoded_frame_index_ + 1) {
    int64_t pts = get_pts(frame_index);
    set_cursor_pts_in_seconds(pts_to_seconds(pts, stream_info.time_base));
  }

  auto result = get_next_frame_internal(pre_allocated_output_tensor);
  last_decoded_frame_index_ = frame_index;
  return result;
}

FrameBatchOutput SingleStreamDecoder::get_frames_at_indices(
    const torch::stable::Tensor& frame_indices) {
  validate_active_stream(AVMEDIA_TYPE_VIDEO);

  auto frame_indices_data = const_accessor<int64_t, 1>(frame_indices);

  bool indices_are_sorted = true;
  for (int64_t i = 1; i < frame_indices.numel(); ++i) {
    if (frame_indices_data[i] < frame_indices_data[i - 1]) {
      indices_are_sorted = false;
      break;
    }
  }

  std::vector<size_t> argsort;
  if (!indices_are_sorted) {
    // if frameIndices is [13, 10, 12, 11]
    // when sorted, it's  [10, 11, 12, 13] <-- this is the sorted order we want
    //                                         to use to decode the frames
    // and argsort is     [ 1,  3,  2,  0]
    argsort.resize(frame_indices.numel());
    for (size_t i = 0; i < argsort.size(); ++i) {
      argsort[i] = i;
    }
    std::sort(
        argsort.begin(),
        argsort.end(),
        [&frame_indices_data](size_t a, size_t b) {
          return frame_indices_data[a] < frame_indices_data[b];
        });
  }

  const auto& stream_info = stream_infos_[active_stream_index_];
  const auto& video_stream_options = stream_info.video_stream_options;
  FrameBatchOutput frame_batch_output(
      frame_indices.numel(),
      get_output_dims(),
      video_stream_options.device,
      device_interface_->get_pre_allocation_dtype(
          video_stream_options.output_dtype));

  auto frame_batch_output_pts_seconds =
      mutable_accessor<double, 1>(frame_batch_output.pts_seconds);
  auto frame_batch_output_duration_seconds =
      mutable_accessor<double, 1>(frame_batch_output.duration_seconds);

  auto previous_index_in_video = -1;
  for (int64_t f = 0; f < frame_indices.numel(); ++f) {
    auto index_in_output = indices_are_sorted ? f : argsort[f];
    auto index_in_video = frame_indices_data[index_in_output];

    if ((f > 0) && (index_in_video == previous_index_in_video)) {
      // Avoid decoding the same frame twice
      auto previous_index_in_output =
          indices_are_sorted ? f - 1 : argsort[f - 1];
      copy_frame(
          frame_batch_output.data,
          index_in_output,
          frame_batch_output.data,
          previous_index_in_output);
      frame_batch_output_pts_seconds[index_in_output] =
          frame_batch_output_pts_seconds[previous_index_in_output];
      frame_batch_output_duration_seconds[index_in_output] =
          frame_batch_output_duration_seconds[previous_index_in_output];
    } else {
      FrameOutput frame_output = get_frame_at_index_internal(
          index_in_video, select_row(frame_batch_output.data, index_in_output));
      frame_batch_output_pts_seconds[index_in_output] =
          frame_output.pts_seconds;
      frame_batch_output_duration_seconds[index_in_output] =
          frame_output.duration_seconds;
    }
    previous_index_in_video = index_in_video;
  }
  frame_batch_output.data =
      maybe_permute_and_convert_dtype(frame_batch_output.data);
  return frame_batch_output;
}

FrameBatchOutput SingleStreamDecoder::get_frames_in_range(
    int64_t start,
    int64_t stop,
    int64_t step) {
  validate_active_stream(AVMEDIA_TYPE_VIDEO);

  const auto& stream_metadata =
      container_metadata_.all_stream_metadata[active_stream_index_];
  const auto& stream_info = stream_infos_[active_stream_index_];
  STD_TORCH_CHECK(
      start >= 0, "Range start, " + std::to_string(start) + " is less than 0.");
  STD_TORCH_CHECK(
      step > 0, "Step must be greater than 0; is " + std::to_string(step));

  // Note that if we do not have the number of frames available in our
  // metadata, then we assume that the upper part of the range is valid.
  std::optional<int64_t> num_frames =
      stream_metadata.get_num_frames(seek_mode_);
  if (num_frames.has_value()) {
    STD_TORCH_CHECK(
        stop <= num_frames.value(),
        "Range stop, " + std::to_string(stop) +
            ", is more than the number of frames, " +
            std::to_string(num_frames.value()));
  }

  int64_t num_output_frames = std::ceil((stop - start) / double(step));
  const auto& video_stream_options = stream_info.video_stream_options;
  FrameBatchOutput frame_batch_output(
      num_output_frames,
      get_output_dims(),
      video_stream_options.device,
      device_interface_->get_pre_allocation_dtype(
          video_stream_options.output_dtype));

  auto frame_batch_output_pts_seconds =
      mutable_accessor<double, 1>(frame_batch_output.pts_seconds);
  auto frame_batch_output_duration_seconds =
      mutable_accessor<double, 1>(frame_batch_output.duration_seconds);
  for (int64_t i = start, f = 0; i < stop; i += step, ++f) {
    FrameOutput frame_output =
        get_frame_at_index_internal(i, select_row(frame_batch_output.data, f));
    frame_batch_output_pts_seconds[f] = frame_output.pts_seconds;
    frame_batch_output_duration_seconds[f] = frame_output.duration_seconds;
  }
  frame_batch_output.data =
      maybe_permute_and_convert_dtype(frame_batch_output.data);
  return frame_batch_output;
}

FrameOutput SingleStreamDecoder::get_frame_played_at(double seconds) {
  validate_active_stream(AVMEDIA_TYPE_VIDEO);
  StreamInfo& stream_info = stream_infos_[active_stream_index_];
  double last_decoded_start_time =
      pts_to_seconds(last_decoded_av_frame_pts_, stream_info.time_base);
  double last_decoded_end_time = pts_to_seconds(
      last_decoded_av_frame_pts_ + last_decoded_av_frame_duration_,
      stream_info.time_base);
  if (seconds >= last_decoded_start_time && seconds < last_decoded_end_time) {
    // We are in the same frame as the one we just returned. However, since we
    // don't cache it locally, we have to rewind back.
    seconds = last_decoded_start_time;
  }

  set_cursor_pts_in_seconds(seconds);
  UniqueAVFrame av_frame =
      decode_av_frame([seconds, this](const AVFrame& av_frame) {
        StreamInfo& stream_info = stream_infos_[active_stream_index_];
        double frame_start_time =
            pts_to_seconds(get_pts_or_dts(av_frame), stream_info.time_base);
        double frame_end_time = pts_to_seconds(
            get_pts_or_dts(av_frame) + get_duration(av_frame),
            stream_info.time_base);
        if (frame_start_time > seconds) {
          // FFMPEG seeked past the frame we are looking for even though we
          // set max_ts to be our needed timestamp in avformat_seek_file()
          // in maybeSeekToBeforeDesiredPts().
          // This could be a bug in FFMPEG:
          // https://trac.ffmpeg.org/ticket/11137 In this case we return the
          // very next frame instead of throwing an exception.
          // TODO: Maybe log to stderr for Debug builds?
          return true;
        }
        return seconds >= frame_start_time && seconds < frame_end_time;
      });

  // Convert the frame to tensor.
  FrameOutput frame_output = convert_av_frame_to_frame_output(*av_frame);
  frame_output.data = maybe_permute_and_convert_dtype(frame_output.data);
  return frame_output;
}

FrameBatchOutput SingleStreamDecoder::get_frames_played_at(
    const torch::stable::Tensor& timestamps) {
  validate_active_stream(AVMEDIA_TYPE_VIDEO);

  const auto& stream_metadata =
      container_metadata_.all_stream_metadata[active_stream_index_];

  double min_seconds = stream_metadata.get_begin_stream_seconds(seek_mode_);
  std::optional<double> max_seconds =
      stream_metadata.get_end_stream_seconds(seek_mode_);

  // The frame played at timestamp t and the one played at timestamp `t +
  // eps` are probably the same frame, with the same index. The easiest way to
  // avoid decoding that unique frame twice is to convert the input timestamps
  // to indices, and leverage the de-duplication logic of getFramesAtIndices.

  torch::stable::Tensor frame_indices =
      torch::stable::empty({timestamps.numel()}, kStableInt64);
  auto frame_indices_accessor = mutable_accessor<int64_t, 1>(frame_indices);
  auto timestamps_accessor = const_accessor<double, 1>(timestamps);

  for (int64_t i = 0; i < timestamps.numel(); ++i) {
    auto frame_seconds = timestamps_accessor[i];
    STD_TORCH_CHECK(
        frame_seconds >= min_seconds,
        "frame pts is " + std::to_string(frame_seconds) +
            "; must be greater than or equal to " +
            std::to_string(min_seconds) + ".");

    // Note that if we can't determine the maximum number of seconds from the
    // metadata, then we assume the frame's pts is valid.
    if (max_seconds.has_value()) {
      STD_TORCH_CHECK(
          frame_seconds < max_seconds.value(),
          "frame pts is " + std::to_string(frame_seconds) +
              "; must be less than " + std::to_string(max_seconds.value()) +
              ".");
    }

    frame_indices_accessor[i] = seconds_to_index_lower_bound(frame_seconds);
  }

  return get_frames_at_indices(frame_indices);
}

FrameBatchOutput SingleStreamDecoder::get_frames_played_in_range(
    double start_seconds,
    double stop_seconds,
    std::optional<double> fps) {
  validate_active_stream(AVMEDIA_TYPE_VIDEO);
  const auto& stream_metadata =
      container_metadata_.all_stream_metadata[active_stream_index_];
  STD_TORCH_CHECK(
      start_seconds <= stop_seconds,
      "Start seconds (" + std::to_string(start_seconds) +
          ") must be less than or equal to stop seconds (" +
          std::to_string(stop_seconds) + ".");

  const auto& stream_info = stream_infos_[active_stream_index_];
  const auto& video_stream_options = stream_info.video_stream_options;

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
  // which by our abstract player, means that both intervals map to frame 0.
  // By the definition of a half open interval, interval A should return no
  // frames. Interval B should return frame 0. However, for both A and B, the
  // individual values of the intervals will map to the same frame indices
  // below. Hence, we need this special case below.
  if (start_seconds == stop_seconds) {
    FrameBatchOutput frame_batch_output(
        0,
        get_output_dims(),
        video_stream_options.device,
        device_interface_->get_pre_allocation_dtype(
            video_stream_options.output_dtype));
    frame_batch_output.data =
        maybe_permute_and_convert_dtype(frame_batch_output.data);
    return frame_batch_output;
  }

  double min_seconds = stream_metadata.get_begin_stream_seconds(seek_mode_);
  STD_TORCH_CHECK(
      start_seconds >= min_seconds,
      "Start seconds is " + std::to_string(start_seconds) +
          "; must be greater than or equal to " + std::to_string(min_seconds) +
          ".");

  // Note that if we can't determine the maximum seconds from the metadata,
  // then we assume upper range is valid.
  std::optional<double> max_seconds =
      stream_metadata.get_end_stream_seconds(seek_mode_);
  if (max_seconds.has_value()) {
    STD_TORCH_CHECK(
        start_seconds < max_seconds.value(),
        "Start seconds is " + std::to_string(start_seconds) +
            "; must be less than " + std::to_string(max_seconds.value()) + ".");
    STD_TORCH_CHECK(
        stop_seconds <= max_seconds.value(),
        "Stop seconds (" + std::to_string(stop_seconds) +
            "; must be less than or equal to " +
            std::to_string(max_seconds.value()) + ").");
  }

  // Resample frames to match the target frame rate
  if (fps.has_value()) {
    STD_TORCH_CHECK(
        fps.value() > 0,
        "fps must be positive, got " + std::to_string(fps.value()));

    // TODO: add an early break if requested fps is the same as the current fps

    double fps_val = fps.value();
    double frame_duration_seconds = 1.0 / fps_val;

    double product = (stop_seconds - start_seconds) * fps_val;
    int64_t num_output_frames = static_cast<int64_t>(std::round(product));

    FrameBatchOutput frame_batch_output(
        num_output_frames,
        get_output_dims(),
        video_stream_options.device,
        device_interface_->get_pre_allocation_dtype(
            video_stream_options.output_dtype));

    auto frame_batch_output_pts_seconds =
        mutable_accessor<double, 1>(frame_batch_output.pts_seconds);
    auto frame_batch_output_duration_seconds =
        mutable_accessor<double, 1>(frame_batch_output.duration_seconds);

    // Decode frames, reusing already-decoded frames for duplicates
    int64_t last_decoded_source_index = -1;

    for (int64_t i = 0; i < num_output_frames; ++i) {
      double target_pts_seconds = start_seconds + i * frame_duration_seconds;
      int64_t source_idx = seconds_to_index_lower_bound(target_pts_seconds);

      if (source_idx == last_decoded_source_index &&
          last_decoded_source_index >= 0) {
        copy_frame(frame_batch_output.data, i, frame_batch_output.data, i - 1);
      } else {
        get_frame_at_index_internal(
            source_idx, select_row(frame_batch_output.data, i));
        last_decoded_source_index = source_idx;
      }

      frame_batch_output_pts_seconds[i] = target_pts_seconds;
      frame_batch_output_duration_seconds[i] = frame_duration_seconds;
    }

    frame_batch_output.data =
        maybe_permute_and_convert_dtype(frame_batch_output.data);
    return frame_batch_output;
  } else {
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

    int64_t start_frame_index = seconds_to_index_lower_bound(start_seconds);
    int64_t stop_frame_index = seconds_to_index_upper_bound(stop_seconds);
    int64_t num_frames = stop_frame_index - start_frame_index;

    FrameBatchOutput frame_batch_output(
        num_frames,
        get_output_dims(),
        video_stream_options.device,
        device_interface_->get_pre_allocation_dtype(
            video_stream_options.output_dtype));
    auto frame_batch_output_pts_seconds =
        mutable_accessor<double, 1>(frame_batch_output.pts_seconds);
    auto frame_batch_output_duration_seconds =
        mutable_accessor<double, 1>(frame_batch_output.duration_seconds);
    for (int64_t i = start_frame_index, f = 0; i < stop_frame_index; ++i, ++f) {
      FrameOutput frame_output = get_frame_at_index_internal(
          i, select_row(frame_batch_output.data, f));
      frame_batch_output_pts_seconds[f] = frame_output.pts_seconds;
      frame_batch_output_duration_seconds[f] = frame_output.duration_seconds;
    }
    frame_batch_output.data =
        maybe_permute_and_convert_dtype(frame_batch_output.data);

    return frame_batch_output;
  }
}

// clang-format off
//
// Note [Audio Decoding Design]
// This note explains why audio decoding is implemented the way it is, and why
// it inherently differs from video decoding.
//
// Like for video, FFmpeg exposes the concept of a frame for audio streams. An
// audio frame is a contiguous sequence of samples, where a sample consists of
// `numChannels` values. An audio frame, or a sequence thereof, is always
// converted into a tensor of shape `(numChannels, numSamplesPerChannel)`.
//
// The notion of 'frame' in audio isn't what users want to interact with.
// Users want to interact with samples. The C++ and core APIs return frames,
// because we want those to be close to FFmpeg concepts, but the higher-level
// public APIs expose samples. As a result:
// - We don't expose index-based APIs for audio, because that would mean
//   exposing the concept of audio frame. For now, we think exposing
//   time-based APIs is more natural.
// - We never perform a scan for audio streams. We don't need to, since we
//   won't be converting timestamps to indices. That's why we enforce the
//   seek_mode to be "approximate" (which is slightly misleading, because
//   technically the output samples will be at their exact positions. But
//   this incongruence is only exposed at the C++/core private levels).
//
// Audio frames are of variable dimensions: in the same stream, a frame can
// contain 1024 samples and the next one may contain 512 [1]. This makes it
// impossible to stack audio frames in the same way we can stack video frames.
// This is one of the main reasons we cannot reuse the same pre-allocation
// logic we have for videos in getFramesPlayedInRange(): pre-allocating a
// batch requires constant (and known) frame dimensions. That's also why
// *concatenated* along the samples dimension, not stacked.
//
// Note [Audio pre-roll and post-roll]
//
// get_samples_played_in_range(a, b) decodes more than [a, b):
// target_preroll_seconds before a, and target_postroll_seconds after b, which
// the Python layer then trims off. Two independent data dependencies need this
// extra margin:
//
// 1. The codec. Most lossy codecs (AAC, MP3, Vorbis, AC-3...) use MDCT with
// overlap-add the output of frame i depends on state accumulated from frame
// i-1, sometimes earlier. avcodec_flush_buffers() empties it on a seek, so the
// first frames decoded after one come out wrong. Decoding k_num_preroll_frames
// frames before the target primes it back - libmpg123 calls these
// "ignoreframes" [1][2] and defaults to 4, which is where our 4 comes from.  1
// is believed to be enough for aac, vorbis and mp3. If frame_size is unknown we
// fall back to k_fallback_preroll_seconds. Lossless codecs don't need such a
// pre-roll, so we avoid it for those.
//
// 2. The resampler, when resampling. It is an interpolation filter: the sample
// it emits at a given instant is a weighted sum of the input samples on both
// sides of it. That is a data dependency of its own, and it holds whether or
// not the codec has one. On top of it, a freshly created resampler skips up to
// num_samples_to_skip input samples to land on the grid, see [Audio resampling
// and frame alignment]. So it needs:
//
//   before a:  num_samples_to_skip + filter_length   input samples
//   after b:                         filter_length   input samples
//
// where filter_length is computed like in FFmpeg [3]. To be on the safe side,
// we actually use 2 * filter_length.
//
// Typical values look like:
//
//     isr -> osr      num_samples_to_skip         filter_length    target_preroll_seconds
//     24000 -> 16000                    3                    50      102 smp =    4.25 ms
//     44100 -> 16000                  441                    91      622 smp =   14.10 ms
//     44100 ->  8000                  441                   182      804 smp =   18.23 ms
//     44100 -> 16001                44100                    91    44281 smp = 1004.10 ms
//
// Note that at worst, we need to preroll ~1s worth of samples: by definition,
// the input and output sample rates align every second.
//
// target_preroll_seconds is the max of the two: the preroll needed by the
// codec, and the preroll needed by the resampler.
//
// [1]
// https://github.com/gypified/libmpg123/blob/8cbf2faf994bd999ce2b45869093bd61ecf8416f/src/libmpg123/frame.c#L884-L894
// [2]
// https://github.com/gypified/libmpg123/blob/8cbf2faf994bd999ce2b45869093bd61ecf8416f/src/libmpg123/libmpg123.c#L586
// [3] https://github.com/FFmpeg/FFmpeg/blob/844e10e1a74929c27dfe0aac1c34e92bb6edbf5a/libswresample/resample.c#L188-L193
//
// clang-format on
AudioFramesOutput SingleStreamDecoder::get_frames_played_in_range_audio(
    double start_seconds,
    std::optional<double> stop_seconds_optional) {
  validate_active_stream(AVMEDIA_TYPE_AUDIO);

  if (stop_seconds_optional.has_value()) {
    STD_TORCH_CHECK(
        start_seconds <= *stop_seconds_optional,
        "Start seconds (" + std::to_string(start_seconds) +
            ") must be less than or equal to stop seconds (" +
            std::to_string(*stop_seconds_optional) + ").");
  }

  StreamInfo& stream_info = stream_infos_[active_stream_index_];

  if (stop_seconds_optional.has_value() &&
      start_seconds == *stop_seconds_optional) {
    // For consistency with video
    int num_channels = get_num_channels(stream_info.codec_context);
    return AudioFramesOutput{torch::stable::empty({num_channels, 0}), 0.0};
  }

  auto start_pts = seconds_to_closest_pts(start_seconds, stream_info.time_base);

  // We must figure out if we need to seek, and where to. The following figures
  // out the value of target_preroll_seconds. See the [Audio pre-roll and
  // post-roll]  note.
  // Codec pre-roll
  static constexpr int k_num_preroll_frames = 4;
  static constexpr double k_fallback_preroll_seconds = 1.0;
  int src_sample_rate = stream_info.codec_context->sample_rate;
  int frame_size = stream_info.codec_context->frame_size;

  const AVCodecDescriptor* codec_descriptor =
      avcodec_descriptor_get(stream_info.codec_context->codec_id);
  bool is_lossless = codec_descriptor != nullptr &&
      (codec_descriptor->props & AV_CODEC_PROP_LOSSLESS) != 0;

  double target_preroll_seconds;
  double target_postroll_seconds = 0.0;
  if (is_lossless) {
    target_preroll_seconds = 0.0;
  } else if (frame_size > 0) {
    target_preroll_seconds = static_cast<double>(k_num_preroll_frames) *
        frame_size / src_sample_rate;
  } else {
    target_preroll_seconds = k_fallback_preroll_seconds;
  }

  // Resampler pre-roll
  int out_sample_rate =
      stream_info.audio_stream_options.sample_rate.value_or(src_sample_rate);
  bool is_resampling = out_sample_rate != src_sample_rate;
  if (is_resampling) {
    // See
    // https://github.com/FFmpeg/FFmpeg/blob/844e10e1a74929c27dfe0aac1c34e92bb6edbf5a/libswresample/resample.c#L188-L193
    // for constants, and the note mentioned above.
    static constexpr int k_swr_filter_size = 32;
    static constexpr double k_swr_cutoff = 0.97;
    double factor =
        std::min(1.0, out_sample_rate * k_swr_cutoff / src_sample_rate);
    double filter_length = std::ceil(k_swr_filter_size / factor);
    // alignment is k in the  [Audio resampling and frame alignment] note,
    // alignment - 1 bounds num_samples_to_skip so it's safe to use in our
    // preroll calculation.
    int64_t alignment =
        src_sample_rate / std::gcd(src_sample_rate, out_sample_rate);

    // double filter length to be safe.
    target_preroll_seconds = std::max(
        target_preroll_seconds,
        (alignment - 1 + 2 * filter_length) / src_sample_rate);

    target_postroll_seconds = 2 * filter_length / src_sample_rate;
  }
  auto target_seek_pts = seconds_to_closest_pts(
      start_seconds - target_preroll_seconds, stream_info.time_base);

  int64_t min_pts = stream_info.stream->start_time != AV_NOPTS_VALUE
      ? stream_info.stream->start_time
      : 0;

  bool needs_seek;
  if (last_decoded_av_frame_pts_ == INT64_MIN) {
    // Fresh decoder: in theory we'd always seek, but we can't because seeking
    // to INT64_MIN (the priming-packet path below) fails on some formats like
    // FLAC, see test_fresh_decoder_seek
    needs_seek = target_seek_pts > min_pts;
  } else {
    auto current_end =
        last_decoded_av_frame_pts_ + last_decoded_av_frame_duration_;
    // We seek if we need to go backwards, or if the target is far enough
    // forward that decoding every intermediate frame would be wasteful.
    needs_seek = start_pts < current_end ||
        start_pts > current_end + (start_pts - target_seek_pts);
  }

  if (needs_seek) {
    if (target_seek_pts <= min_pts) {
      // Edge case: when seeking to the very beginning of the stream, there
      // are no earlier frames to use as preroll. In that case we seek with
      // INT64_MIN, which lets the demuxer land on the true first packet
      // (including any priming packets with negative PTS, such as the AAC
      // priming frame). Not super clear why this is needed, but we have tests
      // that fail without this.
      set_cursor(INT64_MIN);
    } else {
      set_cursor(target_seek_pts);
    }
  }

  // TODO-AUDIO Pre-allocate a long-enough tensor instead of creating a vec +
  // cat(). This would save a copy. We know the duration of the output and the
  // sample rate, so in theory we know the number of output samples.
  std::vector<torch::stable::Tensor> frames;

  std::optional<double> first_frame_pts_seconds = std::nullopt;
  auto stop_pts = stop_seconds_optional.has_value()
      ? seconds_to_closest_pts(*stop_seconds_optional, stream_info.time_base)
      : INT64_MAX;

  // When resampling, we must send the prerolled and postrolled frames through
  // the resampler!
  auto output_start_pts =
      is_resampling ? std::min(start_pts, target_seek_pts) : start_pts;
  auto output_stop_pts = (stop_pts == INT64_MAX) ? stop_pts
                                                 : stop_pts +
          seconds_to_closest_pts(target_postroll_seconds,
                                 stream_info.time_base);

  auto finished = false;
  while (!finished) {
    try {
      UniqueAVFrame av_frame = decode_av_frame(
          [output_start_pts, output_stop_pts](const AVFrame& av_frame) {
            return output_start_pts <
                get_pts_or_dts(av_frame) + get_duration(av_frame) &&
                output_stop_pts > get_pts_or_dts(av_frame);
          });
      auto frame_output = convert_av_frame_to_frame_output(*av_frame);
      if (!first_frame_pts_seconds.has_value()) {
        first_frame_pts_seconds = frame_output.pts_seconds;
      }
      frames.push_back(frame_output.data);
    } catch (const EndOfFileException&) {
      finished = true;
    }

    // We're done once output_stop_pts falls within the last decoded frame,
    // bounds
    // included. Excluding the upper bound would look more natural, but when
    // output_stop_pts lands exactly on a frame boundary we'd keep going: the
    // frame starting there is rejected by the pts filter above, as is every
    // frame after it, so we'd decode to EOF for the same output.

    auto last_decoded_av_frame_end =
        last_decoded_av_frame_pts_ + last_decoded_av_frame_duration_;
    finished |= (last_decoded_av_frame_pts_) <= output_stop_pts &&
        (output_stop_pts <= last_decoded_av_frame_end);
  }

  auto last_samples = device_interface_->maybe_flush_audio_buffers();
  if (last_samples.has_value()) {
    frames.push_back(*last_samples);
  }

  STD_TORCH_CHECK(
      frames.size() > 0 && first_frame_pts_seconds.has_value(),
      "No audio frames were decoded. ",
      "This is probably because start_seconds is too high(",
      start_seconds,
      "),",
      "or because stop_seconds(",
      stop_seconds_optional.has_value() ? std::to_string(*stop_seconds_optional)
                                        : "nullopt",
      ") is too low.");

  return AudioFramesOutput{stable_cat(frames, 1), *first_frame_pts_seconds};
}

// --------------------------------------------------------------------------
// SEEKING APIs
// --------------------------------------------------------------------------

void SingleStreamDecoder::set_cursor_pts_in_seconds(double seconds) {
  // Audio seeking is handled internally by getFramesPlayedInRangeAudio()
  // with preroll, see [Audio pre-roll and post-roll].
  validate_active_stream(AVMEDIA_TYPE_VIDEO);
  set_cursor(seconds_to_closest_pts(
      seconds, stream_infos_[active_stream_index_].time_base));
}

void SingleStreamDecoder::set_cursor(int64_t pts) {
  cursor_was_just_set_ = true;
  cursor_ = pts;
}

bool SingleStreamDecoder::can_we_avoid_seeking() const {
  // Returns true if we can avoid seeking in the AVFormatContext based on
  // heuristics that rely on the target cursor_ and the last decoded frame.
  // Seeking is expensive, so we try to avoid it when possible.
  const StreamInfo& stream_info = stream_infos_.at(active_stream_index_);
  if (stream_info.av_media_type == AVMEDIA_TYPE_AUDIO) {
    // For audio, seeking is handled internally by
    // getFramesPlayedInRangeAudio(). See [Audio pre-roll and post-roll].
    return !cursor_was_just_set_;
  } else if (!cursor_was_just_set_) {
    // For videos, when decoding consecutive frames, we don't need to seek.
    return true;
  }

  if (cursor_ < last_decoded_av_frame_pts_) {
    // We can never skip a seek if we are seeking backwards.
    return false;
  }
  if (last_decoded_av_frame_pts_ == cursor_) {
    // We are seeking to the exact same frame as we are currently at. Without
    // caching we have to rewind back and decode the frame again.
    return false;
  }
  // We are seeking forward, from the current frame x (lastDecodedAvFramePts_)
  // to a target frame y (cursor_). Seeking means jumping to the last keyframe
  // before y, which we call j:
  //
  //   .........x...............j...............y.......
  //            ^               ^               ^
  //       current frame   keyframe we'd     frame we
  //       (last decoded)  seek to           want
  //
  // Whether seeking is worth it depends on how far j is from x. Seeking flushes
  // the decoder, which throws away the `has_b_frames` frames already sitting in
  // the reorder buffer plus the `thread_count - 1` frames in flight in the
  // threading pipeline -- exactly the frames we'd decode next anyway. So we
  // only seek if j is beyond them, i.e. if decoding straight through would cost
  // more than the flush:
  //
  //   seek iff (j - x) > has_b_frames + thread_count - 1
  //
  // (the thread_count term only applies to CPU decoding and to FRAME threading,
  // see below.)
  // See https://github.com/meta-pytorch/torchcodec/issues/1488 for details.

  // These "identifiers" only let us tell whether x and y share the same
  // keyframe. They are not necessarily frame indices (see
  // getKeyFrameIdentifier()), so we use them for equality only, not for the
  // (j - x) distance below.
  int last_key_frame_id = get_key_frame_identifier(last_decoded_av_frame_pts_);
  int target_key_frame_id = get_key_frame_identifier(cursor_);
  if (last_key_frame_id < 0 || target_key_frame_id < 0) {
    return false;
  }
  if (last_key_frame_id == target_key_frame_id) {
    // x and y share the same keyframe (as in `...j...x...y`): seeking would go
    // backwards to j and re-decode up to x, landing us back where we are.
    return true;
  }

  // x
  int64_t last_decoded_frame_index = seconds_to_index_lower_bound(
      pts_to_seconds(last_decoded_av_frame_pts_, stream_info.time_base));

  // j
  int64_t target_key_frame_index;
  if (!stream_info.key_frames.empty()) {
    // exact and custom_frame_mappings modes: getKeyFrameIdentifier() returned
    // getKeyFrameIndexForPtsUsingScannedIndex() for these modes, so
    // targetKeyFrameId is already the position of j in the scanned keyFrames
    // vector.
    target_key_frame_index =
        stream_info.key_frames[target_key_frame_id].frame_index;
  } else {
    // approximate mode: keyFrames is empty, so we can't locate j. We use y
    // instead. This makes the heuristic more conservative, i.e. err towards
    // seeking even more, which is safe.
    target_key_frame_index = seconds_to_index_lower_bound(
        pts_to_seconds(cursor_, stream_info.time_base));
  }

  int64_t frame_reorder_buffer_size =
      std::max(stream_info.codec_context->has_b_frames, 0);

  // The reorder buffer (has_b_frames) is a codec property that applies
  // in all scenarios, but the `thread_count - 1` in-flight frames only exist
  // with FFmpeg FRAME threading, in the CPU decoding path. On other devices
  // (e.g. CUDA) decoding doesn't depend on thread_count, and with SLICE
  // threading the threads cooperate on a single frame rather than pipelining
  // several frames. Note that FRAME threading is the default across the vast
  // majority of codecs, so our check is just to be on the safe side.
  int64_t in_flight_frames = 0;
  if (stream_info.video_stream_options.device == kStableCPU &&
      (stream_info.codec_context->active_thread_type & FF_THREAD_FRAME)) {
    in_flight_frames = std::max(stream_info.codec_context->thread_count, 1) - 1;
  }
  return (target_key_frame_index - last_decoded_frame_index) <=
      frame_reorder_buffer_size + in_flight_frames;
}

// This method looks at currentPts and desiredPts and seeks in the
// AVFormatContext if it is needed. We can skip seeking in certain cases. See
// the comment of canWeAvoidSeeking() for details.
// Returns true iff we seeked.
bool SingleStreamDecoder::maybe_seek_to_before_desired_pts() {
  validate_active_stream();
  StreamInfo& stream_info = stream_infos_[active_stream_index_];

  decode_stats_.num_seeks_attempted++;
  if (can_we_avoid_seeking()) {
    decode_stats_.num_seeks_skipped++;
    return false;
  }

  int64_t desired_pts = cursor_;

  // For some encodings like H265, FFMPEG sometimes seeks past the point we
  // set as the max_ts. So we use our own index to give it the exact pts of
  // the key frame that we want to seek to.
  // See https://github.com/pytorch/torchcodec/issues/179 for more details.
  // See https://trac.ffmpeg.org/ticket/11137 for the underlying ffmpeg bug.
  if (!stream_info.key_frames.empty()) {
    int desired_key_frame_index =
        get_key_frame_index_for_pts_using_scanned_index(
            stream_info.key_frames, desired_pts);
    if (desired_key_frame_index >= 0) {
      desired_pts = stream_info.key_frames[desired_key_frame_index].pts;
    }
    // Otherwise the target precedes the first *scanned* keyframe. This happens
    // when the keyframe needed to decode the target was discarded / trimmed by
    // an edit list (AV_PKT_FLAG_DISCARD) and thus not recorded in our index. We
    // leave desired_pts as the target pts and let FFmpeg seek to the preceding
    // keyframe itself (same as approximate mode, which has no scanned index).
  }

  int status = avformat_seek_file(
      format_context_.get(),
      stream_info.stream_index,
      INT64_MIN,
      desired_pts,
      desired_pts,
      0);
  STD_TORCH_CHECK(
      status >= 0,
      get_seek_error_message(format_context_.get(), desired_pts, status));

  decode_stats_.num_flushes++;
  device_interface_->flush();
  return true;
}

// --------------------------------------------------------------------------
// LOW-LEVEL DECODING
// --------------------------------------------------------------------------

UniqueAVFrame SingleStreamDecoder::decode_av_frame(
    std::function<bool(const AVFrame&)> filter_function) {
  validate_active_stream();

  reset_decode_stats();

  bool seeked = maybe_seek_to_before_desired_pts();
  cursor_was_just_set_ = false;

  // See below where it's used
  bool packet_data_may_be_misaligned = seeked && is_mpeg_ps_;

  UniqueAVFrame av_frame(av_frame_alloc());
  AutoAVPacket auto_av_packet;
  int status = AVSUCCESS;
  bool reached_eof = false;

  // The default implementation uses avcodec_receive_frame and
  // avcodec_send_packet, while specialized interfaces can override for
  // hardware-specific optimizations.
  while (true) {
    status = device_interface_->receive_frame(av_frame);

    if (status != AVSUCCESS && status != AVERROR(EAGAIN)) {
      // Non-retriable error
      break;
    }

    decode_stats_.num_frames_received_by_decoder++;
    // Is this the kind of frame we're looking for?
    if (status == AVSUCCESS && filter_function(*av_frame)) {
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

    if (reached_eof) {
      // We don't have any more packets to receive. So keep on pulling frames
      // from decoder's internal buffers.
      continue;
    }

    // We still haven't found the frame we're looking for. So let's read more
    // packets and send them to the decoder. (read_next_packet is shared with
    // the Demuxer building block.)
    ReferenceAVPacket packet(auto_av_packet);
    status =
        read_next_packet(format_context_.get(), active_stream_index_, *packet);
    decode_stats_.num_packets_read++;

    if (status == AVERROR_EOF) {
      // End of file reached. We must drain the decoder
      status = device_interface_->send_eof_packet();
      STD_TORCH_CHECK(
          status >= AVSUCCESS,
          "Could not flush decoder: ",
          get_ffmpeg_error_string_from_error_code(status));

      reached_eof = true;
    } else {
      STD_TORCH_CHECK(
          status >= AVSUCCESS,
          "Could not read frame from input file: ",
          get_ffmpeg_error_string_from_error_code(status));
    }

    if (reached_eof) {
      // We don't have any more packets to send to the decoder. So keep on
      // pulling frames from its internal buffers.
      continue;
    }

    // We got a valid packet. Send it to the decoder, and we'll receive it in
    // the next iteration.
    status = device_interface_->send_packet(packet);

    if (status == AVERROR_INVALIDDATA && packet_data_may_be_misaligned) {
      // The MPEG-PS demuxer doesn't return proper packets just after a seek, so
      // we skip (potentially more than one) packets until we find a valid one.
      // We *only* do this for MPEG-PS, not for other demuxers. It is
      // unfortunate that we need a format-specific condition in this generic
      // decoding loop.
      continue;
    }
    STD_TORCH_CHECK(
        status >= AVSUCCESS,
        "Could not push packet to decoder: ",
        get_ffmpeg_error_string_from_error_code(status));
    // The decoder accepted a packet, so we're aligned again: from now on
    // invalid data means the file is corrupt, and we want to report it.
    packet_data_may_be_misaligned = false;

    decode_stats_.num_packets_sent_to_decoder++;
  }

  if (status < AVSUCCESS) {
    if (reached_eof || status == AVERROR_EOF) {
      throw SingleStreamDecoder::EndOfFileException(
          "Requested next frame while there are no more frames left to "
          "decode.");
    }
    STD_TORCH_CHECK(
        false,
        "Could not receive frame from decoder: ",
        get_ffmpeg_error_string_from_error_code(status));
  }

  // Note that we don't flush the decoder when we reach EOF (even though
  // that's mentioned in
  // https://ffmpeg.org/doxygen/trunk/group__lavc__encdec.html). This is
  // because we may have packets internally in the decoder that we haven't
  // received as frames. Eventually we will either hit AVERROR_EOF from
  // av_receive_frame() or the user will have seeked to a different location
  // in the file and that will flush the decoder.
  last_decoded_av_frame_pts_ = get_pts_or_dts(*av_frame);
  last_decoded_av_frame_duration_ = get_duration(*av_frame);

  return av_frame;
}

// --------------------------------------------------------------------------
// AVFRAME <-> FRAME OUTPUT CONVERSION
// --------------------------------------------------------------------------

FrameOutput SingleStreamDecoder::convert_av_frame_to_frame_output(
    const AVFrame& av_frame,
    std::optional<torch::stable::Tensor> pre_allocated_output_tensor) {
  // Convert the frame to tensor.
  FrameOutput frame_output;
  frame_output.pts_seconds = pts_to_seconds(
      get_pts_or_dts(av_frame),
      format_context_->streams[active_stream_index_]->time_base);
  frame_output.duration_seconds = pts_to_seconds(
      get_duration(av_frame),
      format_context_->streams[active_stream_index_]->time_base);
  device_interface_->convert_av_frame_to_frame_output(
      av_frame, frame_output, std::move(pre_allocated_output_tensor));
  return frame_output;
}

// --------------------------------------------------------------------------
// OUTPUT ALLOCATION AND SHAPE CONVERSION
// --------------------------------------------------------------------------

torch::stable::Tensor SingleStreamDecoder::maybe_permute_and_convert_dtype(
    torch::stable::Tensor& hwc_tensor) {
  // Permute HWC to CHW if needed. Returns a view of the input tensor, the
  // leading batch-dimension [N] is optional i.e. the input tensor can be 3D or
  // 4D.
  torch::stable::Tensor tensor = hwc_tensor;
  if (stream_infos_[active_stream_index_]
          .video_stream_options.dimension_order != "NHWC") {
    auto num_dimensions = hwc_tensor.dim();
    auto shape = hwc_tensor.sizes();
    if (num_dimensions == 3) {
      STD_TORCH_CHECK(
          shape[2] == 3, "Not a HWC tensor: ", int_array_ref_to_string(shape));
      tensor = stable_permute(hwc_tensor, {2, 0, 1});
    } else if (num_dimensions == 4) {
      STD_TORCH_CHECK(
          shape[3] == 3, "Not a NHWC tensor: ", int_array_ref_to_string(shape));
      tensor = stable_permute(hwc_tensor, {0, 3, 1, 2});
    } else {
      STD_TORCH_CHECK(
          false,
          "Expected tensor with 3 or 4 dimensions, got ",
          num_dimensions);
    }
  }

  return convert_to_output_dtype(
      tensor,
      stream_infos_[active_stream_index_].video_stream_options.output_dtype);
}

// --------------------------------------------------------------------------
// PTS <-> INDEX CONVERSIONS
// --------------------------------------------------------------------------

int SingleStreamDecoder::get_key_frame_identifier(int64_t pts) const {
  // This function "identifies" a key frame for a given pts value.
  // We use the term "identifier" rather than "index" because the nature of the
  // index that is returned depends on various factors:
  // - If seek_mode is exact, we return the index of the key frame in the
  //   scanned key-frame vector (stream_info.keyFrames). So the returned value
  //   is in [0, num_key_frames).
  // - If seek_mode is approximate, we use av_index_search_timestamp() which
  //   may return a value in [0, num_key_frames) like for mkv, but also a value
  //   in [0, num_frames) like for mp4. It really depends on the container.
  //
  //  The range of the "identifier" doesn't matter that much, for now we only
  //  use it to uniquely identify a key frame in canWeAvoidSeeking().
  const StreamInfo& stream_info = stream_infos_.at(active_stream_index_);
  if (stream_info.key_frames.empty()) {
    return av_index_search_timestamp(
        stream_info.stream, pts, AVSEEK_FLAG_BACKWARD);
  } else {
    return get_key_frame_index_for_pts_using_scanned_index(
        stream_info.key_frames, pts);
  }
}

int SingleStreamDecoder::get_key_frame_index_for_pts_using_scanned_index(
    const std::vector<SingleStreamDecoder::FrameInfo>& key_frames,
    int64_t pts) const {
  auto upper_bound = std::upper_bound(
      key_frames.begin(),
      key_frames.end(),
      pts,
      [](int64_t pts, const SingleStreamDecoder::FrameInfo& frame_info) {
        return pts < frame_info.pts;
      });
  if (upper_bound == key_frames.begin()) {
    return -1;
  }
  return upper_bound - 1 - key_frames.begin();
}

int64_t SingleStreamDecoder::seconds_to_index_lower_bound(
    double seconds) const {
  const auto& stream_info = stream_infos_.at(active_stream_index_);
  switch (seek_mode_) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact: {
      auto frame = std::lower_bound(
          stream_info.all_frames.begin(),
          stream_info.all_frames.end(),
          seconds,
          [&stream_info](const FrameInfo& info, double start) {
            return pts_to_seconds(info.next_pts, stream_info.time_base) <=
                start;
          });

      return frame - stream_info.all_frames.begin();
    }
    case SeekMode::approximate: {
      auto& stream_metadata =
          container_metadata_.all_stream_metadata[active_stream_index_];
      STD_TORCH_CHECK(
          stream_metadata.average_fps_from_header.has_value(),
          "Cannot use approximate mode since we couldn't find the average fps from the metadata.");
      double begin_seconds =
          stream_metadata.get_begin_stream_seconds(seek_mode_);
      double relative_seconds = seconds - begin_seconds;
      return std::floor(
          relative_seconds * stream_metadata.average_fps_from_header.value());
    }
    default:
      STD_TORCH_CHECK(false, "Unknown SeekMode");
  }
}

int64_t SingleStreamDecoder::seconds_to_index_upper_bound(double seconds) {
  auto& stream_info = stream_infos_[active_stream_index_];
  switch (seek_mode_) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact: {
      auto frame = std::upper_bound(
          stream_info.all_frames.begin(),
          stream_info.all_frames.end(),
          seconds,
          [&stream_info](double stop, const FrameInfo& info) {
            return stop <= pts_to_seconds(info.pts, stream_info.time_base);
          });

      return frame - stream_info.all_frames.begin();
    }
    case SeekMode::approximate: {
      auto& stream_metadata =
          container_metadata_.all_stream_metadata[active_stream_index_];
      STD_TORCH_CHECK(
          stream_metadata.average_fps_from_header.has_value(),
          "Cannot use approximate mode since we couldn't find the average fps from the metadata.");
      double begin_seconds =
          stream_metadata.get_begin_stream_seconds(seek_mode_);
      double relative_seconds = seconds - begin_seconds;
      return std::ceil(
          relative_seconds * stream_metadata.average_fps_from_header.value());
    }
    default:
      STD_TORCH_CHECK(false, "Unknown SeekMode");
  }
}

int64_t SingleStreamDecoder::get_pts(int64_t frame_index) {
  auto& stream_info = stream_infos_[active_stream_index_];
  switch (seek_mode_) {
    case SeekMode::custom_frame_mappings:
    case SeekMode::exact:
      return stream_info.all_frames[frame_index].pts;
    case SeekMode::approximate: {
      auto& stream_metadata =
          container_metadata_.all_stream_metadata[active_stream_index_];
      STD_TORCH_CHECK(
          stream_metadata.average_fps_from_header.has_value(),
          "Cannot use approximate mode since we couldn't find the average fps from the metadata.");
      return seconds_to_closest_pts(
          stream_metadata.get_begin_stream_seconds(seek_mode_) +
              (frame_index / stream_metadata.average_fps_from_header.value()),
          stream_info.time_base);
    }
    default:
      STD_TORCH_CHECK(false, "Unknown SeekMode");
  }
}

FrameDims SingleStreamDecoder::get_output_dims() const {
  const auto& stream_metadata =
      container_metadata_.all_stream_metadata[active_stream_index_];
  Rotation rotation = rotation_from_degrees(stream_metadata.rotation);
  // If there is a rotation, then resizedOutputDims_ is necessarily non-null
  // (the rotation transform would have set it).
  if (rotation != Rotation::NONE) {
    STD_TORCH_CHECK(
        resized_output_dims_.has_value(),
        "Internal error: rotation is applied but resizedOutputDims_ is not set");
  }
  return resized_output_dims_.value_or(pre_rotation_dims_);
}

// --------------------------------------------------------------------------
// STREAM AND METADATA APIS
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// VALIDATION UTILS
// --------------------------------------------------------------------------

void SingleStreamDecoder::validate_active_stream(
    std::optional<AVMediaType> av_media_type) {
  auto error_msg =
      "Provided stream index=" + std::to_string(active_stream_index_) +
      " was not previously added.";
  STD_TORCH_CHECK(active_stream_index_ != no_active_stream_, error_msg);
  STD_TORCH_CHECK(stream_infos_.count(active_stream_index_) > 0, error_msg);

  int all_stream_metadata_size =
      static_cast<int>(container_metadata_.all_stream_metadata.size());
  STD_TORCH_CHECK(
      active_stream_index_ >= 0 &&
          active_stream_index_ < all_stream_metadata_size,
      "Invalid stream index=" + std::to_string(active_stream_index_) +
          "; valid indices are in the range [0, " +
          std::to_string(all_stream_metadata_size) + ").");

  if (av_media_type.has_value()) {
    STD_TORCH_CHECK(
        stream_infos_[active_stream_index_].av_media_type ==
            av_media_type.value(),
        "The method you called isn't supported. ",
        "If you're seeing this error, you are probably trying to call an ",
        "unsupported method on an audio stream.");
  }
}

void SingleStreamDecoder::validate_scanned_all_streams(const std::string& msg) {
  STD_TORCH_CHECK(
      scanned_all_streams_,
      "Must scan all streams to update metadata before calling ",
      msg);
}

void SingleStreamDecoder::validate_frame_index(
    const StreamMetadata& stream_metadata,
    int64_t frame_index) {
  STABLE_CHECK_INDEX(
      frame_index >= 0,
      "Invalid frame index=" + std::to_string(frame_index) +
          " for stream_index=" + std::to_string(stream_metadata.stream_index) +
          "; negative indices must have an absolute value less than the number of frames, "
          "and the number of frames must be known.");

  // Note that if we do not have the number of frames available in our
  // metadata, then we assume that the frameIndex is valid.
  std::optional<int64_t> num_frames =
      stream_metadata.get_num_frames(seek_mode_);
  if (num_frames.has_value()) {
    STABLE_CHECK_INDEX(
        frame_index < num_frames.value(),
        "Invalid frame index=" + std::to_string(frame_index) +
            " for stream_index=" +
            std::to_string(stream_metadata.stream_index) +
            "; must be less than " + std::to_string(num_frames.value()));
  }
}

// --------------------------------------------------------------------------
// MORALLY PRIVATE UTILS
// --------------------------------------------------------------------------

SingleStreamDecoder::DecodeStats SingleStreamDecoder::get_decode_stats() const {
  return decode_stats_;
}

std::ostream& operator<<(
    std::ostream& os,
    const SingleStreamDecoder::DecodeStats& stats) {
  os << "DecodeStats{"
     << "numFramesReceivedByDecoder=" << stats.num_frames_received_by_decoder
     << ", numPacketsRead=" << stats.num_packets_read
     << ", numPacketsSentToDecoder=" << stats.num_packets_sent_to_decoder
     << ", numSeeksAttempted=" << stats.num_seeks_attempted
     << ", numSeeksSkipped=" << stats.num_seeks_skipped
     << ", numFlushes=" << stats.num_flushes << "}";

  return os;
}

void SingleStreamDecoder::reset_decode_stats() {
  decode_stats_ = DecodeStats{};
}

double SingleStreamDecoder::get_pts_seconds_for_frame(int64_t frame_index) {
  validate_active_stream(AVMEDIA_TYPE_VIDEO);
  validate_scanned_all_streams("getPtsSecondsForFrame");

  const auto& stream_info = stream_infos_[active_stream_index_];
  const auto& stream_metadata =
      container_metadata_.all_stream_metadata[active_stream_index_];
  validate_frame_index(stream_metadata, frame_index);

  return pts_to_seconds(
      stream_info.all_frames[frame_index].pts, stream_info.time_base);
}

std::string SingleStreamDecoder::get_device_interface_details() const {
  STD_TORCH_CHECK(
      device_interface_ != nullptr, "Device interface doesn't exist.");
  return device_interface_->get_details();
}

} // namespace facebook::torchcodec
