// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "Demuxer.h"
#include "DeviceInterface.h"
#include "FFMPEGCommon.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

// Creates, configures and opens a codec context for `stream` using `av_codec`,
// registering the hardware device (if any) via `device_interface`. Shared by
// SingleStreamDecoder and PacketDecoder to avoid duplicating codec setup.
SharedAVCodecContext create_and_open_codec_context(
    AVStream* stream,
    const AVCodec* av_codec,
    DeviceInterface* device_interface,
    std::optional<int> thread_count);

// Decode building block: turns compressed packets into decoded frames - (YUV)
// pictures for a video stream, samples in the codec's own format for an audio
// one. Configured from one of a Demuxer's streams; stateful. Not thread-safe.
class FORCE_PUBLIC_VISIBILITY PacketDecoder {
 public:
  explicit PacketDecoder(
      const Demuxer& demuxer,
      std::optional<int> stream_index = std::nullopt,
      const StableDevice& device = StableDevice(kStableCPU),
      std::optional<int> ffmpeg_thread_count = std::nullopt);

  // Feed one packet to the decoder. Borrows `packet` (does not take ownership).
  int send_packet(const AVPacket& packet);
  // Signal end-of-stream so the decoder flushes its remaining frames.
  int send_eof();
  // Pull one frame. Returns AVSUCCESS with `av_frame` filled, AVERROR(EAGAIN)
  // if more input is needed, AVERROR_EOF at end, or a negative error code.
  int receive_frame(UniqueAVFrame& av_frame);
  // Drop the codec's buffered state (reference frames, in-flight frames) and
  // start over. Needed after the demuxer seeked, and after send_eof(), which
  // otherwise leaves the codec permanently in its drained state.
  // This is called 'reset()' and not 'flush()', because this is publicly
  // exposed flush() is slightly ambiguous and could mean 'flush the frames out
  // of the decoder' rather than meaning 'flush the decoder internal state'.
  void reset();

  std::optional<torch::stable::Tensor> get_frame_storage(
      const AVFrame& av_frame) const {
    return device_interface_->get_frame_storage(av_frame);
  }

  const StableDevice& device() const {
    return device_interface_->device();
  }

  // The stream time base, used to convert frame pts/duration to seconds.
  AVRational time_base() const {
    return time_base_;
  }

  AVMediaType media_type() const {
    return media_type_;
  }

 private:
  std::unique_ptr<DeviceInterface> device_interface_;
  SharedAVCodecContext codec_context_;
  AVRational time_base_ = {};
  AVMediaType media_type_ = AVMEDIA_TYPE_VIDEO;
  // Stamped onto every frame we hand out, so downstream blocks can read the
  // rotation off the frame itself instead of knowing about the stream. Held by
  // value: we're only handed the Demuxer at construction and it may well be
  // gone by the time we decode.
  std::optional<std::array<int32_t, 9>> display_matrix_;
  // The MPEG-PS demuxer doesn't return proper packets just after a seek, so the
  // first ones we're fed may not be decodable. See send_packet().
  bool is_mpeg_ps_ = false;
  bool packet_data_may_be_misaligned_ = false;
};

// How a decoded frame's samples are laid out and how they should be
// interpreted, before any color conversion.
struct FrameMetadata {
  std::string pix_fmt;
  std::string colorspace;
  std::string color_range;
  int64_t bit_depth = 8;
  // The dimensions of the samples as they were decoded, i.e. before rotation.
  int64_t width = 0;
  int64_t height = 0;
  // Degrees counter-clockwise needed to make the frame upright. 0 when the
  // frame carries no display matrix.
  double rotation_degrees = 0;
};

// TODO_API_BREAKDOWN CC P1 these should bet get_*

// Describes `av_frame` without touching its samples. Unlike frame_planes(),
// this works for every pixel format, so callers can ask what a frame is before
// asking for views they may not be able to get.
FORCE_PUBLIC_VISIBILITY FrameMetadata frame_metadata(const AVFrame& av_frame);

// A decoded frame's own samples, before any color conversion: one view per
// component, in the frame's native order: (Y, U, V) for YUV, (R, G, B) for RGB
// codecs, (Y,) for grayscale, plus a trailing alpha view when the format has
// one.
FORCE_PUBLIC_VISIBILITY std::vector<torch::stable::Tensor> frame_planes(
    const AVFrame& av_frame,
    const StableDevice& device,
    const torch::stable::Tensor& tensor_handle);

// A decoded audio frame's samples as a contiguous [num_channels, num_samples]
// tensor whose dtype is the frame's own sample type: uint8 for u8, int16 for
// s16, float32 for flt, and so on, planar or not. This is a copy rather than a
// view: planar formats put each channel in its own allocation and packed ones
// interleave them, so neither is a [C, N] tensor as it stands. An audio frame
// is a few kB, so normalizing here buys a uniform layout for the price of a
// memcpy - and it means a converter can treat the result as planar-of-dtype.
// TODO_API_BREAKDOWN DESIGN P1: do we want to copy? Should we just keep the
// original layout?
FORCE_PUBLIC_VISIBILITY torch::stable::Tensor audio_samples(
    const AVFrame& av_frame);

} // namespace facebook::torchcodec
