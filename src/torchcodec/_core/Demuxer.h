// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "AVIOContextHolder.h"
#include "FFMPEGCommon.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

// Reads the next packet belonging to active_stream_index from format_context
// into `packet`. Returns AVSUCCESS with `packet` filled, AVERROR_EOF at end of
// stream, or a negative error code otherwise. Shared by Demuxer and
// SingleStreamDecoder so the demux + stream-filter loop lives in one place.
// `packet` is only written to, never owned: it can be a scope-bound
// ReferenceAVPacket read into repeatedly, or a UniqueAVPacket the caller is
// about to hand out.
int read_next_packet(
    AVFormatContext* format_context,
    int active_stream_index,
    AVPacket& packet);

// Same, for a demuxer following more than one stream: the packet comes from
// whichever of `active_stream_indices` has the next one.
int read_next_packet(
    AVFormatContext* format_context,
    const std::vector<int>& active_stream_indices,
    AVPacket& packet);

std::string get_seek_error_message(
    const AVFormatContext* format_context,
    int64_t desired_pts,
    int status);

// One demuxed packet, as a scan records it before sorting.
//
// Deliberately the same name as SingleStreamDecoder::FrameInfo, which its own
// scan produces: these describe the same thing, one entry per frame of a
// stream, and differ only in how they say it - `duration` here against
// `next_pts` there, and the frame's index implicit in the vector position
// rather than stored. The two scans should probably be unified eventually;
// until then the shared name is the reminder.
struct FrameInfo {
  int64_t pts;
  int64_t duration;
  bool is_key_frame;
};

// The frames of one stream, in presentation order: three parallel tensors of
// length N, plus the time base `pts` and `duration` are expressed in.
struct FrameIndex {
  torch::stable::Tensor pts; // int64 [N]
  torch::stable::Tensor duration; // int64 [N]
  torch::stable::Tensor is_key_frame; // bool [N]
  AVRational time_base;
};

// Demux building block: owns an AVFormatContext, follows one or more of its
// streams, and yields their (compressed) packets. Does no decoding. Not
// thread-safe.
class FORCE_PUBLIC_VISIBILITY Demuxer {
 public:
  // Both constructors open the container and probe it, without following any
  // stream: add_stream() must be called before anything can be demuxed.
  explicit Demuxer(const std::string& file_path);

  explicit Demuxer(std::unique_ptr<AVIOContextHolder> avio_context_holder);

  // Starts following a stream, and returns its index (absolute across all
  // media types) together with its media type. Identify it by exactly one of
  // `stream_index`, whose media type is then looked up and must be audio or
  // video, or `media_type`, which follows the best stream of that type.
  std::pair<int, AVMediaType> add_stream(
      std::optional<int> stream_index = std::nullopt,
      std::optional<AVMediaType> media_type = std::nullopt);

  // int64 [K]: the index of every audio and video stream in the container, in
  // container order. Anything else - subtitles, data, attachments - is left
  // out: those can be seen in the metadata but cannot be followed. Raises if
  // the container has nothing that can be demuxed at all.
  torch::stable::Tensor get_audio_video_stream_indices() const;

  // Returns the next packet of whichever followed stream has one, as a
  // freshly-allocated packet, or a null packet at end of stream. Which stream
  // it belongs to is on the packet itself, as `stream_index`.
  UniqueAVPacket next_packet();

  // Moves the *container*: every followed stream jumps, so every decoder fed by
  // this demuxer must be reset. The target is resolved against `stream_index`,
  // or against get_reference_stream() when it is left unspecified.
  void seek(double seconds, std::optional<int> stream_index = std::nullopt);

  // Demuxes one stream entirely, without decoding, and returns one entry per
  // frame sorted by pts. Leaves the demuxer back at the start of the container.
  //
  // The demux pass this needs covers every followed video stream at once and
  // happens only once per demuxer, so indexing a second stream costs no I/O.
  // Only legal before the first packet is demuxed, which is what makes the
  // rewind harmless: nothing has been fed to a decoder yet.
  FrameIndex scan(std::optional<int> stream_index = std::nullopt);

  // The index passed in, validated, or the only followed stream when it is left
  // unspecified.
  int resolve_stream_index(std::optional<int> stream_index) const;

  const std::vector<int>& active_stream_indices() const {
    return active_stream_indices_;
  }

  const UniqueDecodingAVFormatContext& format_context() const {
    return format_context_;
  }

 private:
  void find_stream_info();
  // Checks `stream_index` is in range and that it is an audio or video stream.
  void validate_requested_stream(int stream_index);
  // The stream a seek is resolved against when the caller doesn't name one:
  // simply the first one that was added. A seek is resolved in a single
  // stream's time base and lands on that stream's keyframes, so which one it is
  // does matter - but picking by media type would make the default depend on
  // what the container happens to hold, where first-added is the caller's own
  // ordering. Pass a stream to seek() to override it.
  AVStream* get_reference_stream() const;
  // Phase one of a scan: read the container once, bucketing the packets of
  // every followed video stream. Idempotent.
  void scan_all_video_streams();

  // Declared before format_context_ so that it outlives it: the format context
  // reads through the AVIOContext this holds.
  std::unique_ptr<AVIOContextHolder> avio_context_holder_;
  UniqueDecodingAVFormatContext format_context_;
  // In the order they were added. A handful of entries at most, so this is
  // scanned linearly.
  std::vector<int> active_stream_indices_;
  bool has_demuxed_ = false;
  // Filled by scan_all_video_streams(), keyed by stream index. Sorted lazily,
  // per stream, by scan().
  std::map<int, std::vector<FrameInfo>> scanned_packets_;
  bool has_scanned_ = false;
  AutoAVPacket auto_packet_;
};

} // namespace facebook::torchcodec
