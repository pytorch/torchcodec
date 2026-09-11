// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "Demuxer.h"

#include <algorithm>
#include <sstream>

#include "StableABICompat.h"

namespace facebook::torchcodec {

// FFmpeg reports "this seek cannot be performed" as a bare -1, i.e. EPERM,
// which renders as the very misleading "Operation not permitted". It covers
// both a target that the demuxer can't reach and a demuxer with no seeking
// support whatsoever.
std::string get_seek_error_message(
    const AVFormatContext* format_context,
    int64_t desired_pts,
    int status) {
  std::stringstream ss;
  ss << "Could not seek file to pts=" << desired_pts << ": "
     << get_ffmpeg_error_string_from_error_code(status) << ".";
  if (status == AVERROR(EPERM)) {
    ss << " This is either because that timestamp is out of range, or because"
       << " the '" << format_context->iformat->name << "' format does not"
       << " support seeking.";
  }
  return ss.str();
}

namespace {
template <typename IsActive>
int read_next_active_packet(
    AVFormatContext* format_context,
    AVPacket& packet,
    IsActive is_active) {
  int status = AVSUCCESS;
  do {
    // av_read_frame() requires a packet with nothing to free in it, so drop
    // what the previous iteration read from a stream we're not following. A
    // no-op on the first iteration.
    av_packet_unref(&packet);
    status = av_read_frame(format_context, &packet);
    if (status == AVERROR_EOF || status < AVSUCCESS) {
      return status;
    }
  } while (!is_active(packet.stream_index));
  return AVSUCCESS;
}
} // namespace

int read_next_packet(
    AVFormatContext* format_context,
    int active_stream_index,
    AVPacket& packet) {
  return read_next_active_packet(
      format_context, packet, [active_stream_index](int stream_index) {
        return stream_index == active_stream_index;
      });
}

int read_next_packet(
    AVFormatContext* format_context,
    const std::vector<int>& active_stream_indices,
    AVPacket& packet) {
  return read_next_active_packet(
      format_context, packet, [&active_stream_indices](int stream_index) {
        return std::find(
                   active_stream_indices.begin(),
                   active_stream_indices.end(),
                   stream_index) != active_stream_indices.end();
      });
}

namespace {
// "video" / "audio", for error messages.
const char* printable(AVMediaType media_type) {
  const char* name = av_get_media_type_string(media_type);
  return name == nullptr ? "unknown" : name;
}
} // namespace

Demuxer::Demuxer(const std::string& file_path) {
  set_ffmpeg_log_level();

  AVFormatContext* raw_context = nullptr;
  int status =
      avformat_open_input(&raw_context, file_path.c_str(), nullptr, nullptr);
  STD_TORCH_CHECK(
      status == 0,
      "Could not open input file: " + file_path + " " +
          get_ffmpeg_error_string_from_error_code(status));
  STD_TORCH_CHECK(raw_context != nullptr, "Failed to allocate AVFormatContext");
  format_context_.reset(raw_context);

  find_stream_info();
}

Demuxer::Demuxer(std::unique_ptr<AVIOContextHolder> avio_context_holder)
    : avio_context_holder_(std::move(avio_context_holder)) {
  set_ffmpeg_log_level();

  STD_TORCH_CHECK(avio_context_holder_ != nullptr, "Context holder is null");

  // FFmpeg takes a reference to the pointer in the call to open, so we can't
  // hand it a unique_ptr. That means we must free the context ourselves if the
  // open fails.
  AVFormatContext* raw_context = avformat_alloc_context();
  STD_TORCH_CHECK(raw_context != nullptr, "Failed to allocate AVFormatContext");
  raw_context->pb = avio_context_holder_->get_avio_context();

  int status = avformat_open_input(&raw_context, nullptr, nullptr, nullptr);
  if (status != 0) {
    avformat_free_context(raw_context);
    STD_TORCH_CHECK(
        false,
        "Could not open input buffer: " +
            get_ffmpeg_error_string_from_error_code(status));
  }
  format_context_.reset(raw_context);

  find_stream_info();
}

void Demuxer::find_stream_info() {
  int status = avformat_find_stream_info(format_context_.get(), nullptr);
  STD_TORCH_CHECK(
      status >= 0,
      "Failed to find stream info: ",
      get_ffmpeg_error_string_from_error_code(status));

  // We only want packets from the streams we're asked to follow, so discard
  // every stream until add_stream() says otherwise. Note av_read_frame() may
  // still return some of them under certain conditions, which is why
  // read_next_packet() also filters by stream index.
  for (unsigned int i = 0; i < format_context_->nb_streams; ++i) {
    format_context_->streams[i]->discard = AVDISCARD_ALL;
  }
}

void Demuxer::validate_requested_stream(int stream_index) {
  int num_streams = static_cast<int>(format_context_->nb_streams);
  STD_TORCH_CHECK(
      stream_index >= 0 && stream_index < num_streams,
      "The stream index ",
      stream_index,
      " is not a valid stream. The file has ",
      num_streams,
      " streams, so the index must be in [0, ",
      num_streams - 1,
      "].");

  AVMediaType stream_media_type =
      format_context_->streams[stream_index]->codecpar->codec_type;
  STD_TORCH_CHECK(
      stream_media_type == AVMEDIA_TYPE_VIDEO ||
          stream_media_type == AVMEDIA_TYPE_AUDIO,
      "The stream at index ",
      stream_index,
      " is of type '",
      printable(stream_media_type),
      "', which cannot be decoded. Only audio and video streams can.");
}

std::pair<int, AVMediaType> Demuxer::add_stream(
    std::optional<int> stream_index,
    std::optional<AVMediaType> media_type) {
  STD_TORCH_CHECK(
      stream_index.has_value() != media_type.has_value(),
      "A stream must be identified by either its index or its media type, "
      "not both and not neither.");
  STD_TORCH_CHECK(
      !has_demuxed_,
      "Streams must all be added before the first packet is demuxed: a stream "
      "added now would start at wherever the container currently is, not at "
      "the beginning.");

  int index;
  if (stream_index.has_value()) {
    validate_requested_stream(*stream_index);
    index = *stream_index;
  } else {
    index = av_find_best_stream(
        format_context_.get(),
        *media_type,
        /*wanted_stream_nb=*/-1,
        /*related_stream=*/-1,
        /*decoder_ret=*/nullptr,
        /*flags=*/0);
    STD_TORCH_CHECK(
        index >= 0,
        "No valid ",
        printable(*media_type),
        " stream found in input file.");
  }

  STD_TORCH_CHECK(
      std::find(
          active_stream_indices_.begin(),
          active_stream_indices_.end(),
          index) == active_stream_indices_.end(),
      "The stream at index ",
      index,
      " is already being demuxed.");

  format_context_->streams[index]->discard = AVDISCARD_DEFAULT;
  active_stream_indices_.push_back(index);
  return {index, format_context_->streams[index]->codecpar->codec_type};
}

torch::stable::Tensor Demuxer::get_audio_video_stream_indices() const {
  std::vector<int> indices;
  for (unsigned int i = 0; i < format_context_->nb_streams; ++i) {
    AVMediaType type = format_context_->streams[i]->codecpar->codec_type;
    if (type == AVMEDIA_TYPE_VIDEO || type == AVMEDIA_TYPE_AUDIO) {
      indices.push_back(static_cast<int>(i));
    }
  }
  STD_TORCH_CHECK(
      !indices.empty(),
      "This container has no audio or video stream at all, so there is "
      "nothing that could be demuxed.");

  auto out = torch::stable::empty(
      {static_cast<int64_t>(indices.size())}, kStableInt64);
  auto accessor = mutable_accessor<int64_t, 1>(out);
  for (size_t i = 0; i < indices.size(); ++i) {
    accessor[i] = indices[i];
  }
  return out;
}

int Demuxer::resolve_stream_index(std::optional<int> stream_index) const {
  if (stream_index.has_value()) {
    STD_TORCH_CHECK(
        std::find(
            active_stream_indices_.begin(),
            active_stream_indices_.end(),
            *stream_index) != active_stream_indices_.end(),
        "The stream at index ",
        *stream_index,
        " is not being demuxed.");
    return *stream_index;
  }
  STD_TORCH_CHECK(
      active_stream_indices_.size() == 1,
      "This demuxer follows ",
      active_stream_indices_.size(),
      " streams, so the stream index must be specified.");
  return active_stream_indices_[0];
}

AVStream* Demuxer::get_reference_stream() const {
  STD_TORCH_CHECK(
      !active_stream_indices_.empty(),
      "This demuxer isn't following any stream yet.");
  return format_context_->streams[active_stream_indices_[0]];
}

void Demuxer::seek(double seconds, std::optional<int> stream_index) {
  AVStream* reference = stream_index.has_value()
      ? format_context_->streams[resolve_stream_index(stream_index)]
      : get_reference_stream();
  int64_t desired_pts = seconds_to_closest_pts(seconds, reference->time_base);

  int status = avformat_seek_file(
      format_context_.get(),
      reference->index,
      INT64_MIN,
      desired_pts,
      desired_pts,
      0);
  STD_TORCH_CHECK(
      status >= 0,
      get_seek_error_message(format_context_.get(), desired_pts, status));
}

void Demuxer::scan_all_video_streams() {
  if (has_scanned_) {
    return;
  }
  STD_TORCH_CHECK(
      !has_demuxed_,
      "A scan reads the container from end to end and rewinds it, so it has to "
      "happen before any packet is demuxed. Scan up front, or build a second "
      "demuxer for it.");

  std::vector<int> video_stream_indices;
  for (int index : active_stream_indices_) {
    AVStream* candidate = format_context_->streams[index];
    if (candidate->codecpar->codec_type != AVMEDIA_TYPE_VIDEO) {
      continue;
    }
    video_stream_indices.push_back(index);
    auto& packets = scanned_packets_[index];
    if (candidate->nb_frames > 0) {
      packets.reserve(static_cast<size_t>(candidate->nb_frames));
    }
  }

  seek(0);

  // One pass, however many video streams are being followed: the I/O is what a
  // scan costs, and it is the same read for all of them.
  while (true) {
    ReferenceAVPacket packet(auto_packet_);
    int status =
        read_next_packet(format_context_.get(), video_stream_indices, *packet);
    if (status == AVERROR_EOF) {
      break;
    }
    STD_TORCH_CHECK(
        status >= AVSUCCESS,
        "Could not read frame from input file: ",
        get_ffmpeg_error_string_from_error_code(status));

    if (packet->flags & AV_PKT_FLAG_DISCARD) {
      continue;
    }

    scanned_packets_[packet->stream_index].push_back(
        {get_pts_or_dts(packet),
         packet->duration,
         (packet->flags & AV_PKT_FLAG_KEY) != 0});
  }

  seek(0);
  has_scanned_ = true;
}

// TODO_API_BREAKDOWN DESIGN P1 I'm starting to wonder if this should be named
// 'scan()' in the python API. We only really scan once, but this is a
// per-stream call. demuxer.scan(0) pays for the scan and demuxer.scan(1)
// doesn't, but the 'scan()' name suggests that it does. Maybe this should be
// .get_frame_index() (but 'index' is still overloaded). Ah.
FrameIndex Demuxer::scan(std::optional<int> stream_index) {
  AVStream* scanned_stream =
      format_context_->streams[resolve_stream_index(stream_index)];
  STD_TORCH_CHECK(
      scanned_stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO,
      "Only a video stream can be scanned: a frame index describes keyframes "
      "and frame positions, and the stream at index ",
      scanned_stream->index,
      " has neither.");

  scan_all_video_streams();

  // Sorting and building the tensors is per-stream work, so it is only done for
  // the streams whose index is actually asked for.
  std::vector<FrameInfo>& packets = scanned_packets_.at(scanned_stream->index);
  std::stable_sort(
      packets.begin(),
      packets.end(),
      [](const FrameInfo& a, const FrameInfo& b) { return a.pts < b.pts; });

  auto num_frames = static_cast<int64_t>(packets.size());
  FrameIndex index{
      torch::stable::empty({num_frames}, kStableInt64),
      torch::stable::empty({num_frames}, kStableInt64),
      torch::stable::empty({num_frames}, kStableBool),
      scanned_stream->time_base};

  auto pts = mutable_accessor<int64_t, 1>(index.pts);
  auto duration = mutable_accessor<int64_t, 1>(index.duration);
  auto is_key_frame = mutable_accessor<bool, 1>(index.is_key_frame);
  for (int64_t i = 0; i < num_frames; ++i) {
    pts[i] = packets[i].pts;
    duration[i] = packets[i].duration;
    is_key_frame[i] = packets[i].is_key_frame;
  }

  return index;
}

UniqueAVPacket Demuxer::next_packet() {
  STD_TORCH_CHECK(
      !active_stream_indices_.empty(),
      "This demuxer isn't following any stream yet.");
  has_demuxed_ = true;

  // Read straight into a fresh packet the caller owns, rather than into the
  // reusable one a scan reads into: it is what makes the packet safe to hand to
  // another thread.
  UniqueAVPacket packet(av_packet_alloc());
  STD_TORCH_CHECK(packet != nullptr, "Failed to allocate AVPacket");

  int status =
      read_next_packet(format_context_.get(), active_stream_indices_, *packet);
  if (status == AVERROR_EOF) {
    return UniqueAVPacket{};
  }
  STD_TORCH_CHECK(
      status >= AVSUCCESS,
      "Could not read frame from input file: ",
      get_ffmpeg_error_string_from_error_code(status));

  return packet;
}

} // namespace facebook::torchcodec
