// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/SingleStreamDecoder.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include "src/torchcodec/_core/DeviceInterface.h"
#include "torch/types.h"

extern "C" {
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/imgutils.h>
#include <libavutil/log.h>
}

namespace facebook::torchcodec {
namespace {

double pts_to_seconds(int64_t pts, int den) {
  return static_cast<double>(pts) / den;
}

double pts_to_seconds(int64_t pts, const AVRational& time_base) {
  return pts_to_seconds(pts, time_base.den);
}

int64_t seconds_to_closest_pts(double seconds, const AVRational& time_base) {
  return static_cast<int64_t>(std::round(seconds * time_base.den));
}

} // namespace

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
  TORCH_CHECK(
      status == 0,
      "Could not open input file: " + video_file_path + " " +
          get_ffmpeg_error_string_from_error_code(status));
  TORCH_CHECK(rawContext != nullptr);
  format_context_.reset(raw_context);

  initialize_decoder();
}

SingleStreamDecoder::SingleStreamDecoder(
    std::unique_ptr<AVIOContext_holder> context,
    SeekMode seek_mode)
    : seek_mode_(seek_mode), avio_context_holder_(std::move(context)) {
  set_ffmpeg_log_level();

  TORCH_CHECK(avioContextHolder_, "Context holder cannot be null");

  // Because FFmpeg requires a reference to a pointer in the call to open, we
  // can't use a unique pointer here. Note that means we must call free if open
  // fails.
  AVFormatContext* raw_context = avformat_alloc_context();
  TORCH_CHECK(rawContext != nullptr, "Unable to alloc avformat context");

  raw_context->pb = avio_context_holder_->get_avio_context();
  int status = avformat_open_input(&raw_context, nullptr, nullptr, nullptr);
  if (status != 0) {
    avformat_free_context(raw_context);
    TORCH_CHECK(
        false,
        "Failed to open input buffer: " +
            get_ffmpeg_error_string_from_error_code(status));
  }

  format_context_.reset(raw_context);

  initialize_decoder();
}

SingleStreamDecoder::~SingleStreamDecoder() {
  for (auto& [streamIndex, stream_info] : stream_infos_) {
    auto& device = stream_info.video_stream_options.device;
    if (device.type() == torch::kCPU) {
    } else if (device.type() == torch::kCUDA) {
      release_context_on_cuda(device, stream_info.codec_context.get());
    } else {
      TORCH_CHECK(false, "Invalid device type: " + device.str());
    }
  }
}

void SingleStreamDecoder::initializeDecoder() {
  TORCH_CHECK(!initialized_, "Attempted double initialization.");

  // In principle, the AVFormatContext should be filled in by the call to
  // avformat_open_input() which reads the header. However, some formats do not
  // store enough info in the header, so we call avformat_find_stream_info()
  // which decodes a few frames to get missing info. For more, see:
  // https://ffmpeg.org/doxygen/7.0/group__lavf__decoding.html
  int status = avformat_find_stream_info(format_context_.get(), nullptr);
  if (status < 0) {
    throw std::runtime_error(
        "Failed to find stream info: " +
        get_ffmpeg_error_string_from_error_code(status));
  }

  for (unsigned int i = 0; i < format_context_->nb_streams; i++) {
    AVStream* av_stream = format_context_->streams[i];
    StreamMetadata stream_metadata;

    TORCH_CHECK(
        static_cast<int>(i) == av_stream->index,
        "Our stream index, " + std::to_string(i) +
            ", does not match AVStream's index, " +
            std::to_string(av_stream->index) + ".");
    stream_metadata.stream_index = i;
    stream_metadata.media_type = av_stream->codecpar->codec_type;
    stream_metadata.codec_name =
        avcodec_get_name(av_stream->codecpar->codec_id);
    stream_metadata.bit_rate = av_stream->codecpar->bit_rate;

    int64_t frame_count = av_stream->nb_frames;
    if (frameCount > 0) {
      stream_metadata.num_frames = frame_count;
    }

    if (avStream->duration > 0 && av_stream->time_base.den > 0) {
      stream_metadata.duration_seconds =
          av_q2d(av_stream->time_base) * av_stream->duration;
    }
    if (avStream->start_time != AV_NOPTS_VALUE) {
      stream_metadata.begin_stream_from_header =
          av_q2d(av_stream->time_base) * av_stream->start_time;
    }

    if (avStream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      double fps = av_q2d(av_stream->r_frame_rate);
      if (fps > 0) {
        stream_metadata.average_fps = fps;
      }
      container_metadata_.num_video_streams++;
    } else if (avStream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
      AVSampleFormat format =
          static_cast<_avsample_format>(av_stream->codecpar->format);

      // If the AVSampleFormat is not recognized, we get back nullptr. We have
      // to make sure we don't initialize a std::string with nullptr. There's
      // nothing to do on the else branch because we're already using an
      // optional; it'll just remain empty.
      const char* raw_sample_format = av_get_sample_fmt_name(format);
      if (rawSampleFormat != nullptr) {
        stream_metadata.sample_format = std::string(raw_sample_format);
      }
      container_metadata_.num_audio_streams++;
    }

    container_metadata_.all_stream_metadata.push_back(stream_metadata);
  }

  if (formatContext_->duration > 0) {
    container_metadata_.duration_seconds =
        pts_to_seconds(format_context_->duration, AV_TIME_BASE);
  }

  if (formatContext_->bit_rate > 0) {
    container_metadata_.bit_rate = format_context_->bit_rate;
  }

  int best_video_stream =
      get_best_stream_index(_avm_e_d_i_a__t_y_p_e__v_i_d_e_o);
  if (bestVideoStream >= 0) {
    container_metadata_.best_video_stream_index = best_video_stream;
  }

  int best_audio_stream =
      get_best_stream_index(_avm_e_d_i_a__t_y_p_e__a_u_d_i_o);
  if (bestAudioStream >= 0) {
    container_metadata_.best_audio_stream_index = best_audio_stream;
  }

  if (seekMode_ == SeekMode::exact) {
    scan_file_and_update_metadata_and_index();
  }

  initialized_ = true;
}

void SingleStreamDecoder::setFFmpegLogLevel() {
  auto log_level = AV_LOG_QUIET;
  const char* log_level_env = std::getenv("TORCHCODEC_FFMPEG_LOG_LEVEL");
  if (logLevelEnv != nullptr) {
    if (std::strcmp(logLevelEnv, "QUIET") == 0) {
      log_level = AV_LOG_QUIET;
    } else if (std::strcmp(logLevelEnv, "PANIC") == 0) {
      log_level = AV_LOG_PANIC;
    } else if (std::strcmp(logLevelEnv, "FATAL") == 0) {
      log_level = AV_LOG_FATAL;
    } else if (std::strcmp(logLevelEnv, "ERROR") == 0) {
      log_level = AV_LOG_ERROR;
    } else if (std::strcmp(logLevelEnv, "WARNING") == 0) {
      log_level = AV_LOG_WARNING;
    } else if (std::strcmp(logLevelEnv, "INFO") == 0) {
      log_level = AV_LOG_INFO;
    } else if (std::strcmp(logLevelEnv, "VERBOSE") == 0) {
      log_level = AV_LOG_VERBOSE;
    } else if (std::strcmp(logLevelEnv, "DEBUG") == 0) {
      log_level = AV_LOG_DEBUG;
    } else if (std::strcmp(logLevelEnv, "TRACE") == 0) {
      log_level = AV_LOG_TRACE;
    } else {
      TORCH_CHECK(
          false,
          "Invalid TORCHCODEC_FFMPEG_LOG_LEVEL: ",
          log_level_env,
          ". Use e.g. 'QUIET', 'PANIC', 'VERBOSE', etc.");
    }
  }
  av_log_set_level(log_level);
}

int SingleStreamDecoder::getBestStreamIndex(AVMediaType media_type) {
  AVCodecOnlyUseForCallingAVFindBestStream av_codec = nullptr;
  int stream_index = av_find_best_stream(
      format_context_.get(), media_type, -1, -1, &avCodec, 0);
  return stream_index;
}

// --------------------------------------------------------------------------
// VIDEO METADATA QUERY API
// --------------------------------------------------------------------------

void SingleStreamDecoder::scanFileAndUpdateMetadataAndIndex() {
  if (scannedAllStreams_) {
    return;
  }

  for (unsigned int i = 0; i < format_context_->nb_streams; ++i) {
    // We want to scan and update the metadata of all streams.
    TORCH_CHECK(
        format_context_->streams[i]->discard != AVDISCARD_ALL,
        "Did you add a stream before you called for a scan?");
  }

  AutoAVPacket auto_avpacket;
  while (true) {
    ReferenceAVPacket packet(auto_avpacket);

    // av_read_frame is a misleading name: it gets the next **packet**.
    int status = av_read_frame(format_context_.get(), packet.get());

    if (status == AVERROR_EOF) {
      break;
    }

    if (status != AVSUCCESS) {
      throw std::runtime_error(
          "Failed to read frame from input file: " +
          get_ffmpeg_error_string_from_error_code(status));
    }

    if (packet->flags & AV_PKT_FLAG_DISCARD) {
      continue;
    }

    // We got a valid packet. Let's figure out what stream it belongs to and
    // record its relevant metadata.
    int stream_index = packet->stream_index;
    auto& stream_metadata =
        container_metadata_.all_stream_metadata[stream_index];
    stream_metadata.min_pts_from_scan = std::min(
        stream_metadata.min_pts_from_scan.value_or(_i_n_t64__m_a_x),
        packet->pts);
    stream_metadata.max_pts_from_scan = std::max(
        stream_metadata.max_pts_from_scan.value_or(_i_n_t64__m_i_n),
        packet->pts + packet->duration);
    stream_metadata.num_frames_from_scan =
        stream_metadata.num_frames_from_scan.value_or(0) + 1;

    // Note that we set the other value in this struct, nextPts, only after
    // we have scanned all packets and sorted by pts.
    FrameInfo frame_info = {packet->pts};
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
       ++streamIndex) {
    auto& stream_metadata =
        container_metadata_.all_stream_metadata[stream_index];
    auto av_stream = format_context_->streams[stream_index];

    stream_metadata.num_frames_from_scan =
        stream_infos_[stream_index].all_frames.size();

    if (streamMetadata.minPtsFromScan.has_value()) {
      stream_metadata.min_pts_seconds_from_scan =
          *streamMetadata.minPtsFromScan * av_q2d(av_stream->time_base);
    }
    if (streamMetadata.maxPtsFromScan.has_value()) {
      stream_metadata.max_pts_seconds_from_scan =
          *streamMetadata.maxPtsFromScan * av_q2d(av_stream->time_base);
    }
  }

  // Reset the seek-cursor back to the beginning.
  int status = avformat_seek_file(format_context_.get(), 0, INT64_MIN, 0, 0, 0);
  if (status < 0) {
    throw std::runtime_error(
        "Could not seek file to pts=0: " +
        get_ffmpeg_error_string_from_error_code(status));
  }

  // Sort all frames by their pts.
  for (auto& [streamIndex, stream_info] : stream_infos_) {
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
      if (streamInfo.allFrames[i].isKeyFrame) {
        TORCH_CHECK(
            key_frame_index < stream_info.key_frames.size(),
            "The all_frames vec claims it has MORE key_frames than the key_frames vec. There's a bug in torchcodec.");
        stream_info.key_frames[key_frame_index].frame_index = i;
        ++keyFrameIndex;
      }
      if (i + 1 < stream_info.all_frames.size()) {
        stream_info.all_frames[i].next_pts = stream_info.all_frames[i + 1].pts;
      }
    }
    TORCH_CHECK(
        key_frame_index == stream_info.key_frames.size(),
        "The all_frames vec claims it has LESS key_frames than the key_frames vec. There's a bug in torchcodec.");
  }

  scanned_all_streams_ = true;
}

SingleStreamDecoder::ContainerMetadata
SingleStreamDecoder::getContainerMetadata() const {
  return container_metadata_;
}

torch::Tensor SingleStreamDecoder::getKeyFrameIndices() {
  validate_active_stream(_avm_e_d_i_a__t_y_p_e__v_i_d_e_o);
  validate_scanned_all_streams("get_key_frame_indices");

  const std::vector<_frame_info>& key_frames =
      stream_infos_[active_stream_index_].key_frames;
  torch::Tensor key_frame_indices =
      torch::empty({static_cast<int64_t>(keyFrames.size())}, {torch::kInt64});
  for (size_t i = 0; i < key_frames.size(); ++i) {
    key_frame_indices[i] = key_frames[i].frame_index;
  }

  return key_frame_indices;
}

// --------------------------------------------------------------------------
// ADDING STREAMS API
// --------------------------------------------------------------------------

void SingleStreamDecoder::addStream(
    int stream_index,
    AVMediaType media_type,
    const torch::Device& device,
    std::optional<int> ffmpeg_thread_count) {
  TORCH_CHECK(
      active_stream_index_ == NO_ACTIVE_STREAM,
      "Can only add one single stream.");
  TORCH_CHECK(
      media_type == AVMEDIA_TYPE_VIDEO || media_type == AVMEDIA_TYPE_AUDIO,
      "Can only add video or audio streams.");
  TORCH_CHECK(formatContext_.get() != nullptr);

  AVCodecOnlyUseForCallingAVFindBestStream av_codec = nullptr;

  active_stream_index_ = av_find_best_stream(
      format_context_.get(), media_type, stream_index, -1, &avCodec, 0);

  if (activeStreamIndex_ < 0) {
    throw std::invalid_argument(
        "No valid stream found in input file. Is " +
        std::to_string(stream_index) + " of the desired media type?");
  }

  TORCH_CHECK(avCodec != nullptr);

  StreamInfo& stream_info = stream_infos_[active_stream_index_];
  stream_info.stream_index = active_stream_index_;
  stream_info.time_base =
      format_context_->streams[active_stream_index_]->time_base;
  stream_info.stream = format_context_->streams[active_stream_index_];
  stream_info.av_media_type = media_type;

  // This should never happen, checking just to be safe.
  TORCH_CHECK(
      stream_info.stream->codecpar->codec_type == media_type,
      "FFmpeg found stream with index ",
      active_stream_index_,
      " which is of the wrong media type.");

  // TODO_CODE_QUALITY it's pretty meh to have a video-specific logic within
  // addStream() which is supposed to be generic
  if (mediaType == AVMEDIA_TYPE_VIDEO && device.type() == torch::kCUDA) {
    av_codec = make_avcodec_only_use_for_calling_avfind_best_stream(
        find_cuda_codec(device, stream_info.stream->codecpar->codec_id)
            .value_or(avCodec));
  }

  AVCodecContext* codec_context = avcodec_alloc_context3(av_codec);
  TORCH_CHECK(codecContext != nullptr);
  stream_info.codec_context.reset(codec_context);

  int ret_val = avcodec_parameters_to_context(
      stream_info.codec_context.get(), stream_info.stream->codecpar);
  TORCH_CHECK_EQ(retVal, AVSUCCESS);

  stream_info.codec_context->thread_count = ffmpeg_thread_count.value_or(0);
  stream_info.codec_context->pkt_timebase = stream_info.stream->time_base;

  // TODO_CODE_QUALITY same as above.
  if (mediaType == AVMEDIA_TYPE_VIDEO && device.type() == torch::kCUDA) {
    initialize_context_on_cuda(device, codec_context);
  }

  ret_val = avcodec_open2(stream_info.codec_context.get(), av_codec, nullptr);
  if (retVal < AVSUCCESS) {
    throw std::invalid_argument(
        get_ffmpeg_error_string_from_error_code(ret_val));
  }

  codec_context->time_base = stream_info.stream->time_base;
  container_metadata_.all_stream_metadata[active_stream_index_].codec_name =
      std::string(avcodec_get_name(codec_context->codec_id));

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

void SingleStreamDecoder::addVideoStream(
    int stream_index,
    const VideoStreamOptions& video_stream_options) {
  TORCH_CHECK(
      video_stream_options.device.type() == torch::kCPU ||
          video_stream_options.device.type() == torch::kCUDA,
      "Invalid device type: " + video_stream_options.device.str());

  add_stream(
      stream_index,
      AVMEDIA_TYPE_VIDEO,
      video_stream_options.device,
      video_stream_options.ffmpeg_thread_count);

  auto& stream_metadata =
      container_metadata_.all_stream_metadata[active_stream_index_];

  if (seekMode_ == SeekMode::approximate &&
      !streamMetadata.averageFps.has_value()) {
    throw std::runtime_error(
        "Seek mode is approximate, but stream " +
        std::to_string(active_stream_index_) +
        " does not have an average fps in its metadata.");
  }

  auto& stream_info = stream_infos_[active_stream_index_];
  stream_info.video_stream_options = video_stream_options;

  stream_metadata.width = stream_info.codec_context->width;
  stream_metadata.height = stream_info.codec_context->height;

  // By default, we want to use swscale for color conversion because it is
  // faster. However, it has width requirements, so we may need to fall back
  // to filtergraph. We also need to respect what was requested from the
  // options; we respect the options unconditionally, so it's possible for
  // swscale's width requirements to be violated. We don't expose the ability to
  // choose color conversion library publicly; we only use this ability
  // internally.
  int width =
      video_stream_options.width.value_or(stream_info.codec_context->width);

  // swscale requires widths to be multiples of 32:
  // https://stackoverflow.com/questions/74351955/turn-off-sw-scale-conversion-to-planar-yuv-32-byte-alignment-requirements
  // so we fall back to filtergraph if the width is not a multiple of 32.
  auto default_library = (width % 32 == 0)
      ? SingleStreamDecoder::ColorConversionLibrary::SWSCALE
      : SingleStreamDecoder::ColorConversionLibrary::FILTERGRAPH;

  stream_info.color_conversion_library =
      video_stream_options.color_conversion_library.value_or(default_library);
}

void SingleStreamDecoder::addAudioStream(
    int stream_index,
    const AudioStreamOptions& audio_stream_options) {
  TORCH_CHECK(
      seek_mode_ == SeekMode::approximate,
      "seek_mode must be 'approximate' for audio streams.");

  add_stream(stream_index, AVMEDIA_TYPE_AUDIO);

  auto& stream_info = stream_infos_[active_stream_index_];
  stream_info.audio_stream_options = audio_stream_options;

  auto& stream_metadata =
      container_metadata_.all_stream_metadata[active_stream_index_];
  stream_metadata.sample_rate =
      static_cast<int64_t>(stream_info.codec_context->sample_rate);
  stream_metadata.num_channels =
      static_cast<int64_t>(get_num_channels(stream_info.codec_context));

  // FFmpeg docs say that the decoder will try to decode natively in this
  // format, if it can. Docs don't say what the decoder does when it doesn't
  // support that format, but it looks like it does nothing, so this probably
  // doesn't hurt.
  stream_info.codec_context->request_sample_fmt = AV_SAMPLE_FMT_FLTP;
}

// --------------------------------------------------------------------------
// HIGH-LEVEL DECODING ENTRY-POINTS
// --------------------------------------------------------------------------

SingleStreamDecoder::FrameOutput SingleStreamDecoder::getNextFrame() {
  auto output = get_next_frame_internal();
  if (streamInfos_[activeStreamIndex_].avMediaType == AVMEDIA_TYPE_VIDEO) {
    output.data = maybe_permute_h_w_c2_c_h_w(output.data);
  }
  return output;
}

SingleStreamDecoder::FrameOutput SingleStreamDecoder::getNextFrameInternal(
    std::optional<torch::Tensor> pre_allocated_output_tensor) {
  validate_active_stream();
  UniqueAVFrame avframe = decode_avframe(
      [this](const UniqueAVFrame& avframe) { return avframe->pts >= cursor_; });
  return convert_avframe_to_frame_output(avframe, pre_allocated_output_tensor);
}

SingleStreamDecoder::FrameOutput SingleStreamDecoder::getFrameAtIndex(
    int64_t frame_index) {
  auto frame_output = get_frame_at_index_internal(frame_index);
  frame_output.data = maybe_permute_h_w_c2_c_h_w(frame_output.data);
  return frame_output;
}

SingleStreamDecoder::FrameOutput SingleStreamDecoder::getFrameAtIndexInternal(
    int64_t frame_index,
    std::optional<torch::Tensor> pre_allocated_output_tensor) {
  validate_active_stream(_avm_e_d_i_a__t_y_p_e__v_i_d_e_o);

  const auto& stream_info = stream_infos_[active_stream_index_];
  const auto& stream_metadata =
      container_metadata_.all_stream_metadata[active_stream_index_];
  validate_frame_index(stream_metadata, frame_index);

  int64_t pts = get_pts(frame_index);
  set_cursor_pts_in_seconds(pts_to_seconds(pts, stream_info.time_base));
  return get_next_frame_internal(pre_allocated_output_tensor);
}

SingleStreamDecoder::FrameBatchOutput SingleStreamDecoder::getFramesAtIndices(
    const std::vector<int64_t>& frame_indices) {
  validate_active_stream(_avm_e_d_i_a__t_y_p_e__v_i_d_e_o);

  auto indices_are_sorted =
      std::is_sorted(frame_indices.begin(), frame_indices.end());

  std::vector<size_t> argsort;
  if (!indicesAreSorted) {
    // if frameIndices is [13, 10, 12, 11]
    // when sorted, it's [10, 11, 12, 13] <-- this is the sorted order we want
    // to use to decode the frames
    // and argsort is [ 1, 3, 2, 0]
    argsort.resize(frame_indices.size());
    for (size_t i = 0; i < argsort.size(); ++i) {
      argsort[i] = i;
    }
    std::sort(
        argsort.begin(), argsort.end(), [&frameIndices](size_t a, size_t b) {
          return frame_indices[a] < frame_indices[b];
        });
  }

  const auto& stream_metadata =
      container_metadata_.all_stream_metadata[active_stream_index_];
  const auto& stream_info = stream_infos_[active_stream_index_];
  const auto& video_stream_options = stream_info.video_stream_options;
  FrameBatchOutput frame_batch_output(
      frame_indices.size(), video_stream_options, stream_metadata);

  auto previous_index_in_video = -1;
  for (size_t f = 0; f < frame_indices.size(); ++f) {
    auto index_in_output = indices_are_sorted ? f : argsort[f];
    auto index_in_video = frame_indices[index_in_output];

    validate_frame_index(stream_metadata, index_in_video);

    if ((f > 0) && (indexInVideo == previous_index_in_video)) {
      // Avoid decoding the same frame twice
      auto previous_index_in_output =
          indices_are_sorted ? f - 1 : argsort[f - 1];
      frame_batch_output.data[index_in_output].copy_(
          frame_batch_output.data[previous_index_in_output]);
      frame_batch_output.pts_seconds[index_in_output] =
          frame_batch_output.pts_seconds[previous_index_in_output];
      frame_batch_output.duration_seconds[index_in_output] =
          frame_batch_output.duration_seconds[previous_index_in_output];
    } else {
      FrameOutput frame_output = get_frame_at_index_internal(
          index_in_video, frame_batch_output.data[index_in_output]);
      frame_batch_output.pts_seconds[index_in_output] =
          frame_output.pts_seconds;
      frame_batch_output.duration_seconds[index_in_output] =
          frame_output.duration_seconds;
    }
    previous_index_in_video = index_in_video;
  }
  frame_batch_output.data = maybe_permute_h_w_c2_c_h_w(frame_batch_output.data);
  return frame_batch_output;
}

SingleStreamDecoder::FrameBatchOutput SingleStreamDecoder::getFramesInRange(
    int64_t start,
    int64_t stop,
    int64_t step) {
  validate_active_stream(_avm_e_d_i_a__t_y_p_e__v_i_d_e_o);

  const auto& stream_metadata =
      container_metadata_.all_stream_metadata[active_stream_index_];
  const auto& stream_info = stream_infos_[active_stream_index_];
  int64_t num_frames = get_num_frames(stream_metadata);
  TORCH_CHECK(
      start >= 0, "Range start, " + std::to_string(start) + " is less than 0.");
  TORCH_CHECK(
      stop <= num_frames,
      "Range stop, " + std::to_string(stop) +
          ", is more than the number of frames, " + std::to_string(num_frames));
  TORCH_CHECK(
      step > 0, "Step must be greater than 0; is " + std::to_string(step));

  int64_t num_output_frames = std::ceil((stop - start) / double(step));
  const auto& video_stream_options = stream_info.video_stream_options;
  FrameBatchOutput frame_batch_output(
      num_output_frames, video_stream_options, stream_metadata);

  for (int64_t i = start, f = 0; i < stop; i += step, ++f) {
    FrameOutput frame_output =
        get_frame_at_index_internal(i, frame_batch_output.data[f]);
    frame_batch_output.pts_seconds[f] = frame_output.pts_seconds;
    frame_batch_output.duration_seconds[f] = frame_output.duration_seconds;
  }
  frame_batch_output.data = maybe_permute_h_w_c2_c_h_w(frame_batch_output.data);
  return frame_batch_output;
}

SingleStreamDecoder::FrameOutput SingleStreamDecoder::getFramePlayedAt(
    double seconds) {
  validate_active_stream(_avm_e_d_i_a__t_y_p_e__v_i_d_e_o);
  StreamInfo& stream_info = stream_infos_[active_stream_index_];
  double frame_start_time = pts_to_seconds(
      stream_info.last_decoded_avframe_pts, stream_info.time_base);
  double frame_end_time = pts_to_seconds(
      stream_info.last_decoded_avframe_pts +
          stream_info.last_decoded_avframe_duration,
      stream_info.time_base);
  if (seconds >= frame_start_time && seconds < frame_end_time) {
    // We are in the same frame as the one we just returned. However, since we
    // don't cache it locally, we have to rewind back.
    seconds = frame_start_time;
  }

  set_cursor_pts_in_seconds(seconds);
  UniqueAVFrame avframe =
      decode_avframe([seconds, this](const UniqueAVFrame& avframe) {
        StreamInfo& stream_info = stream_infos_[active_stream_index_];
        double frame_start_time =
            pts_to_seconds(avframe->pts, stream_info.time_base);
        double frame_end_time = pts_to_seconds(
            avframe->pts + get_duration(avframe), stream_info.time_base);
        if (frameStartTime > seconds) {
          // FFMPEG seeked past the frame we are looking for even though we
          // set max_ts to be our needed timestamp in avformat_seek_file()
          // in maybeSeekToBeforeDesiredPts().
          // This could be a bug in FFMPEG: https://trac.ffmpeg.org/ticket/11137
          // In this case we return the very next frame instead of throwing an
          // exception.
          // TODO: Maybe log to stderr for Debug builds?
          return true;
        }
        return seconds >= frame_start_time && seconds < frame_end_time;
      });

  // Convert the frame to tensor.
  FrameOutput frame_output = convert_avframe_to_frame_output(avframe);
  frame_output.data = maybe_permute_h_w_c2_c_h_w(frame_output.data);
  return frame_output;
}

SingleStreamDecoder::FrameBatchOutput SingleStreamDecoder::getFramesPlayedAt(
    const std::vector<double>& timestamps) {
  validate_active_stream(_avm_e_d_i_a__t_y_p_e__v_i_d_e_o);

  const auto& stream_metadata =
      container_metadata_.all_stream_metadata[active_stream_index_];

  double min_seconds = get_min_seconds(stream_metadata);
  double max_seconds = get_max_seconds(stream_metadata);

  // The frame played at timestamp t and the one played at timestamp `t +
  // eps` are probably the same frame, with the same index. The easiest way to
  // avoid decoding that unique frame twice is to convert the input timestamps
  // to indices, and leverage the de-duplication logic of getFramesAtIndices.

  std::vector<int64_t> frame_indices(timestamps.size());
  for (size_t i = 0; i < timestamps.size(); ++i) {
    auto frame_seconds = timestamps[i];
    TORCH_CHECK(
        frame_seconds >= min_seconds && frame_seconds < max_seconds,
        "frame pts is " + std::to_string(frame_seconds) +
            "; must be in range [" + std::to_string(min_seconds) + ", " +
            std::to_string(max_seconds) + ").");

    frame_indices[i] = seconds_to_index_lower_bound(frame_seconds);
  }

  return get_frames_at_indices(frame_indices);
}

SingleStreamDecoder::FrameBatchOutput
SingleStreamDecoder::getFramesPlayedInRange(
    double start_seconds,
    double stop_seconds) {
  validate_active_stream(_avm_e_d_i_a__t_y_p_e__v_i_d_e_o);
  const auto& stream_metadata =
      container_metadata_.all_stream_metadata[active_stream_index_];
  TORCH_CHECK(
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
  // frame=0, pts=0.0
  // frame=1, pts=0.3
  //
  // interval A: [0.2, 0.2)
  // interval B: [0.2, 0.15)
  //
  // Both intervals take place between the pts values for frame 0 and frame 1,
  // which by our abstract player, means that both intervals map to frame 0. By
  // the definition of a half open interval, interval A should return no frames.
  // Interval B should return frame 0. However, for both A and B, the individual
  // values of the intervals will map to the same frame indices below. Hence, we
  // need this special case below.
  if (startSeconds == stop_seconds) {
    FrameBatchOutput frame_batch_output(
        0, video_stream_options, stream_metadata);
    frame_batch_output.data =
        maybe_permute_h_w_c2_c_h_w(frame_batch_output.data);
    return frame_batch_output;
  }

  double min_seconds = get_min_seconds(stream_metadata);
  double max_seconds = get_max_seconds(stream_metadata);
  TORCH_CHECK(
      start_seconds >= min_seconds && start_seconds < max_seconds,
      "Start seconds is " + std::to_string(start_seconds) +
          "; must be in range [" + std::to_string(min_seconds) + ", " +
          std::to_string(max_seconds) + ").");
  TORCH_CHECK(
      stop_seconds <= max_seconds,
      "Stop seconds (" + std::to_string(stop_seconds) +
          "; must be less than or equal to " + std::to_string(max_seconds) +
          ").");

  // Note that we look at nextPts for a frame, and not its pts or duration.
  // Our abstract player displays frames starting at the pts for that frame
  // until the pts for the next frame. There are two consequences:
  //
  // 1. We ignore the duration for a frame. A frame is played until the
  // next frame replaces it. This model is robust to durations being 0 or
  // incorrect; our source of truth is the pts for frames. If duration is
  // accurate, the nextPts for a frame would be equivalent to pts +
  // duration.
  // 2. In order to establish if the start of an interval maps to a
  // particular frame, we need to figure out if it is ordered after the
  // frame's pts, but before the next frames's pts.

  int64_t start_frame_index = seconds_to_index_lower_bound(start_seconds);
  int64_t stop_frame_index = seconds_to_index_upper_bound(stop_seconds);
  int64_t num_frames = stop_frame_index - start_frame_index;

  FrameBatchOutput frame_batch_output(
      num_frames, video_stream_options, stream_metadata);
  for (int64_t i = start_frame_index, f = 0; i < stop_frame_index; ++i, ++f) {
    FrameOutput frame_output =
        get_frame_at_index_internal(i, frame_batch_output.data[f]);
    frame_batch_output.pts_seconds[f] = frame_output.pts_seconds;
    frame_batch_output.duration_seconds[f] = frame_output.duration_seconds;
  }
  frame_batch_output.data = maybe_permute_h_w_c2_c_h_w(frame_batch_output.data);

  return frame_batch_output;
}

// Note [Audio Decoding Design]
// This note explains why audio decoding is implemented the way it is, and why
// it inherently differs from video decoding.
//
// Like for video, FFmpeg exposes the concept of a frame for audio streams. An
// audio frame is a contiguous sequence of samples, where a sample consists of
// `numChannels` values. An audio frame, or a sequence thereof, is always
// converted into a tensor of shape `(numChannels, numSamplesPerChannel)`.
//
// The notion of 'frame' in audio isn't what users want to interact with. Users
// want to interact with samples. The C++ and core APIs return frames, because
// we want those to be close to FFmpeg concepts, but the higher-level public
// APIs expose samples. As a result:
// - We don't expose index-based APIs for audio, because that would mean
// exposing the concept of audio frame. For now, we think exposing time-based
// APIs is more natural.
// - We never perform a scan for audio streams. We don't need to, since we won't
// be converting timestamps to indices. That's why we enforce the seek_mode
// to be "approximate" (which is slightly misleading, because technically the
// output samples will be at their exact positions. But this incongruence is
// only exposed at the C++/core private levels).
//
// Audio frames are of variable dimensions: in the same stream, a frame can
// contain 1024 samples and the next one may contain 512 [1]. This makes it
// impossible to stack audio frames in the same way we can stack video frames.
// This is one of the main reasons we cannot reuse the same pre-allocation logic
// we have for videos in getFramesPlayedInRange(): pre-allocating a batch
// requires constant (and known) frame dimensions. That's also why
// *concatenated* along the samples dimension, not stacked.
//
// [IMPORTANT!] There is one key invariant that we must respect when decoding
// audio frames:
//
// BEFORE DECODING FRAME i, WE MUST DECODE ALL FRAMES j < i.
//
// Always. Why? We don't know. What we know is that if we don't, we get clipped,
// incorrect audio as output [2]. All other (correct) libraries like TorchAudio
// or Decord do something similar, whether it was intended or not. This has a
// few implications:
// - The **only** place we're allowed to seek to in an audio stream is the
// stream's beginning. This ensures that if we need a frame, we'll have
// decoded all previous frames.
// - Because of that, we don't allow the public APIs to seek. Public APIs can
// call next() and `getFramesPlayedInRangeAudio()`, but they cannot manually
// seek.
// - We try not to seek, when we can avoid it. Typically if the next frame we
// need is in the future, we don't seek back to the beginning, we just decode
// all the frames in-between.
//
// [2] If you're brave and curious, you can read the long "Seek offset for
// audio" note in https://github.com/pytorch/torchcodec/pull/507/files, which
// sums up past (and failed) attemps at working around this issue.
SingleStreamDecoder::AudioFramesOutput
SingleStreamDecoder::getFramesPlayedInRangeAudio(
    double start_seconds,
    std::optional<double> stop_seconds_optional) {
  validate_active_stream(_avm_e_d_i_a__t_y_p_e__a_u_d_i_o);

  if (stopSecondsOptional.has_value()) {
    TORCH_CHECK(
        start_seconds <= *stopSecondsOptional,
        "Start seconds (" + std::to_string(start_seconds) +
            ") must be less than or equal to stop seconds (" +
            std::to_string(*stop_seconds_optional) + ").");
  }

  if (stopSecondsOptional.has_value() &&
      start_seconds == *stopSecondsOptional) {
    // For consistency with video
    return AudioFramesOutput{torch::empty({0, 0}), 0.0};
  }

  StreamInfo& stream_info = stream_infos_[active_stream_index_];

  auto start_pts = seconds_to_closest_pts(start_seconds, stream_info.time_base);
  if (startPts < stream_info.last_decoded_avframe_pts +
          stream_info.last_decoded_avframe_duration) {
    // If we need to seek backwards, then we have to seek back to the beginning
    // of the stream.
    // See [Audio Decoding Design].
    set_cursor(_i_n_t64__m_i_n);
  }

  // TODO-AUDIO Pre-allocate a long-enough tensor instead of creating a vec +
  // cat(). This would save a copy. We know the duration of the output and the
  // sample rate, so in theory we know the number of output samples.
  std::vector<torch::Tensor> frames;

  std::optional<double> first_frame_pts_seconds = std::nullopt;
  auto stop_pts = stop_seconds_optional.has_value()
      ? seconds_to_closest_pts(*stop_seconds_optional, stream_info.time_base)
      : INT64_MAX;
  auto finished = false;
  while (!finished) {
    try {
      UniqueAVFrame avframe =
          decode_avframe([start_pts](const UniqueAVFrame& avframe) {
            return start_pts < avframe->pts + get_duration(avframe);
          });
      auto frame_output = convert_avframe_to_frame_output(avframe);
      if (!firstFramePtsSeconds.has_value()) {
        first_frame_pts_seconds = frame_output.pts_seconds;
      }
      frames.push_back(frame_output.data);
    } catch (const EndOfFileException& e) {
      finished = true;
    }

    // If stopSeconds is in [begin, end] of the last decoded frame, we should
    // stop decoding more frames. Note that if we were to use [begin, end),
    // which may seem more natural, then we would decode the frame starting at
    // stopSeconds, which isn't what we want!
    auto last_decoded_avframe_end = stream_info.last_decoded_avframe_pts +
        stream_info.last_decoded_avframe_duration;
    finished |= (streamInfo.lastDecodedAvFramePts) <= stop_pts &&
        (stopPts <= last_decoded_avframe_end);
  }

  auto last_samples = maybe_flush_swr_buffers();
  if (lastSamples.has_value()) {
    frames.push_back(*last_samples);
  }

  TORCH_CHECK(
      frames.size() > 0 && first_frame_pts_seconds.has_value(),
      "No audio frames were decoded. ",
      "This is probably because start_seconds is too high? ",
      "Current value is ",
      start_seconds);

  return AudioFramesOutput{torch::cat(frames, 1), *firstFramePtsSeconds};
}

// --------------------------------------------------------------------------
// SEEKING APIs
// --------------------------------------------------------------------------

void SingleStreamDecoder::setCursorPtsInSeconds(double seconds) {
  // We don't allow public audio decoding APIs to seek, see [Audio Decoding
  // Design]
  validate_active_stream(_avm_e_d_i_a__t_y_p_e__v_i_d_e_o);
  set_cursor(seconds_to_closest_pts(
      seconds, stream_infos_[active_stream_index_].time_base));
}

void SingleStreamDecoder::setCursor(int64_t pts) {
  cursor_was_just_set_ = true;
  cursor_ = pts;
}

/*
Videos have I frames and non-_i frames (P and B frames). Non-I frames need data
from the previous I frame to be decoded.

Imagine the cursor is at a random frame with PTS=lastDecodedAvFramePts (x for
brevity) and we wish to seek to a user-specified PTS=y.

If y < x, we don't have a choice but to seek backwards to the highest I frame
before y.

If y > x, we have two choices:

1. We could keep decoding forward until we hit y. Illustrated below:

I P P P I P P P I P P I P P I P
x y

2. We could try to jump to an I frame between x and y (indicated by j below).
And then start decoding until we encounter y. Illustrated below:

I P P P I P P P I P P I P P I P
x j y

(2) is more efficient than (1) if there is an I frame between x and y.
*/
bool SingleStreamDecoder::canWeAvoidSeeking() const {
  const StreamInfo& stream_info = stream_infos_.at(active_stream_index_);
  if (streamInfo.avMediaType == AVMEDIA_TYPE_AUDIO) {
    // For audio, we only need to seek if a backwards seek was requested within
    // getFramesPlayedInRangeAudio(), when setCursorPtsInSeconds() was called.
    // For more context, see [Audio Decoding Design]
    return !cursorWasJustSet_;
  }
  int64_t last_decoded_avframe_pts =
      stream_infos_.at(active_stream_index_).last_decoded_avframe_pts;
  if (cursor_ < last_decoded_avframe_pts) {
    // We can never skip a seek if we are seeking backwards.
    return false;
  }
  if (lastDecodedAvFramePts == cursor_) {
    // We are seeking to the exact same frame as we are currently at. Without
    // caching we have to rewind back and decode the frame again.
    // TODO: https://github.com/pytorch-labs/torchcodec/issues/84 we could
    // implement caching.
    return false;
  }
  // We are seeking forwards.
  // We can only skip a seek if both lastDecodedAvFramePts and
  // cursor_ share the same keyframe.
  int last_decoded_avframe_index =
      get_key_frame_index_for_pts(last_decoded_avframe_pts);
  int target_key_frame_index = get_key_frame_index_for_pts(cursor_);
  return last_decoded_avframe_index >= 0 && target_key_frame_index >= 0 &&
      last_decoded_avframe_index == target_key_frame_index;
}

// This method looks at currentPts and desiredPts and seeks in the
// AVFormatContext if it is needed. We can skip seeking in certain cases. See
// the comment of canWeAvoidSeeking() for details.
void SingleStreamDecoder::maybeSeekToBeforeDesiredPts() {
  validate_active_stream();
  StreamInfo& stream_info = stream_infos_[active_stream_index_];

  decode_stats_.num_seeks_attempted++;
  if (canWeAvoidSeeking()) {
    decode_stats_.num_seeks_skipped++;
    return;
  }

  int64_t desired_pts = cursor_;

  // For some encodings like H265, FFMPEG sometimes seeks past the point we
  // set as the max_ts. So we use our own index to give it the exact pts of
  // the key frame that we want to seek to.
  // See https://github.com/pytorch/torchcodec/issues/179 for more details.
  // See https://trac.ffmpeg.org/ticket/11137 for the underlying ffmpeg bug.
  if (!streamInfo.keyFrames.empty()) {
    int desired_key_frame_index =
        get_key_frame_index_for_pts_using_scanned_index(
            stream_info.key_frames, desired_pts);
    desired_key_frame_index = std::max(desired_key_frame_index, 0);
    desired_pts = stream_info.key_frames[desired_key_frame_index].pts;
  }

  int status = avformat_seek_file(
      format_context_.get(),
      stream_info.stream_index,
      INT64_MIN,
      desired_pts,
      desired_pts,
      0);
  if (status < 0) {
    throw std::runtime_error(
        "Could not seek file to pts=" + std::to_string(desired_pts) + ": " +
        get_ffmpeg_error_string_from_error_code(status));
  }
  decode_stats_.num_flushes++;
  avcodec_flush_buffers(stream_info.codec_context.get());
}

// --------------------------------------------------------------------------
// LOW-LEVEL DECODING
// --------------------------------------------------------------------------

UniqueAVFrame SingleStreamDecoder::decodeAVFrame(
    std::function<bool(const UniqueAVFrame&)> filter_function) {
  validate_active_stream();

  reset_decode_stats();

  if (cursorWasJustSet_) {
    maybe_seek_to_before_desired_pts();
    cursor_was_just_set_ = false;
  }

  StreamInfo& stream_info = stream_infos_[active_stream_index_];

  // Need to get the next frame or error from PopFrame.
  UniqueAVFrame avframe(avframe_alloc());
  AutoAVPacket auto_avpacket;
  int status = AVSUCCESS;
  bool reached_e_o_f = false;
  while (true) {
    status =
        avcodec_receive_frame(stream_info.codec_context.get(), avframe.get());

    if (status != AVSUCCESS && status != AVERROR(EAGAIN)) {
      // Non-retriable error
      break;
    }

    decode_stats_.num_frames_received_by_decoder++;
    // Is this the kind of frame we're looking for?
    if (status == AVSUCCESS && filter_function(avframe)) {
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

    if (reachedEOF) {
      // We don't have any more packets to receive. So keep on pulling frames
      // from its internal buffers.
      continue;
    }

    // We still haven't found the frame we're looking for. So let's read more
    // packets and send them to the decoder.
    ReferenceAVPacket packet(auto_avpacket);
    do {
      status = av_read_frame(format_context_.get(), packet.get());
      decode_stats_.num_packets_read++;

      if (status == AVERROR_EOF) {
        // End of file reached. We must drain the codec by sending a nullptr
        // packet.
        status = avcodec_send_packet(
            stream_info.codec_context.get(),
            /*avpkt=*/nullptr);
        if (status < AVSUCCESS) {
          throw std::runtime_error(
              "Could not flush decoder: " +
              get_ffmpeg_error_string_from_error_code(status));
        }

        reached_e_o_f = true;
        break;
      }

      if (status < AVSUCCESS) {
        throw std::runtime_error(
            "Could not read frame from input file: " +
            get_ffmpeg_error_string_from_error_code(status));
      }
    } while (packet->stream_index != active_stream_index_);

    if (reachedEOF) {
      // We don't have any more packets to send to the decoder. So keep on
      // pulling frames from its internal buffers.
      continue;
    }

    // We got a valid packet. Send it to the decoder, and we'll receive it in
    // the next iteration.
    status = avcodec_send_packet(stream_info.codec_context.get(), packet.get());
    if (status < AVSUCCESS) {
      throw std::runtime_error(
          "Could not push packet to decoder: " +
          get_ffmpeg_error_string_from_error_code(status));
    }

    decode_stats_.num_packets_sent_to_decoder++;
  }

  if (status < AVSUCCESS) {
    if (reachedEOF || status == AVERROR_EOF) {
      throw SingleStreamDecoder::EndOfFileException(
          "Requested next frame while there are no more frames left to "
          "decode.");
    }
    throw std::runtime_error(
        "Could not receive frame from decoder: " +
        get_ffmpeg_error_string_from_error_code(status));
  }

  // Note that we don't flush the decoder when we reach EOF (even though that's
  // mentioned in https://ffmpeg.org/doxygen/trunk/group__lavc__encdec.html).
  // This is because we may have packets internally in the decoder that we
  // haven't received as frames. Eventually we will either hit AVERROR_EOF from
  // av_receive_frame() or the user will have seeked to a different location in
  // the file and that will flush the decoder.
  stream_info.last_decoded_avframe_pts = avframe->pts;
  stream_info.last_decoded_avframe_duration = get_duration(avframe);

  return avframe;
}

// --------------------------------------------------------------------------
// AVFRAME <-> FRAME OUTPUT CONVERSION
// --------------------------------------------------------------------------

SingleStreamDecoder::FrameOutput
SingleStreamDecoder::convertAVFrameToFrameOutput(
    UniqueAVFrame& avframe,
    std::optional<torch::Tensor> pre_allocated_output_tensor) {
  // Convert the frame to tensor.
  FrameOutput frame_output;
  auto& stream_info = stream_infos_[active_stream_index_];
  frame_output.pts_seconds = pts_to_seconds(
      avframe->pts, format_context_->streams[active_stream_index_]->time_base);
  frame_output.duration_seconds = pts_to_seconds(
      get_duration(avframe),
      format_context_->streams[active_stream_index_]->time_base);
  if (streamInfo.avMediaType == AVMEDIA_TYPE_AUDIO) {
    convert_audio_avframe_to_frame_output_on_c_p_u(avframe, frame_output);
  } else if (streamInfo.videoStreamOptions.device.type() == torch::kCPU) {
    convert_avframe_to_frame_output_on_c_p_u(
        avframe, frame_output, pre_allocated_output_tensor);
  } else if (streamInfo.videoStreamOptions.device.type() == torch::kCUDA) {
    convert_avframe_to_frame_output_on_cuda(
        stream_info.video_stream_options.device,
        stream_info.video_stream_options,
        avframe,
        frame_output,
        pre_allocated_output_tensor);
  } else {
    TORCH_CHECK(
        false,
        "Invalid device type: " +
            stream_info.video_stream_options.device.str());
  }
  return frame_output;
}

// Note [preAllocatedOutputTensor with swscale and filtergraph]:
// Callers may pass a pre-allocated tensor, where the output.data tensor will
// be stored. This parameter is honored in any case, but it only leads to a
// speed-up when swscale is used. With swscale, we can tell ffmpeg to place the
// decoded frame directly into `preAllocatedtensor.data_ptr()`. We haven't yet
// found a way to do that with filtegraph.
// TODO: Figure out whether that's possible!
// Dimension order of the preAllocatedOutputTensor must be HWC, regardless of
// `dimension_order` parameter. It's up to callers to re-shape it if needed.
void SingleStreamDecoder::convertAVFrameToFrameOutputOnCPU(
    UniqueAVFrame& avframe,
    FrameOutput& frame_output,
    std::optional<torch::Tensor> pre_allocated_output_tensor) {
  auto& stream_info = stream_infos_[active_stream_index_];

  auto frame_dims = get_height_and_width_from_options_or_avframe(
      stream_info.video_stream_options, avframe);
  int expected_output_height = frame_dims.height;
  int expected_output_width = frame_dims.width;

  if (preAllocatedOutputTensor.has_value()) {
    auto shape = pre_allocated_output_tensor.value().sizes();
    TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == expected_output_height) &&
            (shape[1] == expected_output_width) && (shape[2] == 3),
        "Expected pre-allocated tensor of shape ",
        expected_output_height,
        "x",
        expected_output_width,
        "x3, got ",
        shape);
  }

  torch::Tensor output_tensor;
  // We need to compare the current frame context with our previous frame
  // context. If they are different, then we need to re-create our colorspace
  // conversion objects. We create our colorspace conversion objects late so
  // that we don't have to depend on the unreliable metadata in the header.
  // And we sometimes re-create them because it's possible for frame
  // resolution to change mid-stream. Finally, we want to reuse the colorspace
  // conversion objects as much as possible for performance reasons.
  enum AVPixelFormat frame_format =
      static_cast<enum AVPixelFormat>(avFrame->format);
  auto frame_context = DecodedFrameContext{
      avframe->width,
      avframe->height,
      frame_format,
      expected_output_width,
      expected_output_height};

  if (streamInfo.colorConversionLibrary == ColorConversionLibrary::SWSCALE) {
    output_tensor =
        pre_allocated_output_tensor.value_or(allocate_empty_h_w_c_tensor(
            expected_output_height, expected_output_width, torch::kCPU));

    if (!streamInfo.swsContext ||
        stream_info.prev_frame_context != frame_context) {
      create_sws_context(stream_info, frame_context, avframe->colorspace);
      stream_info.prev_frame_context = frame_context;
    }
    int result_height =
        convert_avframe_to_tensor_using_sws_scale(avframe, output_tensor);
    // If this check failed, it would mean that the frame wasn't reshaped to
    // the expected height.
    // TODO: Can we do the same check for width?
    TORCH_CHECK(
        result_height == expected_output_height,
        "resultHeight != expected_output_height: ",
        result_height,
        " != ",
        expected_output_height);

    frame_output.data = output_tensor;
  } else if (
      stream_info.color_conversion_library ==
      ColorConversionLibrary::FILTERGRAPH) {
    if (!streamInfo.filterGraphContext.filterGraph ||
        stream_info.prev_frame_context != frame_context) {
      create_filter_graph(
          stream_info, expected_output_height, expected_output_width);
      stream_info.prev_frame_context = frame_context;
    }
    output_tensor = convert_avframe_to_tensor_using_filter_graph(avframe);

    // Similarly to above, if this check fails it means the frame wasn't
    // reshaped to its expected dimensions by filtergraph.
    auto shape = output_tensor.sizes();
    TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == expected_output_height) &&
            (shape[1] == expected_output_width) && (shape[2] == 3),
        "Expected output tensor of shape ",
        expected_output_height,
        "x",
        expected_output_width,
        "x3, got ",
        shape);

    if (preAllocatedOutputTensor.has_value()) {
      // We have already validated that preAllocatedOutputTensor and
      // outputTensor have the same shape.
      pre_allocated_output_tensor.value().copy_(output_tensor);
      frame_output.data = pre_allocated_output_tensor.value();
    } else {
      frame_output.data = output_tensor;
    }
  } else {
    throw std::runtime_error(
        "Invalid color conversion library: " +
        std::to_string(static_cast<int>(stream_info.color_conversion_library)));
  }
}

int SingleStreamDecoder::convertAVFrameToTensorUsingSwsScale(
    const UniqueAVFrame& avframe,
    torch::Tensor& output_tensor) {
  StreamInfo& active_stream_info = stream_infos_[active_stream_index_];
  SwsContext* sws_context = active_stream_info.sws_context.get();
  uint8_t* pointers[4] = {
      output_tensor.data_ptr<uint8_t>(), nullptr, nullptr, nullptr};
  int expected_output_width = output_tensor.sizes()[1];
  int linesizes[4] = {expectedOutputWidth * 3, 0, 0, 0};
  int result_height = sws_scale(
      sws_context,
      avframe->data,
      avframe->linesize,
      0,
      avframe->height,
      pointers,
      linesizes);
  return result_height;
}

torch::Tensor SingleStreamDecoder::convertAVFrameToTensorUsingFilterGraph(
    const UniqueAVFrame& avframe) {
  FilterGraphContext& filter_graph_context =
      stream_infos_[active_stream_index_].filter_graph_context;
  int status = av_buffersrc_write_frame(
      filter_graph_context.source_context, avframe.get());
  if (status < AVSUCCESS) {
    throw std::runtime_error("_failed to add frame to buffer source context");
  }

  UniqueAVFrame filtered_avframe(avframe_alloc());
  status = av_buffersink_get_frame(
      filter_graph_context.sink_context, filtered_avframe.get());
  TORCH_CHECK_EQ(filteredAVFrame->format, AV_PIX_FMT_RGB24);

  auto frame_dims =
      get_height_and_width_from_resized_avframe(*filtered_avframe.get());
  int height = frame_dims.height;
  int width = frame_dims.width;
  std::vector<int64_t> shape = {height, width, 3};
  std::vector<int64_t> strides = {filteredAVFrame->linesize[0], 3, 1};
  AVFrame* filtered_avframe_ptr = filtered_avframe.release();
  auto deleter = [filteredAVFramePtr](void*) {
    UniqueAVFrame avframe_to_delete(filtered_avframe_ptr);
  };
  return torch::from_blob(
      filtered_avframe_ptr->data[0], shape, strides, deleter, {torch::kUInt8});
}

void SingleStreamDecoder::convertAudioAVFrameToFrameOutputOnCPU(
    UniqueAVFrame& src_avframe,
    FrameOutput& frame_output) {
  AVSampleFormat source_sample_format =
      static_cast<_avsample_format>(src_avframe->format);
  AVSampleFormat desired_sample_format = AV_SAMPLE_FMT_FLTP;

  int source_sample_rate = src_avframe->sample_rate;
  int desired_sample_rate =
      stream_infos_[active_stream_index_]
          .audio_stream_options.sample_rate.value_or(source_sample_rate);

  bool must_convert =
      (sourceSampleFormat != desired_sample_format ||
       source_sample_rate != desired_sample_rate);

  UniqueAVFrame converted_avframe;
  if (mustConvert) {
    converted_avframe = convert_audio_avframe_sample_format_and_sample_rate(
        src_avframe,
        source_sample_format,
        desired_sample_format,
        source_sample_rate,
        desired_sample_rate);
  }
  const UniqueAVFrame& avframe = must_convert ? converted_avframe : src_avframe;

  AVSampleFormat format = static_cast<_avsample_format>(avframe->format);
  TORCH_CHECK(
      format == desired_sample_format,
      "Something went wrong, the frame didn't get converted to the desired format. ",
      "Desired format = ",
      av_get_sample_fmt_name(desired_sample_format),
      "source format = ",
      av_get_sample_fmt_name(format));

  auto num_samples = avframe->nb_samples; // per channel
  auto num_channels = get_num_channels(avframe);

  frame_output.data = torch::empty({numChannels, num_samples}, torch::kFloat32);

  if (numSamples > 0) {
    uint8_t* output_channel_data =
        static_cast<uint8_t*>(frame_output.data.data_ptr());
    auto num_bytes_per_channel = num_samples * av_get_bytes_per_sample(format);
    for (auto channel = 0; channel < num_channels;
         ++channel, output_channel_data += num_bytes_per_channel) {
      memcpy(
          output_channel_data,
          avframe->extended_data[channel],
          num_bytes_per_channel);
    }
  }
}

UniqueAVFrame SingleStreamDecoder::convertAudioAVFrameSampleFormatAndSampleRate(
    const UniqueAVFrame& src_avframe,
    AVSampleFormat source_sample_format,
    AVSampleFormat desired_sample_format,
    int source_sample_rate,
    int desired_sample_rate) {
  auto& stream_info = stream_infos_[active_stream_index_];

  if (!streamInfo.swrContext) {
    create_swr_context(
        stream_info,
        source_sample_format,
        desired_sample_format,
        source_sample_rate,
        desired_sample_rate);
  }

  UniqueAVFrame converted_avframe(avframe_alloc());
  TORCH_CHECK(
      converted_avframe,
      "Could not allocate frame for sample format conversion.");

  set_channel_layout(converted_avframe, src_avframe);
  converted_avframe->format = static_cast<int>(desired_sample_format);
  converted_avframe->sample_rate = desired_sample_rate;
  if (sourceSampleRate != desired_sample_rate) {
    // Note that this is an upper bound on the number of output samples.
    // `swr_convert()` will likely not fill convertedAVFrame with that many
    // samples if sample rate conversion is needed. It will buffer the last few
    // ones because those require future samples. That's also why we reset
    // nb_samples after the call to `swr_convert()`.
    // We could also use `swr_get_out_samples()` to determine the number of
    // output samples, but empirically `av_rescale_rnd()` seems to provide a
    // tighter bound.
    converted_avframe->nb_samples = av_rescale_rnd(
        swr_get_delay(stream_info.swr_context.get(), source_sample_rate) +
            src_avframe->nb_samples,
        desired_sample_rate,
        source_sample_rate,
        AV_ROUND_UP);
  } else {
    converted_avframe->nb_samples = src_avframe->nb_samples;
  }

  auto status = avframe_get_buffer(converted_avframe.get(), 0);
  TORCH_CHECK(
      status == AVSUCCESS,
      "Could not allocate frame buffers for sample format conversion: ",
      get_ffmpeg_error_string_from_error_code(status));

  auto num_converted_samples = swr_convert(
      stream_info.swr_context.get(),
      converted_avframe->data,
      converted_avframe->nb_samples,
      static_cast<const uint8_t**>(
          const_cast<const uint8_t**>(src_avframe->data)),
      src_avframe->nb_samples);
  // numConvertedSamples can be 0 if we're downsampling by a great factor and
  // the first frame doesn't contain a lot of samples. It should be handled
  // properly by the caller.
  TORCH_CHECK(
      num_converted_samples >= 0,
      "Error in swr_convert: ",
      get_ffmpeg_error_string_from_error_code(num_converted_samples));

  // See comment above about nb_samples
  converted_avframe->nb_samples = num_converted_samples;

  return converted_avframe;
}

std::optional<torch::Tensor> SingleStreamDecoder::maybeFlushSwrBuffers() {
  // When sample rate conversion is involved, swresample buffers some of the
  // samples in-between calls to swr_convert (see the libswresample docs).
  // That's because the last few samples in a given frame require future samples
  // from the next frame to be properly converted. This function flushes out the
  // samples that are stored in swresample's buffers.
  auto& stream_info = stream_infos_[active_stream_index_];
  if (!streamInfo.swrContext) {
    return std::nullopt;
  }
  auto num_remaining_samples = // this is an upper bound
      swr_get_out_samples(stream_info.swr_context.get(), 0);

  if (numRemainingSamples == 0) {
    return std::nullopt;
  }

  auto num_channels = get_num_channels(stream_info.codec_context);
  torch::Tensor last_samples =
      torch::empty({numChannels, num_remaining_samples}, torch::kFloat32);

  std::vector<uint8_t*> output_buffers(num_channels);
  for (auto i = 0; i < num_channels; i++) {
    output_buffers[i] = static_cast<uint8_t*>(last_samples[i].data_ptr());
  }

  auto actual_num_remaining_samples = swr_convert(
      stream_info.swr_context.get(),
      output_buffers.data(),
      num_remaining_samples,
      nullptr,
      0);

  return last_samples.narrow(
      /*dim=*/1, /*start=*/0, /*length=*/actualNumRemainingSamples);
}

// --------------------------------------------------------------------------
// OUTPUT ALLOCATION AND SHAPE CONVERSION
// --------------------------------------------------------------------------

SingleStreamDecoder::FrameBatchOutput::FrameBatchOutput(
    int64_t num_frames,
    const VideoStreamOptions& video_stream_options,
    const StreamMetadata& stream_metadata)
    : ptsSeconds(torch::empty({numFrames}, {torch::kFloat64})),
      durationSeconds(torch::empty({numFrames}, {torch::kFloat64})) {
  auto frame_dims = get_height_and_width_from_options_or_metadata(
      video_stream_options, stream_metadata);
  int height = frame_dims.height;
  int width = frame_dims.width;
  data = allocate_empty_h_w_c_tensor(
      height, width, video_stream_options.device, num_frames);
}

torch::Tensor allocate_empty_h_w_c_tensor(
    int height,
    int width,
    torch::Device device,
    std::optional<int> num_frames) {
  auto tensor_options = torch::TensorOptions()
                            .dtype(torch::kUInt8)
                            .layout(torch::kStrided)
                            .device(device);
  TORCH_CHECK(height > 0, "height must be > 0, got: ", height);
  TORCH_CHECK(width > 0, "width must be > 0, got: ", width);
  if (numFrames.has_value()) {
    auto num_frames_value = num_frames.value();
    TORCH_CHECK(
        num_frames_value >= 0,
        "numFrames must be >= 0, got: ",
        num_frames_value);
    return torch::empty({numFramesValue, height, width, 3}, tensor_options);
  } else {
    return torch::empty({height, width, 3}, tensor_options);
  }
}

// Returns a [N]CHW *view* of a [N]HWC input tensor, if the options require so.
// The [N] leading batch-dimension is optional i.e. the input tensor can be 3D
// or 4D.
// Calling permute() is guaranteed to return a view as per the docs:
// https://pytorch.org/docs/stable/generated/torch.permute.html
torch::Tensor SingleStreamDecoder::maybePermuteHWC2CHW(
    torch::Tensor& hwc_tensor) {
  if (streamInfos_[activeStreamIndex_].videoStreamOptions.dimensionOrder ==
      "NHWC") {
    return hwc_tensor;
  }
  auto num_dimensions = hwc_tensor.dim();
  auto shape = hwc_tensor.sizes();
  if (numDimensions == 3) {
    TORCH_CHECK(shape[2] == 3, "Not a HWC tensor: ", shape);
    return hwc_tensor.permute({2, 0, 1});
  } else if (numDimensions == 4) {
    TORCH_CHECK(shape[3] == 3, "Not a NHWC tensor: ", shape);
    return hwc_tensor.permute({0, 3, 1, 2});
  } else {
    TORCH_CHECK(
        false, "Expected tensor with 3 or 4 dimensions, got ", num_dimensions);
  }
}

// --------------------------------------------------------------------------
// COLOR CONVERSION UTILS AND INITIALIZERS
// --------------------------------------------------------------------------

bool SingleStreamDecoder::DecodedFrameContext::operator==(
    const SingleStreamDecoder::DecodedFrameContext& other) {
  return decoded_width == other.decoded_width &&
      decoded_height == other.decoded_height &&
      decoded_format == other.decoded_format &&
      expected_width == other.expected_width &&
      expected_height == other.expected_height;
}

bool SingleStreamDecoder::DecodedFrameContext::operator!=(
    const SingleStreamDecoder::DecodedFrameContext& other) {
  return !(*this == other);
}

void SingleStreamDecoder::createFilterGraph(
    StreamInfo& stream_info,
    int expected_output_height,
    int expected_output_width) {
  FilterGraphContext& filter_graph_context = stream_info.filter_graph_context;
  filter_graph_context.filter_graph.reset(avfilter_graph_alloc());
  TORCH_CHECK(filterGraphContext.filterGraph.get() != nullptr);

  if (streamInfo.videoStreamOptions.ffmpegThreadCount.has_value()) {
    filter_graph_context.filter_graph->nb_threads =
        stream_info.video_stream_options.ffmpeg_thread_count.value();
  }

  const AVFilter* buffersrc = avfilter_get_by_name("buffer");
  const AVFilter* buffersink = avfilter_get_by_name("buffersink");
  AVCodecContext* codec_context = stream_info.codec_context.get();

  std::stringstream filter_args;
  filter_args << "video_size=" << codec_context->width << "x"
              << codec_context->height;
  filter_args << ":pix_fmt=" << codec_context->pix_fmt;
  filter_args << ":time_base=" << stream_info.stream->time_base.num << "/"
              << stream_info.stream->time_base.den;
  filter_args << ":pixel_aspect=" << codec_context->sample_aspect_ratio.num
              << "/" << codec_context->sample_aspect_ratio.den;

  int status = avfilter_graph_create_filter(
      &filterGraphContext.sourceContext,
      buffersrc,
      "in",
      filter_args.str().c_str(),
      nullptr,
      filter_graph_context.filter_graph.get());
  if (status < 0) {
    throw std::runtime_error(
        std::string("_failed to create filter graph: ") + filter_args.str() +
        ": " + get_ffmpeg_error_string_from_error_code(status));
  }

  status = avfilter_graph_create_filter(
      &filterGraphContext.sinkContext,
      buffersink,
      "out",
      nullptr,
      nullptr,
      filter_graph_context.filter_graph.get());
  if (status < 0) {
    throw std::runtime_error(
        "Failed to create filter graph: " +
        get_ffmpeg_error_string_from_error_code(status));
  }

  enum AVPixelFormat pix_fmts[] = {AV_PIX_FMT_RGB24, AV_PIX_FMT_NONE};

  status = av_opt_set_int_list(
      filter_graph_context.sink_context,
      "pix_fmts",
      pix_fmts,
      AV_PIX_FMT_NONE,
      AV_OPT_SEARCH_CHILDREN);
  if (status < 0) {
    throw std::runtime_error(
        "Failed to set output pixel formats: " +
        get_ffmpeg_error_string_from_error_code(status));
  }

  UniqueAVFilterInOut outputs(avfilter_inout_alloc());
  UniqueAVFilterInOut inputs(avfilter_inout_alloc());

  outputs->name = av_strdup("in");
  outputs->filter_ctx = filter_graph_context.source_context;
  outputs->pad_idx = 0;
  outputs->next = nullptr;
  inputs->name = av_strdup("out");
  inputs->filter_ctx = filter_graph_context.sink_context;
  inputs->pad_idx = 0;
  inputs->next = nullptr;

  std::stringstream description;
  description << "scale=" << expected_output_width << ":"
              << expected_output_height;
  description << ":sws_flags=bilinear";

  AVFilterInOut* outputs_tmp = outputs.release();
  AVFilterInOut* inputs_tmp = inputs.release();
  status = avfilter_graph_parse_ptr(
      filter_graph_context.filter_graph.get(),
      description.str().c_str(),
      &inputsTmp,
      &outputsTmp,
      nullptr);
  outputs.reset(outputs_tmp);
  inputs.reset(inputs_tmp);
  if (status < 0) {
    throw std::runtime_error(
        "Failed to parse filter description: " +
        get_ffmpeg_error_string_from_error_code(status));
  }

  status =
      avfilter_graph_config(filter_graph_context.filter_graph.get(), nullptr);
  if (status < 0) {
    throw std::runtime_error(
        "Failed to configure filter graph: " +
        get_ffmpeg_error_string_from_error_code(status));
  }
}

void SingleStreamDecoder::createSwsContext(
    StreamInfo& stream_info,
    const DecodedFrameContext& frame_context,
    const enum AVColorSpace colorspace) {
  SwsContext* sws_context = sws_get_context(
      frame_context.decoded_width,
      frame_context.decoded_height,
      frame_context.decoded_format,
      frame_context.expected_width,
      frame_context.expected_height,
      AV_PIX_FMT_RGB24,
      SWS_BILINEAR,
      nullptr,
      nullptr,
      nullptr);
  TORCH_CHECK(swsContext, "sws_getContext() returned nullptr");

  int* inv_table = nullptr;
  int* table = nullptr;
  int src_range, dst_range, brightness, contrast, saturation;
  int ret = sws_get_colorspace_details(
      sws_context,
      &invTable,
      &srcRange,
      &table,
      &dstRange,
      &brightness,
      &contrast,
      &saturation);
  TORCH_CHECK(ret != -1, "sws_getColorspaceDetails returned -1");

  const int* colorspace_table = sws_get_coefficients(colorspace);
  ret = sws_set_colorspace_details(
      sws_context,
      colorspace_table,
      src_range,
      colorspace_table,
      dst_range,
      brightness,
      contrast,
      saturation);
  TORCH_CHECK(ret != -1, "sws_setColorspaceDetails returned -1");

  stream_info.sws_context.reset(sws_context);
}

void SingleStreamDecoder::createSwrContext(
    StreamInfo& stream_info,
    AVSampleFormat source_sample_format,
    AVSampleFormat desired_sample_format,
    int source_sample_rate,
    int desired_sample_rate) {
  auto swr_context = allocate_swr_context(
      stream_info.codec_context,
      source_sample_format,
      desired_sample_format,
      source_sample_rate,
      desired_sample_rate);

  auto status = swr_init(swr_context);
  TORCH_CHECK(
      status == AVSUCCESS,
      "Couldn't initialize SwrContext: ",
      get_ffmpeg_error_string_from_error_code(status),
      ". If the error says 'Invalid argument', it's likely that you are using "
      "a buggy FFmpeg version. FFmpeg4 is known to fail here in some "
      "valid scenarios. Try to upgrade FFmpeg?");
  stream_info.swr_context.reset(swr_context);
}

// --------------------------------------------------------------------------
// PTS <-> INDEX CONVERSIONS
// --------------------------------------------------------------------------

int SingleStreamDecoder::getKeyFrameIndexForPts(int64_t pts) const {
  const StreamInfo& stream_info = stream_infos_.at(active_stream_index_);
  if (streamInfo.keyFrames.empty()) {
    return av_index_search_timestamp(
        stream_info.stream, pts, AVSEEK_FLAG_BACKWARD);
  } else {
    return get_key_frame_index_for_pts_using_scanned_index(
        stream_info.key_frames, pts);
  }
}

int SingleStreamDecoder::getKeyFrameIndexForPtsUsingScannedIndex(
    const std::vector<_single_stream_decoder::_frame_info>& key_frames,
    int64_t pts) const {
  auto upper_bound = std::upper_bound(
      key_frames.begin(),
      key_frames.end(),
      pts,
      [](int64_t pts, const SingleStreamDecoder::FrameInfo& frame_info) {
        return pts < frame_info.pts;
      });
  if (upperBound == key_frames.begin()) {
    return -1;
  }
  return upper_bound - 1 - key_frames.begin();
}

int64_t SingleStreamDecoder::secondsToIndexLowerBound(double seconds) {
  auto& stream_info = stream_infos_[active_stream_index_];
  switch (seekMode_) {
    case SeekMode::exact: {
      auto frame = std::lower_bound(
          stream_info.all_frames.begin(),
          stream_info.all_frames.end(),
          seconds,
          [&streamInfo](const FrameInfo& info, double start) {
            return pts_to_seconds(info.next_pts, stream_info.time_base) <=
                start;
          });

      return frame - stream_info.all_frames.begin();
    }
    case SeekMode::approximate: {
      auto& stream_metadata =
          container_metadata_.all_stream_metadata[active_stream_index_];
      TORCH_CHECK(
          stream_metadata.average_fps.has_value(),
          "Cannot use approximate mode since we couldn't find the average fps from the metadata.");
      return std::floor(seconds * stream_metadata.average_fps.value());
    }
    default:
      throw std::runtime_error("_unknown SeekMode");
  }
}

int64_t SingleStreamDecoder::secondsToIndexUpperBound(double seconds) {
  auto& stream_info = stream_infos_[active_stream_index_];
  switch (seekMode_) {
    case SeekMode::exact: {
      auto frame = std::upper_bound(
          stream_info.all_frames.begin(),
          stream_info.all_frames.end(),
          seconds,
          [&streamInfo](double stop, const FrameInfo& info) {
            return stop <= pts_to_seconds(info.pts, stream_info.time_base);
          });

      return frame - stream_info.all_frames.begin();
    }
    case SeekMode::approximate: {
      auto& stream_metadata =
          container_metadata_.all_stream_metadata[active_stream_index_];
      TORCH_CHECK(
          stream_metadata.average_fps.has_value(),
          "Cannot use approximate mode since we couldn't find the average fps from the metadata.");
      return std::ceil(seconds * stream_metadata.average_fps.value());
    }
    default:
      throw std::runtime_error("_unknown SeekMode");
  }
}

int64_t SingleStreamDecoder::getPts(int64_t frame_index) {
  auto& stream_info = stream_infos_[active_stream_index_];
  switch (seekMode_) {
    case SeekMode::exact:
      return stream_info.all_frames[frame_index].pts;
    case SeekMode::approximate: {
      auto& stream_metadata =
          container_metadata_.all_stream_metadata[active_stream_index_];
      TORCH_CHECK(
          stream_metadata.average_fps.has_value(),
          "Cannot use approximate mode since we couldn't find the average fps from the metadata.");
      return seconds_to_closest_pts(
          frame_index / stream_metadata.average_fps.value(),
          stream_info.time_base);
    }
    default:
      throw std::runtime_error("_unknown SeekMode");
  }
}

// --------------------------------------------------------------------------
// STREAM AND METADATA APIS
// --------------------------------------------------------------------------

int64_t SingleStreamDecoder::getNumFrames(
    const StreamMetadata& stream_metadata) {
  switch (seekMode_) {
    case SeekMode::exact:
      return stream_metadata.num_frames_from_scan.value();
    case SeekMode::approximate: {
      TORCH_CHECK(
          stream_metadata.num_frames.has_value(),
          "Cannot use approximate mode since we couldn't find the number of frames from the metadata.");
      return stream_metadata.num_frames.value();
    }
    default:
      throw std::runtime_error("_unknown SeekMode");
  }
}

double SingleStreamDecoder::getMinSeconds(
    const StreamMetadata& stream_metadata) {
  switch (seekMode_) {
    case SeekMode::exact:
      return stream_metadata.min_pts_seconds_from_scan.value();
    case SeekMode::approximate:
      return 0;
    default:
      throw std::runtime_error("_unknown SeekMode");
  }
}

double SingleStreamDecoder::getMaxSeconds(
    const StreamMetadata& stream_metadata) {
  switch (seekMode_) {
    case SeekMode::exact:
      return stream_metadata.max_pts_seconds_from_scan.value();
    case SeekMode::approximate: {
      TORCH_CHECK(
          stream_metadata.duration_seconds.has_value(),
          "Cannot use approximate mode since we couldn't find the duration from the metadata.");
      return stream_metadata.duration_seconds.value();
    }
    default:
      throw std::runtime_error("_unknown SeekMode");
  }
}

// --------------------------------------------------------------------------
// VALIDATION UTILS
// --------------------------------------------------------------------------

void SingleStreamDecoder::validateActiveStream(
    std::optional<_avmedia_type> av_media_type) {
  auto error_msg =
      "Provided stream index=" + std::to_string(active_stream_index_) +
      " was not previously added.";
  TORCH_CHECK(activeStreamIndex_ != NO_ACTIVE_STREAM, error_msg);
  TORCH_CHECK(streamInfos_.count(activeStreamIndex_) > 0, error_msg);

  int all_stream_metadata_size =
      static_cast<int>(container_metadata_.all_stream_metadata.size());
  TORCH_CHECK(
      active_stream_index_ >= 0 &&
          active_stream_index_ < all_stream_metadata_size,
      "Invalid stream index=" + std::to_string(active_stream_index_) +
          "; valid indices are in the range [0, " +
          std::to_string(all_stream_metadata_size) + ").");

  if (avMediaType.has_value()) {
    TORCH_CHECK(
        stream_infos_[active_stream_index_].av_media_type ==
            av_media_type.value(),
        "The method you called isn't supported. ",
        "If you're seeing this error, you are probably trying to call an ",
        "unsupported method on an audio stream.");
  }
}

void SingleStreamDecoder::validateScannedAllStreams(const std::string& msg) {
  if (!scannedAllStreams_) {
    throw std::runtime_error(
        "Must scan all streams to update metadata before calling " + msg);
  }
}

void SingleStreamDecoder::validateFrameIndex(
    const StreamMetadata& stream_metadata,
    int64_t frame_index) {
  int64_t num_frames = get_num_frames(stream_metadata);
  TORCH_CHECK(
      frame_index >= 0 && frame_index < num_frames,
      "Invalid frame index=" + std::to_string(frame_index) +
          " for stream_index=" + std::to_string(stream_metadata.stream_index) +
          " num_frames=" + std::to_string(num_frames));
}

// --------------------------------------------------------------------------
// MORALLY PRIVATE UTILS
// --------------------------------------------------------------------------

SingleStreamDecoder::DecodeStats SingleStreamDecoder::getDecodeStats() const {
  return decode_stats_;
}

std::ostream& operator<<(
    std::ostream& os,
    const SingleStreamDecoder::DecodeStats& stats) {
  os << "DecodeStats{"
     << "numFramesReceivedByDecoder=" << stats.num_frames_received_by_decoder
     << ", num_packets_read=" << stats.num_packets_read
     << ", num_packets_sent_to_decoder=" << stats.num_packets_sent_to_decoder
     << ", num_seeks_attempted=" << stats.num_seeks_attempted
     << ", num_seeks_skipped=" << stats.num_seeks_skipped
     << ", num_flushes=" << stats.num_flushes << "}";

  return os;
}

void SingleStreamDecoder::resetDecodeStats() {
  decode_stats_ = DecodeStats{};
}

double SingleStreamDecoder::getPtsSecondsForFrame(int64_t frame_index) {
  validate_active_stream(_avm_e_d_i_a__t_y_p_e__v_i_d_e_o);
  validate_scanned_all_streams("get_pts_seconds_for_frame");

  const auto& stream_info = stream_infos_[active_stream_index_];
  const auto& stream_metadata =
      container_metadata_.all_stream_metadata[active_stream_index_];
  validate_frame_index(stream_metadata, frame_index);

  return pts_to_seconds(
      stream_info.all_frames[frame_index].pts, stream_info.time_base);
}

// --------------------------------------------------------------------------
// FrameDims APIs
// --------------------------------------------------------------------------

FrameDims get_height_and_width_from_resized_avframe(
    const AVFrame& resized_avframe) {
  return FrameDims(resizedAVFrame.height, resized_avframe.width);
}

FrameDims get_height_and_width_from_options_or_metadata(
    const SingleStreamDecoder::VideoStreamOptions& video_stream_options,
    const SingleStreamDecoder::StreamMetadata& stream_metadata) {
  return FrameDims(
      video_stream_options.height.value_or(*stream_metadata.height),
      video_stream_options.width.value_or(*stream_metadata.width));
}

FrameDims get_height_and_width_from_options_or_avframe(
    const SingleStreamDecoder::VideoStreamOptions& video_stream_options,
    const UniqueAVFrame& avframe) {
  return FrameDims(
      video_stream_options.height.value_or(avframe->height),
      video_stream_options.width.value_or(avframe->width));
}

SingleStreamDecoder::SeekMode seek_mode_from_string(
    std::string_view seek_mode) {
  if (seekMode == "exact") {
    return SingleStreamDecoder::SeekMode::exact;
  } else if (seekMode == "approximate") {
    return SingleStreamDecoder::SeekMode::approximate;
  } else {
    TORCH_CHECK(false, "Invalid seek mode: " + std::string(seek_mode));
  }
}

} // namespace facebook::torchcodec
