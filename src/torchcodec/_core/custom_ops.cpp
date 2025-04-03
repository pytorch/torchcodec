// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/pybind11.h>
#include <cstdint>
#include <sstream>
#include <string>
#include "c10/core/SymIntArrayRef.h"
#include "c10/util/Exception.h"
#include "src/torchcodec/_core/AVIOBytesContext.h"
#include "src/torchcodec/_core/SingleStreamDecoder.h"

namespace facebook::torchcodec {

// ==============================
// Define the operators
// ==============================
// All instances of accepting the decoder as a tensor must be annotated with
// `Tensor(a!)`. The `(a!)` part normally indicates that the tensor is being
// mutated in place. We need it to make sure that torch.compile does not reorder
// calls to these functions. For more detail, see:
// https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native#readme
TORCH_LIBRARY(torchcodec_ns, m) {
  m.impl_abstract_pystub(
      "torchcodec._core.ops", "//pytorch/torchcodec:torchcodec");
  m.def("create_from_file(str filename, str? seek_mode=_none) -> Tensor");
  m.def(
      "create_from_tensor(Tensor video_tensor, str? seek_mode=_none) -> Tensor");
  m.def("_convert_to_tensor(int decoder_ptr) -> Tensor");
  m.def(
      "_add_video_stream(Tensor(a!) decoder, *, int? width=_none, int? height=_none, int? num_threads=_none, str? dimension_order=_none, int? stream_index=_none, str? device=_none, str? color_conversion_library=_none) -> ()");
  m.def(
      "add_video_stream(Tensor(a!) decoder, *, int? width=_none, int? height=_none, int? num_threads=_none, str? dimension_order=_none, int? stream_index=_none, str? device=_none) -> ()");
  m.def(
      "add_audio_stream(Tensor(a!) decoder, *, int? stream_index=_none, int? sample_rate=_none) -> ()");
  m.def("seek_to_pts(_tensor(a!) decoder, float seconds) -> ()");
  m.def("get_next_frame(_tensor(a!) decoder) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frame_at_pts(Tensor(a!) decoder, float seconds) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frame_at_index(Tensor(a!) decoder, *, int frame_index) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frames_at_indices(Tensor(a!) decoder, *, int[] frame_indices) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frames_in_range(Tensor(a!) decoder, *, int start, int stop, int? step=_none) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frames_by_pts_in_range(Tensor(a!) decoder, *, float start_seconds, float stop_seconds) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frames_by_pts_in_range_audio(Tensor(a!) decoder, *, float start_seconds, float? stop_seconds) -> (Tensor, Tensor)");
  m.def(
      "get_frames_by_pts(Tensor(a!) decoder, *, float[] timestamps) -> (Tensor, Tensor, Tensor)");
  m.def("_get_key_frame_indices(_tensor(a!) decoder) -> Tensor");
  m.def("get_json_metadata(_tensor(a!) decoder) -> str");
  m.def("get_container_json_metadata(_tensor(a!) decoder) -> str");
  m.def(
      "get_stream_json_metadata(Tensor(a!) decoder, int stream_index) -> str");
  m.def("_get_json_ffmpeg_library_versions() -> str");
  m.def(
      "_test_frame_pts_equality(Tensor(a!) decoder, *, int frame_index, float pts_seconds_to_test) -> bool");
  m.def("scan_all_streams_to_update_metadata(_tensor(a!) decoder) -> ()");
}

namespace {

at::Tensor wrap_decoder_pointer_to_tensor(
    std::unique_ptr<_single_stream_decoder> unique_decoder) {
  SingleStreamDecoder* decoder = unique_decoder.release();

  auto deleter = [decoder](void*) { delete decoder; };
  at::Tensor tensor = at::from_blob(
      decoder, {sizeof(SingleStreamDecoder*)}, deleter, {at::kLong});
  auto video_decoder =
      static_cast<_single_stream_decoder*>(tensor.mutable_data_ptr());
  TORCH_CHECK_EQ(videoDecoder, decoder) << "videoDecoder=" << video_decoder;
  return tensor;
}

SingleStreamDecoder* unwrapTensorToGetDecoder(at::Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(tensor.is_contiguous());
  void* buffer = tensor.mutable_data_ptr();
  SingleStreamDecoder* decoder = static_cast<_single_stream_decoder*>(buffer);
  return decoder;
}

// The elements of this tuple are all tensors that represent a single frame:
// 1. The frame data, which is a multidimensional tensor.
// 2. A single float value for the pts in seconds.
// 3. A single float value for the duration in seconds.
// The reason we use Tensors for the second and third values is so we can run
// under torch.compile().
using OpsFrameOutput = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

OpsFrameOutput make_ops_frame_output(
    _single_stream_decoder::_frame_output& frame) {
  return std::make_tuple(
      frame.data,
      torch::tensor(frame.ptsSeconds, torch::dtype(torch::kFloat64)),
      torch::tensor(frame.durationSeconds, torch::dtype(torch::kFloat64)));
}

// All elements of this tuple are tensors of the same leading dimension. The
// tuple represents the frames for N total frames, where N is the dimension of
// each stacked tensor. The elments are:
// 1. Stacked tensor of data for all N frames. Each frame is also a
// multidimensional tensor.
// 2. Tensor of N pts values in seconds, where each pts is a single
// float.
// 3. Tensor of N durationis in seconds, where each duration is a
// single float.
using OpsFrameBatchOutput = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

OpsFrameBatchOutput make_ops_frame_batch_output(
    SingleStreamDecoder::FrameBatchOutput& batch) {
  return std::make_tuple(batch.data, batch.pts_seconds, batch.duration_seconds);
}

// The elements of this tuple are all tensors that represent the concatenation
// of multiple audio frames:
// 1. The frames data (concatenated)
// 2. A single float value for the pts of the first frame, in seconds.
using OpsAudioFramesOutput = std::tuple<at::Tensor, at::Tensor>;

OpsAudioFramesOutput make_ops_audio_frames_output(
    SingleStreamDecoder::AudioFramesOutput& audio_frames) {
  return std::make_tuple(
      audio_frames.data,
      torch::tensor(audioFrames.ptsSeconds, torch::dtype(torch::kFloat64)));
}

std::string quote_value(const std::string& value) {
  return "\"" + value + "\"";
}

std::string map_to_json(
    const std::map<std::string, std::string>& metadata_map) {
  std::stringstream ss;
  ss << "{\n";
  auto it = metadata_map.begin();
  while (it != metadata_map.end()) {
    ss << "\"" << it->first << "\": " << it->second;
    ++it;
    if (it != metadata_map.end()) {
      ss << ",\n";
    } else {
      ss << "\n";
    }
  }
  ss << "}";

  return ss.str();
}

} // namespace

// ==============================
// Implementations for the operators
// ==============================

// Create a SingleStreamDecoder from file and wrap the pointer in a tensor.
at::Tensor create_from_file(
    std::string_view filename,
    std::optional<std::string_view> seek_mode = std::nullopt) {
  std::string filename_str(filename);

  SingleStreamDecoder::SeekMode real_seek =
      SingleStreamDecoder::SeekMode::exact;
  if (seek_mode.has_value()) {
    real_seek = seek_mode_from_string(seek_mode.value());
  }

  std::unique_ptr<_single_stream_decoder> unique_decoder =
      std::make_unique<_single_stream_decoder>(filename_str, real_seek);

  return wrap_decoder_pointer_to_tensor(std::move(unique_decoder));
}

// Create a SingleStreamDecoder from the actual bytes of a video and wrap the
// pointer in a tensor. The SingleStreamDecoder will decode the provided bytes.
at::Tensor create_from_tensor(
    at::Tensor video_tensor,
    std::optional<std::string_view> seek_mode = std::nullopt) {
  TORCH_CHECK(video_tensor.is_contiguous(), "video_tensor must be contiguous");
  TORCH_CHECK(
      video_tensor.scalar_type() == torch::kUInt8,
      "video_tensor must be k_u_int8");
  void* data = video_tensor.mutable_data_ptr();
  size_t length = video_tensor.numel();

  SingleStreamDecoder::SeekMode real_seek =
      SingleStreamDecoder::SeekMode::exact;
  if (seek_mode.has_value()) {
    real_seek = seek_mode_from_string(seek_mode.value());
  }

  auto context_holder = std::make_unique<_avio_bytes_context>(data, length);

  std::unique_ptr<_single_stream_decoder> unique_decoder =
      std::make_unique<_single_stream_decoder>(
          std::move(context_holder), real_seek);
  return wrap_decoder_pointer_to_tensor(std::move(unique_decoder));
}

at::Tensor _convert_to_tensor(int64_t decoder_ptr) {
  auto decoder = reinterpret_cast<_single_stream_decoder*>(decoder_ptr);
  std::unique_ptr<_single_stream_decoder> unique_decoder(decoder);
  return wrap_decoder_pointer_to_tensor(std::move(unique_decoder));
}

void _add_video_stream(
    at::Tensor& decoder,
    std::optional<int64_t> width = std::nullopt,
    std::optional<int64_t> height = std::nullopt,
    std::optional<int64_t> num_threads = std::nullopt,
    std::optional<std::string_view> dimension_order = std::nullopt,
    std::optional<int64_t> stream_index = std::nullopt,
    std::optional<std::string_view> device = std::nullopt,
    std::optional<std::string_view> color_conversion_library = std::nullopt) {
  SingleStreamDecoder::VideoStreamOptions video_stream_options;
  video_stream_options.width = width;
  video_stream_options.height = height;
  video_stream_options.ffmpeg_thread_count = num_threads;

  if (dimension_order.has_value()) {
    std::string std_dimension_order{dimension_order.value()};
    TORCH_CHECK(stdDimensionOrder == "NHWC" || std_dimension_order == "NCHW");
    video_stream_options.dimension_order = std_dimension_order;
  }
  if (color_conversion_library.has_value()) {
    std::string std_color_conversion_library{color_conversion_library.value()};
    if (stdColorConversionLibrary == "filtergraph") {
      video_stream_options.color_conversion_library =
          SingleStreamDecoder::ColorConversionLibrary::FILTERGRAPH;
    } else if (stdColorConversionLibrary == "swscale") {
      video_stream_options.color_conversion_library =
          SingleStreamDecoder::ColorConversionLibrary::SWSCALE;
    } else {
      throw std::runtime_error(
          "Invalid color_conversion_library=" + std_color_conversion_library +
          ". color_conversion_library must be either filtergraph or swscale.");
    }
  }
  if (device.has_value()) {
    if (device.value() == "cpu") {
      video_stream_options.device = torch::Device(torch::kCPU);
    } else if (device.value().rfind("cuda", 0) == 0) { // starts with "cuda"
      std::string device_str(device.value());
      video_stream_options.device = torch::Device(deviceStr);
    } else {
      throw std::runtime_error(
          "Invalid device=" + std::string(device.value()) +
          ". device must be either cpu or cuda.");
    }
  }

  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  video_decoder->add_video_stream(
      stream_index.value_or(-1), video_stream_options);
}

// Add a new video stream at `stream_index` using the provided options.
void add_video_stream(
    at::Tensor& decoder,
    std::optional<int64_t> width = std::nullopt,
    std::optional<int64_t> height = std::nullopt,
    std::optional<int64_t> num_threads = std::nullopt,
    std::optional<std::string_view> dimension_order = std::nullopt,
    std::optional<int64_t> stream_index = std::nullopt,
    std::optional<std::string_view> device = std::nullopt) {
  _add_video_stream(
      decoder,
      width,
      height,
      num_threads,
      dimension_order,
      stream_index,
      device);
}

void add_audio_stream(
    at::Tensor& decoder,
    std::optional<int64_t> stream_index = std::nullopt,
    std::optional<int64_t> sample_rate = std::nullopt) {
  SingleStreamDecoder::AudioStreamOptions audio_stream_options;
  audio_stream_options.sample_rate = sample_rate;

  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  video_decoder->add_audio_stream(
      stream_index.value_or(-1), audio_stream_options);
}

// Seek to a particular presentation timestamp in the video in seconds.
void seek_to_pts(at::Tensor& decoder, double seconds) {
  auto video_decoder =
      static_cast<_single_stream_decoder*>(decoder.mutable_data_ptr());
  video_decoder->set_cursor_pts_in_seconds(seconds);
}

// Get the next frame from the video as a tuple that has the frame data, pts and
// duration as tensors.
OpsFrameOutput get_next_frame(at::Tensor& decoder) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  SingleStreamDecoder::FrameOutput result;
  try {
    result = video_decoder->get_next_frame();
  } catch (const SingleStreamDecoder::EndOfFileException& e) {
    C10_THROW_ERROR(IndexError, e.what());
  }
  return make_ops_frame_output(result);
}

// Return the frame that is visible at a given timestamp in seconds. Each frame
// in FFMPEG has a presentation timestamp and a duration. The frame visible at a
// given timestamp T has T >= PTS and T < PTS + Duration.
OpsFrameOutput get_frame_at_pts(at::Tensor& decoder, double seconds) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  SingleStreamDecoder::FrameOutput result;
  try {
    result = video_decoder->get_frame_played_at(seconds);
  } catch (const SingleStreamDecoder::EndOfFileException& e) {
    C10_THROW_ERROR(IndexError, e.what());
  }
  return make_ops_frame_output(result);
}

// Return the frame that is visible at a given index in the video.
OpsFrameOutput get_frame_at_index(at::Tensor& decoder, int64_t frame_index) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  auto result = video_decoder->get_frame_at_index(frame_index);
  return make_ops_frame_output(result);
}

// Return the frames at given indices for a given stream
OpsFrameBatchOutput get_frames_at_indices(
    at::Tensor& decoder,
    at::IntArrayRef frame_indices) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  std::vector<int64_t> frame_indices_vec(
      frame_indices.begin(), frame_indices.end());
  auto result = video_decoder->get_frames_at_indices(frame_indices_vec);
  return make_ops_frame_batch_output(result);
}

// Return the frames inside a range as a single stacked Tensor. The range is
// defined as [start, stop).
OpsFrameBatchOutput get_frames_in_range(
    at::Tensor& decoder,
    int64_t start,
    int64_t stop,
    std::optional<int64_t> step = std::nullopt) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  auto result =
      video_decoder->get_frames_in_range(start, stop, step.value_or(1));
  return make_ops_frame_batch_output(result);
}

// Return the frames at given ptss for a given stream
OpsFrameBatchOutput get_frames_by_pts(
    at::Tensor& decoder,
    at::ArrayRef<double> timestamps) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  std::vector<double> timestamps_vec(timestamps.begin(), timestamps.end());
  auto result = video_decoder->get_frames_played_at(timestamps_vec);
  return make_ops_frame_batch_output(result);
}

// Return the frames inside the range as a single stacked Tensor. The range is
// defined as [start_seconds, stop_seconds). The frames are stacked in pts
// order.
OpsFrameBatchOutput get_frames_by_pts_in_range(
    at::Tensor& decoder,
    double start_seconds,
    double stop_seconds) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  auto result =
      video_decoder->get_frames_played_in_range(start_seconds, stop_seconds);
  return make_ops_frame_batch_output(result);
}

OpsAudioFramesOutput get_frames_by_pts_in_range_audio(
    at::Tensor& decoder,
    double start_seconds,
    std::optional<double> stop_seconds = std::nullopt) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  auto result = video_decoder->get_frames_played_in_range_audio(
      start_seconds, stop_seconds);
  return make_ops_audio_frames_output(result);
}

// For testing only. We need to implement this operation as a core library
// function because what we're testing is round-tripping pts values as
// double-precision floating point numbers from C++ to Python and back to C++.
// We want to make sure that the value is preserved exactly, bit-for-bit, during
// this process.
//
// Returns true if for the given decoder, the pts
// value when converted to seconds as a double is exactly pts_seconds_to_test.
// Returns false otherwise.
bool _test_frame_pts_equality(
    at::Tensor& decoder,
    int64_t frame_index,
    double pts_seconds_to_test) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  return pts_seconds_to_test ==
      video_decoder->get_pts_seconds_for_frame(frame_index);
}

torch::Tensor _get_key_frame_indices(at::Tensor& decoder) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  return video_decoder->get_key_frame_indices();
}

// Get the metadata from the video as a string.
std::string get_json_metadata(at::Tensor& decoder) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);

  SingleStreamDecoder::ContainerMetadata video_metadata =
      video_decoder->get_container_metadata();
  auto maybe_best_video_stream_index = video_metadata.best_video_stream_index;

  std::map<std::string, std::string> metadata_map;
  // serialize the metadata into a string std::stringstream ss;
  double duration_seconds = 0;
  if (maybeBestVideoStreamIndex.has_value() &&
      video_metadata.all_stream_metadata[*maybe_best_video_stream_index]
          .durationSeconds.has_value()) {
    duration_seconds =
        video_metadata.all_stream_metadata[*maybe_best_video_stream_index]
            .durationSeconds.value_or(0);
  } else {
    // Fallback to container-level duration if stream duration is not found.
    duration_seconds = video_metadata.duration_seconds.value_or(0);
  }
  metadata_map["duration_seconds"] = std::to_string(duration_seconds);

  if (videoMetadata.bitRate.has_value()) {
    metadata_map["bit_rate"] = std::to_string(video_metadata.bit_rate.value());
  }

  if (maybeBestVideoStreamIndex.has_value()) {
    auto stream_metadata =
        video_metadata.all_stream_metadata[*maybe_best_video_stream_index];
    if (streamMetadata.numFramesFromScan.has_value()) {
      metadata_map["num_frames"] =
          std::to_string(*stream_metadata.num_frames_from_scan);
    } else if (streamMetadata.numFrames.has_value()) {
      metadata_map["num_frames"] = std::to_string(*stream_metadata.num_frames);
    }
    if (streamMetadata.minPtsSecondsFromScan.has_value()) {
      metadata_map["min_pts_seconds_from_scan"] =
          std::to_string(*stream_metadata.min_pts_seconds_from_scan);
    }
    if (streamMetadata.maxPtsSecondsFromScan.has_value()) {
      metadata_map["max_pts_seconds_from_scan"] =
          std::to_string(*stream_metadata.max_pts_seconds_from_scan);
    }
    if (streamMetadata.codecName.has_value()) {
      metadata_map["codec"] = quote_value(stream_metadata.codec_name.value());
    }
    if (streamMetadata.width.has_value()) {
      metadata_map["width"] = std::to_string(*stream_metadata.width);
    }
    if (streamMetadata.height.has_value()) {
      metadata_map["height"] = std::to_string(*stream_metadata.height);
    }
    if (streamMetadata.averageFps.has_value()) {
      metadata_map["average_fps"] =
          std::to_string(*stream_metadata.average_fps);
    }
  }
  if (videoMetadata.bestVideoStreamIndex.has_value()) {
    metadata_map["best_video_stream_index"] =
        std::to_string(*video_metadata.best_video_stream_index);
  }
  if (videoMetadata.bestAudioStreamIndex.has_value()) {
    metadata_map["best_audio_stream_index"] =
        std::to_string(*video_metadata.best_audio_stream_index);
  }

  return map_to_json(metadata_map);
}

// Get the container metadata as a string.
std::string get_container_json_metadata(at::Tensor& decoder) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);

  auto container_metadata = video_decoder->get_container_metadata();

  std::map<std::string, std::string> map;

  if (containerMetadata.durationSeconds.has_value()) {
    map["duration_seconds"] =
        std::to_string(*container_metadata.duration_seconds);
  }

  if (containerMetadata.bitRate.has_value()) {
    map["bit_rate"] = std::to_string(*container_metadata.bit_rate);
  }

  if (containerMetadata.bestVideoStreamIndex.has_value()) {
    map["best_video_stream_index"] =
        std::to_string(*container_metadata.best_video_stream_index);
  }
  if (containerMetadata.bestAudioStreamIndex.has_value()) {
    map["best_audio_stream_index"] =
        std::to_string(*container_metadata.best_audio_stream_index);
  }

  map["num_streams"] =
      std::to_string(container_metadata.all_stream_metadata.size());

  return map_to_json(map);
}

// Get the stream metadata as a string.
std::string get_stream_json_metadata(
    at::Tensor& decoder,
    int64_t stream_index) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  auto all_stream_metadata =
      video_decoder->get_container_metadata().all_stream_metadata;
  if (stream_index < 0 ||
      stream_index >= static_cast<int64_t>(all_stream_metadata.size())) {
    throw std::out_of_range(
        "stream_index out of bounds: " + std::to_string(stream_index));
  }
  auto stream_metadata = all_stream_metadata[stream_index];

  std::map<std::string, std::string> map;

  if (streamMetadata.durationSeconds.has_value()) {
    map["duration_seconds"] = std::to_string(*stream_metadata.duration_seconds);
  }
  if (streamMetadata.bitRate.has_value()) {
    map["bit_rate"] = std::to_string(*stream_metadata.bit_rate);
  }
  if (streamMetadata.numFramesFromScan.has_value()) {
    map["num_frames_from_scan"] =
        std::to_string(*stream_metadata.num_frames_from_scan);
  }
  if (streamMetadata.numFrames.has_value()) {
    map["num_frames"] = std::to_string(*stream_metadata.num_frames);
  }
  if (streamMetadata.beginStreamFromHeader.has_value()) {
    map["begin_stream_from_header"] =
        std::to_string(*stream_metadata.begin_stream_from_header);
  }
  if (streamMetadata.minPtsSecondsFromScan.has_value()) {
    map["min_pts_seconds_from_scan"] =
        std::to_string(*stream_metadata.min_pts_seconds_from_scan);
  }
  if (streamMetadata.maxPtsSecondsFromScan.has_value()) {
    map["max_pts_seconds_from_scan"] =
        std::to_string(*stream_metadata.max_pts_seconds_from_scan);
  }
  if (streamMetadata.codecName.has_value()) {
    map["codec"] = quote_value(stream_metadata.codec_name.value());
  }
  if (streamMetadata.width.has_value()) {
    map["width"] = std::to_string(*stream_metadata.width);
  }
  if (streamMetadata.height.has_value()) {
    map["height"] = std::to_string(*stream_metadata.height);
  }
  if (streamMetadata.averageFps.has_value()) {
    map["average_fps"] = std::to_string(*stream_metadata.average_fps);
  }
  if (streamMetadata.sampleRate.has_value()) {
    map["sample_rate"] = std::to_string(*stream_metadata.sample_rate);
  }
  if (streamMetadata.numChannels.has_value()) {
    map["num_channels"] = std::to_string(*stream_metadata.num_channels);
  }
  if (streamMetadata.sampleFormat.has_value()) {
    map["sample_format"] = quote_value(stream_metadata.sample_format.value());
  }
  if (streamMetadata.mediaType == AVMEDIA_TYPE_VIDEO) {
    map["media_type"] = quote_value("video");
  } else if (streamMetadata.mediaType == AVMEDIA_TYPE_AUDIO) {
    map["media_type"] = quote_value("audio");
  } else {
    map["media_type"] = quote_value("other");
  }
  return map_to_json(map);
}

// Returns version information about the various FFMPEG libraries that are
// loaded in the program's address space.
std::string _get_json_ffmpeg_library_versions() {
  std::stringstream ss;
  ss << "{\n";

  unsigned int version = avfilter_version();
  ss << "\"libavfilter\": [" << AV_VERSION_MAJOR(version) << ", "
     << AV_VERSION_MINOR(version) << ", " << AV_VERSION_MICRO(version)
     << "],\n";
  version = avutil_version();
  ss << "\"libavutil\": [" << AV_VERSION_MAJOR(version) << ", "
     << AV_VERSION_MINOR(version) << ", " << AV_VERSION_MICRO(version)
     << "],\n";
  version = avcodec_version();
  ss << "\"libavcodec\": [" << AV_VERSION_MAJOR(version) << ", "
     << AV_VERSION_MINOR(version) << ", " << AV_VERSION_MICRO(version)
     << "],\n";
  version = avformat_version();
  ss << "\"libavformat\": [" << AV_VERSION_MAJOR(version) << ", "
     << AV_VERSION_MINOR(version) << ", " << AV_VERSION_MICRO(version)
     << "],\n";
  ss << "\"ffmpeg_version\": \"" << av_version_info() << "\"\n";
  ss << "}\n";

  return ss.str();
}

// Scans video packets to get more accurate metadata like frame count, exact
// keyframe positions, etc. Exact keyframe positions are useful for efficient
// accurate seeking. Note that this function reads the entire video but it does
// not decode frames. Reading a video file is much cheaper than decoding it.
void scan_all_streams_to_update_metadata(at::Tensor& decoder) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  video_decoder->scan_file_and_update_metadata_and_index();
}

TORCH_LIBRARY_IMPL(torchcodec_ns, BackendSelect, m) {
  m.impl("create_from_file", &create_from_file);
  m.impl("create_from_tensor", &create_from_tensor);
  m.impl("_convert_to_tensor", &_convert_to_tensor);
  m.impl(
      "_get_json_ffmpeg_library_versions", &_get_json_ffmpeg_library_versions);
}

TORCH_LIBRARY_IMPL(torchcodec_ns, CPU, m) {
  m.impl("seek_to_pts", &seek_to_pts);
  m.impl("add_video_stream", &add_video_stream);
  m.impl("_add_video_stream", &_add_video_stream);
  m.impl("add_audio_stream", &add_audio_stream);
  m.impl("get_next_frame", &get_next_frame);
  m.impl("_get_key_frame_indices", &_get_key_frame_indices);
  m.impl("get_json_metadata", &get_json_metadata);
  m.impl("get_container_json_metadata", &get_container_json_metadata);
  m.impl("get_stream_json_metadata", &get_stream_json_metadata);
  m.impl("get_frame_at_pts", &get_frame_at_pts);
  m.impl("get_frame_at_index", &get_frame_at_index);
  m.impl("get_frames_at_indices", &get_frames_at_indices);
  m.impl("get_frames_in_range", &get_frames_in_range);
  m.impl("get_frames_by_pts_in_range", &get_frames_by_pts_in_range);
  m.impl("get_frames_by_pts_in_range_audio", &get_frames_by_pts_in_range_audio);
  m.impl("get_frames_by_pts", &get_frames_by_pts);
  m.impl("_test_frame_pts_equality", &_test_frame_pts_equality);
  m.impl(
      "scan_all_streams_to_update_metadata",
      &scan_all_streams_to_update_metadata);
}

} // namespace facebook::torchcodec
