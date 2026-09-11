// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// fmt and pybind11 headers from torch error out if TORCH_TARGET_VERSION is
// defined, so we temporarily undefine it.
// See https://github.com/pytorch/pytorch/pull/174372 for context
#pragma push_macro("TORCH_TARGET_VERSION")
#undef TORCH_TARGET_VERSION
#include <fmt/format.h>
#include <pybind11/pybind11.h>
#pragma pop_macro("TORCH_TARGET_VERSION")
#include <cstdint>
#include <string>

extern "C" {
#include <libavutil/pixdesc.h>
}

#include "AVIOContextHolder.h"
#include "AudioConverter.h"
#include "ColorConverter.h"
#include "Demuxer.h"
#include "Encoder.h"
#include "FileIO.h"
#include "IOInterface.h"
#include "Logging.h"
#include "NVDECCacheConfig.h"
#include "PacketDecoder.h"
#include "SingleStreamDecoder.h"
#include "StableABICompat.h"
#include "TensorIO.h"
#include "ValidationUtils.h"
#include "WavDecoder.h"

namespace facebook::torchcodec {

// ==============================
// Define the operators
// ==============================
// All instances of accepting the decoder as a tensor must be annotated with
// `Tensor(a!)`. The `(a!)` part normally indicates that the tensor is being
// mutated in place. We need it to make sure that torch.compile does not reorder
// calls to these functions. For more detail, see:
//   https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native#readme
//
STABLE_TORCH_LIBRARY_FRAGMENT(torchcodec_ns, m) {
  m.def("create_from_file(str filename, str? seek_mode=None) -> Tensor");
  m.def(
      "create_from_tensor(Tensor video_tensor, str? seek_mode=None) -> Tensor");
  m.def(
      "_create_from_file_like(int file_like_context, str? seek_mode=None) -> Tensor");
  m.def(
      "_add_video_stream(Tensor(a!) decoder, *, int? num_threads=None, str? dimension_order=None, int? stream_index=None, str device=\"cpu\", str device_variant=\"default\", str transform_specs=\"\", Tensor? custom_frame_mappings_pts=None, Tensor? custom_frame_mappings_duration=None, Tensor? custom_frame_mappings_keyframe_indices=None, str? color_conversion_library=None, str output_dtype=\"uint8\") -> ()");
  m.def(
      "add_video_stream(Tensor(a!) decoder, *, int? num_threads=None, str? dimension_order=None, int? stream_index=None, str device=\"cpu\", str device_variant=\"default\", str transform_specs=\"\", Tensor? custom_frame_mappings_pts=None, Tensor? custom_frame_mappings_duration=None, Tensor? custom_frame_mappings_keyframe_indices=None, str output_dtype=\"uint8\") -> ()");
  m.def(
      "add_audio_stream(Tensor(a!) decoder, *, int? stream_index=None, int? sample_rate=None, int? num_channels=None) -> ()");
  m.def("seek_to_pts(Tensor(a!) decoder, float seconds) -> ()");
  m.def("get_next_frame(Tensor(a!) decoder) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frame_at_pts(Tensor(a!) decoder, float seconds) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frame_at_index(Tensor(a!) decoder, *, int frame_index) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frames_at_indices(Tensor(a!) decoder, *, Tensor frame_indices) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frames_in_range(Tensor(a!) decoder, *, int start, int stop, int? step=None) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frames_by_pts_in_range(Tensor(a!) decoder, *, float start_seconds, float stop_seconds, float? fps=None) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frames_by_pts_in_range_audio(Tensor(a!) decoder, *, float start_seconds, float? stop_seconds) -> (Tensor, Tensor)");
  m.def(
      "get_frames_by_pts(Tensor(a!) decoder, *, Tensor timestamps) -> (Tensor, Tensor, Tensor)");
  m.def("_blocks_create_demuxer_from_file(str filename) -> Tensor");
  m.def("_blocks_create_demuxer_from_tensor(Tensor video_tensor) -> Tensor");
  m.def(
      "_blocks_create_demuxer_from_file_like(int file_like_context) -> Tensor");
  m.def(
      "_blocks_demuxer_add_stream(Tensor(a!) demuxer, int? stream_index=None, str? media_type=None) -> (int, str)");
  m.def(
      "_blocks_demuxer_get_audio_video_stream_indices(Tensor demuxer) -> Tensor");
  m.def("_blocks_demuxer_container_json_metadata(Tensor demuxer) -> str");
  m.def(
      "_blocks_demuxer_stream_json_metadata(Tensor demuxer, int stream_index) -> str");
  m.def(
      "_blocks_demuxer_next_packet(Tensor(a!) demuxer) -> (Tensor, bool, int)");
  m.def(
      "_blocks_demuxer_seek(Tensor(a!) demuxer, float seconds, int? stream_index=None) -> ()");
  m.def(
      "_blocks_demuxer_scan(Tensor(a!) demuxer, int? stream_index=None) -> (Tensor, Tensor, Tensor, int, int)");
  m.def(
      "_blocks_create_packet_decoder(Tensor demuxer, *, int? stream_index=None, int? num_threads=None, str device=\"cpu\") -> Tensor");
  m.def(
      "_blocks_packet_decoder_send_packet(Tensor(a!) decoder, Tensor packet) -> int");
  m.def("_blocks_packet_decoder_send_eof(Tensor(a!) decoder) -> int");
  m.def("_blocks_packet_decoder_reset(Tensor(a!) decoder) -> ()");
  m.def(
      "_blocks_packet_decoder_receive_frame(Tensor(a!) decoder) -> (Tensor, int, float, float, Device, Tensor)");
  m.def(
      "_blocks_audio_packet_decoder_receive_frame(Tensor(a!) decoder) -> (Tensor, int, float, float, int, str)");
  m.def(
      "_blocks_create_color_converter(str device=\"cpu\", str output_dtype=\"uint8\") -> Tensor");
  m.def(
      "_blocks_create_audio_converter(int? sample_rate=None, int? num_channels=None) -> Tensor");
  m.def(
      "_blocks_audio_converter_convert(Tensor(a!) converter, Tensor samples, int sample_rate) -> Tensor");
  m.def("_blocks_audio_converter_drain(Tensor(a!) converter) -> Tensor");
  m.def("_blocks_audio_converter_reset(Tensor(a!) converter) -> ()");
  m.def(
      "_blocks_convert_frame(Tensor(a!) converter, Tensor frame, Device device) -> Tensor");
  m.def(
      "_blocks_frame_metadata(Tensor frame) -> (str, str, str, int, int, int, float)");
  m.def(
      "_blocks_frame_planes(Tensor frame, Device device) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("_get_key_frame_indices(Tensor(a!) decoder) -> Tensor");
  m.def("get_json_metadata(Tensor(a!) decoder) -> str");
  m.def("get_container_json_metadata(Tensor(a!) decoder) -> str");
  m.def(
      "get_stream_json_metadata(Tensor(a!) decoder, int stream_index) -> str");
  m.def("_get_json_ffmpeg_library_versions() -> str");
  m.def("_get_backend_details(Tensor(a!) decoder) -> str");
  m.def(
      "_test_frame_pts_equality(Tensor(a!) decoder, *, int frame_index, float pts_seconds_to_test) -> bool");
  m.def("scan_all_streams_to_update_metadata(Tensor(a!) decoder) -> ()");
  m.def("create_streaming_encoder() -> Tensor");
  m.def("streaming_encoder_close(Tensor(a!) encoder) -> ()");
  m.def(
      "streaming_encoder_add_video_stream(Tensor(a!) encoder, int height, int width, float frame_rate, str device=\"cpu\", str? codec=None, str? pixel_format=None, float? crf=None, str? preset=None, str[]? extra_options=None) -> int");
  m.def(
      "streaming_encoder_add_audio_stream(Tensor(a!) encoder, int sample_rate, int num_channels, int? bit_rate=None, int? output_num_channels=None, int? output_sample_rate=None) -> int");
  m.def("streaming_encoder_open_file(Tensor(a!) encoder, str filename) -> ()");
  m.def(
      "streaming_encoder_open_file_like(Tensor(a!) encoder, str format, int file_like_context) -> ()");
  m.def(
      "streaming_encoder_add_frames(Tensor(a!) encoder, Tensor frames, int stream_index) -> ()");
  m.def(
      "streaming_encoder_add_samples(Tensor(a!) encoder, Tensor samples, int stream_index) -> ()");
  m.def("set_nvdec_cache_capacity(int capacity) -> ()");
  m.def("get_nvdec_cache_capacity() -> int");
  m.def("_get_nvdec_cache_size(int device_index) -> int");
  m.def("_set_cpp_log_level(int level) -> ()");
  m.def("_get_log_level() -> int");
  m.def("create_wav_decoder_from_file(str filename) -> Tensor");
  m.def("create_wav_decoder_from_tensor(Tensor data) -> Tensor");
  m.def("_create_wav_decoder_from_file_like(int file_like_context) -> Tensor");
  m.def(
      "get_wav_samples_in_range(Tensor(a!) decoder, float start_seconds, float? stop_seconds) -> (Tensor, Tensor)");
  m.def("get_wav_metadata_from_decoder(Tensor(a!) decoder) -> str");
}

namespace {

torch::stable::Tensor wrap_decoder_pointer_to_tensor(
    std::unique_ptr<SingleStreamDecoder> unique_decoder) {
  SingleStreamDecoder* decoder = unique_decoder.release();

  auto deleter = [decoder](void*) { delete decoder; };
  int64_t sizes[] = {static_cast<int64_t>(sizeof(SingleStreamDecoder*))};
  int64_t strides[] = {1};
  torch::stable::Tensor tensor = torch::stable::from_blob(
      decoder,
      {sizes, 1},
      {strides, 1},
      StableDevice(kStableCPU),
      kStableInt64,
      deleter);
  auto video_decoder =
      static_cast<SingleStreamDecoder*>(tensor.mutable_data_ptr());
  STD_TORCH_CHECK(video_decoder == decoder, "videoDecoder != decoder");
  return tensor;
}

torch::stable::Tensor wrap_wav_decoder_pointer_to_tensor(
    std::unique_ptr<WavDecoder> unique_decoder) {
  WavDecoder* decoder = unique_decoder.release();

  auto deleter = [decoder](void*) { delete decoder; };
  int64_t sizes[] = {static_cast<int64_t>(sizeof(WavDecoder*))};
  int64_t strides[] = {1};
  torch::stable::Tensor tensor = torch::stable::from_blob(
      decoder,
      {sizes, 1},
      {strides, 1},
      StableDevice(kStableCPU),
      kStableInt64,
      deleter);
  auto wav_decoder = static_cast<WavDecoder*>(tensor.mutable_data_ptr());
  STD_TORCH_CHECK(wav_decoder == decoder, "wavDecoder != decoder");
  return tensor;
}

WavDecoder* unwrap_tensor_to_get_wav_decoder(torch::stable::Tensor& tensor) {
  STD_TORCH_CHECK(
      tensor.is_contiguous(),
      "fake decoder tensor must be contiguous! This is an internal error, please report on the torchcodec issue tracker.");
  WavDecoder* decoder = static_cast<WavDecoder*>(tensor.mutable_data_ptr());
  return decoder;
}

SingleStreamDecoder* unwrap_tensor_to_get_decoder(
    torch::stable::Tensor& tensor) {
  STD_TORCH_CHECK(
      tensor.is_contiguous(),
      "fake decoder tensor must be contiguous! This is an internal error, please report on the torchcodec issue tracker.");
  void* buffer = tensor.mutable_data_ptr();
  SingleStreamDecoder* decoder = static_cast<SingleStreamDecoder*>(buffer);
  return decoder;
}

// Generic pointer<->tensor laundering for the building-block handle types
// (Demuxer / PacketDecoder / ColorConverter / AVPacket / AVFrame). Same trick
// as wrap_decoder_pointer_to_tensor: the tensor's data pointer IS the raw
// pointer, and the tensor is the owner. Taking the unique_ptr by value is what
// makes the ownership transfer explicit at the call site.
template <typename T, typename D>
torch::stable::Tensor wrap_pointer_to_tensor(std::unique_ptr<T, D> ptr) {
  D object_deleter = ptr.get_deleter();
  T* raw = ptr.release();
  auto deleter = [raw, object_deleter](void*) { object_deleter(raw); };
  int64_t sizes[] = {static_cast<int64_t>(sizeof(T*))};
  int64_t strides[] = {1};
  return torch::stable::from_blob(
      raw,
      {sizes, 1},
      {strides, 1},
      StableDevice(kStableCPU),
      kStableInt64,
      deleter);
}

template <typename T>
T* unwrap_tensor_to_pointer(torch::stable::Tensor& tensor) {
  STD_TORCH_CHECK(tensor.is_contiguous(), "handle tensor must be contiguous");
  return static_cast<T*>(tensor.mutable_data_ptr());
}

torch::stable::Tensor wrap_multi_stream_encoder_pointer_to_tensor(
    std::unique_ptr<MultiStreamEncoder> unique_encoder) {
  MultiStreamEncoder* encoder = unique_encoder.release();
  auto deleter = [encoder](void*) { delete encoder; };
  int64_t sizes[] = {static_cast<int64_t>(sizeof(MultiStreamEncoder*))};
  int64_t strides[] = {1};
  torch::stable::Tensor tensor = torch::stable::from_blob(
      encoder,
      {sizes, 1},
      {strides, 1},
      StableDevice(kStableCPU),
      kStableInt64,
      deleter);
  auto multi_stream_encoder =
      static_cast<MultiStreamEncoder*>(tensor.mutable_data_ptr());
  STD_TORCH_CHECK(
      multi_stream_encoder == encoder, "multiStreamEncoder != encoder");
  return tensor;
}

MultiStreamEncoder* unwrap_tensor_to_get_multi_stream_encoder(
    torch::stable::Tensor& tensor) {
  STD_TORCH_CHECK(
      tensor.is_contiguous(),
      "fake encoder tensor must be contiguous! This is an internal error, please report on the torchcodec issue tracker.");
  void* buffer = tensor.mutable_data_ptr();
  MultiStreamEncoder* encoder = static_cast<MultiStreamEncoder*>(buffer);
  return encoder;
}

// The elements of this tuple are all tensors that represent a single frame:
//   1. The frame data, which is a multidimensional tensor.
//   2. A single float value for the pts in seconds.
//   3. A single float value for the duration in seconds.
// The reason we use Tensors for the second and third values is so we can run
// under torch.compile().
using OpsFrameOutput = std::
    tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor>;

OpsFrameOutput make_ops_frame_output(FrameOutput& frame) {
  return std::make_tuple(
      frame.data,
      torch::stable::full({}, frame.pts_seconds, kStableFloat64),
      torch::stable::full({}, frame.duration_seconds, kStableFloat64));
}

// All elements of this tuple are tensors of the same leading dimension. The
// tuple represents the frames for N total frames, where N is the dimension of
// each stacked tensor. The elments are:
//   1. Stacked tensor of data for all N frames. Each frame is also a
//   multidimensional tensor.
//   2. Tensor of N pts values in seconds, where each pts is a single
//   float.
//   3. Tensor of N durationis in seconds, where each duration is a
//   single float.
using OpsFrameBatchOutput = std::
    tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor>;

OpsFrameBatchOutput make_ops_frame_batch_output(FrameBatchOutput& batch) {
  return std::make_tuple(batch.data, batch.pts_seconds, batch.duration_seconds);
}

// The elements of this tuple are all tensors that represent the concatenation
// of multiple audio frames:
//   1. The frames data (concatenated)
//   2. A single float value for the pts of the first frame, in seconds.
using OpsAudioFramesOutput =
    std::tuple<torch::stable::Tensor, torch::stable::Tensor>;

OpsAudioFramesOutput make_ops_audio_frames_output(
    AudioFramesOutput& audio_frames) {
  return std::make_tuple(
      audio_frames.data,
      torch::stable::full({}, audio_frames.pts_seconds, kStableFloat64));
}

std::string quote_value(const std::string& value) {
  return "\"" + value + "\"";
}

// Helper function to unflatten extra_options, alternating keys and values
std::map<std::string, std::string> unflatten_extra_options(
    const std::vector<std::string>& opts) {
  std::map<std::string, std::string> options_map;
  for (size_t i = 0; i < opts.size(); i += 2) {
    options_map[opts[i]] = opts[i + 1];
  }
  return options_map;
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

SeekMode seek_mode_from_string(std::string_view seek_mode) {
  if (seek_mode == "exact") {
    return SeekMode::exact;
  } else if (seek_mode == "approximate") {
    return SeekMode::approximate;
  } else if (seek_mode == "custom_frame_mappings") {
    return SeekMode::custom_frame_mappings;
  } else {
    STD_TORCH_CHECK(false, "Invalid seek mode: " + std::string(seek_mode));
  }
}

void write_fallback_based_metadata(
    std::map<std::string, std::string>& map,
    const StreamMetadata& stream_metadata,
    SeekMode seek_mode) {
  auto duration_seconds = stream_metadata.get_duration_seconds(seek_mode);
  if (duration_seconds.has_value()) {
    map["durationSeconds"] = fmt::to_string(duration_seconds.value());
  }

  auto num_frames = stream_metadata.get_num_frames(seek_mode);
  if (num_frames.has_value()) {
    map["numFrames"] = std::to_string(num_frames.value());
  }

  double begin_stream_seconds =
      stream_metadata.get_begin_stream_seconds(seek_mode);
  map["beginStreamSeconds"] = fmt::to_string(begin_stream_seconds);

  auto end_stream_seconds = stream_metadata.get_end_stream_seconds(seek_mode);
  if (end_stream_seconds.has_value()) {
    map["endStreamSeconds"] = fmt::to_string(end_stream_seconds.value());
  }

  auto average_fps = stream_metadata.get_average_fps(seek_mode);
  if (average_fps.has_value()) {
    map["averageFps"] = fmt::to_string(average_fps.value());
  }
}

int checked_to_positive_int(const std::string& str) {
  int ret = 0;
  try {
    ret = std::stoi(str);
  } catch (const std::invalid_argument&) {
    STD_TORCH_CHECK(false, "String cannot be converted to an int:" + str);
  } catch (const std::out_of_range&) {
    STD_TORCH_CHECK(false, "String would become integer out of range:" + str);
  }
  STD_TORCH_CHECK(ret > 0, "String must be a positive integer:" + str);
  return ret;
}

int checked_to_non_negative_int(const std::string& str) {
  int ret = 0;
  try {
    ret = std::stoi(str);
  } catch (const std::invalid_argument&) {
    STD_TORCH_CHECK(false, "String cannot be converted to an int:" + str);
  } catch (const std::out_of_range&) {
    STD_TORCH_CHECK(false, "String would become integer out of range:" + str);
  }
  STD_TORCH_CHECK(ret >= 0, "String must be a non-negative integer:" + str);
  return ret;
}

// Resize transform specs take the form:
//
//   "resize, <height>, <width>"
//
// Where "resize" is the string literal and <height> and <width> are positive
// integers.
Transform* make_resize_transform(
    const std::vector<std::string>& resize_transform_spec) {
  STD_TORCH_CHECK(
      resize_transform_spec.size() == 3,
      "resizeTransformSpec must have 3 elements including its name");
  int height = checked_to_positive_int(resize_transform_spec[1]);
  int width = checked_to_positive_int(resize_transform_spec[2]);
  return new ResizeTransform(FrameDims(height, width));
}

// Crop transform specs take the form:
//
//   "crop, <height>, <width>, <x>, <y>"
//
// Where "crop" is the string literal and <height>, <width>, <x> and <y> are
// positive integers. <x> and <y> are the x and y coordinates of the top left
// corner of the crop. Note that we follow the PyTorch convention of (height,
// width) for specifying image dimensions; FFmpeg uses (width, height).
Transform* make_crop_transform(
    const std::vector<std::string>& crop_transform_spec) {
  STD_TORCH_CHECK(
      crop_transform_spec.size() == 5,
      "cropTransformSpec must have 5 elements including its name");
  int height = checked_to_positive_int(crop_transform_spec[1]);
  int width = checked_to_positive_int(crop_transform_spec[2]);
  int x = checked_to_non_negative_int(crop_transform_spec[3]);
  int y = checked_to_non_negative_int(crop_transform_spec[4]);
  return new CropTransform(FrameDims(height, width), x, y);
}

// CenterCrop transform specs take the form:
//
//   "center_crop, <height>, <width>"
//
// Where "center_crop" is the string literal and <height>, <width> are
// positive integers. Note that we follow the PyTorch convention of (height,
// width) for specifying image dimensions; FFmpeg uses (width, height).
Transform* make_center_crop_transform(
    const std::vector<std::string>& crop_transform_spec) {
  STD_TORCH_CHECK(
      crop_transform_spec.size() == 3,
      "cropTransformSpec must have 3 elements including its name");
  int height = checked_to_positive_int(crop_transform_spec[1]);
  int width = checked_to_positive_int(crop_transform_spec[2]);
  return new CropTransform(FrameDims(height, width));
}

std::vector<std::string> split(const std::string& str, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream token_stream(str);
  while (std::getline(token_stream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

// The transformSpecsRaw string is always in the format:
//
//   "name1, param1, param2, ...; name2, param1, param2, ...; ..."
//
// Where "nameX" is the name of the transform, and "paramX" are the parameters.
std::vector<Transform*> make_transforms(
    const std::string& transform_specs_raw) {
  std::vector<Transform*> transforms;
  std::vector<std::string> transform_specs = split(transform_specs_raw, ';');
  for (const std::string& transform_spec_raw : transform_specs) {
    std::vector<std::string> transform_spec = split(transform_spec_raw, ',');
    STD_TORCH_CHECK(
        transform_spec.size() >= 1,
        "Invalid transform spec: " + transform_spec_raw);

    auto name = transform_spec[0];
    if (name == "resize") {
      transforms.push_back(make_resize_transform(transform_spec));
    } else if (name == "crop") {
      transforms.push_back(make_crop_transform(transform_spec));
    } else if (name == "center_crop") {
      transforms.push_back(make_center_crop_transform(transform_spec));
    } else {
      STD_TORCH_CHECK(false, "Invalid transform name: " + name);
    }
  }
  return transforms;
}

} // namespace

// ==============================
// Implementations for the operators
// ==============================

// Create a SingleStreamDecoder from file and wrap the pointer in a tensor.
torch::stable::Tensor create_from_file(
    std::string filename,
    std::optional<std::string> seek_mode = std::nullopt) {
  SeekMode real_seek = SeekMode::exact;
  if (seek_mode.has_value()) {
    real_seek = seek_mode_from_string(seek_mode.value());
  }

  std::unique_ptr<SingleStreamDecoder> unique_decoder =
      std::make_unique<SingleStreamDecoder>(filename, real_seek);

  return wrap_decoder_pointer_to_tensor(std::move(unique_decoder));
}

// Create a SingleStreamDecoder from the actual bytes of a video and wrap the
// pointer in a tensor. The SingleStreamDecoder will decode the provided bytes.
torch::stable::Tensor create_from_tensor(
    const torch::stable::Tensor& video_tensor,
    std::optional<std::string> seek_mode = std::nullopt) {
  STD_TORCH_CHECK(
      video_tensor.is_contiguous(), "video_tensor must be contiguous");
  STD_TORCH_CHECK(
      video_tensor.scalar_type() == kStableUInt8,
      "video_tensor must be kUInt8");

  SeekMode real_seek = SeekMode::exact;
  if (seek_mode.has_value()) {
    real_seek = seek_mode_from_string(seek_mode.value());
  }

  auto avio_context_holder = std::make_unique<AVIOContextHolder>(
      std::make_unique<TensorReadIO>(video_tensor), /*is_for_writing=*/false);

  std::unique_ptr<SingleStreamDecoder> unique_decoder =
      std::make_unique<SingleStreamDecoder>(
          std::move(avio_context_holder), real_seek);
  return wrap_decoder_pointer_to_tensor(std::move(unique_decoder));
}

torch::stable::Tensor _create_from_file_like(
    int64_t file_like_context,
    std::optional<std::string> seek_mode) {
  auto file_like_context_ptr =
      reinterpret_cast<IOInterface*>(file_like_context);
  STD_TORCH_CHECK(
      file_like_context_ptr != nullptr,
      "file_like_context must be a valid pointer");
  auto avio_context_holder = std::make_unique<AVIOContextHolder>(
      std::unique_ptr<IOInterface>(file_like_context_ptr),
      /*is_for_writing=*/false);

  SeekMode real_seek = SeekMode::exact;
  if (seek_mode.has_value()) {
    real_seek = seek_mode_from_string(seek_mode.value());
  }

  std::unique_ptr<SingleStreamDecoder> unique_decoder =
      std::make_unique<SingleStreamDecoder>(
          std::move(avio_context_holder), real_seek);
  return wrap_decoder_pointer_to_tensor(std::move(unique_decoder));
}

OutputDtypeConfig parse_output_dtype_config(const std::string& output_dtype) {
  if (output_dtype == "uint8") {
    return OutputDtypeConfig::UINT8;
  } else if (output_dtype == "float32") {
    return OutputDtypeConfig::FLOAT32;
  } else if (output_dtype == "auto") {
    return OutputDtypeConfig::AUTO;
  }
  STD_TORCH_CHECK(
      false,
      "Invalid output_dtype=",
      output_dtype,
      ". Supported values are: uint8, float32, auto.");
}

void _add_video_stream(
    torch::stable::Tensor& decoder,
    std::optional<int64_t> num_threads = std::nullopt,
    std::optional<std::string> dimension_order = std::nullopt,
    std::optional<int64_t> stream_index = std::nullopt,
    std::string device = "cpu",
    std::string device_variant = "default",
    std::string transform_specs = "",
    std::optional<torch::stable::Tensor> custom_frame_mappings_pts =
        std::nullopt,
    std::optional<torch::stable::Tensor> custom_frame_mappings_duration =
        std::nullopt,
    std::optional<torch::stable::Tensor>
        custom_frame_mappings_keyframe_indices = std::nullopt,
    std::optional<std::string> color_conversion_library = std::nullopt,
    std::string output_dtype = "uint8") {
  VideoStreamOptions video_stream_options;
  video_stream_options.ffmpeg_thread_count = num_threads;

  video_stream_options.output_dtype_config =
      parse_output_dtype_config(output_dtype);

  if (dimension_order.has_value()) {
    STD_TORCH_CHECK(
        *dimension_order == "NHWC" || *dimension_order == "NCHW",
        "dimension_order must be NHWC or NCHW");
    video_stream_options.dimension_order = std::move(*dimension_order);
  }
  if (color_conversion_library.has_value()) {
    const std::string& std_color_conversion_library =
        color_conversion_library.value();
    if (std_color_conversion_library == "filtergraph") {
      video_stream_options.color_conversion_library =
          ColorConversionLibrary::FILTERGRAPH;
    } else if (std_color_conversion_library == "swscale") {
      video_stream_options.color_conversion_library =
          ColorConversionLibrary::SWSCALE;
    } else {
      STD_TORCH_CHECK(
          false,
          "Invalid color_conversion_library=",
          std_color_conversion_library,
          ". color_conversion_library must be either filtergraph or swscale.");
    }
  }

  validate_device_interface(device, device_variant);

  video_stream_options.device = StableDevice(std::move(device));
  video_stream_options.device_variant = std::move(device_variant);

  std::vector<Transform*> transforms =
      make_transforms(std::move(transform_specs));

  bool has_pts = custom_frame_mappings_pts.has_value();
  bool has_duration = custom_frame_mappings_duration.has_value();
  bool has_keyframe_indices =
      custom_frame_mappings_keyframe_indices.has_value();
  STD_TORCH_CHECK(
      (has_pts == has_duration) && (has_duration == has_keyframe_indices),
      "custom_frame_mappings_pts, custom_frame_mappings_duration, and "
      "custom_frame_mappings_keyframe_indices must all be provided or all be "
      "None. This is a bug in TorchCodec, please report it.");

  std::optional<SingleStreamDecoder::FrameMappings> converted_mappings = has_pts
      ? std::make_optional(SingleStreamDecoder::FrameMappings{
            std::move(*custom_frame_mappings_pts),
            std::move(*custom_frame_mappings_keyframe_indices),
            std::move(*custom_frame_mappings_duration)})
      : std::nullopt;
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  video_decoder->add_video_stream(
      stream_index.value_or(-1),
      transforms,
      video_stream_options,
      converted_mappings);
}

// Add a new video stream at `stream_index` using the provided options.
void add_video_stream(
    torch::stable::Tensor& decoder,
    std::optional<int64_t> num_threads = std::nullopt,
    std::optional<std::string> dimension_order = std::nullopt,
    std::optional<int64_t> stream_index = std::nullopt,
    std::string device = "cpu",
    std::string device_variant = "default",
    std::string transform_specs = "",
    std::optional<torch::stable::Tensor> custom_frame_mappings_pts =
        std::nullopt,
    std::optional<torch::stable::Tensor> custom_frame_mappings_duration =
        std::nullopt,
    std::optional<torch::stable::Tensor>
        custom_frame_mappings_keyframe_indices = std::nullopt,
    std::string output_dtype = "uint8") {
  _add_video_stream(
      decoder,
      num_threads,
      std::move(dimension_order),
      stream_index,
      std::move(device),
      std::move(device_variant),
      std::move(transform_specs),
      std::move(custom_frame_mappings_pts),
      std::move(custom_frame_mappings_duration),
      std::move(custom_frame_mappings_keyframe_indices),
      /*color_conversion_library=*/std::nullopt,
      std::move(output_dtype));
}

void add_audio_stream(
    torch::stable::Tensor& decoder,
    std::optional<int64_t> stream_index = std::nullopt,
    std::optional<int64_t> sample_rate = std::nullopt,
    std::optional<int64_t> num_channels = std::nullopt) {
  AudioStreamOptions audio_stream_options;
  audio_stream_options.sample_rate = sample_rate;
  audio_stream_options.num_channels = num_channels;

  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  video_decoder->add_audio_stream(
      stream_index.value_or(-1), audio_stream_options);
}

// Seek to a particular presentation timestamp in the video in seconds.
void seek_to_pts(torch::stable::Tensor& decoder, double seconds) {
  auto video_decoder =
      static_cast<SingleStreamDecoder*>(decoder.mutable_data_ptr());
  video_decoder->set_cursor_pts_in_seconds(seconds);
}

// Get the next frame from the video as a tuple that has the frame data, pts and
// duration as tensors.
OpsFrameOutput get_next_frame(torch::stable::Tensor& decoder) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  FrameOutput result;
  try {
    result = video_decoder->get_next_frame();
  } catch (const SingleStreamDecoder::EndOfFileException& e) {
    STABLE_CHECK_INDEX(false, e.what());
  }
  return make_ops_frame_output(result);
}

// Return the frame that is visible at a given timestamp in seconds. Each frame
// in FFMPEG has a presentation timestamp and a duration. The frame visible at a
// given timestamp T has T >= PTS and T < PTS + Duration.
OpsFrameOutput get_frame_at_pts(
    torch::stable::Tensor& decoder,
    double seconds) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  FrameOutput result;
  try {
    result = video_decoder->get_frame_played_at(seconds);
  } catch (const SingleStreamDecoder::EndOfFileException& e) {
    STABLE_CHECK_INDEX(false, e.what());
  }
  return make_ops_frame_output(result);
}

// Return the frame that is visible at a given index in the video.
OpsFrameOutput get_frame_at_index(
    torch::stable::Tensor& decoder,
    int64_t frame_index) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  auto result = video_decoder->get_frame_at_index(frame_index);
  return make_ops_frame_output(result);
}

// Return the frames at given indices for a given stream
OpsFrameBatchOutput get_frames_at_indices(
    torch::stable::Tensor& decoder,
    const torch::stable::Tensor& frame_indices) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  auto result = video_decoder->get_frames_at_indices(frame_indices);
  return make_ops_frame_batch_output(result);
}

// Return the frames inside a range as a single stacked Tensor. The range is
// defined as [start, stop).
OpsFrameBatchOutput get_frames_in_range(
    torch::stable::Tensor& decoder,
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
    torch::stable::Tensor& decoder,
    const torch::stable::Tensor& timestamps) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  auto result = video_decoder->get_frames_played_at(timestamps);
  return make_ops_frame_batch_output(result);
}

// Return the frames inside the range as a single stacked Tensor. The range is
// defined as [start_seconds, stop_seconds). The frames are stacked in pts
// order.
// If fps is specified, frames are resampled to match the target frame
// rate by duplicating or dropping frames as necessary.
OpsFrameBatchOutput get_frames_by_pts_in_range(
    torch::stable::Tensor& decoder,
    double start_seconds,
    double stop_seconds,
    std::optional<double> fps = std::nullopt) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  auto result = video_decoder->get_frames_played_in_range(
      start_seconds, stop_seconds, fps);
  return make_ops_frame_batch_output(result);
}

OpsAudioFramesOutput get_frames_by_pts_in_range_audio(
    torch::stable::Tensor& decoder,
    double start_seconds,
    std::optional<double> stop_seconds = std::nullopt) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  auto result = video_decoder->get_frames_played_in_range_audio(
      start_seconds, stop_seconds);
  return make_ops_audio_frames_output(result);
}

// ==============================
// Building-block ops (torchcodec.decoders._blocks)
// ==============================

std::optional<int> to_optional_int(std::optional<int64_t> value) {
  if (!value.has_value()) {
    return std::nullopt;
  }
  return static_cast<int>(value.value());
}

AVMediaType parse_media_type(const std::string& media_type) {
  if (media_type == "video") {
    return AVMEDIA_TYPE_VIDEO;
  }
  STD_TORCH_CHECK(
      media_type == "audio",
      "Invalid media_type: '",
      media_type,
      "'. Expected 'video' or 'audio'.");
  return AVMEDIA_TYPE_AUDIO;
}

torch::stable::Tensor _blocks_create_demuxer_from_file(std::string filename) {
  auto demuxer = std::make_unique<Demuxer>(filename);
  return wrap_pointer_to_tensor<Demuxer>(std::move(demuxer));
}

torch::stable::Tensor _blocks_create_demuxer_from_tensor(
    const torch::stable::Tensor& video_tensor) {
  STD_TORCH_CHECK(
      video_tensor.is_contiguous(), "video_tensor must be contiguous");
  STD_TORCH_CHECK(
      video_tensor.scalar_type() == kStableUInt8,
      "video_tensor must be kUInt8");

  auto avio_context_holder = std::make_unique<AVIOContextHolder>(
      std::make_unique<TensorReadIO>(video_tensor), /*is_for_writing=*/false);
  auto demuxer = std::make_unique<Demuxer>(std::move(avio_context_holder));
  return wrap_pointer_to_tensor<Demuxer>(std::move(demuxer));
}

torch::stable::Tensor _blocks_create_demuxer_from_file_like(
    int64_t file_like_context) {
  auto file_like_context_ptr =
      reinterpret_cast<IOInterface*>(file_like_context);
  STD_TORCH_CHECK(
      file_like_context_ptr != nullptr,
      "file_like_context must be a valid pointer");

  auto avio_context_holder = std::make_unique<AVIOContextHolder>(
      std::unique_ptr<IOInterface>(file_like_context_ptr),
      /*is_for_writing=*/false);
  auto demuxer = std::make_unique<Demuxer>(std::move(avio_context_holder));
  return wrap_pointer_to_tensor<Demuxer>(std::move(demuxer));
}

torch::stable::Tensor _blocks_demuxer_get_audio_video_stream_indices(
    torch::stable::Tensor& demuxer) {
  return unwrap_tensor_to_pointer<Demuxer>(demuxer)
      ->get_audio_video_stream_indices();
}

// (stream index, media type) of the stream that is now being followed. The
// media type comes back so the caller doesn't have to ask for it separately
// when it identified the stream by index.
std::tuple<int64_t, std::string> _blocks_demuxer_add_stream(
    torch::stable::Tensor& demuxer,
    std::optional<int64_t> stream_index,
    std::optional<std::string> media_type) {
  auto [index, type] = unwrap_tensor_to_pointer<Demuxer>(demuxer)->add_stream(
      to_optional_int(stream_index),
      media_type.has_value()
          ? std::optional<AVMediaType>(parse_media_type(*media_type))
          : std::nullopt);
  return {
      static_cast<int64_t>(index),
      type == AVMEDIA_TYPE_VIDEO ? "video" : "audio"};
}

// (packet_handle, is_eof, stream_index). On EOF the packet_handle is a dummy
// tensor that must not be used and stream_index is -1. Native bool avoids
// per-frame .item() overhead in Python.
using OpsPacketOutput = std::tuple<torch::stable::Tensor, bool, int64_t>;

OpsPacketOutput _blocks_demuxer_next_packet(torch::stable::Tensor& demuxer) {
  Demuxer* demuxer_ptr = unwrap_tensor_to_pointer<Demuxer>(demuxer);
  UniqueAVPacket packet = demuxer_ptr->next_packet();
  if (packet == nullptr) {
    return std::make_tuple(torch::stable::full({1}, 0, kStableInt64), true, -1);
  }
  int64_t stream_index = packet->stream_index;
  return std::make_tuple(
      wrap_pointer_to_tensor(std::move(packet)), false, stream_index);
}

void _blocks_demuxer_seek(
    torch::stable::Tensor& demuxer,
    double seconds,
    std::optional<int64_t> stream_index) {
  unwrap_tensor_to_pointer<Demuxer>(demuxer)->seek(
      seconds, to_optional_int(stream_index));
}

// (pts, duration, is_key_frame, time_base_num, time_base_den). pts and duration
// are int64 [N] in time-base units rather than seconds
using OpsScanOutput = std::tuple<
    torch::stable::Tensor,
    torch::stable::Tensor,
    torch::stable::Tensor,
    int64_t,
    int64_t>;

OpsScanOutput _blocks_demuxer_scan(
    torch::stable::Tensor& demuxer,
    std::optional<int64_t> stream_index) {
  FrameIndex index = unwrap_tensor_to_pointer<Demuxer>(demuxer)->scan(
      to_optional_int(stream_index));
  return std::make_tuple(
      index.pts,
      index.duration,
      index.is_key_frame,
      static_cast<int64_t>(index.time_base.num),
      static_cast<int64_t>(index.time_base.den));
}

torch::stable::Tensor _blocks_create_packet_decoder(
    torch::stable::Tensor& demuxer,
    std::optional<int64_t> stream_index,
    std::optional<int64_t> num_threads,
    std::string device) {
  Demuxer* demuxer_ptr = unwrap_tensor_to_pointer<Demuxer>(demuxer);
  validate_device_interface(device);
  std::optional<int> thread_count;
  if (num_threads.has_value()) {
    thread_count = static_cast<int>(num_threads.value());
  }
  auto decoder = std::make_unique<PacketDecoder>(
      *demuxer_ptr,
      to_optional_int(stream_index),
      StableDevice(device),
      thread_count);
  return wrap_pointer_to_tensor<PacketDecoder>(std::move(decoder));
}

int64_t _blocks_packet_decoder_send_packet(
    torch::stable::Tensor& decoder,
    torch::stable::Tensor& packet) {
  PacketDecoder* decoder_ptr = unwrap_tensor_to_pointer<PacketDecoder>(decoder);
  return static_cast<int64_t>(
      decoder_ptr->send_packet(*unwrap_tensor_to_pointer<AVPacket>(packet)));
}

int64_t _blocks_packet_decoder_send_eof(torch::stable::Tensor& decoder) {
  PacketDecoder* decoder_ptr = unwrap_tensor_to_pointer<PacketDecoder>(decoder);
  return static_cast<int64_t>(decoder_ptr->send_eof());
}

void _blocks_packet_decoder_reset(torch::stable::Tensor& decoder) {
  unwrap_tensor_to_pointer<PacketDecoder>(decoder)->reset();
}

// (frame_handle, status, pts_seconds, duration_seconds, device, storage).
using OpsReceiveFrameOutput = std::tuple<
    torch::stable::Tensor,
    int64_t,
    double,
    double,
    StableDevice,
    torch::stable::Tensor>;

OpsReceiveFrameOutput _blocks_packet_decoder_receive_frame(
    torch::stable::Tensor& decoder) {
  PacketDecoder* decoder_ptr = unwrap_tensor_to_pointer<PacketDecoder>(decoder);
  UniqueAVFrame av_frame(av_frame_alloc());
  STD_TORCH_CHECK(av_frame != nullptr, "Failed to allocate AVFrame");
  int status = decoder_ptr->receive_frame(av_frame);
  if (status != AVSUCCESS) {
    return std::make_tuple(
        torch::stable::full({1}, 0, kStableInt64),
        static_cast<int64_t>(status),
        0.0,
        0.0,
        StableDevice(kStableCPU),
        torch::stable::empty({int64_t(0)}, kStableUInt8));
  }
  AVRational time_base = decoder_ptr->time_base();
  double pts_seconds = pts_to_seconds(get_pts_or_dts(*av_frame), time_base);
  double duration_seconds = pts_to_seconds(get_duration(*av_frame), time_base);
  StableDevice device = decoder_ptr->device();
  torch::stable::Tensor storage =
      decoder_ptr->get_frame_storage(*av_frame).value_or(
          torch::stable::empty({int64_t(0)}, kStableUInt8));
  return std::make_tuple(
      wrap_pointer_to_tensor(std::move(av_frame)),
      static_cast<int64_t>(0),
      pts_seconds,
      duration_seconds,
      device,
      storage);
}

// (samples, status, pts_seconds, duration_seconds, sample_rate,
// sample_format). `samples` is [num_channels, num_samples] in the frame's own
// sample type; there is no frame handle because, unlike a video frame, nothing
// downstream needs the AVFrame itself.
using OpsReceiveAudioFrameOutput = std::
    tuple<torch::stable::Tensor, int64_t, double, double, int64_t, std::string>;

OpsReceiveAudioFrameOutput _blocks_audio_packet_decoder_receive_frame(
    torch::stable::Tensor& decoder) {
  PacketDecoder* decoder_ptr = unwrap_tensor_to_pointer<PacketDecoder>(decoder);
  STD_TORCH_CHECK(
      decoder_ptr->media_type() == AVMEDIA_TYPE_AUDIO,
      "This PacketDecoder decodes video, not audio.");

  UniqueAVFrame av_frame(av_frame_alloc());
  STD_TORCH_CHECK(av_frame != nullptr, "Failed to allocate AVFrame");
  int status = decoder_ptr->receive_frame(av_frame);
  if (status != AVSUCCESS) {
    return std::make_tuple(
        torch::stable::empty({int64_t(0)}, kStableUInt8),
        static_cast<int64_t>(status),
        0.0,
        0.0,
        static_cast<int64_t>(0),
        std::string(""));
  }

  AVRational time_base = decoder_ptr->time_base();
  const char* sample_format_name =
      av_get_sample_fmt_name(static_cast<AVSampleFormat>(av_frame->format));
  return std::make_tuple(
      audio_samples(*av_frame),
      static_cast<int64_t>(0),
      pts_to_seconds(get_pts_or_dts(*av_frame), time_base),
      pts_to_seconds(get_duration(*av_frame), time_base),
      static_cast<int64_t>(av_frame->sample_rate),
      std::string(
          sample_format_name == nullptr ? "unknown" : sample_format_name));
}

torch::stable::Tensor _blocks_create_color_converter(
    std::string device,
    std::string output_dtype) {
  validate_device_interface(device);
  auto converter = std::make_unique<ColorConverter>(
      StableDevice(device), parse_output_dtype_config(output_dtype));
  return wrap_pointer_to_tensor<ColorConverter>(std::move(converter));
}

torch::stable::Tensor _blocks_convert_frame(
    torch::stable::Tensor& converter,
    torch::stable::Tensor& frame,
    StableDevice device) {
  ColorConverter* converter_ptr =
      unwrap_tensor_to_pointer<ColorConverter>(converter);
  return converter_ptr->convert(
      *unwrap_tensor_to_pointer<AVFrame>(frame), device);
}

torch::stable::Tensor _blocks_create_audio_converter(
    std::optional<int64_t> sample_rate,
    std::optional<int64_t> num_channels) {
  auto converter = std::make_unique<AudioConverter>(
      to_optional_int(sample_rate), to_optional_int(num_channels));
  return wrap_pointer_to_tensor<AudioConverter>(std::move(converter));
}

torch::stable::Tensor _blocks_audio_converter_convert(
    torch::stable::Tensor& converter,
    const torch::stable::Tensor& samples,
    int64_t sample_rate) {
  return unwrap_tensor_to_pointer<AudioConverter>(converter)->convert(
      samples, static_cast<int>(sample_rate));
}

torch::stable::Tensor _blocks_audio_converter_drain(
    torch::stable::Tensor& converter) {
  return unwrap_tensor_to_pointer<AudioConverter>(converter)->drain();
}

void _blocks_audio_converter_reset(torch::stable::Tensor& converter) {
  unwrap_tensor_to_pointer<AudioConverter>(converter)->reset();
}

using OpsFrameMetadataOutput = std::tuple<
    std::string, // pixel-format
    std::string, // colorspace
    std::string, // color range
    int64_t, // bit depth
    int64_t, // width
    int64_t, // height
    double>; // rotation, in degrees

OpsFrameMetadataOutput _blocks_frame_metadata(
    torch::stable::Tensor& tensor_handle) {
  FrameMetadata metadata =
      frame_metadata(*unwrap_tensor_to_pointer<AVFrame>(tensor_handle));
  return std::make_tuple(
      metadata.pix_fmt,
      metadata.colorspace,
      metadata.color_range,
      metadata.bit_depth,
      metadata.width,
      metadata.height,
      metadata.rotation_degrees);
}

using OpsFramePlanesOutput = std::tuple<
    torch::stable::Tensor,
    torch::stable::Tensor,
    torch::stable::Tensor,
    torch::stable::Tensor>;

OpsFramePlanesOutput _blocks_frame_planes(
    torch::stable::Tensor& tensor_handle,
    StableDevice device) {
  AVFrame* av_frame = unwrap_tensor_to_pointer<AVFrame>(tensor_handle);
  std::vector<torch::stable::Tensor> planes =
      frame_planes(*av_frame, device, tensor_handle);

  // Op schema wants a fixed number of planes, so we pad with empty tensors that
  // then get removed at the Python level.
  planes.resize(4, torch::stable::empty({int64_t(0)}, kStableUInt8));
  return std::make_tuple(planes[0], planes[1], planes[2], planes[3]);
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
    torch::stable::Tensor& decoder,
    int64_t frame_index,
    double pts_seconds_to_test) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  return pts_seconds_to_test ==
      video_decoder->get_pts_seconds_for_frame(frame_index);
}

torch::stable::Tensor _get_key_frame_indices(torch::stable::Tensor& decoder) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  return video_decoder->get_key_frame_indices();
}

// Get the metadata from the video as a string.
std::string get_json_metadata(torch::stable::Tensor& decoder) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);

  ContainerMetadata video_metadata = video_decoder->get_container_metadata();
  auto maybe_best_video_stream_index = video_metadata.best_video_stream_index;

  std::map<std::string, std::string> metadata_map;
  // serialize the metadata into a string std::stringstream ss;
  double duration_seconds_from_header = 0;
  if (maybe_best_video_stream_index.has_value() &&
      video_metadata.all_stream_metadata[*maybe_best_video_stream_index]
          .duration_seconds_from_header.has_value()) {
    duration_seconds_from_header =
        video_metadata.all_stream_metadata[*maybe_best_video_stream_index]
            .duration_seconds_from_header.value_or(0);
  } else {
    // Fallback to container-level duration if stream duration is not found.
    duration_seconds_from_header =
        video_metadata.duration_seconds_from_header.value_or(0);
  }
  metadata_map["durationSecondsFromHeader"] =
      fmt::to_string(duration_seconds_from_header);

  if (video_metadata.bit_rate.has_value()) {
    metadata_map["bitRate"] = fmt::to_string(video_metadata.bit_rate.value());
  }

  if (maybe_best_video_stream_index.has_value()) {
    auto stream_metadata =
        video_metadata.all_stream_metadata[*maybe_best_video_stream_index];
    if (stream_metadata.num_frames_from_content.has_value()) {
      metadata_map["numFramesFromHeader"] =
          std::to_string(*stream_metadata.num_frames_from_content);
    } else if (stream_metadata.num_frames_from_header.has_value()) {
      metadata_map["numFramesFromHeader"] =
          std::to_string(*stream_metadata.num_frames_from_header);
    }
    if (stream_metadata.begin_stream_pts_seconds_from_content.has_value()) {
      metadata_map["beginStreamSecondsFromContent"] = fmt::to_string(
          *stream_metadata.begin_stream_pts_seconds_from_content);
    }
    if (stream_metadata.end_stream_pts_seconds_from_content.has_value()) {
      metadata_map["endStreamSecondsFromContent"] =
          fmt::to_string(*stream_metadata.end_stream_pts_seconds_from_content);
    }
    if (stream_metadata.codec_name.has_value()) {
      metadata_map["codec"] = quote_value(stream_metadata.codec_name.value());
    }
    if (stream_metadata.post_rotation_width.has_value()) {
      metadata_map["width"] =
          std::to_string(*stream_metadata.post_rotation_width);
    }
    if (stream_metadata.post_rotation_height.has_value()) {
      metadata_map["height"] =
          std::to_string(*stream_metadata.post_rotation_height);
    }
    if (stream_metadata.average_fps_from_header.has_value()) {
      metadata_map["averageFpsFromHeader"] =
          fmt::to_string(*stream_metadata.average_fps_from_header);
    }
  }
  if (video_metadata.best_video_stream_index.has_value()) {
    metadata_map["bestVideoStreamIndex"] =
        std::to_string(*video_metadata.best_video_stream_index);
  }
  if (video_metadata.best_audio_stream_index.has_value()) {
    metadata_map["bestAudioStreamIndex"] =
        std::to_string(*video_metadata.best_audio_stream_index);
  }

  return map_to_json(metadata_map);
}

// Get the container metadata as a string.
namespace {
// The half of a stream's metadata that comes from the container header. Shared
// by get_stream_json_metadata, which then adds the content-derived and
// fallback-computed fields, and by the Demuxer building block, which has only
// this half to report.
void write_header_based_metadata(
    std::map<std::string, std::string>& map,
    const StreamMetadata& stream_metadata) {
  if (stream_metadata.duration_seconds_from_header.has_value()) {
    map["durationSecondsFromHeader"] =
        fmt::to_string(*stream_metadata.duration_seconds_from_header);
  }
  if (stream_metadata.bit_rate.has_value()) {
    map["bitRate"] = fmt::to_string(*stream_metadata.bit_rate);
  }
  if (stream_metadata.num_frames_from_header.has_value()) {
    map["numFramesFromHeader"] =
        std::to_string(*stream_metadata.num_frames_from_header);
  }
  if (stream_metadata.begin_stream_seconds_from_header.has_value()) {
    map["beginStreamSecondsFromHeader"] =
        fmt::to_string(*stream_metadata.begin_stream_seconds_from_header);
  }
  if (stream_metadata.codec_name.has_value()) {
    map["codec"] = quote_value(stream_metadata.codec_name.value());
  }
  if (stream_metadata.post_rotation_width.has_value()) {
    map["width"] = std::to_string(*stream_metadata.post_rotation_width);
  }
  if (stream_metadata.post_rotation_height.has_value()) {
    map["height"] = std::to_string(*stream_metadata.post_rotation_height);
  }
  if (stream_metadata.sample_aspect_ratio.has_value()) {
    map["sampleAspectRatioNum"] =
        std::to_string((*stream_metadata.sample_aspect_ratio).num);
    map["sampleAspectRatioDen"] =
        std::to_string((*stream_metadata.sample_aspect_ratio).den);
  }
  if (stream_metadata.rotation.has_value()) {
    map["rotation"] = std::to_string(*stream_metadata.rotation);
  }
  if (auto name = stream_metadata.get_color_primaries_name()) {
    map["colorPrimaries"] = quote_value(*name);
  }
  if (auto name = stream_metadata.get_color_space_name()) {
    map["colorSpace"] = quote_value(*name);
  }
  if (auto name = stream_metadata.get_color_transfer_characteristic_name()) {
    map["colorTransferCharacteristic"] = quote_value(*name);
  }
  if (stream_metadata.pixel_format.has_value()) {
    map["pixelFormat"] = quote_value(stream_metadata.pixel_format.value());
  }
  if (stream_metadata.average_fps_from_header.has_value()) {
    map["averageFpsFromHeader"] =
        fmt::to_string(*stream_metadata.average_fps_from_header);
  }
  if (stream_metadata.sample_rate.has_value()) {
    map["sampleRate"] = std::to_string(*stream_metadata.sample_rate);
  }
  if (stream_metadata.num_channels.has_value()) {
    map["numChannels"] = std::to_string(*stream_metadata.num_channels);
  }
  if (stream_metadata.sample_format.has_value()) {
    map["sampleFormat"] = quote_value(stream_metadata.sample_format.value());
  }
  if (stream_metadata.media_type == AVMEDIA_TYPE_VIDEO) {
    map["mediaType"] = quote_value("video");
  } else if (stream_metadata.media_type == AVMEDIA_TYPE_AUDIO) {
    map["mediaType"] = quote_value("audio");
  } else {
    map["mediaType"] = quote_value("other");
  }
}
} // namespace

// The container header, as the Demuxer building block reports it: container
// facts plus one entry per stream, and nothing derived from content. The
// building blocks never merge the two, so a caller always knows which of the
// two sources a number came from.
std::string _blocks_demuxer_container_json_metadata(
    torch::stable::Tensor& demuxer) {
  ContainerMetadata metadata = get_container_metadata_from_format_context(
      unwrap_tensor_to_pointer<Demuxer>(demuxer)->format_context().get());

  std::map<std::string, std::string> map;
  if (metadata.duration_seconds_from_header.has_value()) {
    map["durationSecondsFromHeader"] =
        fmt::to_string(*metadata.duration_seconds_from_header);
  }
  if (metadata.bit_rate.has_value()) {
    map["bitRate"] = fmt::to_string(*metadata.bit_rate);
  }
  if (metadata.best_video_stream_index.has_value()) {
    map["bestVideoStreamIndex"] =
        std::to_string(*metadata.best_video_stream_index);
  }
  if (metadata.best_audio_stream_index.has_value()) {
    map["bestAudioStreamIndex"] =
        std::to_string(*metadata.best_audio_stream_index);
  }
  map["numStreams"] = std::to_string(metadata.all_stream_metadata.size());
  return map_to_json(map);
}

std::string _blocks_demuxer_stream_json_metadata(
    torch::stable::Tensor& demuxer,
    int64_t stream_index) {
  ContainerMetadata metadata = get_container_metadata_from_format_context(
      unwrap_tensor_to_pointer<Demuxer>(demuxer)->format_context().get());
  STABLE_CHECK_INDEX(
      stream_index >= 0 &&
          stream_index <
              static_cast<int64_t>(metadata.all_stream_metadata.size()),
      "stream_index out of bounds: " + std::to_string(stream_index));

  std::map<std::string, std::string> map;
  write_header_based_metadata(map, metadata.all_stream_metadata[stream_index]);
  return map_to_json(map);
}

std::string get_container_json_metadata(torch::stable::Tensor& decoder) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);

  auto container_metadata = video_decoder->get_container_metadata();

  std::map<std::string, std::string> map;

  if (container_metadata.duration_seconds_from_header.has_value()) {
    map["durationSecondsFromHeader"] =
        fmt::to_string(*container_metadata.duration_seconds_from_header);
  }

  if (container_metadata.bit_rate.has_value()) {
    map["bitRate"] = fmt::to_string(*container_metadata.bit_rate);
  }

  if (container_metadata.best_video_stream_index.has_value()) {
    map["bestVideoStreamIndex"] =
        std::to_string(*container_metadata.best_video_stream_index);
  }
  if (container_metadata.best_audio_stream_index.has_value()) {
    map["bestAudioStreamIndex"] =
        std::to_string(*container_metadata.best_audio_stream_index);
  }

  map["numStreams"] =
      std::to_string(container_metadata.all_stream_metadata.size());

  return map_to_json(map);
}

// Get the stream metadata as a string.
std::string get_stream_json_metadata(
    torch::stable::Tensor& decoder,
    int64_t stream_index) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  auto all_stream_metadata =
      video_decoder->get_container_metadata().all_stream_metadata;
  STABLE_CHECK_INDEX(
      stream_index >= 0 &&
          stream_index < static_cast<int64_t>(all_stream_metadata.size()),
      "stream_index out of bounds: " + std::to_string(stream_index));

  auto stream_metadata = all_stream_metadata[stream_index];
  auto seek_mode = video_decoder->get_seek_mode();
  int active_stream_index = video_decoder->get_active_stream_index();

  std::map<std::string, std::string> map;

  write_header_based_metadata(map, stream_metadata);
  if (stream_metadata.num_frames_from_content.has_value()) {
    map["numFramesFromContent"] =
        std::to_string(*stream_metadata.num_frames_from_content);
  }
  if (stream_metadata.begin_stream_pts_seconds_from_content.has_value()) {
    map["beginStreamSecondsFromContent"] =
        fmt::to_string(*stream_metadata.begin_stream_pts_seconds_from_content);
  }
  if (stream_metadata.end_stream_pts_seconds_from_content.has_value()) {
    map["endStreamSecondsFromContent"] =
        fmt::to_string(*stream_metadata.end_stream_pts_seconds_from_content);
  }

  // Check whether content-based metadata is available for this stream.
  // In exact mode: content-based metadata exists for all streams.
  // In approximate mode: content-based metadata does not exist for any stream.
  // In custom_frame_mappings: content-based metadata exists only for the active
  // stream.
  //
  // Our fallback logic assumes content-based metadata is available.
  // It is available for decoding on the active stream, but would break
  // when getting metadata from non-active streams.
  if ((seek_mode != SeekMode::custom_frame_mappings) ||
      (seek_mode == SeekMode::custom_frame_mappings &&
       stream_index == active_stream_index)) {
    write_fallback_based_metadata(map, stream_metadata, seek_mode);
  } else if (seek_mode == SeekMode::custom_frame_mappings) {
    // If this is not the active stream, then we don't have content-based
    // metadata for custom frame mappings. In that case, we want the same
    // behavior as we would get with approximate mode. Encoding this behavior in
    // the fallback logic itself is tricky and not worth it for this corner
    // case. So we hardcode in approximate mode.
    //
    // TODO: This hacky behavior is only necessary because the custom frame
    //       mapping is supplied in SingleStreamDecoder::addVideoStream() rather
    //       than in the constructor. And it's supplied to addVideoStream() and
    //       not the constructor because we need to know the stream index. If we
    //       can encode the relevant stream indices into custom frame mappings
    //       itself, then we can put it in the constructor.
    write_fallback_based_metadata(map, stream_metadata, SeekMode::approximate);
  }

  return map_to_json(map);
}

// Returns version information about the various FFMPEG libraries that are
// loaded in the program's address space.
// TODO: ideally we'd have a more robust way of getting the ffmpeg version,
// we're using av_version_info() which is not standardized and shouldn't be
// parsed by code (which we do!). See
// https://github.com/pytorch/torchcodec/issues/100
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

std::string get_backend_details(torch::stable::Tensor& decoder) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  return video_decoder->get_device_interface_details();
}

// Scans video packets to get more accurate metadata like frame count, exact
// keyframe positions, etc. Exact keyframe positions are useful for efficient
// accurate seeking. Note that this function reads the entire video but it does
// not decode frames. Reading a video file is much cheaper than decoding it.
void scan_all_streams_to_update_metadata(torch::stable::Tensor& decoder) {
  auto video_decoder = unwrap_tensor_to_get_decoder(decoder);
  video_decoder->scan_file_and_update_metadata_and_index();
}

torch::stable::Tensor create_streaming_encoder() {
  auto encoder = std::make_unique<MultiStreamEncoder>();
  return wrap_multi_stream_encoder_pointer_to_tensor(std::move(encoder));
}

void streaming_encoder_close(torch::stable::Tensor& encoder) {
  unwrap_tensor_to_get_multi_stream_encoder(encoder)->close();
}

int64_t streaming_encoder_add_video_stream(
    torch::stable::Tensor& encoder,
    int64_t height,
    int64_t width,
    double frame_rate,
    std::string device = "cpu",
    std::optional<std::string> codec = std::nullopt,
    std::optional<std::string> pixel_format = std::nullopt,
    std::optional<double> crf = std::nullopt,
    std::optional<std::string> preset = std::nullopt,
    std::optional<std::vector<std::string>> extra_options = std::nullopt) {
  std::optional<std::map<std::string, std::string>> extra_options_map;
  if (extra_options.has_value()) {
    extra_options_map = unflatten_extra_options(extra_options.value());
  }
  return static_cast<int64_t>(
      unwrap_tensor_to_get_multi_stream_encoder(encoder)->add_video_stream(
          static_cast<int>(height),
          static_cast<int>(width),
          frame_rate,
          std::move(device),
          std::move(codec),
          std::move(pixel_format),
          crf,
          std::move(preset),
          std::move(extra_options_map)));
}

void streaming_encoder_open_file(
    torch::stable::Tensor& encoder,
    std::string filename) {
  unwrap_tensor_to_get_multi_stream_encoder(encoder)->open(filename);
}

void streaming_encoder_open_file_like(
    torch::stable::Tensor& encoder,
    std::string format,
    int64_t file_like_context) {
  auto file_like_context_ptr =
      reinterpret_cast<IOInterface*>(file_like_context);
  STD_TORCH_CHECK(
      file_like_context_ptr != nullptr,
      "file_like_context must be a valid pointer");
  auto avio_context_holder = std::make_unique<AVIOContextHolder>(
      std::unique_ptr<IOInterface>(file_like_context_ptr),
      /*is_for_writing=*/true);
  unwrap_tensor_to_get_multi_stream_encoder(encoder)->open(
      format, std::move(avio_context_holder));
}

int64_t streaming_encoder_add_audio_stream(
    torch::stable::Tensor& encoder,
    int64_t sample_rate,
    int64_t num_channels,
    std::optional<int64_t> bit_rate = std::nullopt,
    std::optional<int64_t> output_num_channels = std::nullopt,
    std::optional<int64_t> output_sample_rate = std::nullopt) {
  return static_cast<int64_t>(
      unwrap_tensor_to_get_multi_stream_encoder(encoder)->add_audio_stream(
          validate_int64_to_int(sample_rate, "sample_rate"),
          validate_int64_to_int(num_channels, "num_channels"),
          validate_optional_int64_to_int(bit_rate, "bit_rate"),
          validate_optional_int64_to_int(
              output_num_channels, "output_num_channels"),
          validate_optional_int64_to_int(
              output_sample_rate, "output_sample_rate")));
}

void streaming_encoder_add_frames(
    torch::stable::Tensor& encoder,
    const torch::stable::Tensor& frames,
    int64_t stream_index) {
  unwrap_tensor_to_get_multi_stream_encoder(encoder)->add_frames(
      frames, static_cast<int>(stream_index));
}

void streaming_encoder_add_samples(
    torch::stable::Tensor& encoder,
    const torch::stable::Tensor& samples,
    int64_t stream_index) {
  unwrap_tensor_to_get_multi_stream_encoder(encoder)->add_samples(
      samples, static_cast<int>(stream_index));
}

torch::stable::Tensor create_wav_decoder_from_file(
    const std::string& filename) {
  auto io = std::make_unique<FileIO>(filename, FileIO::Mode::Read);
  auto decoder = std::make_unique<WavDecoder>(std::move(io));
  return wrap_wav_decoder_pointer_to_tensor(std::move(decoder));
}

torch::stable::Tensor create_wav_decoder_from_tensor(
    const torch::stable::Tensor& data) {
  auto io = std::make_unique<TensorReadIO>(data);
  auto decoder = std::make_unique<WavDecoder>(std::move(io));
  return wrap_wav_decoder_pointer_to_tensor(std::move(decoder));
}

torch::stable::Tensor _create_wav_decoder_from_file_like(
    int64_t file_like_context) {
  auto file_like_context_ptr =
      reinterpret_cast<IOInterface*>(file_like_context);
  std::unique_ptr<IOInterface> io(file_like_context_ptr);
  auto decoder = std::make_unique<WavDecoder>(std::move(io));
  return wrap_wav_decoder_pointer_to_tensor(std::move(decoder));
}

OpsAudioFramesOutput get_wav_samples_in_range(
    torch::stable::Tensor& decoder,
    double start_seconds,
    std::optional<double> stop_seconds) {
  auto wav_decoder = unwrap_tensor_to_get_wav_decoder(decoder);
  AudioFramesOutput audio_frames =
      wav_decoder->get_samples_in_range(start_seconds, stop_seconds);
  return make_ops_audio_frames_output(audio_frames);
}

std::string get_wav_metadata_from_decoder(torch::stable::Tensor& decoder) {
  auto wav_decoder = unwrap_tensor_to_get_wav_decoder(decoder);
  StreamMetadata stream_metadata = wav_decoder->get_stream_metadata();

  std::map<std::string, std::string> metadata_map;

  metadata_map["sampleRate"] = std::to_string(*stream_metadata.sample_rate);
  metadata_map["numChannels"] = std::to_string(*stream_metadata.num_channels);
  metadata_map["sampleFormat"] = quote_value(*stream_metadata.sample_format);
  metadata_map["codec"] = quote_value(*stream_metadata.codec_name);
  metadata_map["durationSeconds"] =
      fmt::to_string(*stream_metadata.duration_seconds_from_header);
  metadata_map["durationSecondsFromHeader"] =
      fmt::to_string(*stream_metadata.duration_seconds_from_header);
  metadata_map["bitRate"] = fmt::to_string(*stream_metadata.bit_rate);
  metadata_map["streamIndex"] = std::to_string(stream_metadata.stream_index);
  metadata_map["mediaType"] = quote_value("audio");
  metadata_map["beginStreamSeconds"] =
      fmt::to_string(*stream_metadata.begin_stream_pts_seconds_from_content);
  return map_to_json(metadata_map);
}

void _set_nvdec_cache_capacity(int64_t capacity) {
  int capacity_int = validate_int64_to_int(capacity, "capacity");
  STD_TORCH_CHECK(
      capacity_int >= 0,
      "NVDEC cache capacity must be non-negative, got ",
      capacity_int);
  set_nvdec_cache_capacity(capacity_int);
}

int64_t _get_nvdec_cache_capacity() {
  return static_cast<int64_t>(get_nvdec_cache_capacity());
}

int64_t _get_nvdec_cache_size(int64_t device_index) {
  int device_index_int = validate_int64_to_int(device_index, "device_index");
  STD_TORCH_CHECK(
      device_index_int >= 0,
      "device_index must be non-negative, got ",
      device_index_int);
  return static_cast<int64_t>(get_nvdec_cache_size(device_index_int));
}

void _set_cpp_log_level(int64_t level) {
  set_cpp_log_level(static_cast<LogLevel>(level));
}

int64_t _get_log_level() {
  return static_cast<int64_t>(get_log_level());
}

STABLE_TORCH_LIBRARY_IMPL(torchcodec_ns, BackendSelect, m) {
  m.impl("create_from_file", TORCH_BOX(&create_from_file));
  m.impl("create_from_tensor", TORCH_BOX(&create_from_tensor));
  m.impl("_create_from_file_like", TORCH_BOX(&_create_from_file_like));
  m.impl(
      "_blocks_create_demuxer_from_file",
      TORCH_BOX(&_blocks_create_demuxer_from_file));
  m.impl(
      "_blocks_create_demuxer_from_tensor",
      TORCH_BOX(&_blocks_create_demuxer_from_tensor));
  m.impl(
      "_blocks_create_demuxer_from_file_like",
      TORCH_BOX(&_blocks_create_demuxer_from_file_like));
  m.impl(
      "_blocks_create_color_converter",
      TORCH_BOX(&_blocks_create_color_converter));
  m.impl(
      "_blocks_create_audio_converter",
      TORCH_BOX(&_blocks_create_audio_converter));
  m.impl(
      "_get_json_ffmpeg_library_versions",
      TORCH_BOX(&_get_json_ffmpeg_library_versions));
  m.impl("create_streaming_encoder", TORCH_BOX(&create_streaming_encoder));
  m.impl(
      "streaming_encoder_add_video_stream",
      TORCH_BOX(&streaming_encoder_add_video_stream));
  m.impl(
      "streaming_encoder_add_audio_stream",
      TORCH_BOX(&streaming_encoder_add_audio_stream));
  m.impl(
      "streaming_encoder_open_file", TORCH_BOX(&streaming_encoder_open_file));
  m.impl(
      "streaming_encoder_open_file_like",
      TORCH_BOX(&streaming_encoder_open_file_like));
  m.impl(
      "streaming_encoder_add_frames", TORCH_BOX(&streaming_encoder_add_frames));
  m.impl(
      "streaming_encoder_add_samples",
      TORCH_BOX(&streaming_encoder_add_samples));
  m.impl("set_nvdec_cache_capacity", TORCH_BOX(&_set_nvdec_cache_capacity));
  m.impl("get_nvdec_cache_capacity", TORCH_BOX(&_get_nvdec_cache_capacity));
  m.impl("_get_nvdec_cache_size", TORCH_BOX(&_get_nvdec_cache_size));
  m.impl("_set_cpp_log_level", TORCH_BOX(&_set_cpp_log_level));
  m.impl("_get_log_level", TORCH_BOX(&_get_log_level));
  m.impl(
      "create_wav_decoder_from_file", TORCH_BOX(&create_wav_decoder_from_file));
  m.impl(
      "create_wav_decoder_from_tensor",
      TORCH_BOX(&create_wav_decoder_from_tensor));
  m.impl(
      "_create_wav_decoder_from_file_like",
      TORCH_BOX(&_create_wav_decoder_from_file_like));
  m.impl("get_wav_samples_in_range", TORCH_BOX(&get_wav_samples_in_range));
  m.impl(
      "get_wav_metadata_from_decoder",
      TORCH_BOX(&get_wav_metadata_from_decoder));
}

STABLE_TORCH_LIBRARY_IMPL(torchcodec_ns, CPU, m) {
  m.impl("seek_to_pts", TORCH_BOX(&seek_to_pts));
  m.impl("add_video_stream", TORCH_BOX(&add_video_stream));
  m.impl("_add_video_stream", TORCH_BOX(&_add_video_stream));
  m.impl("add_audio_stream", TORCH_BOX(&add_audio_stream));
  m.impl("get_next_frame", TORCH_BOX(&get_next_frame));
  m.impl("_get_key_frame_indices", TORCH_BOX(&_get_key_frame_indices));
  m.impl("get_json_metadata", TORCH_BOX(&get_json_metadata));
  m.impl(
      "get_container_json_metadata", TORCH_BOX(&get_container_json_metadata));
  m.impl("get_stream_json_metadata", TORCH_BOX(&get_stream_json_metadata));
  m.impl("get_frame_at_pts", TORCH_BOX(&get_frame_at_pts));
  m.impl("get_frame_at_index", TORCH_BOX(&get_frame_at_index));
  m.impl("get_frames_at_indices", TORCH_BOX(&get_frames_at_indices));
  m.impl("get_frames_in_range", TORCH_BOX(&get_frames_in_range));
  m.impl("get_frames_by_pts_in_range", TORCH_BOX(&get_frames_by_pts_in_range));
  m.impl(
      "get_frames_by_pts_in_range_audio",
      TORCH_BOX(&get_frames_by_pts_in_range_audio));
  m.impl("get_frames_by_pts", TORCH_BOX(&get_frames_by_pts));
  m.impl("_blocks_demuxer_add_stream", TORCH_BOX(&_blocks_demuxer_add_stream));
  m.impl(
      "_blocks_demuxer_get_audio_video_stream_indices",
      TORCH_BOX(&_blocks_demuxer_get_audio_video_stream_indices));
  m.impl(
      "_blocks_demuxer_container_json_metadata",
      TORCH_BOX(&_blocks_demuxer_container_json_metadata));
  m.impl(
      "_blocks_demuxer_stream_json_metadata",
      TORCH_BOX(&_blocks_demuxer_stream_json_metadata));
  m.impl(
      "_blocks_demuxer_next_packet", TORCH_BOX(&_blocks_demuxer_next_packet));
  m.impl("_blocks_demuxer_seek", TORCH_BOX(&_blocks_demuxer_seek));
  m.impl("_blocks_demuxer_scan", TORCH_BOX(&_blocks_demuxer_scan));
  m.impl(
      "_blocks_create_packet_decoder",
      TORCH_BOX(&_blocks_create_packet_decoder));
  m.impl(
      "_blocks_packet_decoder_send_packet",
      TORCH_BOX(&_blocks_packet_decoder_send_packet));
  m.impl(
      "_blocks_packet_decoder_send_eof",
      TORCH_BOX(&_blocks_packet_decoder_send_eof));
  m.impl(
      "_blocks_packet_decoder_reset", TORCH_BOX(&_blocks_packet_decoder_reset));
  m.impl(
      "_blocks_packet_decoder_receive_frame",
      TORCH_BOX(&_blocks_packet_decoder_receive_frame));
  m.impl(
      "_blocks_audio_packet_decoder_receive_frame",
      TORCH_BOX(&_blocks_audio_packet_decoder_receive_frame));
  m.impl("_blocks_convert_frame", TORCH_BOX(&_blocks_convert_frame));
  m.impl(
      "_blocks_audio_converter_convert",
      TORCH_BOX(&_blocks_audio_converter_convert));
  m.impl(
      "_blocks_audio_converter_drain",
      TORCH_BOX(&_blocks_audio_converter_drain));
  m.impl(
      "_blocks_audio_converter_reset",
      TORCH_BOX(&_blocks_audio_converter_reset));
  m.impl("_blocks_frame_metadata", TORCH_BOX(&_blocks_frame_metadata));
  m.impl("_blocks_frame_planes", TORCH_BOX(&_blocks_frame_planes));
  m.impl("_test_frame_pts_equality", TORCH_BOX(&_test_frame_pts_equality));
  m.impl(
      "scan_all_streams_to_update_metadata",
      TORCH_BOX(&scan_all_streams_to_update_metadata));

  m.impl("_get_backend_details", TORCH_BOX(&get_backend_details));
  m.impl("create_streaming_encoder", TORCH_BOX(&create_streaming_encoder));
  m.impl(
      "streaming_encoder_open_file", TORCH_BOX(&streaming_encoder_open_file));
  m.impl(
      "streaming_encoder_open_file_like",
      TORCH_BOX(&streaming_encoder_open_file_like));
  m.impl("streaming_encoder_close", TORCH_BOX(&streaming_encoder_close));
  m.impl(
      "streaming_encoder_add_video_stream",
      TORCH_BOX(&streaming_encoder_add_video_stream));
  m.impl(
      "streaming_encoder_add_audio_stream",
      TORCH_BOX(&streaming_encoder_add_audio_stream));
  m.impl(
      "streaming_encoder_add_frames", TORCH_BOX(&streaming_encoder_add_frames));
  m.impl(
      "streaming_encoder_add_samples",
      TORCH_BOX(&streaming_encoder_add_samples));
}

} // namespace facebook::torchcodec
