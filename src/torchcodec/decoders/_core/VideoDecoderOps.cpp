// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/decoders/_core/VideoDecoderOps.h"
#include <pybind11/pybind11.h>
#include <cstdint>
#include <sstream>
#include <string>
#include "c10/core/SymIntArrayRef.h"
#include "c10/util/Exception.h"
#include "src/torchcodec/decoders/_core/VideoDecoder.h"

namespace facebook::torchcodec {

// ==============================
// Define the operators
// ==============================
// All instances of accepting the decoder as a tensor must be annotated with
// `Tensor(a!)`. The `(a!)` part normally indicates that the tensor is being
// mutated in place. We need it to make sure that torch.compile does not reorder
// calls to these functions. For more detail, see:
//   https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native#readme
TORCH_LIBRARY(torchcodec_ns, m) {
  m.impl_abstract_pystub(
      "torchcodec.decoders._core.video_decoder_ops",
      "//pytorch/torchcodec:torchcodec");
  m.def("create_from_file(str filename) -> Tensor");
  m.def("create_from_tensor(Tensor video_tensor) -> Tensor");
  m.def(
      "_add_video_stream(Tensor(a!) decoder, *, int? width=None, int? height=None, int? num_threads=None, str? dimension_order=None, int? stream_index=None, str? device=None, str? color_conversion_library=None) -> ()");
  m.def(
      "add_video_stream(Tensor(a!) decoder, *, int? width=None, int? height=None, int? num_threads=None, str? dimension_order=None, int? stream_index=None, str? device=None) -> ()");
  m.def("seek_to_pts(Tensor(a!) decoder, float seconds) -> ()");
  m.def("get_next_frame(Tensor(a!) decoder) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frame_at_pts(Tensor(a!) decoder, float seconds) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frame_at_index(Tensor(a!) decoder, *, int stream_index, int frame_index) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frames_at_indices(Tensor(a!) decoder, *, int stream_index, int[] frame_indices) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frames_in_range(Tensor(a!) decoder, *, int stream_index, int start, int stop, int? step=None) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frames_by_pts_in_range(Tensor(a!) decoder, *, int stream_index, float start_seconds, float stop_seconds) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frames_by_pts(Tensor(a!) decoder, *, int stream_index, float[] timestamps) -> (Tensor, Tensor, Tensor)");
  m.def("get_json_metadata(Tensor(a!) decoder) -> str");
  m.def("get_container_json_metadata(Tensor(a!) decoder) -> str");
  m.def(
      "get_stream_json_metadata(Tensor(a!) decoder, int stream_index) -> str");
  m.def("_get_json_ffmpeg_library_versions() -> str");
  m.def(
      "_test_frame_pts_equality(Tensor(a!) decoder, *, int stream_index, int frame_index, float pts_seconds_to_test) -> bool");
  m.def("scan_all_streams_to_update_metadata(Tensor(a!) decoder) -> ()");
}

namespace {
at::Tensor wrapDecoderPointerToTensor(
    std::unique_ptr<VideoDecoder> uniqueDecoder) {
  VideoDecoder* decoder = uniqueDecoder.release();

  auto deleter = [decoder](void*) { delete decoder; };
  at::Tensor tensor =
      at::from_blob(decoder, {sizeof(VideoDecoder)}, deleter, {at::kLong});
  auto videoDecoder = static_cast<VideoDecoder*>(tensor.mutable_data_ptr());
  TORCH_CHECK_EQ(videoDecoder, decoder) << "videoDecoder=" << videoDecoder;
  return tensor;
}

VideoDecoder* unwrapTensorToGetDecoder(at::Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(tensor.is_contiguous());
  void* buffer = tensor.mutable_data_ptr();
  VideoDecoder* decoder = static_cast<VideoDecoder*>(buffer);
  return decoder;
}

OpsDecodedOutput makeOpsDecodedOutput(VideoDecoder::DecodedOutput& frame) {
  return std::make_tuple(
      frame.frame,
      torch::tensor(frame.ptsSeconds, torch::dtype(torch::kFloat64)),
      torch::tensor(frame.durationSeconds, torch::dtype(torch::kFloat64)));
}

OpsBatchDecodedOutput makeOpsBatchDecodedOutput(
    VideoDecoder::BatchDecodedOutput& batch) {
  return std::make_tuple(batch.frames, batch.ptsSeconds, batch.durationSeconds);
}
} // namespace

// ==============================
// Implementations for the operators
// ==============================

at::Tensor create_from_file(c10::string_view filename) {
  std::string filenameStr(filename);
  std::unique_ptr<VideoDecoder> uniqueDecoder =
      VideoDecoder::createFromFilePath(filenameStr);
  return wrapDecoderPointerToTensor(std::move(uniqueDecoder));
}

at::Tensor create_from_tensor(at::Tensor video_tensor) {
  TORCH_CHECK(video_tensor.is_contiguous(), "video_tensor must be contiguous");
  void* buffer = video_tensor.mutable_data_ptr();
  size_t length = video_tensor.numel();
  std::unique_ptr<VideoDecoder> videoDecoder =
      VideoDecoder::createFromBuffer(buffer, length);
  return wrapDecoderPointerToTensor(std::move(videoDecoder));
}

at::Tensor create_from_buffer(const void* buffer, size_t length) {
  std::unique_ptr<VideoDecoder> uniqueDecoder =
      VideoDecoder::createFromBuffer(buffer, length);
  return wrapDecoderPointerToTensor(std::move(uniqueDecoder));
}

void add_video_stream(
    at::Tensor& decoder,
    std::optional<int64_t> width,
    std::optional<int64_t> height,
    std::optional<int64_t> num_threads,
    std::optional<c10::string_view> dimension_order,
    std::optional<int64_t> stream_index,
    std::optional<c10::string_view> device) {
  _add_video_stream(
      decoder,
      width,
      height,
      num_threads,
      dimension_order,
      stream_index,
      device);
}

void _add_video_stream(
    at::Tensor& decoder,
    std::optional<int64_t> width,
    std::optional<int64_t> height,
    std::optional<int64_t> num_threads,
    std::optional<c10::string_view> dimension_order,
    std::optional<int64_t> stream_index,
    std::optional<c10::string_view> device,
    std::optional<c10::string_view> color_conversion_library) {
  VideoDecoder::VideoStreamDecoderOptions options;
  options.width = width;
  options.height = height;
  options.ffmpegThreadCount = num_threads;

  if (dimension_order.has_value()) {
    std::string stdDimensionOrder{dimension_order.value()};
    TORCH_CHECK(stdDimensionOrder == "NHWC" || stdDimensionOrder == "NCHW");
    options.dimensionOrder = stdDimensionOrder;
  }
  if (color_conversion_library.has_value()) {
    std::string stdColorConversionLibrary{color_conversion_library.value()};
    if (stdColorConversionLibrary == "filtergraph") {
      options.colorConversionLibrary =
          VideoDecoder::ColorConversionLibrary::FILTERGRAPH;
    } else if (stdColorConversionLibrary == "swscale") {
      options.colorConversionLibrary =
          VideoDecoder::ColorConversionLibrary::SWSCALE;
    } else {
      throw std::runtime_error(
          "Invalid color_conversion_library=" + stdColorConversionLibrary +
          ". color_conversion_library must be either filtergraph or swscale.");
    }
  }
  if (device.has_value()) {
    if (device.value() == "cpu") {
      options.device = torch::Device(torch::kCPU);
    } else if (device.value().rfind("cuda", 0) == 0) { // starts with "cuda"
      std::string deviceStr(device.value());
      options.device = torch::Device(deviceStr);
    } else {
      throw std::runtime_error(
          "Invalid device=" + std::string(device.value()) +
          ". device must be either cpu or cuda.");
    }
  }

  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  videoDecoder->addVideoStreamDecoder(stream_index.value_or(-1), options);
}

void seek_to_pts(at::Tensor& decoder, double seconds) {
  auto videoDecoder = static_cast<VideoDecoder*>(decoder.mutable_data_ptr());
  videoDecoder->setCursorPtsInSeconds(seconds);
}

OpsDecodedOutput get_next_frame(at::Tensor& decoder) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  VideoDecoder::DecodedOutput result;
  try {
    result = videoDecoder->getNextFrameNoDemux();
  } catch (const VideoDecoder::EndOfFileException& e) {
    C10_THROW_ERROR(IndexError, e.what());
  }
  if (result.frame.sizes().size() != 3) {
    throw std::runtime_error(
        "image_size is unexpected. Expected 3, got: " +
        std::to_string(result.frame.sizes().size()));
  }
  return makeOpsDecodedOutput(result);
}

OpsDecodedOutput get_frame_at_pts(at::Tensor& decoder, double seconds) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  auto result = videoDecoder->getFramePlayedAtTimestampNoDemux(seconds);
  return makeOpsDecodedOutput(result);
}

OpsDecodedOutput get_frame_at_index(
    at::Tensor& decoder,
    int64_t stream_index,
    int64_t frame_index) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  auto result = videoDecoder->getFrameAtIndex(stream_index, frame_index);
  return makeOpsDecodedOutput(result);
}

OpsBatchDecodedOutput get_frames_at_indices(
    at::Tensor& decoder,
    int64_t stream_index,
    at::IntArrayRef frame_indices) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  std::vector<int64_t> frameIndicesVec(
      frame_indices.begin(), frame_indices.end());
  auto result = videoDecoder->getFramesAtIndices(stream_index, frameIndicesVec);
  return makeOpsBatchDecodedOutput(result);
}

OpsBatchDecodedOutput get_frames_in_range(
    at::Tensor& decoder,
    int64_t stream_index,
    int64_t start,
    int64_t stop,
    std::optional<int64_t> step) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  auto result = videoDecoder->getFramesInRange(
      stream_index, start, stop, step.value_or(1));
  return makeOpsBatchDecodedOutput(result);
}
OpsBatchDecodedOutput get_frames_by_pts(
    at::Tensor& decoder,
    int64_t stream_index,
    at::ArrayRef<double> timestamps) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  std::vector<double> timestampsVec(timestamps.begin(), timestamps.end());
  auto result =
      videoDecoder->getFramesPlayedByTimestamps(stream_index, timestampsVec);
  return makeOpsBatchDecodedOutput(result);
}

OpsBatchDecodedOutput get_frames_by_pts_in_range(
    at::Tensor& decoder,
    int64_t stream_index,
    double start_seconds,
    double stop_seconds) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  auto result = videoDecoder->getFramesPlayedByTimestampInRange(
      stream_index, start_seconds, stop_seconds);
  return makeOpsBatchDecodedOutput(result);
}

std::string quoteValue(const std::string& value) {
  return "\"" + value + "\"";
}

std::string mapToJson(const std::map<std::string, std::string>& metadataMap) {
  std::stringstream ss;
  ss << "{\n";
  auto it = metadataMap.begin();
  while (it != metadataMap.end()) {
    ss << "\"" << it->first << "\": " << it->second;
    ++it;
    if (it != metadataMap.end()) {
      ss << ",\n";
    } else {
      ss << "\n";
    }
  }
  ss << "}";

  return ss.str();
}

bool _test_frame_pts_equality(
    at::Tensor& decoder,
    int64_t stream_index,
    int64_t frame_index,
    double pts_seconds_to_test) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  return pts_seconds_to_test ==
      videoDecoder->getPtsSecondsForFrame(stream_index, frame_index);
}

std::string get_json_metadata(at::Tensor& decoder) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);

  VideoDecoder::ContainerMetadata videoMetadata =
      videoDecoder->getContainerMetadata();
  auto maybeBestVideoStreamIndex = videoMetadata.bestVideoStreamIndex;

  std::map<std::string, std::string> metadataMap;
  // serialize the metadata into a string std::stringstream ss;
  double durationSeconds = 0;
  if (maybeBestVideoStreamIndex.has_value() &&
      videoMetadata.streams[*maybeBestVideoStreamIndex]
          .durationSeconds.has_value()) {
    durationSeconds = videoMetadata.streams[*maybeBestVideoStreamIndex]
                          .durationSeconds.value_or(0);
  } else {
    // Fallback to container-level duration if stream duration is not found.
    durationSeconds = videoMetadata.durationSeconds.value_or(0);
  }
  metadataMap["durationSeconds"] = std::to_string(durationSeconds);

  if (videoMetadata.bitRate.has_value()) {
    metadataMap["bitRate"] = std::to_string(videoMetadata.bitRate.value());
  }

  if (maybeBestVideoStreamIndex.has_value()) {
    auto streamMetadata = videoMetadata.streams[*maybeBestVideoStreamIndex];
    if (streamMetadata.numFramesFromScan.has_value()) {
      metadataMap["numFrames"] =
          std::to_string(*streamMetadata.numFramesFromScan);
    } else if (streamMetadata.numFrames.has_value()) {
      metadataMap["numFrames"] = std::to_string(*streamMetadata.numFrames);
    }
    if (streamMetadata.minPtsSecondsFromScan.has_value()) {
      metadataMap["minPtsSecondsFromScan"] =
          std::to_string(*streamMetadata.minPtsSecondsFromScan);
    }
    if (streamMetadata.maxPtsSecondsFromScan.has_value()) {
      metadataMap["maxPtsSecondsFromScan"] =
          std::to_string(*streamMetadata.maxPtsSecondsFromScan);
    }
    if (streamMetadata.codecName.has_value()) {
      metadataMap["codec"] = quoteValue(streamMetadata.codecName.value());
    }
    if (streamMetadata.width.has_value()) {
      metadataMap["width"] = std::to_string(*streamMetadata.width);
    }
    if (streamMetadata.height.has_value()) {
      metadataMap["height"] = std::to_string(*streamMetadata.height);
    }
    if (streamMetadata.averageFps.has_value()) {
      metadataMap["averageFps"] = std::to_string(*streamMetadata.averageFps);
    }
  }
  if (videoMetadata.bestVideoStreamIndex.has_value()) {
    metadataMap["bestVideoStreamIndex"] =
        std::to_string(*videoMetadata.bestVideoStreamIndex);
  }
  if (videoMetadata.bestAudioStreamIndex.has_value()) {
    metadataMap["bestAudioStreamIndex"] =
        std::to_string(*videoMetadata.bestAudioStreamIndex);
  }

  return mapToJson(metadataMap);
}

std::string get_container_json_metadata(at::Tensor& decoder) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);

  auto containerMetadata = videoDecoder->getContainerMetadata();

  std::map<std::string, std::string> map;

  if (containerMetadata.durationSeconds.has_value()) {
    map["durationSeconds"] = std::to_string(*containerMetadata.durationSeconds);
  }

  if (containerMetadata.bitRate.has_value()) {
    map["bitRate"] = std::to_string(*containerMetadata.bitRate);
  }

  if (containerMetadata.bestVideoStreamIndex.has_value()) {
    map["bestVideoStreamIndex"] =
        std::to_string(*containerMetadata.bestVideoStreamIndex);
  }
  if (containerMetadata.bestAudioStreamIndex.has_value()) {
    map["bestAudioStreamIndex"] =
        std::to_string(*containerMetadata.bestAudioStreamIndex);
  }

  map["numStreams"] = std::to_string(containerMetadata.streams.size());

  return mapToJson(map);
}

std::string get_stream_json_metadata(
    at::Tensor& decoder,
    int64_t stream_index) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  auto streams = videoDecoder->getContainerMetadata().streams;
  if (stream_index < 0 || stream_index >= streams.size()) {
    throw std::out_of_range(
        "stream_index out of bounds: " + std::to_string(stream_index));
  }
  auto streamMetadata = streams[stream_index];

  std::map<std::string, std::string> map;

  if (streamMetadata.durationSeconds.has_value()) {
    map["durationSeconds"] = std::to_string(*streamMetadata.durationSeconds);
  }
  if (streamMetadata.bitRate.has_value()) {
    map["bitRate"] = std::to_string(*streamMetadata.bitRate);
  }
  if (streamMetadata.numFramesFromScan.has_value()) {
    map["numFramesFromScan"] =
        std::to_string(*streamMetadata.numFramesFromScan);
  }
  if (streamMetadata.numFrames.has_value()) {
    map["numFrames"] = std::to_string(*streamMetadata.numFrames);
  }
  if (streamMetadata.minPtsSecondsFromScan.has_value()) {
    map["minPtsSecondsFromScan"] =
        std::to_string(*streamMetadata.minPtsSecondsFromScan);
  }
  if (streamMetadata.maxPtsSecondsFromScan.has_value()) {
    map["maxPtsSecondsFromScan"] =
        std::to_string(*streamMetadata.maxPtsSecondsFromScan);
  }
  if (streamMetadata.codecName.has_value()) {
    map["codec"] = quoteValue(streamMetadata.codecName.value());
  }
  if (streamMetadata.width.has_value()) {
    map["width"] = std::to_string(*streamMetadata.width);
  }
  if (streamMetadata.height.has_value()) {
    map["height"] = std::to_string(*streamMetadata.height);
  }
  if (streamMetadata.averageFps.has_value()) {
    map["averageFps"] = std::to_string(*streamMetadata.averageFps);
  }
  return mapToJson(map);
}

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

void scan_all_streams_to_update_metadata(at::Tensor& decoder) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  videoDecoder->scanFileAndUpdateMetadataAndIndex();
}

TORCH_LIBRARY_IMPL(torchcodec_ns, BackendSelect, m) {
  m.impl("create_from_file", &create_from_file);
  m.impl("create_from_tensor", &create_from_tensor);
  m.impl(
      "_get_json_ffmpeg_library_versions", &_get_json_ffmpeg_library_versions);
}

TORCH_LIBRARY_IMPL(torchcodec_ns, CPU, m) {
  m.impl("seek_to_pts", &seek_to_pts);
  m.impl("add_video_stream", &add_video_stream);
  m.impl("_add_video_stream", &_add_video_stream);
  m.impl("get_next_frame", &get_next_frame);
  m.impl("get_json_metadata", &get_json_metadata);
  m.impl("get_container_json_metadata", &get_container_json_metadata);
  m.impl("get_stream_json_metadata", &get_stream_json_metadata);
  m.impl("get_frame_at_pts", &get_frame_at_pts);
  m.impl("get_frame_at_index", &get_frame_at_index);
  m.impl("get_frames_at_indices", &get_frames_at_indices);
  m.impl("get_frames_in_range", &get_frames_in_range);
  m.impl("get_frames_by_pts_in_range", &get_frames_by_pts_in_range);
  m.impl("get_frames_by_pts", &get_frames_by_pts);
  m.impl("_test_frame_pts_equality", &_test_frame_pts_equality);
  m.impl(
      "scan_all_streams_to_update_metadata",
      &scan_all_streams_to_update_metadata);
}

} // namespace facebook::torchcodec
