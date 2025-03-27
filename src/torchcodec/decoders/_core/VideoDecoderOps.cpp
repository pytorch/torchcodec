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
#include "src/torchcodec/decoders/_core/AVIOBytesContext.h"
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
      "torchcodec.decoders._core.ops", "//pytorch/torchcodec:torchcodec");
  m.def("create_from_file(str filename, str? seek_mode=None) -> Tensor");
  m.def(
      "create_from_tensor(Tensor video_tensor, str? seek_mode=None) -> Tensor");
  m.def("_convert_to_tensor(int decoder_ptr) -> Tensor");
  m.def(
      "_add_video_stream(Tensor(a!) decoder, *, int? width=None, int? height=None, int? num_threads=None, str? dimension_order=None, int? stream_index=None, str? device=None, str? color_conversion_library=None) -> ()");
  m.def(
      "add_video_stream(Tensor(a!) decoder, *, int? width=None, int? height=None, int? num_threads=None, str? dimension_order=None, int? stream_index=None, str? device=None) -> ()");
  m.def(
      "add_audio_stream(Tensor(a!) decoder, *, int? stream_index=None, int? sample_rate=None) -> ()");
  m.def("seek_to_pts(Tensor(a!) decoder, float seconds) -> ()");
  m.def("get_next_frame(Tensor(a!) decoder) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frame_at_pts(Tensor(a!) decoder, float seconds) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frame_at_index(Tensor(a!) decoder, *, int frame_index) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frames_at_indices(Tensor(a!) decoder, *, int[] frame_indices) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frames_in_range(Tensor(a!) decoder, *, int start, int stop, int? step=None) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frames_by_pts_in_range(Tensor(a!) decoder, *, float start_seconds, float stop_seconds) -> (Tensor, Tensor, Tensor)");
  m.def(
      "get_frames_by_pts_in_range_audio(Tensor(a!) decoder, *, float start_seconds, float? stop_seconds) -> (Tensor, Tensor)");
  m.def(
      "get_frames_by_pts(Tensor(a!) decoder, *, float[] timestamps) -> (Tensor, Tensor, Tensor)");
  m.def("_get_key_frame_indices(Tensor(a!) decoder) -> Tensor");
  m.def("get_json_metadata(Tensor(a!) decoder) -> str");
  m.def("get_container_json_metadata(Tensor(a!) decoder) -> str");
  m.def(
      "get_stream_json_metadata(Tensor(a!) decoder, int stream_index) -> str");
  m.def("_get_json_ffmpeg_library_versions() -> str");
  m.def(
      "_test_frame_pts_equality(Tensor(a!) decoder, *, int frame_index, float pts_seconds_to_test) -> bool");
  m.def("scan_all_streams_to_update_metadata(Tensor(a!) decoder) -> ()");
}

namespace {

at::Tensor wrapDecoderPointerToTensor(
    std::unique_ptr<VideoDecoder> uniqueDecoder) {
  VideoDecoder* decoder = uniqueDecoder.release();

  auto deleter = [decoder](void*) { delete decoder; };
  at::Tensor tensor =
      at::from_blob(decoder, {sizeof(VideoDecoder*)}, deleter, {at::kLong});
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

OpsFrameOutput makeOpsFrameOutput(VideoDecoder::FrameOutput& frame) {
  return std::make_tuple(
      frame.data,
      torch::tensor(frame.ptsSeconds, torch::dtype(torch::kFloat64)),
      torch::tensor(frame.durationSeconds, torch::dtype(torch::kFloat64)));
}

OpsFrameBatchOutput makeOpsFrameBatchOutput(
    VideoDecoder::FrameBatchOutput& batch) {
  return std::make_tuple(batch.data, batch.ptsSeconds, batch.durationSeconds);
}

OpsAudioFramesOutput makeOpsAudioFramesOutput(
    VideoDecoder::AudioFramesOutput& audioFrames) {
  return std::make_tuple(
      audioFrames.data,
      torch::tensor(audioFrames.ptsSeconds, torch::dtype(torch::kFloat64)));
}
} // namespace

// ==============================
// Implementations for the operators
// ==============================

at::Tensor create_from_file(
    std::string_view filename,
    std::optional<std::string_view> seek_mode) {
  std::string filenameStr(filename);

  VideoDecoder::SeekMode realSeek = VideoDecoder::SeekMode::exact;
  if (seek_mode.has_value()) {
    realSeek = seekModeFromString(seek_mode.value());
  }

  std::unique_ptr<VideoDecoder> uniqueDecoder =
      std::make_unique<VideoDecoder>(filenameStr, realSeek);

  return wrapDecoderPointerToTensor(std::move(uniqueDecoder));
}

at::Tensor create_from_tensor(
    at::Tensor video_tensor,
    std::optional<std::string_view> seek_mode) {
  TORCH_CHECK(video_tensor.is_contiguous(), "video_tensor must be contiguous");
  TORCH_CHECK(
      video_tensor.scalar_type() == torch::kUInt8,
      "video_tensor must be kUInt8");
  void* data = video_tensor.mutable_data_ptr();
  size_t length = video_tensor.numel();

  VideoDecoder::SeekMode realSeek = VideoDecoder::SeekMode::exact;
  if (seek_mode.has_value()) {
    realSeek = seekModeFromString(seek_mode.value());
  }

  auto contextHolder = std::make_unique<AVIOBytesContext>(data, length);

  std::unique_ptr<VideoDecoder> uniqueDecoder =
      std::make_unique<VideoDecoder>(std::move(contextHolder), realSeek);
  return wrapDecoderPointerToTensor(std::move(uniqueDecoder));
}

at::Tensor _convert_to_tensor(int64_t decoder_ptr) {
  auto decoder = reinterpret_cast<VideoDecoder*>(decoder_ptr);
  std::unique_ptr<VideoDecoder> uniqueDecoder(decoder);
  return wrapDecoderPointerToTensor(std::move(uniqueDecoder));
}

void add_video_stream(
    at::Tensor& decoder,
    std::optional<int64_t> width,
    std::optional<int64_t> height,
    std::optional<int64_t> num_threads,
    std::optional<std::string_view> dimension_order,
    std::optional<int64_t> stream_index,
    std::optional<std::string_view> device) {
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
    std::optional<std::string_view> dimension_order,
    std::optional<int64_t> stream_index,
    std::optional<std::string_view> device,
    std::optional<std::string_view> color_conversion_library) {
  VideoDecoder::VideoStreamOptions videoStreamOptions;
  videoStreamOptions.width = width;
  videoStreamOptions.height = height;
  videoStreamOptions.ffmpegThreadCount = num_threads;

  if (dimension_order.has_value()) {
    std::string stdDimensionOrder{dimension_order.value()};
    TORCH_CHECK(stdDimensionOrder == "NHWC" || stdDimensionOrder == "NCHW");
    videoStreamOptions.dimensionOrder = stdDimensionOrder;
  }
  if (color_conversion_library.has_value()) {
    std::string stdColorConversionLibrary{color_conversion_library.value()};
    if (stdColorConversionLibrary == "filtergraph") {
      videoStreamOptions.colorConversionLibrary =
          VideoDecoder::ColorConversionLibrary::FILTERGRAPH;
    } else if (stdColorConversionLibrary == "swscale") {
      videoStreamOptions.colorConversionLibrary =
          VideoDecoder::ColorConversionLibrary::SWSCALE;
    } else {
      throw std::runtime_error(
          "Invalid color_conversion_library=" + stdColorConversionLibrary +
          ". color_conversion_library must be either filtergraph or swscale.");
    }
  }
  if (device.has_value()) {
    if (device.value() == "cpu") {
      videoStreamOptions.device = torch::Device(torch::kCPU);
    } else if (device.value().rfind("cuda", 0) == 0) { // starts with "cuda"
      std::string deviceStr(device.value());
      videoStreamOptions.device = torch::Device(deviceStr);
    } else if (device.value().rfind("xpu", 0) == 0) { // starts with "xpu"
      std::string deviceStr(device.value());
      videoStreamOptions.device = torch::Device(deviceStr);
    } else {
      throw std::runtime_error(
          "Invalid device=" + std::string(device.value()) +
          ". device must be either cpu, cuda or xpu.");
    }
  }

  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  videoDecoder->addVideoStream(stream_index.value_or(-1), videoStreamOptions);
}

void add_audio_stream(
    at::Tensor& decoder,
    std::optional<int64_t> stream_index,
    std::optional<int64_t> sample_rate) {
  VideoDecoder::AudioStreamOptions audioStreamOptions;
  audioStreamOptions.sampleRate = sample_rate;

  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  videoDecoder->addAudioStream(stream_index.value_or(-1), audioStreamOptions);
}

void seek_to_pts(at::Tensor& decoder, double seconds) {
  auto videoDecoder = static_cast<VideoDecoder*>(decoder.mutable_data_ptr());
  videoDecoder->setCursorPtsInSeconds(seconds);
}

OpsFrameOutput get_next_frame(at::Tensor& decoder) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  VideoDecoder::FrameOutput result;
  try {
    result = videoDecoder->getNextFrame();
  } catch (const VideoDecoder::EndOfFileException& e) {
    C10_THROW_ERROR(IndexError, e.what());
  }
  return makeOpsFrameOutput(result);
}

OpsFrameOutput get_frame_at_pts(at::Tensor& decoder, double seconds) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  VideoDecoder::FrameOutput result;
  try {
    result = videoDecoder->getFramePlayedAt(seconds);
  } catch (const VideoDecoder::EndOfFileException& e) {
    C10_THROW_ERROR(IndexError, e.what());
  }
  return makeOpsFrameOutput(result);
}

OpsFrameOutput get_frame_at_index(at::Tensor& decoder, int64_t frame_index) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  auto result = videoDecoder->getFrameAtIndex(frame_index);
  return makeOpsFrameOutput(result);
}

OpsFrameBatchOutput get_frames_at_indices(
    at::Tensor& decoder,
    at::IntArrayRef frame_indices) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  std::vector<int64_t> frameIndicesVec(
      frame_indices.begin(), frame_indices.end());
  auto result = videoDecoder->getFramesAtIndices(frameIndicesVec);
  return makeOpsFrameBatchOutput(result);
}

OpsFrameBatchOutput get_frames_in_range(
    at::Tensor& decoder,
    int64_t start,
    int64_t stop,
    std::optional<int64_t> step) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  auto result = videoDecoder->getFramesInRange(start, stop, step.value_or(1));
  return makeOpsFrameBatchOutput(result);
}

OpsFrameBatchOutput get_frames_by_pts(
    at::Tensor& decoder,
    at::ArrayRef<double> timestamps) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  std::vector<double> timestampsVec(timestamps.begin(), timestamps.end());
  auto result = videoDecoder->getFramesPlayedAt(timestampsVec);
  return makeOpsFrameBatchOutput(result);
}

OpsFrameBatchOutput get_frames_by_pts_in_range(
    at::Tensor& decoder,
    double start_seconds,
    double stop_seconds) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  auto result =
      videoDecoder->getFramesPlayedInRange(start_seconds, stop_seconds);
  return makeOpsFrameBatchOutput(result);
}

OpsAudioFramesOutput get_frames_by_pts_in_range_audio(
    at::Tensor& decoder,
    double start_seconds,
    std::optional<double> stop_seconds) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  auto result =
      videoDecoder->getFramesPlayedInRangeAudio(start_seconds, stop_seconds);
  return makeOpsAudioFramesOutput(result);
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
    int64_t frame_index,
    double pts_seconds_to_test) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  return pts_seconds_to_test ==
      videoDecoder->getPtsSecondsForFrame(frame_index);
}

torch::Tensor _get_key_frame_indices(at::Tensor& decoder) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  return videoDecoder->getKeyFrameIndices();
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
      videoMetadata.allStreamMetadata[*maybeBestVideoStreamIndex]
          .durationSeconds.has_value()) {
    durationSeconds =
        videoMetadata.allStreamMetadata[*maybeBestVideoStreamIndex]
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
    auto streamMetadata =
        videoMetadata.allStreamMetadata[*maybeBestVideoStreamIndex];
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

  map["numStreams"] =
      std::to_string(containerMetadata.allStreamMetadata.size());

  return mapToJson(map);
}

std::string get_stream_json_metadata(
    at::Tensor& decoder,
    int64_t stream_index) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  auto allStreamMetadata =
      videoDecoder->getContainerMetadata().allStreamMetadata;
  if (stream_index < 0 ||
      stream_index >= static_cast<int64_t>(allStreamMetadata.size())) {
    throw std::out_of_range(
        "stream_index out of bounds: " + std::to_string(stream_index));
  }
  auto streamMetadata = allStreamMetadata[stream_index];

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
  if (streamMetadata.beginStreamFromHeader.has_value()) {
    map["beginStreamFromHeader"] =
        std::to_string(*streamMetadata.beginStreamFromHeader);
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
  if (streamMetadata.sampleRate.has_value()) {
    map["sampleRate"] = std::to_string(*streamMetadata.sampleRate);
  }
  if (streamMetadata.numChannels.has_value()) {
    map["numChannels"] = std::to_string(*streamMetadata.numChannels);
  }
  if (streamMetadata.sampleFormat.has_value()) {
    map["sampleFormat"] = quoteValue(streamMetadata.sampleFormat.value());
  }
  if (streamMetadata.mediaType == AVMEDIA_TYPE_VIDEO) {
    map["mediaType"] = quoteValue("video");
  } else if (streamMetadata.mediaType == AVMEDIA_TYPE_AUDIO) {
    map["mediaType"] = quoteValue("audio");
  } else {
    map["mediaType"] = quoteValue("other");
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
