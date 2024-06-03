// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cstdint>
#include <sstream>
#include <string>
#include "c10/core/SymIntArrayRef.h"
#include "src/torchcodec/decoders/_core/VideoDecoder.h"

namespace facebook::torchcodec {

// ==============================
// Define the operators
// ==============================

torch::Tensor plus_one(torch::Tensor t) {
  return t + 1;
}

TORCH_LIBRARY(plusoneops, m) {
  m.def("plus_one", plus_one);
}

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
      "add_video_stream(Tensor(a!) decoder, *, int? width=None, int? height=None, int? num_threads=None, str? shape=None, int? stream_index=None) -> ()");
  m.def("seek_to_pts(Tensor(a!) decoder, float seconds) -> ()");
  m.def("get_next_frame(Tensor(a!) decoder) -> Tensor");
  m.def("get_frame_at_pts(Tensor(a!) decoder, float seconds) -> Tensor");
  m.def(
      "get_frame_at_index(Tensor(a!) decoder, *, int frame_index, int? stream_index=None) -> Tensor");
  m.def(
      "get_frames_at_indices(Tensor(a!) decoder, *, int[] frame_indices, int? stream_index=None) -> Tensor");
  m.def("get_json_metadata(Tensor(a!) decoder) -> str");
}

// ==============================
// Implementations for the operators
// ==============================

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

at::Tensor create_from_file(c10::string_view filename) {
  std::string filenameStr(filename);
  std::unique_ptr<VideoDecoder> uniqueDecoder =
      VideoDecoder::createFromFilePath(filenameStr);
  uniqueDecoder->scanFileAndUpdateMetadataAndIndex();
  return wrapDecoderPointerToTensor(std::move(uniqueDecoder));
}

at::Tensor create_from_tensor(at::Tensor video_tensor) {
  TORCH_CHECK(video_tensor.is_contiguous(), "video_tensor must be contiguous");
  void* buffer = video_tensor.mutable_data_ptr();
  size_t length = video_tensor.numel();
  std::unique_ptr<VideoDecoder> videoDecoder =
      VideoDecoder::createFromBuffer(buffer, length);
  videoDecoder->scanFileAndUpdateMetadataAndIndex();
  return wrapDecoderPointerToTensor(std::move(videoDecoder));
}

at::Tensor create_from_buffer(const void* buffer, size_t length) {
  std::unique_ptr<VideoDecoder> uniqueDecoder =
      VideoDecoder::createFromBuffer(buffer, length);
  uniqueDecoder->scanFileAndUpdateMetadataAndIndex();
  return wrapDecoderPointerToTensor(std::move(uniqueDecoder));
}

void add_video_stream(
    at::Tensor& decoder,
    std::optional<int64_t> width = std::nullopt,
    std::optional<int64_t> height = std::nullopt,
    std::optional<int64_t> num_threads = std::nullopt,
    std::optional<c10::string_view> shape = std::nullopt,
    std::optional<int64_t> stream_index = std::nullopt) {
  VideoDecoder::VideoStreamDecoderOptions options;
  options.width = width;
  options.height = height;
  options.ffmpegThreadCount = num_threads;

  if (shape.has_value()) {
    std::string stdShape{shape.value()};
    TORCH_CHECK(stdShape == "NHWC" || stdShape == "NCHW");
    options.shape = stdShape;
  }

  auto videoDecoder = static_cast<VideoDecoder*>(decoder.mutable_data_ptr());
  videoDecoder->addVideoStreamDecoder(stream_index.value_or(-1), options);
}

void seek_to_pts(at::Tensor& decoder, double seconds) {
  auto videoDecoder = static_cast<VideoDecoder*>(decoder.mutable_data_ptr());
  videoDecoder->setCursorPtsInSeconds(seconds);
}

at::Tensor get_next_frame(at::Tensor& decoder) {
  auto videoDecoder = static_cast<VideoDecoder*>(decoder.mutable_data_ptr());
  auto result = videoDecoder->getNextDecodedOutput().frame;
  if (result.sizes().size() != 3) {
    throw std::runtime_error(
        "image_size is unexpected. Expected 3, got: " +
        std::to_string(result.sizes().size()));
  }
  return result;
}

at::Tensor get_frame_at_pts(at::Tensor& decoder, double seconds) {
  auto videoDecoder = static_cast<VideoDecoder*>(decoder.mutable_data_ptr());
  auto result = videoDecoder->getFrameDisplayedAtTimestamp(seconds);
  return result.frame;
}

at::Tensor get_frame_at_index(
    at::Tensor& decoder,
    int64_t frame_index,
    std::optional<int64_t> stream_index) {
  auto videoDecoder = static_cast<VideoDecoder*>(decoder.mutable_data_ptr());
  auto result =
      videoDecoder->getFrameAtIndex(stream_index.value_or(-1), frame_index);
  return result.frame;
}

at::Tensor get_frames_at_indices(
    at::Tensor& decoder,
    at::IntArrayRef frame_indices,
    std::optional<int64_t> stream_index) {
  auto videoDecoder = static_cast<VideoDecoder*>(decoder.mutable_data_ptr());
  std::vector<int64_t> frameIndicesVec(
      frame_indices.begin(), frame_indices.end());
  auto result = videoDecoder->getFramesAtIndexes(
      stream_index.value_or(-1), frameIndicesVec);
  return result.frames;
}

std::string quoteValue(const std::string& value) {
  return "\"" + value + "\"";
}

std::string get_json_metadata(at::Tensor& decoder) {
  auto videoDecoder = static_cast<VideoDecoder*>(decoder.mutable_data_ptr());

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
    // Just overwrite has we have a better definition for video
    if (streamMetadata.bitRate.has_value()) {
      metadataMap["bitRate"] = std::to_string(*streamMetadata.bitRate);
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

TORCH_LIBRARY_IMPL(torchcodec_ns, BackendSelect, m) {
  m.impl("create_from_file", &create_from_file);
  m.impl("create_from_tensor", &create_from_tensor);
}

TORCH_LIBRARY_IMPL(torchcodec_ns, CPU, m) {
  m.impl("seek_to_pts", &seek_to_pts);
  m.impl("add_video_stream", &add_video_stream);
  m.impl("get_next_frame", &get_next_frame);
  m.impl("get_json_metadata", &get_json_metadata);
  m.impl("get_frame_at_pts", &get_frame_at_pts);
  m.impl("get_frame_at_index", &get_frame_at_index);
  m.impl("get_frames_at_indices", &get_frames_at_indices);
}

} // namespace facebook::torchcodec
