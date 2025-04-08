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
#include "src/torchcodec/_core/Encoder.h"
#include "src/torchcodec/_core/SingleStreamDecoder.h"

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
      "torchcodec._core.ops", "//pytorch/torchcodec:torchcodec");
  m.def("create_from_file(str filename, str? seek_mode=None) -> Tensor");
  m.def(
      "create_audio_encoder(Tensor wf, int sample_rate, str filename, int? bit_rate=None) -> Tensor");
  m.def("encode_audio(Tensor(a!) encoder) -> ()");
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
    std::unique_ptr<SingleStreamDecoder> uniqueDecoder) {
  SingleStreamDecoder* decoder = uniqueDecoder.release();

  auto deleter = [decoder](void*) { delete decoder; };
  at::Tensor tensor = at::from_blob(
      decoder, {sizeof(SingleStreamDecoder*)}, deleter, {at::kLong});
  auto videoDecoder =
      static_cast<SingleStreamDecoder*>(tensor.mutable_data_ptr());
  TORCH_CHECK_EQ(videoDecoder, decoder) << "videoDecoder=" << videoDecoder;
  return tensor;
}

SingleStreamDecoder* unwrapTensorToGetDecoder(at::Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(tensor.is_contiguous());
  void* buffer = tensor.mutable_data_ptr();
  SingleStreamDecoder* decoder = static_cast<SingleStreamDecoder*>(buffer);
  return decoder;
}

// The elements of this tuple are all tensors that represent a single frame:
//   1. The frame data, which is a multidimensional tensor.
//   2. A single float value for the pts in seconds.
//   3. A single float value for the duration in seconds.
// The reason we use Tensors for the second and third values is so we can run
// under torch.compile().
using OpsFrameOutput = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

OpsFrameOutput makeOpsFrameOutput(FrameOutput& frame) {
  return std::make_tuple(
      frame.data,
      torch::tensor(frame.ptsSeconds, torch::dtype(torch::kFloat64)),
      torch::tensor(frame.durationSeconds, torch::dtype(torch::kFloat64)));
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
using OpsFrameBatchOutput = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

OpsFrameBatchOutput makeOpsFrameBatchOutput(FrameBatchOutput& batch) {
  return std::make_tuple(batch.data, batch.ptsSeconds, batch.durationSeconds);
}

// The elements of this tuple are all tensors that represent the concatenation
// of multiple audio frames:
//   1. The frames data (concatenated)
//   2. A single float value for the pts of the first frame, in seconds.
using OpsAudioFramesOutput = std::tuple<at::Tensor, at::Tensor>;

OpsAudioFramesOutput makeOpsAudioFramesOutput(AudioFramesOutput& audioFrames) {
  return std::make_tuple(
      audioFrames.data,
      torch::tensor(audioFrames.ptsSeconds, torch::dtype(torch::kFloat64)));
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

} // namespace

// ==============================
// Implementations for the operators
// ==============================

// Create a SingleStreamDecoder from file and wrap the pointer in a tensor.
at::Tensor create_from_file(
    std::string_view filename,
    std::optional<std::string_view> seek_mode = std::nullopt) {
  std::string filenameStr(filename);

  SingleStreamDecoder::SeekMode realSeek = SingleStreamDecoder::SeekMode::exact;
  if (seek_mode.has_value()) {
    realSeek = seekModeFromString(seek_mode.value());
  }

  std::unique_ptr<SingleStreamDecoder> uniqueDecoder =
      std::make_unique<SingleStreamDecoder>(filenameStr, realSeek);

  return wrapDecoderPointerToTensor(std::move(uniqueDecoder));
}

// Create a SingleStreamDecoder from the actual bytes of a video and wrap the
// pointer in a tensor. The SingleStreamDecoder will decode the provided bytes.
at::Tensor create_from_tensor(
    at::Tensor video_tensor,
    std::optional<std::string_view> seek_mode = std::nullopt) {
  TORCH_CHECK(video_tensor.is_contiguous(), "video_tensor must be contiguous");
  TORCH_CHECK(
      video_tensor.scalar_type() == torch::kUInt8,
      "video_tensor must be kUInt8");
  void* data = video_tensor.mutable_data_ptr();
  size_t length = video_tensor.numel();

  SingleStreamDecoder::SeekMode realSeek = SingleStreamDecoder::SeekMode::exact;
  if (seek_mode.has_value()) {
    realSeek = seekModeFromString(seek_mode.value());
  }

  auto contextHolder = std::make_unique<AVIOBytesContext>(data, length);

  std::unique_ptr<SingleStreamDecoder> uniqueDecoder =
      std::make_unique<SingleStreamDecoder>(std::move(contextHolder), realSeek);
  return wrapDecoderPointerToTensor(std::move(uniqueDecoder));
}

at::Tensor _convert_to_tensor(int64_t decoder_ptr) {
  auto decoder = reinterpret_cast<SingleStreamDecoder*>(decoder_ptr);
  std::unique_ptr<SingleStreamDecoder> uniqueDecoder(decoder);
  return wrapDecoderPointerToTensor(std::move(uniqueDecoder));
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
  VideoStreamOptions videoStreamOptions;
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
          ColorConversionLibrary::FILTERGRAPH;
    } else if (stdColorConversionLibrary == "swscale") {
      videoStreamOptions.colorConversionLibrary =
          ColorConversionLibrary::SWSCALE;
    } else {
      throw std::runtime_error(
          "Invalid color_conversion_library=" + stdColorConversionLibrary +
          ". color_conversion_library must be either filtergraph or swscale.");
    }
  }
  if (device.has_value()) {
    videoStreamOptions.device = createTorchDevice(std::string(device.value()));
  }

  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  videoDecoder->addVideoStream(stream_index.value_or(-1), videoStreamOptions);
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
  AudioStreamOptions audioStreamOptions;
  audioStreamOptions.sampleRate = sample_rate;

  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  videoDecoder->addAudioStream(stream_index.value_or(-1), audioStreamOptions);
}

// Seek to a particular presentation timestamp in the video in seconds.
void seek_to_pts(at::Tensor& decoder, double seconds) {
  auto videoDecoder =
      static_cast<SingleStreamDecoder*>(decoder.mutable_data_ptr());
  videoDecoder->setCursorPtsInSeconds(seconds);
}

// Get the next frame from the video as a tuple that has the frame data, pts and
// duration as tensors.
OpsFrameOutput get_next_frame(at::Tensor& decoder) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  FrameOutput result;
  try {
    result = videoDecoder->getNextFrame();
  } catch (const SingleStreamDecoder::EndOfFileException& e) {
    C10_THROW_ERROR(IndexError, e.what());
  }
  return makeOpsFrameOutput(result);
}

// Return the frame that is visible at a given timestamp in seconds. Each frame
// in FFMPEG has a presentation timestamp and a duration. The frame visible at a
// given timestamp T has T >= PTS and T < PTS + Duration.
OpsFrameOutput get_frame_at_pts(at::Tensor& decoder, double seconds) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  FrameOutput result;
  try {
    result = videoDecoder->getFramePlayedAt(seconds);
  } catch (const SingleStreamDecoder::EndOfFileException& e) {
    C10_THROW_ERROR(IndexError, e.what());
  }
  return makeOpsFrameOutput(result);
}

// Return the frame that is visible at a given index in the video.
OpsFrameOutput get_frame_at_index(at::Tensor& decoder, int64_t frame_index) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  auto result = videoDecoder->getFrameAtIndex(frame_index);
  return makeOpsFrameOutput(result);
}

// Return the frames at given indices for a given stream
OpsFrameBatchOutput get_frames_at_indices(
    at::Tensor& decoder,
    at::IntArrayRef frame_indices) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  std::vector<int64_t> frameIndicesVec(
      frame_indices.begin(), frame_indices.end());
  auto result = videoDecoder->getFramesAtIndices(frameIndicesVec);
  return makeOpsFrameBatchOutput(result);
}

// Return the frames inside a range as a single stacked Tensor. The range is
// defined as [start, stop).
OpsFrameBatchOutput get_frames_in_range(
    at::Tensor& decoder,
    int64_t start,
    int64_t stop,
    std::optional<int64_t> step = std::nullopt) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  auto result = videoDecoder->getFramesInRange(start, stop, step.value_or(1));
  return makeOpsFrameBatchOutput(result);
}

// Return the frames at given ptss for a given stream
OpsFrameBatchOutput get_frames_by_pts(
    at::Tensor& decoder,
    at::ArrayRef<double> timestamps) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  std::vector<double> timestampsVec(timestamps.begin(), timestamps.end());
  auto result = videoDecoder->getFramesPlayedAt(timestampsVec);
  return makeOpsFrameBatchOutput(result);
}

// Return the frames inside the range as a single stacked Tensor. The range is
// defined as [start_seconds, stop_seconds). The frames are stacked in pts
// order.
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
    std::optional<double> stop_seconds = std::nullopt) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  auto result =
      videoDecoder->getFramesPlayedInRangeAudio(start_seconds, stop_seconds);
  return makeOpsAudioFramesOutput(result);
}

at::Tensor wrapAudioEncoderPointerToTensor(
    std::unique_ptr<AudioEncoder> uniqueAudioEncoder) {
  AudioEncoder* encoder = uniqueAudioEncoder.release();

  auto deleter = [encoder](void*) { delete encoder; };
  at::Tensor tensor =
      at::from_blob(encoder, {sizeof(AudioEncoder*)}, deleter, {at::kLong});
  auto encoder_ = static_cast<AudioEncoder*>(tensor.mutable_data_ptr());
  TORCH_CHECK_EQ(encoder_, encoder) << "AudioEncoder=" << encoder_;
  return tensor;
}

AudioEncoder* unwrapTensorToGetAudioEncoder(at::Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(tensor.is_contiguous());
  void* buffer = tensor.mutable_data_ptr();
  AudioEncoder* encoder = static_cast<AudioEncoder*>(buffer);
  return encoder;
}

at::Tensor create_audio_encoder(
    const at::Tensor wf,
    int64_t sample_rate,
    std::string_view file_name,
    std::optional<int64_t> bit_rate = std::nullopt) {
  TORCH_CHECK(
      sample_rate <= std::numeric_limits<int>::max(),
      "sample_rate=",
      sample_rate,
      " is too large to be cast to an int.");
  std::unique_ptr<AudioEncoder> uniqueAudioEncoder =
      std::make_unique<AudioEncoder>(
          wf, static_cast<int>(sample_rate), file_name, bit_rate);
  return wrapAudioEncoderPointerToTensor(std::move(uniqueAudioEncoder));
}

void encode_audio(at::Tensor& encoder) {
  auto encoder_ = unwrapTensorToGetAudioEncoder(encoder);
  encoder_->encode();
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
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  return pts_seconds_to_test ==
      videoDecoder->getPtsSecondsForFrame(frame_index);
}

torch::Tensor _get_key_frame_indices(at::Tensor& decoder) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  return videoDecoder->getKeyFrameIndices();
}

// Get the metadata from the video as a string.
std::string get_json_metadata(at::Tensor& decoder) {
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);

  ContainerMetadata videoMetadata = videoDecoder->getContainerMetadata();
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

// Get the container metadata as a string.
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

// Get the stream metadata as a string.
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
  auto videoDecoder = unwrapTensorToGetDecoder(decoder);
  videoDecoder->scanFileAndUpdateMetadataAndIndex();
}

TORCH_LIBRARY_IMPL(torchcodec_ns, BackendSelect, m) {
  m.impl("create_from_file", &create_from_file);
  m.impl("create_audio_encoder", &create_audio_encoder);
  m.impl("create_from_tensor", &create_from_tensor);
  m.impl("_convert_to_tensor", &_convert_to_tensor);
  m.impl(
      "_get_json_ffmpeg_library_versions", &_get_json_ffmpeg_library_versions);
}

TORCH_LIBRARY_IMPL(torchcodec_ns, CPU, m) {
  m.impl("encode_audio", &encode_audio);
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
