// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <cstdint>
#include <string>

#include "src/torchcodec/_core/AVIOFileLikeContext.h"
#include "src/torchcodec/_core/Encoder.h"
#include "src/torchcodec/_core/SingleStreamDecoder.h"
#include "src/torchcodec/_core/StreamOptions.h"

namespace py = pybind11;
using namespace py::literals;

namespace facebook::torchcodec {

// In principle, this should be able to return a tensor. But when we try that,
// we run into the bug reported here:
//
//   https://github.com/pytorch/pytorch/issues/136664
//
// So we instead launder the pointer through an int, and then use a conversion
// function on the custom ops side to launder that int into a tensor.
int64_t create_from_file_like(
    py::object file_like,
    std::optional<std::string_view> seek_mode) {
  SingleStreamDecoder::SeekMode realSeek = SingleStreamDecoder::SeekMode::exact;
  if (seek_mode.has_value()) {
    realSeek = seekModeFromString(seek_mode.value());
  }

  auto avioContextHolder = std::make_unique<AVIOFileLikeContext>(file_like);

  SingleStreamDecoder* decoder =
      new SingleStreamDecoder(std::move(avioContextHolder), realSeek);
  return reinterpret_cast<int64_t>(decoder);
}

int64_t encode_audio_to_file_like(
    int64_t data_ptr,
    const std::vector<int64_t>& shape,
    int64_t sample_rate,
    std::string_view format,
    py::object file_like,
    std::optional<int64_t> bit_rate = std::nullopt,
    std::optional<int64_t> num_channels = std::nullopt) {
  // Create tensor from existing data pointer (enforcing float32)
  auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32);
  auto samples = torch::from_blob(
      reinterpret_cast<void*>(data_ptr), shape, tensor_options);

  AudioStreamOptions audioStreamOptions;
  audioStreamOptions.bitRate = bit_rate;
  audioStreamOptions.numChannels = num_channels;

  auto avioContextHolder = AVIOFileLikeContext::createForWriting(file_like);

  AudioEncoder encoder(
      samples,
      static_cast<int>(sample_rate),
      format,
      std::move(avioContextHolder),
      audioStreamOptions);
  encoder.encode();

  // Return 0 to indicate success
  return 0;
}

PYBIND11_MODULE(decoder_core_pybind_ops, m) {
  m.def("create_from_file_like", &create_from_file_like);
  m.def(
      "encode_audio_to_file_like",
      &encode_audio_to_file_like,
      "data_ptr"_a,
      "shape"_a,
      "sample_rate"_a,
      "format"_a,
      "file_like"_a,
      "bit_rate"_a = py::none(),
      "num_channels"_a = py::none());
}

} // namespace facebook::torchcodec
