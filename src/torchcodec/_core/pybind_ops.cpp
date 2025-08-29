// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <string>

#include "src/torchcodec/_core/AVIOFileLikeContext.h"
#include "src/torchcodec/_core/Encoder.h"
#include "src/torchcodec/_core/SingleStreamDecoder.h"
#include "src/torchcodec/_core/StreamOptions.h"
#include "src/torchcodec/_core/ValidationUtils.h"

namespace py = pybind11;

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

  auto avioContextHolder =
      std::make_unique<AVIOFileLikeContext>(file_like, /*isForWriting=*/false);

  SingleStreamDecoder* decoder =
      new SingleStreamDecoder(std::move(avioContextHolder), realSeek);
  return reinterpret_cast<int64_t>(decoder);
}

void encode_audio_to_file_like(
    int64_t data_ptr,
    const std::vector<int64_t>& shape,
    int64_t sample_rate,
    std::string_view format,
    py::object file_like,
    std::optional<int64_t> bit_rate = std::nullopt,
    std::optional<int64_t> num_channels = std::nullopt,
    std::optional<int64_t> desired_sample_rate = std::nullopt) {
  // We assume float32 *and* contiguity, this must be enforced by the caller.
  auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32);
  auto samples = torch::from_blob(
      reinterpret_cast<void*>(data_ptr), shape, tensor_options);

  AudioStreamOptions audioStreamOptions;
  audioStreamOptions.bitRate = validateOptionalInt64ToInt(bit_rate, "bit_rate");
  audioStreamOptions.numChannels =
      validateOptionalInt64ToInt(num_channels, "num_channels");
  audioStreamOptions.sampleRate =
      validateOptionalInt64ToInt(desired_sample_rate, "desired_sample_rate");

  auto avioContextHolder =
      std::make_unique<AVIOFileLikeContext>(file_like, /*isForWriting=*/true);

  AudioEncoder encoder(
      samples,
      validateInt64ToInt(sample_rate, "sample_rate"),
      format,
      std::move(avioContextHolder),
      audioStreamOptions);
  encoder.encode();
}

#ifndef PYBIND_OPS_MODULE_NAME
#error PYBIND_OPS_MODULE_NAME must be defined!
#endif

PYBIND11_MODULE(PYBIND_OPS_MODULE_NAME, m) {
  m.def("create_from_file_like", &create_from_file_like);
  m.def("encode_audio_to_file_like", &encode_audio_to_file_like);
}

} // namespace facebook::torchcodec
