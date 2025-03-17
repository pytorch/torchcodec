// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <string>

#include "src/torchcodec/decoders/_core/AVIOFileLikeContext.h"
#include "src/torchcodec/decoders/_core/VideoDecoder.h"

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
  VideoDecoder::SeekMode realSeek = VideoDecoder::SeekMode::exact;
  if (seek_mode.has_value()) {
    realSeek = seekModeFromString(seek_mode.value());
  }

  auto contextHolder = std::make_unique<AVIOFileLikeContext>(file_like);

  VideoDecoder* decoder = new VideoDecoder(std::move(contextHolder), realSeek);
  return reinterpret_cast<int64_t>(decoder);
}

#ifndef TORCHCODEC_PYBIND
#error TORCHCODEC_PYBIND must be defined.
#endif

PYBIND11_MODULE(TORCHCODEC_PYBIND, m) {
  m.def("create_from_file_like", &create_from_file_like);
}

} // namespace facebook::torchcodec
