// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/types.h>
#include "src/torchcodec/_core/AVIOContextHolder.h"

namespace facebook::torchcodec {

namespace detail {

struct TensorContext {
  torch::Tensor data;
  int64_t current;
};

} // namespace detail

// For Decoding: enables users to pass in the entire video or audio as bytes.
// Our read and seek functions then traverse the bytes in memory.
class AVIOFromTensorContext : public AVIOContextHolder {
 public:
  explicit AVIOFromTensorContext(torch::Tensor data);

 private:
  detail::TensorContext tensorContext_;
};

// For Encoding: used to encode into an output uint8 (bytes) tensor.
class AVIOToTensorContext : public AVIOContextHolder {
 public:
  explicit AVIOToTensorContext();
  torch::Tensor getOutputTensor();

 private:
  detail::TensorContext tensorContext_;
};

} // namespace facebook::torchcodec
