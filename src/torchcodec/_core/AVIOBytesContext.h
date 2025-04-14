// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/types.h>
#include "src/torchcodec/_core/AVIOContextHolder.h"

namespace facebook::torchcodec {

// For Decoding: enables users to pass in the entire video or audio as bytes.
// Our read and seek functions then traverse the bytes in memory.
class AVIOBytesContext : public AVIOContextHolder {
 public:
  explicit AVIOBytesContext(const void* data, int64_t dataSize);

 private:
  struct DataContext {
    const uint8_t* data;
    int64_t size;
    int64_t current;
  };

  static int read(void* opaque, uint8_t* buf, int buf_size);
  static int64_t seek(void* opaque, int64_t offset, int whence);

  DataContext dataContext_;
};

// For Encoding: used to encode into an output uint8 (bytes) tensor.
class AVIOToTensorContext : public AVIOContextHolder {
 public:
  explicit AVIOToTensorContext();
  torch::Tensor getOutputTensor();

 private:
  struct DataContext {
    torch::Tensor outputTensor;
    int64_t current;
  };

  static const int INITIAL_TENSOR_SIZE = 10'000'000; // 10MB
  static const int MAX_TENSOR_SIZE = 320'000'000; // 320 MB
  static int write(void* opaque, const uint8_t* buf, int buf_size);
  // We need to expose seek() for some formats like mp3.
  static int64_t seek(void* opaque, int64_t offset, int whence);

  DataContext dataContext_;
};

} // namespace facebook::torchcodec
