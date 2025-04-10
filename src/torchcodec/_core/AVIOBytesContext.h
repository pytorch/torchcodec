// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/types.h>
#include "src/torchcodec/_core/AVIOContextHolder.h"

namespace facebook::torchcodec {

// Enables users to pass in the entire video as bytes. Our read and seek
// functions then traverse the bytes in memory.
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

class AVIOToTensorContext : public AVIOContextHolder {
 public:
  explicit AVIOToTensorContext();
  torch::Tensor getOutputTensor();

 private:
  // Should this class be tensor-aware? Or should we just store a uint8* buffer
  // instead of the tensor? If it's not tensor-aware it means we need to do the
  // (re)allocation outside of it. Same for the call to narrow().
  struct DataContext {
    torch::Tensor outputTensor;
    int64_t current;
  };

  static const int OUTPUT_TENSOR_SIZE = 5'000'000; // TODO-ENCODING handle this
  static int write(void* opaque, uint8_t* buf, int buf_size);
  static int64_t seek(void* opaque, int64_t offset, int whence);

  DataContext dataContext_;
};

} // namespace facebook::torchcodec
