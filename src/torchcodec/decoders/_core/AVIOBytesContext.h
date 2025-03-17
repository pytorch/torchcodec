// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "src/torchcodec/decoders/_core/AVIOContextHolder.h"

namespace facebook::torchcodec {

// TODO: make comment below better
// memory buffer that is passed in.
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

} // namespace facebook::torchcodec
