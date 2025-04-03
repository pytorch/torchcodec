// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/AVIOContextHolder.h"
#include <torch/types.h>

namespace facebook::torchcodec {

void AVIOContextHolder::create_avio_context(
    AVIOReadFunction read,
    AVIOSeekFunction seek,
    void* held_data,
    int buffer_size) {
  TORCH_CHECK(
      buffer_size > 0,
      "Buffer size must be greater than 0; is " + std::to_string(buffer_size));
  auto buffer = static_cast<uint8_t*>(av_malloc(buffer_size));
  TORCH_CHECK(
      buffer != nullptr,
      "Failed to allocate buffer of size " + std::to_string(buffer_size));

  avio_context_.reset(avio_alloc_context(
      buffer,
      buffer_size,
      0,
      held_data,
      read,
      nullptr, // write function; not supported yet
      seek));

  if (!avioContext_) {
    av_freep(&buffer);
    TORCH_CHECK(false, "Failed to allocate AVIOContext");
  }
}

AVIOContextHolder::~AVIOContextHolder() {
  if (avioContext_) {
    av_freep(&avio_context_->buffer);
  }
}

AVIOContext* AVIOContextHolder::get_avio_context() {
  return avio_context_.get();
}

} // namespace facebook::torchcodec
