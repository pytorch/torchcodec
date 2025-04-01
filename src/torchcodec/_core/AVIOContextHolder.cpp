// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/AVIOContextHolder.h"
#include <torch/types.h>

namespace facebook::torchcodec {

void AVIOContextHolder::createAVIOContext(
    AVIOReadFunction read,
    AVIOSeekFunction seek,
    void* heldData,
    int bufferSize) {
  TORCH_CHECK(
      bufferSize > 0,
      "Buffer size must be greater than 0; is " + std::to_string(bufferSize));
  auto buffer = static_cast<uint8_t*>(av_malloc(bufferSize));
  TORCH_CHECK(
      buffer != nullptr,
      "Failed to allocate buffer of size " + std::to_string(bufferSize));

  avioContext_.reset(avio_alloc_context(
      buffer,
      bufferSize,
      0,
      heldData,
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
    av_freep(&avioContext_->buffer);
  }
}

AVIOContext* AVIOContextHolder::getAVIOContext() {
  return avioContext_.get();
}

} // namespace facebook::torchcodec
