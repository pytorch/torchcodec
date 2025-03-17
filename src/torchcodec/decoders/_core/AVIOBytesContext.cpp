// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/types.h>
#include "src/torchcodec/decoders/_core/AVIOBytesContext.h"

namespace facebook::torchcodec {

AVIOBytesContext::AVIOBytesContext(const void* data, int64_t dataSize)
    : dataContext_{static_cast<const uint8_t*>(data), dataSize, 0} {
  TORCH_CHECK(data != nullptr, "Video data buffer cannot be nullptr!");
  TORCH_CHECK(dataSize > 0, "Video data size must be positive");
  createAVIOContext(&read, &seek, &dataContext_);
}

// The signature of this function is defined by FFMPEG.
int AVIOBytesContext::read(void* opaque, uint8_t* buf, int buf_size) {
  auto dataContext = static_cast<DataContext*>(opaque);
  TORCH_CHECK(
      dataContext->current <= dataContext->size,
      "Tried to read outside of the buffer: current=",
      dataContext->current,
      ", size=",
      dataContext->size);

  buf_size = FFMIN(
      buf_size, static_cast<int>(dataContext->size - dataContext->current));
  TORCH_CHECK(
      buf_size >= 0,
      "Tried to read negative bytes: buf_size=",
      buf_size,
      ", size=",
      dataContext->size,
      ", current=",
      dataContext->current);

  if (!buf_size) {
    return AVERROR_EOF;
  }
  memcpy(buf, dataContext->data + dataContext->current, buf_size);
  dataContext->current += buf_size;
  return buf_size;
}

// The signature of this function is defined by FFMPEG.
int64_t AVIOBytesContext::seek(void* opaque, int64_t offset, int whence) {
  auto dataContext = static_cast<DataContext*>(opaque);
  int64_t ret = -1;

  switch (whence) {
    case AVSEEK_SIZE:
      ret = dataContext->size;
      break;
    case SEEK_SET:
      dataContext->current = offset;
      ret = offset;
      break;
    default:
      break;
  }

  return ret;
}

} // namespace facebook::torchcodec
