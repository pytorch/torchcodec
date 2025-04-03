// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/AVIOBytesContext.h"
#include <torch/types.h>

namespace facebook::torchcodec {

AVIOBytesContext::AVIOBytesContext(const void* data, int64_t data_size)
    : data_context_{static_cast<const uint8_t*>(data), data_size, 0} {
  TORCH_CHECK(data != nullptr, "Video data buffer cannot be nullptr!");
  TORCH_CHECK(dataSize > 0, "Video data size must be positive");
  create_avio_context(&read, &seek, &dataContext_);
}

// The signature of this function is defined by FFMPEG.
int AVIOBytesContext::read(void* opaque, uint8_t* buf, int buf_size) {
  auto data_context = static_cast<_data_context*>(opaque);
  TORCH_CHECK(
      data_context->current <= data_context->size,
      "Tried to read outside of the buffer: current=",
      data_context->current,
      ", size=",
      data_context->size);

  int64_t num_bytes_read = std::min(
      static_cast<int64_t>(buf_size),
      data_context->size - data_context->current);

  TORCH_CHECK(
      num_bytes_read >= 0,
      "Tried to read negative bytes: num_bytes_read=",
      num_bytes_read,
      ", size=",
      data_context->size,
      ", current=",
      data_context->current);

  if (numBytesRead == 0) {
    return AVERROR_EOF;
  }

  std::memcpy(buf, data_context->data + data_context->current, num_bytes_read);
  data_context->current += num_bytes_read;
  return num_bytes_read;
}

// The signature of this function is defined by FFMPEG.
int64_t AVIOBytesContext::seek(void* opaque, int64_t offset, int whence) {
  auto data_context = static_cast<_data_context*>(opaque);
  int64_t ret = -1;

  switch (whence) {
    case AVSEEK_SIZE:
      ret = data_context->size;
      break;
    case SEEK_SET:
      data_context->current = offset;
      ret = offset;
      break;
    default:
      break;
  }

  return ret;
}

} // namespace facebook::torchcodec
