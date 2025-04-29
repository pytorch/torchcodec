// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/AVIOBytesContext.h"
#include <torch/types.h>

namespace facebook::torchcodec {

AVIOBytesContext::AVIOBytesContext(const void* data, int64_t dataSize)
    : dataContext_{static_cast<const uint8_t*>(data), dataSize, 0} {
  TORCH_CHECK(data != nullptr, "Video data buffer cannot be nullptr!");
  TORCH_CHECK(dataSize > 0, "Video data size must be positive");
  createAVIOContext(&read, nullptr, &seek, &dataContext_);
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

  int64_t numBytesRead = std::min(
      static_cast<int64_t>(buf_size), dataContext->size - dataContext->current);

  TORCH_CHECK(
      numBytesRead >= 0,
      "Tried to read negative bytes: numBytesRead=",
      numBytesRead,
      ", size=",
      dataContext->size,
      ", current=",
      dataContext->current);

  if (numBytesRead == 0) {
    return AVERROR_EOF;
  }

  std::memcpy(buf, dataContext->data + dataContext->current, numBytesRead);
  dataContext->current += numBytesRead;
  return numBytesRead;
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

AVIOToTensorContext::AVIOToTensorContext()
    : dataContext_{
          torch::empty(
              {AVIOToTensorContext::INITIAL_TENSOR_SIZE},
              {torch::kUInt8}),
          0} {
  createAVIOContext(nullptr, &write, &seek, &dataContext_);
}

// The signature of this function is defined by FFMPEG.
int AVIOToTensorContext::write(void* opaque, const uint8_t* buf, int buf_size) {
  auto dataContext = static_cast<DataContext*>(opaque);

  int64_t bufSize = static_cast<int64_t>(buf_size);
  if (dataContext->current + bufSize > dataContext->outputTensor.numel()) {
    TORCH_CHECK(
        dataContext->outputTensor.numel() * 2 <=
            AVIOToTensorContext::MAX_TENSOR_SIZE,
        "We tried to allocate an output encoded tensor larger than ",
        AVIOToTensorContext::MAX_TENSOR_SIZE,
        " bytes. If you think this should be supported, please report.");

    // We double the size of the outpout tensor. Calling cat() may not be the
    // most efficient, but it's simple.
    dataContext->outputTensor =
        torch::cat({dataContext->outputTensor, dataContext->outputTensor});
  }

  TORCH_CHECK(
      dataContext->current + bufSize <= dataContext->outputTensor.numel(),
      "Re-allocation of the output tensor didn't work. ",
      "This should not happen, please report on TorchCodec bug tracker");

  uint8_t* outputTensorData = dataContext->outputTensor.data_ptr<uint8_t>();
  std::memcpy(outputTensorData + dataContext->current, buf, bufSize);
  dataContext->current += bufSize;
  return buf_size;
}

// The signature of this function is defined by FFMPEG.
// Note: This `seek()` implementation is very similar to that of
// AVIOBytesContext. We could consider merging both classes, or do some kind of
// refac, but this doesn't seem worth it ATM.
int64_t AVIOToTensorContext::seek(void* opaque, int64_t offset, int whence) {
  auto dataContext = static_cast<DataContext*>(opaque);
  int64_t ret = -1;

  switch (whence) {
    case AVSEEK_SIZE:
      ret = dataContext->outputTensor.numel();
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

torch::Tensor AVIOToTensorContext::getOutputTensor() {
  return dataContext_.outputTensor.narrow(
      /*dim=*/0, /*start=*/0, /*length=*/dataContext_.current);
}

} // namespace facebook::torchcodec
