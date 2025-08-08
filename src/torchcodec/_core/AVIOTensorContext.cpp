// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/AVIOTensorContext.h"
#include <torch/types.h>

namespace facebook::torchcodec {

namespace {

constexpr int64_t INITIAL_TENSOR_SIZE = 10'000'000; // 10 MB
constexpr int64_t MAX_TENSOR_SIZE = 320'000'000; // 320 MB

// The signature of this function is defined by FFMPEG.
int read(void* opaque, uint8_t* buf, int buf_size) {
  auto tensorContext = static_cast<detail::TensorContext*>(opaque);
  TORCH_CHECK(
      tensorContext->current <= tensorContext->data.numel(),
      "Tried to read outside of the buffer: current=",
      tensorContext->current,
      ", size=",
      tensorContext->data.numel());

  int64_t numBytesRead = std::min(
      static_cast<int64_t>(buf_size),
      tensorContext->data.numel() - tensorContext->current);

  TORCH_CHECK(
      numBytesRead >= 0,
      "Tried to read negative bytes: numBytesRead=",
      numBytesRead,
      ", size=",
      tensorContext->data.numel(),
      ", current=",
      tensorContext->current);

  if (numBytesRead == 0) {
    return AVERROR_EOF;
  }

  std::memcpy(
      buf,
      tensorContext->data.data_ptr<uint8_t>() + tensorContext->current,
      numBytesRead);
  tensorContext->current += numBytesRead;
  return numBytesRead;
}

// The signature of this function is defined by FFMPEG.
int write(void* opaque, const uint8_t* buf, int buf_size) {
  auto tensorContext = static_cast<detail::TensorContext*>(opaque);

  int64_t bufSize = static_cast<int64_t>(buf_size);
  if (tensorContext->current + bufSize > tensorContext->data.numel()) {
    TORCH_CHECK(
        tensorContext->data.numel() * 2 <= MAX_TENSOR_SIZE,
        "We tried to allocate an output encoded tensor larger than ",
        MAX_TENSOR_SIZE,
        " bytes. If you think this should be supported, please report.");

    // We double the size of the outpout tensor. Calling cat() may not be the
    // most efficient, but it's simple.
    tensorContext->data =
        torch::cat({tensorContext->data, tensorContext->data});
  }

  TORCH_CHECK(
      tensorContext->current + bufSize <= tensorContext->data.numel(),
      "Re-allocation of the output tensor didn't work. ",
      "This should not happen, please report on TorchCodec bug tracker");

  uint8_t* outputTensorData = tensorContext->data.data_ptr<uint8_t>();
  std::memcpy(outputTensorData + tensorContext->current, buf, bufSize);
  tensorContext->current += bufSize;
  return buf_size;
}

// The signature of this function is defined by FFMPEG.
int64_t seek(void* opaque, int64_t offset, int whence) {
  auto tensorContext = static_cast<detail::TensorContext*>(opaque);
  int64_t ret = -1;

  switch (whence) {
    case AVSEEK_SIZE:
      ret = tensorContext->data.numel();
      break;
    case SEEK_SET:
      tensorContext->current = offset;
      ret = offset;
      break;
    default:
      break;
  }

  return ret;
}

} // namespace

AVIOFromTensorContext::AVIOFromTensorContext(torch::Tensor data)
    : tensorContext_{data, 0} {
  TORCH_CHECK(data.numel() > 0, "data must not be empty");
  TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
  TORCH_CHECK(data.scalar_type() == torch::kUInt8, "data must be kUInt8");
  createAVIOContext(
      &read, nullptr, &seek, &tensorContext_, /*isForWriting=*/false);
}

AVIOToTensorContext::AVIOToTensorContext()
    : tensorContext_{torch::empty({INITIAL_TENSOR_SIZE}, {torch::kUInt8}), 0} {
  createAVIOContext(
      nullptr, &write, &seek, &tensorContext_, /*isForWriting=*/true);
}

torch::Tensor AVIOToTensorContext::getOutputTensor() {
  return tensorContext_.data.narrow(
      /*dim=*/0, /*start=*/0, /*length=*/tensorContext_.current);
}

} // namespace facebook::torchcodec
