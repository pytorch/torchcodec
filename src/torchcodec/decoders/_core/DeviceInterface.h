// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/types.h>
#include <memory>
#include <stdexcept>
#include <string>
#include "src/torchcodec/decoders/_core/VideoDecoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
}

namespace facebook::torchcodec {

// Note that all these device functions should only be called if the device is
// not a CPU device. CPU device functions are already implemented in the
// VideoDecoder implementation.
// These functions should only be called from within an if block like this:
// if (device.type() != torch::kCPU) {
//   deviceFunction(device, ...);
// }

// Initialize the hardware device that is specified in `device`. Some builds
// support CUDA and others only support CPU.
void initializeContextOnCuda(
    const torch::Device& device,
    AVCodecContext* codecContext);

void convertAVFrameToDecodedOutputOnCuda(
    const torch::Device& device,
    const VideoDecoder::VideoStreamDecoderOptions& options,
    VideoDecoder::RawDecodedOutput& rawOutput,
    VideoDecoder::DecodedOutput& output,
    std::optional<torch::Tensor> preAllocatedOutputTensor = std::nullopt);

void releaseContextOnCuda(
    const torch::Device& device,
    AVCodecContext* codecContext);

} // namespace facebook::torchcodec
