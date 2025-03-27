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
#include "FFMPEGCommon.h"
#include "src/torchcodec/decoders/_core/VideoDecoder.h"

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

void initializeContextOnXpu(
    const torch::Device& device,
    AVCodecContext* codecContext);

void convertAVFrameToFrameOutputOnCuda(
    const torch::Device& device,
    const VideoDecoder::VideoStreamOptions& videoStreamOptions,
    UniqueAVFrame& avFrame,
    VideoDecoder::FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor = std::nullopt);

void convertAVFrameToFrameOutputOnXpu(
    const torch::Device& device,
    const VideoDecoder::VideoStreamOptions& videoStreamOptions,
    UniqueAVFrame& avFrame,
    VideoDecoder::FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor = std::nullopt);

void releaseContextOnCuda(
    const torch::Device& device,
    AVCodecContext* codecContext);

void releaseContextOnXpu(
    const torch::Device& device,
    AVCodecContext* codecContext);

std::optional<const AVCodec*> findCudaCodec(
    const torch::Device& device,
    const AVCodecID& codecId);

std::optional<const AVCodec*> findXpuCodec(
    const torch::Device& device,
    const AVCodecID& codecId);

} // namespace facebook::torchcodec
