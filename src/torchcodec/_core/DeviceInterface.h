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
#include "src/torchcodec/_core/SingleStreamDecoder.h"

namespace facebook::torchcodec {

// Note that all these device functions should only be called if the device is
// not a CPU device. CPU device functions are already implemented in the
// SingleStreamDecoder implementation.
// These functions should only be called from within an if block like this:
// if (device.type() != torch::kCPU) {
// deviceFunction(device, ...);
// }

// Initialize the hardware device that is specified in `device`. Some builds
// support CUDA and others only support CPU.
void initialize_context_on_cuda(
    const torch::Device& device,
    AVCodecContext* codec_context);

void convert_avframe_to_frame_output_on_cuda(
    const torch::Device& device,
    const SingleStreamDecoder::VideoStreamOptions& video_stream_options,
    UniqueAVFrame& avframe,
    SingleStreamDecoder::FrameOutput& frame_output,
    std::optional<torch::Tensor> pre_allocated_output_tensor = std::nullopt);

void release_context_on_cuda(
    const torch::Device& device,
    AVCodecContext* codec_context);

std::optional<const AVCodec*> find_cuda_codec(
    const torch::Device& device,
    const AVCodecID& codec_id);

} // namespace facebook::torchcodec
