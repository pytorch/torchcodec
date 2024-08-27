// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "src/torchcodec/decoders/_core/VideoDecoder.h"
#include <torch/torch.h>
#include <torch/types.h>

#ifdef ENABLE_NVTX
#include <nvtx3/nvtx3.hpp>
#endif

namespace facebook::torchcodec {

#ifdef ENABLE_NVTX
#define NVTX_SCOPED_RANGE(Annotation) nvtx3::scoped_range loop{Annotation}
#else
#define NVTX_SCOPED_RANGE(Annotation) do {} while (0)
#endif

AVBufferRef* initializeCudaContext(const torch::Device& device);

torch::Tensor convertFrameToTensorUsingCuda(
    const AVCodecContext* codecContext,
    const VideoDecoder::VideoStreamDecoderOptions& options,
    const AVFrame* src);

} // facebook::torchcodec

