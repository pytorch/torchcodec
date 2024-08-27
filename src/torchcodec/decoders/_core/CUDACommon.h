// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/torch.h>
#include <torch/types.h>
#include "src/torchcodec/decoders/_core/VideoDecoder.h"

#ifdef ENABLE_NVTX
#include <nvtx3/nvtx3.hpp>
#endif

// The API for general code to access CUDA specific behaviors. CUDA specific
// behaviors require CUDA specific definitions which are only available on
// systems with CUDA installed. Hence, CUDA specific behaviors have to be
// guarded with ifdefs.
//
// In order to prevent ifdefs in general code, we create an API with a function
// for each behavior we need. General code can call the API, as the correct
// guards happen internally. General code still needs to check in general code
// if CUDA is being used, as the functions will throw an exception if CUDA is
// not available.

namespace facebook::torchcodec {

#ifdef ENABLE_NVTX
#define NVTX_SCOPED_RANGE(Annotation) nvtx3::scoped_range loop{Annotation}
#else
#define NVTX_SCOPED_RANGE(Annotation) \
  do {                                \
  } while (0)
#endif

AVBufferRef* initializeCudaContext(const torch::Device& device);

torch::Tensor convertFrameToTensorUsingCuda(
    const AVCodecContext* codecContext,
    const VideoDecoder::VideoStreamDecoderOptions& options,
    const AVFrame* src);

} // namespace facebook::torchcodec
