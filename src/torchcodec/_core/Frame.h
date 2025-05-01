// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/types.h>
#include "src/torchcodec/_core/FFMPEGCommon.h"
#include "src/torchcodec/_core/Metadata.h"
#include "src/torchcodec/_core/StreamOptions.h"

namespace facebook::torchcodec {

// All public video decoding entry points return either a FrameOutput or a
// FrameBatchOutput.
// They are the equivalent of the user-facing Frame and FrameBatch classes in
// Python. They contain RGB decoded frames along with some associated data
// like PTS and duration.
// FrameOutput is also relevant for audio decoding, typically as the output of
// getNextFrame(), or as a temporary output variable.
struct FrameOutput {
  // data shape is:
  // - 3D (C, H, W) or (H, W, C) for videos
  // - 2D (numChannels, numSamples) for audio
  torch::Tensor data;
  double ptsSeconds;
  double durationSeconds;
};

struct FrameBatchOutput {
  torch::Tensor data; // 4D: of shape NCHW or NHWC.
  torch::Tensor ptsSeconds; // 1D of shape (N,)
  torch::Tensor durationSeconds; // 1D of shape (N,)

  explicit FrameBatchOutput(
      int64_t numFrames,
      const VideoStreamOptions& videoStreamOptions,
      const StreamMetadata& streamMetadata);
};

struct AudioFramesOutput {
  torch::Tensor data; // shape is (numChannels, numSamples)
  double ptsSeconds;
};

// --------------------------------------------------------------------------
// FRAME TENSOR ALLOCATION APIs
// --------------------------------------------------------------------------

// Note [Frame Tensor allocation and height and width]
//
// We always allocate [N]HWC tensors. The low-level decoding functions all
// assume HWC tensors, since this is what FFmpeg natively handles. It's up to
// the high-level decoding entry-points to permute that back to CHW, by calling
// maybePermuteHWC2CHW().
//
// Also, importantly, the way we figure out the the height and width of the
// output frame tensor varies, and depends on the decoding entry-point. In
// *decreasing order of accuracy*, we use the following sources for determining
// height and width:
// - getHeightAndWidthFromResizedAVFrame(). This is the height and width of the
//   AVframe, *post*-resizing. This is only used for single-frame decoding APIs,
//   on CPU, with filtergraph.
// - getHeightAndWidthFromOptionsOrAVFrame(). This is the height and width from
//   the user-specified options if they exist, or the height and width of the
//   AVFrame *before* it is resized. In theory, i.e. if there are no bugs within
//   our code or within FFmpeg code, this should be exactly the same as
//   getHeightAndWidthFromResizedAVFrame(). This is used by single-frame
//   decoding APIs, on CPU with swscale, and on GPU.
// - getHeightAndWidthFromOptionsOrMetadata(). This is the height and width from
//   the user-specified options if they exist, or the height and width form the
//   stream metadata, which itself got its value from the CodecContext, when the
//   stream was added. This is used by batch decoding APIs, for both GPU and
//   CPU.
//
// The source of truth for height and width really is the (resized) AVFrame: it
// comes from the decoded ouptut of FFmpeg. The info from the metadata (i.e.
// from the CodecContext) may not be as accurate. However, the AVFrame is only
// available late in the call stack, when the frame is decoded, while the
// CodecContext is available early when a stream is added. This is why we use
// the CodecContext for pre-allocating batched output tensors (we could
// pre-allocate those only once we decode the first frame to get the info frame
// the AVFrame, but that's a more complex logic).
//
// Because the sources for height and width may disagree, we may end up with
// conflicts: e.g. if we pre-allocate a batch output tensor based on the
// metadata info, but the decoded AVFrame has a different height and width.
// it is very important to check the height and width assumptions where the
// tensors memory is used/filled in order to avoid segfaults.

struct FrameDims {
  int height;
  int width;

  FrameDims(int h, int w) : height(h), width(w) {}
};

// There's nothing preventing you from calling this on a non-resized frame, but
// please don't.
FrameDims getHeightAndWidthFromResizedAVFrame(const AVFrame& resizedAVFrame);

FrameDims getHeightAndWidthFromOptionsOrMetadata(
    const VideoStreamOptions& videoStreamOptions,
    const StreamMetadata& streamMetadata);

FrameDims getHeightAndWidthFromOptionsOrAVFrame(
    const VideoStreamOptions& videoStreamOptions,
    const UniqueAVFrame& avFrame);

torch::Tensor allocateEmptyHWCTensor(
    int height,
    int width,
    torch::Device device,
    std::optional<int> numFrames = std::nullopt);

} // namespace facebook::torchcodec
