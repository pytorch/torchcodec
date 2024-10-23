// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/types.h>
#include <optional>

namespace facebook::torchcodec {

// The following functions are useful for calling the Pytorch C++ ops from C++
// code. For example, the decoder can be created like so:
// auto createDecoderOp =
//      torch::Dispatcher::singleton()
//          .findSchemaOrThrow("torchcodec_ns::create_from_file", "")
//          .typed<decltype(VideoDecoder_create)>();
// auto decoderTensor = createDecoderOp.call(videoPath);

// Create a VideoDecoder from file and wrap the pointer in a tensor.
at::Tensor create_from_file(c10::string_view filename);

at::Tensor create_from_tensor(at::Tensor video_tensor);

// This API is C++ only and will not be exposed via custom ops, use
// videodecoder_create_from_bytes in Python
at::Tensor create_from_buffer(const void* buffer, size_t length);

// Add a new video stream at `stream_index` using the provided options.
void add_video_stream(
    at::Tensor& decoder,
    std::optional<int64_t> width = std::nullopt,
    std::optional<int64_t> height = std::nullopt,
    std::optional<int64_t> num_threads = std::nullopt,
    std::optional<c10::string_view> dimension_order = std::nullopt,
    std::optional<int64_t> stream_index = std::nullopt,
    std::optional<c10::string_view> device = std::nullopt);

void _add_video_stream(
    at::Tensor& decoder,
    std::optional<int64_t> width = std::nullopt,
    std::optional<int64_t> height = std::nullopt,
    std::optional<int64_t> num_threads = std::nullopt,
    std::optional<c10::string_view> dimension_order = std::nullopt,
    std::optional<int64_t> stream_index = std::nullopt,
    std::optional<c10::string_view> device = std::nullopt,
    std::optional<c10::string_view> color_conversion_library = std::nullopt);

// Seek to a particular presentation timestamp in the video in seconds.
void seek_to_pts(at::Tensor& decoder, double seconds);

// The elements of this tuple are all tensors that represent a single frame:
//   1. The frame data, which is a multidimensional tensor.
//   2. A single float value for the pts in seconds.
//   3. A single float value for the duration in seconds.
// The reason we use Tensors for the second and third values is so we can run
// under torch.compile().
using OpsDecodedOutput = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

// All elements of this tuple are tensors of the same leading dimension. The
// tuple represents the frames for N total frames, where N is the dimension of
// each stacked tensor. The elments are:
//   1. Stacked tensor of data for all N frames. Each frame is also a
//   multidimensional tensor.
//   2. Tensor of N pts values in seconds, where each pts is a single
//   float.
//   3. Tensor of N durationis in seconds, where each duration is a
//   single float.
using OpsBatchDecodedOutput = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

// Return the frame that is visible at a given timestamp in seconds. Each frame
// in FFMPEG has a presentation timestamp and a duration. The frame visible at a
// given timestamp T has T >= PTS and T < PTS + Duration.
OpsDecodedOutput get_frame_at_pts(at::Tensor& decoder, double seconds);

// Return the frames at given ptss for a given stream
OpsBatchDecodedOutput get_frames_by_pts(
    at::Tensor& decoder,
    int64_t stream_index,
    at::ArrayRef<double> timestamps);

// Return the frame that is visible at a given index in the video.
OpsDecodedOutput get_frame_at_index(
    at::Tensor& decoder,
    int64_t stream_index,
    int64_t frame_index);

// Get the next frame from the video as a tuple that has the frame data, pts and
// duration as tensors.
OpsDecodedOutput get_next_frame(at::Tensor& decoder);

// Return the frames at given indices for a given stream
OpsBatchDecodedOutput get_frames_at_indices(
    at::Tensor& decoder,
    int64_t stream_index,
    at::IntArrayRef frame_indices);

// Return the frames inside a range as a single stacked Tensor. The range is
// defined as [start, stop).
OpsBatchDecodedOutput get_frames_in_range(
    at::Tensor& decoder,
    int64_t stream_index,
    int64_t start,
    int64_t stop,
    std::optional<int64_t> step = std::nullopt);

// Return the frames inside the range as a single stacked Tensor. The range is
// defined as [start_seconds, stop_seconds). The frames are stacked in pts
// order.
OpsBatchDecodedOutput get_frames_by_pts_in_range(
    at::Tensor& decoder,
    int64_t stream_index,
    double start_seconds,
    double stop_seconds);

// For testing only. We need to implement this operation as a core library
// function because what we're testing is round-tripping pts values as
// double-precision floating point numbers from C++ to Python and back to C++.
// We want to make sure that the value is preserved exactly, bit-for-bit, during
// this process.
//
// Returns true if for the given decoder, in the stream stream_index, the pts
// value when converted to seconds as a double is exactly pts_seconds_to_test.
// Returns false otherwise.
bool _test_frame_pts_equality(
    at::Tensor& decoder,
    int64_t stream_index,
    int64_t frame_index,
    double pts_seconds_to_test);

// Get the metadata from the video as a string.
std::string get_json_metadata(at::Tensor& decoder);

// Get the container metadata as a string.
std::string get_container_json_metadata(at::Tensor& decoder);

// Get the stream metadata as a string.
std::string get_stream_json_metadata(at::Tensor& decoder, int64_t stream_index);

// Returns version information about the various FFMPEG libraries that are
// loaded in the program's address space.
std::string _get_json_ffmpeg_library_versions();

// Scans video packets to get more accurate metadata like frame count, exact
// keyframe positions, etc. Exact keyframe positions are useful for efficient
// accurate seeking. Note that this function reads the entire video but it does
// not decode frames. Reading a video file is much cheaper than decoding it.
void scan_all_streams_to_update_metadata(at::Tensor& decoder);

} // namespace facebook::torchcodec
