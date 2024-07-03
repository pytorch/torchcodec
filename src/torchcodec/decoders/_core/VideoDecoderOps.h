// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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
    std::optional<c10::string_view> shape = std::nullopt,
    std::optional<int64_t> stream_index = std::nullopt);

// Seek to a particular presentation timestamp in the video in seconds.
void seek_to_pts(at::Tensor& decoder, double seconds);

// Return the frame that is visible at a given timestamp in seconds. Each frame
// in FFMPEG has a presentation timestamp and a duration. The frame visible at a
// given timestamp T has T >= PTS and T < PTS + Duration.
at::Tensor get_frame_at_pts(at::Tensor& decoder, double seconds);

// Return the frame that is visible at a given index in the video.
at::Tensor get_frame_at_index(
    at::Tensor& decoder,
    int64_t stream_index,
    int64_t frame_index);

// Return the frame along with pts and duration that is visible at a given index
// in the video.
std::tuple<at::Tensor, double, double> get_frame_with_info_at_index(
    at::Tensor& decoder,
    int64_t stream_index,
    int64_t frame_index);

// Return the frames at a given index for a given stream as a single stacked
// Tensor.
at::Tensor get_frames_at_indices(
    at::Tensor& decoder,
    int64_t stream_index,
    at::IntArrayRef frame_indices);

// Return the frames inside a range as a single stacked Tensor. The range is
// defined as [start, stop).
at::Tensor get_frames_in_range(
    at::Tensor& decoder,
    int64_t stream_index,
    int64_t start,
    int64_t stop,
    std::optional<int64_t> step = std::nullopt);

// Get the next frame from the video as a tensor.
at::Tensor get_next_frame(at::Tensor& decoder);

// Get the metadata from the video as a string.
std::string get_json_metadata(at::Tensor& decoder);

// Get the container metadata as a string.
std::string get_container_json_metadata(at::Tensor& decoder);

// Get the stream metadata as a string.
std::string get_stream_json_metadata(at::Tensor& decoder);

// Returns version information about the various FFMPEG libraries that are
// loaded in the program's address space.
std::string _get_json_ffmpeg_library_versions();

} // namespace facebook::torchcodec
