#include <torch/types.h>
#include "src/torchcodec/_core/DeviceInterface.h"

namespace facebook::torchcodec {

// This file is linked with the CPU-only version of torchcodec.
// So all functions will throw an error because they should only be called if
// the device is not CPU.

[[noreturn]] void throw_unsupported_device_error(const torch::Device& device) {
  TORCH_CHECK(
      device.type() != torch::kCPU,
      "Device functions should only be called if the device is not CPU.")
  TORCH_CHECK(false, "Unsupported device: " + device.str());
}

void convert_avframe_to_frame_output_on_cuda(
    const torch::Device& device,
    [[maybe_unused]] const SingleStreamDecoder::VideoStreamOptions&
        video_stream_options,
    [[maybe_unused]] UniqueAVFrame& avframe,
    [[maybe_unused]] SingleStreamDecoder::FrameOutput& frame_output,
    [[maybe_unused]] std::optional<torch::Tensor> pre_allocated_output_tensor) {
  throw_unsupported_device_error(device);
}

void initialize_context_on_cuda(
    const torch::Device& device,
    [[maybe_unused]] AVCodecContext* codec_context) {
  throw_unsupported_device_error(device);
}

void release_context_on_cuda(
    const torch::Device& device,
    [[maybe_unused]] AVCodecContext* codec_context) {
  throw_unsupported_device_error(device);
}

std::optional<const AVCodec*> find_cuda_codec(
    const torch::Device& device,
    [[maybe_unused]] const AVCodecID& codec_id) {
  throw_unsupported_device_error(device);
}

} // namespace facebook::torchcodec
