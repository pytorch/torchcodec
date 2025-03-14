#include <torch/types.h>
#include "src/torchcodec/decoders/_core/DeviceInterface.h"

namespace facebook::torchcodec {

// This file is linked with the CPU-only version of torchcodec.
// So all functions will throw an error because they should only be called if
// the device is not CPU.

[[noreturn]] void throwUnsupportedDeviceError(const torch::Device& device) {
  TORCH_CHECK(
      device.type() != torch::kCPU,
      "Device functions should only be called if the device is not CPU.")
  TORCH_CHECK(false, "Unsupported device: " + device.str());
}

#ifndef ENABLE_CUDA
void convertAVFrameToFrameOutputOnCuda(
    const torch::Device& device,
    [[maybe_unused]] const VideoDecoder::VideoStreamOptions& videoStreamOptions,
    [[maybe_unused]] VideoDecoder::AVFrameStream& avFrameStream,
    [[maybe_unused]] VideoDecoder::FrameOutput& frameOutput,
    [[maybe_unused]] std::optional<torch::Tensor> preAllocatedOutputTensor) {
  throwUnsupportedDeviceError(device);
}

void initializeContextOnCuda(
    const torch::Device& device,
    [[maybe_unused]] AVCodecContext* codecContext) {
  throwUnsupportedDeviceError(device);
}

void releaseContextOnCuda(
    const torch::Device& device,
    [[maybe_unused]] AVCodecContext* codecContext) {
  throwUnsupportedDeviceError(device);
}

std::optional<const AVCodec*> findCudaCodec(
    const torch::Device& device,
    [[maybe_unused]] const AVCodecID& codecId) {
  throwUnsupportedDeviceError(device);
}
#endif // ENABLE_CUDA

#ifndef ENABLE_XPU
void convertAVFrameToFrameOutputOnXpu(
    const torch::Device& device,
    [[maybe_unused]] const VideoDecoder::VideoStreamOptions& videoStreamOptions,
    [[maybe_unused]] VideoDecoder::AVFrameStream& avFrameStream,
    [[maybe_unused]] VideoDecoder::FrameOutput& frameOutput,
    [[maybe_unused]] std::optional<torch::Tensor> preAllocatedOutputTensor) {
  throwUnsupportedDeviceError(device);
}

void initializeContextOnXpu(
    const torch::Device& device,
    [[maybe_unused]] AVCodecContext* codecContext) {
  throwUnsupportedDeviceError(device);
}

void releaseContextOnXpu(
    const torch::Device& device,
    [[maybe_unused]] AVCodecContext* codecContext) {
  throwUnsupportedDeviceError(device);
}

std::optional<const AVCodec*> findXpuCodec(
    const torch::Device& device,
    [[maybe_unused]] const AVCodecID& codecId) {
  throwUnsupportedDeviceError(device);
}
#endif // ENABLE_XPU

} // namespace facebook::torchcodec
