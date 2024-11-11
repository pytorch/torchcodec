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

void convertAVFrameToDecodedOutputOnCuda(
    const torch::Device& device,
    const VideoDecoder::VideoStreamDecoderOptions& options,
    VideoDecoder::RawDecodedOutput& rawOutput,
    VideoDecoder::DecodedOutput& output,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  throwUnsupportedDeviceError(device);
}

void initializeContextOnCuda(
    const torch::Device& device,
    AVCodecContext* codecContext) {
  throwUnsupportedDeviceError(device);
}

void releaseContextOnCuda(
    const torch::Device& device,
    AVCodecContext* codecContext) {
  throwUnsupportedDeviceError(device);
}

} // namespace facebook::torchcodec
