#include <torch/types.h>
#include "src/torchcodec/decoders/_core/DeviceInterface.h"

namespace facebook::torchcodec {

// This file is linked with the CPU-only version of torchcodec.
// So all functions will throw an error because they should only be called if
// the device is not CPU.

void throwUnsupportedDeviceError(const torch::Device& device) {
  TORCH_CHECK(
      device.type() != torch::kCPU,
      "Device functions should only be called if the device is not CPU.")
  throw std::runtime_error("Unsupported device: " + device.str());
}

VideoDecoder::DecodedOutput convertAVFrameToDecodedOutputOnDevice(
    const torch::Device& device,
    const VideoDecoder::VideoStreamDecoderOptions& options,
    AVCodecContext* codecContext,
    VideoDecoder::RawDecodedOutput& rawOutput) {
  throwUnsupportedDeviceError(device);
  VideoDecoder::DecodedOutput output;
  return output;
}

void initializeDeviceContext(
    const torch::Device& device,
    AVCodecContext* codecContext) {
  throwUnsupportedDeviceError(device);
}

} // namespace facebook::torchcodec
