#include <torch/types.h>
#include "src/torchcodec/decoders/_core/DeviceInterface.h"

namespace facebook::torchcodec {

void maybeInitializeDeviceContext(
    const torch::Device& device,
    AVCodecContext* codecContext) {
  if (device.type() == torch::kCPU) {
    return;
  }
  throw std::runtime_error("Unsupported device: " + device.str());
}

VideoDecoder::DecodedOutput convertAVFrameToDecodedOutputOnDevice(
    const torch::Device& device,
    const VideoDecoder::VideoStreamDecoderOptions& options,
    AVCodecContext* codecContext,
    VideoDecoder::RawDecodedOutput& rawOutput) {
  TORCH_CHECK(false, "We should not run device code on CPU")
}

} // namespace facebook::torchcodec
