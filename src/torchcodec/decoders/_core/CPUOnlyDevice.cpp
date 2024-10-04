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

} // namespace facebook::torchcodec
