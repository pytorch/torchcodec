#include <torch/types.h>

namespace facebook::torchcodec {

void maybeInitializeDeviceContext(const torch::Device& device) {
  if (device.type() == torch::kCPU) {
    return;
  }
  throw std::runtime_error("Unsupported device: " + device.str());
}

} // namespace facebook::torchcodec
