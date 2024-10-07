#include <torch/types.h>

namespace facebook::torchcodec {

void throwUnsupportedDeviceError(const torch::Device& device) {
  throw std::runtime_error("Unsupported device: " + device.str());
}

void maybeInitializeDeviceContext(const torch::Device& device) {
  throwUnsupportedDeviceError(device);
}

} // namespace facebook::torchcodec
