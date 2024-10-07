#include <torch/types.h>

namespace facebook::torchcodec {

// This file is linked with the CPU-only version of torchcodec.
// So all functions will throw an error because they should only be called if
// the device is not CPU.

void throwUnsupportedDeviceError(const torch::Device& device) {
  throw std::runtime_error("Unsupported device: " + device.str());
}

void maybeInitializeDeviceContext(const torch::Device& device) {
  throwUnsupportedDeviceError(device);
}

} // namespace facebook::torchcodec
