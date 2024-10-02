#include <torch/types.h>

namespace facebook::torchcodec {

void maybeInitializeDeviceContext(const torch::Device& device) {
  if (device.type() == torch::kCPU) {
    return;
  } else if (device.type() == torch::kCUDA) {
    throw std::runtime_error("CUDA device is unimplemented");
  }
  throw std::runtime_error("Unsupported device: " + device.str());
}

} // namespace facebook::torchcodec
