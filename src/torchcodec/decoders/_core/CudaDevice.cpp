#include <torch/types.h>

namespace facebook::torchcodec {

void maybeInitializeDeviceContext(const torch::Device& device) {
  if (device.type() == torch::kCPU) {
    return;
  } else if (device.type() == torch::kCUDA) {
    // TODO: https://github.com/pytorch/torchcodec/issues/238: Implement CUDA
    // device.
    throw std::runtime_error(
        "CUDA device is unimplemented. Follow this issue for tracking progress: https://github.com/pytorch/torchcodec/issues/238");
  }
  throw std::runtime_error("Unsupported device: " + device.str());
}

} // namespace facebook::torchcodec
