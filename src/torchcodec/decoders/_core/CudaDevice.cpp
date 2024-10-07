#include <torch/types.h>

namespace facebook::torchcodec {

void throwErrorIfNonCudaDevice(const torch::Device& device) {
  if (device.type() != torch::kCUDA) {
    throw std::runtime_error("Unsupported device: " + device.str());
  }

  void initializeDeviceContext(const torch::Device& device) {
    throwErrorIfNonCudaDevice(device);
    // TODO: https://github.com/pytorch/torchcodec/issues/238: Implement CUDA
    // device.
    throw std::runtime_error(
        "CUDA device is unimplemented. Follow this issue for tracking progress: https://github.com/pytorch/torchcodec/issues/238");
  }

} // namespace facebook::torchcodec
