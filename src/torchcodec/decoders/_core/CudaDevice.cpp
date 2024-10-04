#include <torch/types.h>
#include "src/torchcodec/decoders/_core/DeviceInterface.h"
#include "src/torchcodec/decoders/_core/FFMPEGCommon.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext_cuda.h>
}

namespace facebook::torchcodec {

AVBufferRef* getCudaContext() {
  enum AVHWDeviceType type = av_hwdevice_find_type_by_name("cuda");
  TORCH_CHECK(type != AV_HWDEVICE_TYPE_NONE, "Failed to find cuda device");
  int err = 0;
  AVBufferRef* hw_device_ctx;
  err = av_hwdevice_ctx_create(
      &hw_device_ctx,
      type,
      nullptr,
      nullptr,
  // Introduced in 58.26.100:
  // https://github.com/FFmpeg/FFmpeg/blob/4acb9b7d1046944345ae506165fb55883d04d8a6/doc/APIchanges#L265
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(58, 26, 100)
      AV_CUDA_USE_CURRENT_CONTEXT
#else
      0
#endif
  );
  if (err < 0) {
    TORCH_CHECK(
        false,
        "Failed to create specified HW device",
        getFFMPEGErrorStringFromErrorCode(err));
  }
  return hw_device_ctx;
}

void maybeInitializeDeviceContext(
    const torch::Device& device,
    AVCodecContext* codecContext) {
  if (device.type() == torch::kCPU) {
    return;
  } else if (device.type() == torch::kCUDA) {
    codecContext->hw_device_ctx = av_buffer_ref(getCudaContext());
  }
  throw std::runtime_error("Unsupported device: " + device.str());
}

} // namespace facebook::torchcodec
