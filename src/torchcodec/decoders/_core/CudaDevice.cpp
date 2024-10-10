#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <npp.h>
#include <torch/types.h>
#include <mutex>

#include "src/torchcodec/decoders/_core/DeviceInterface.h"
#include "src/torchcodec/decoders/_core/FFMPEGCommon.h"
#include "src/torchcodec/decoders/_core/VideoDecoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {
namespace {

const int MAX_CUDA_GPUS = 16;
const int MAX_CACHE_SIZE_PER_GPU = 10;
std::set<AVBufferRef*> g_cached_hw_device_ctxs[MAX_CUDA_GPUS];
std::mutex g_cached_hw_device_mutexes[MAX_CUDA_GPUS];

torch::DeviceIndex getFFMPEGCompatibleDeviceIndex(const torch::Device& device) {
  torch::DeviceIndex deviceIndex = device.index();
  deviceIndex = std::max<at::DeviceIndex>(deviceIndex, 0);
  TORCH_CHECK(deviceIndex >= 0, "Device index out of range");
  TORCH_CHECK(deviceIndex < MAX_CUDA_GPUS, "Device index out of range");
  // FFMPEG cannot handle negative device indices.
  // For single GPU- machines libtorch returns -1 for the device index. So for
  // that case we set the device index to 0.
  // TODO: Double check if this works for multi-GPU machines correctly.
  return deviceIndex;
}

void addToCache(const torch::Device& device, AVCodecContext* codecContext) {
  torch::DeviceIndex deviceIndex = getFFMPEGCompatibleDeviceIndex(device);
  std::scoped_lock lock(g_cached_hw_device_mutexes[deviceIndex]);
  if (g_cached_hw_device_ctxs[deviceIndex].size() >= MAX_CACHE_SIZE_PER_GPU) {
    return;
  }
  g_cached_hw_device_ctxs[deviceIndex].insert(codecContext->hw_device_ctx);
  codecContext->hw_device_ctx = nullptr;
}

AVBufferRef* getFromCache(const torch::Device& device) {
  torch::DeviceIndex deviceIndex = getFFMPEGCompatibleDeviceIndex(device);
  std::scoped_lock lock(g_cached_hw_device_mutexes[deviceIndex]);
  if (g_cached_hw_device_ctxs[deviceIndex].size() > 0) {
    auto it = g_cached_hw_device_ctxs[deviceIndex].begin();
    AVBufferRef* hw_device_ctx = *it;
    g_cached_hw_device_ctxs[deviceIndex].erase(it);
    return hw_device_ctx;
  }
  return nullptr;
}

AVBufferRef* getCudaContext(const torch::Device& device) {
  enum AVHWDeviceType type = av_hwdevice_find_type_by_name("cuda");
  TORCH_CHECK(type != AV_HWDEVICE_TYPE_NONE, "Failed to find cuda device");
  torch::DeviceIndex deviceIndex = getFFMPEGCompatibleDeviceIndex(device);

  AVBufferRef* hw_device_ctx = getFromCache(device);
  if (hw_device_ctx != nullptr) {
    return hw_device_ctx;
  }

  std::string deviceOrdinal = std::to_string(deviceIndex);
  int err = av_hwdevice_ctx_create(
      &hw_device_ctx, type, deviceOrdinal.c_str(), nullptr, 0);
  if (err < 0) {
    TORCH_CHECK(
        false,
        "Failed to create specified HW device",
        getFFMPEGErrorStringFromErrorCode(err));
  }
  return hw_device_ctx;
}

torch::Tensor allocateDeviceTensor(
    at::IntArrayRef shape,
    torch::Device device,
    const torch::Dtype dtype = torch::kUInt8) {
  return torch::empty(
      shape,
      torch::TensorOptions()
          .dtype(dtype)
          .layout(torch::kStrided)
          .device(device));
}

void throwErrorIfNonCudaDevice(const torch::Device& device) {
  TORCH_CHECK(
      device.type() != torch::kCPU,
      "Device functions should only be called if the device is not CPU.")
  if (device.type() != torch::kCUDA) {
    throw std::runtime_error("Unsupported device: " + device.str());
  }
}
} // namespace

void releaseContextOnCuda(
    const torch::Device& device,
    AVCodecContext* codecContext) {
  throwErrorIfNonCudaDevice(device);
  AVBufferRef* hw_device_ctx = codecContext->hw_device_ctx;
  addToCache(device, codecContext);
}
void initializeContextOnCuda(
    const torch::Device& device,
    AVCodecContext* codecContext) {
  throwErrorIfNonCudaDevice(device);
  // It is important for pytorch itself to create the cuda context. If ffmpeg
  // creates the context it may not be compatible with pytorch.
  // This is a dummy tensor to initialize the cuda context.
  torch::Tensor dummyTensorForCudaInitialization = torch::empty(
      {1}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
  codecContext->hw_device_ctx = getCudaContext(device);
  return;
}

void convertAVFrameToDecodedOutputOnCuda(
    const torch::Device& device,
    const VideoDecoder::VideoStreamDecoderOptions& options,
    AVCodecContext* codecContext,
    VideoDecoder::RawDecodedOutput& rawOutput,
    VideoDecoder::DecodedOutput& output) {
  AVFrame* src = rawOutput.frame.get();

  TORCH_CHECK(
      src->format == AV_PIX_FMT_CUDA,
      "Expected format to be AV_PIX_FMT_CUDA, got " +
          std::string(av_get_pix_fmt_name((AVPixelFormat)src->format)));
  int width = options.width.value_or(codecContext->width);
  int height = options.height.value_or(codecContext->height);
  NppiSize oSizeROI = {width, height};
  Npp8u* input[2] = {src->data[0], src->data[1]};
  torch::Tensor& dst = output.frame;
  dst = allocateDeviceTensor({height, width, 3}, options.device);

  // Use the user-requested GPU for running the NPP kernel.
  c10::cuda::CUDAGuard deviceGuard(device);

  auto start = std::chrono::high_resolution_clock::now();

  NppStatus status = nppiNV12ToRGB_8u_P2C3R(
      input,
      src->linesize[0],
      static_cast<Npp8u*>(dst.data_ptr()),
      dst.stride(0),
      oSizeROI);
  TORCH_CHECK(status == NPP_SUCCESS, "Failed to convert NV12 frame.");
  // Make the pytorch stream wait for the npp kernel to finish before using the
  // output.
  at::cuda::CUDAEvent nppDoneEvent;
  at::cuda::CUDAStream nppStreamWrapper =
      c10::cuda::getStreamFromExternal(nppGetStream(), device.index());
  nppDoneEvent.record(nppStreamWrapper);
  nppDoneEvent.block(at::cuda::getCurrentCUDAStream());

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::micro> duration = end - start;
  VLOG(9) << "NPP Conversion of frame height=" << height << " width=" << width
          << " took: " << duration.count() << "us" << std::endl;
  if (options.dimensionOrder == "NCHW") {
    // The docs guaranty this to return a view:
    // https://pytorch.org/docs/stable/generated/torch.permute.html
    dst = dst.permute({2, 0, 1});
  }
}

} // namespace facebook::torchcodec
