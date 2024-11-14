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

// We reuse cuda contexts across VideoDeoder instances. This is because
// creating a cuda context is expensive. The cache mechanism is as follows:
// 1. There is a cache of size MAX_CONTEXTS_PER_GPU_IN_CACHE cuda contexts for
//    each GPU.
// 2. When we destroy a VideoDecoder instance we release the cuda context to
//    the cache if the cache is not full.
// 3. When we create a VideoDecoder instance we try to get a cuda context from
//    the cache. If the cache is empty we create a new cuda context.

// Pytorch can only handle up to 128 GPUs.
// https://github.com/pytorch/pytorch/blob/e30c55ee527b40d67555464b9e402b4b7ce03737/c10/cuda/CUDAMacros.h#L44
const int MAX_CUDA_GPUS = 128;
// Set to -1 to have an infinitely sized cache. Set it to 0 to disable caching.
// Set to a positive number to have a cache of that size.
const int MAX_CONTEXTS_PER_GPU_IN_CACHE = -1;
std::vector<AVBufferRef*> g_cached_hw_device_ctxs[MAX_CUDA_GPUS];
std::mutex g_cached_hw_device_mutexes[MAX_CUDA_GPUS];

torch::DeviceIndex getFFMPEGCompatibleDeviceIndex(const torch::Device& device) {
  torch::DeviceIndex deviceIndex = device.index();
  deviceIndex = std::max<at::DeviceIndex>(deviceIndex, 0);
  TORCH_CHECK(deviceIndex >= 0, "Device index out of range");
  // FFMPEG cannot handle negative device indices.
  // For single GPU- machines libtorch returns -1 for the device index. So for
  // that case we set the device index to 0.
  // TODO: Double check if this works for multi-GPU machines correctly.
  return deviceIndex;
}

void addToCacheIfCacheHasCapacity(
    const torch::Device& device,
    AVCodecContext* codecContext) {
  torch::DeviceIndex deviceIndex = getFFMPEGCompatibleDeviceIndex(device);
  if (static_cast<int>(deviceIndex) >= MAX_CUDA_GPUS) {
    return;
  }
  std::scoped_lock lock(g_cached_hw_device_mutexes[deviceIndex]);
  if (MAX_CONTEXTS_PER_GPU_IN_CACHE >= 0 &&
      g_cached_hw_device_ctxs[deviceIndex].size() >=
          MAX_CONTEXTS_PER_GPU_IN_CACHE) {
    return;
  }
  g_cached_hw_device_ctxs[deviceIndex].push_back(codecContext->hw_device_ctx);
  codecContext->hw_device_ctx = nullptr;
}

AVBufferRef* getFromCache(const torch::Device& device) {
  torch::DeviceIndex deviceIndex = getFFMPEGCompatibleDeviceIndex(device);
  if (static_cast<int>(deviceIndex) >= MAX_CUDA_GPUS) {
    return nullptr;
  }
  std::scoped_lock lock(g_cached_hw_device_mutexes[deviceIndex]);
  if (g_cached_hw_device_ctxs[deviceIndex].size() > 0) {
    AVBufferRef* hw_device_ctx = g_cached_hw_device_ctxs[deviceIndex].back();
    g_cached_hw_device_ctxs[deviceIndex].pop_back();
    return hw_device_ctx;
  }
  return nullptr;
}

#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(58, 26, 100)

AVBufferRef* getFFMPEGContextFromExistingCudaContext(
    const torch::Device& device,
    torch::DeviceIndex nonNegativeDeviceIndex,
    enum AVHWDeviceType type) {
  c10::cuda::CUDAGuard deviceGuard(device);
  // Valid values for the argument to cudaSetDevice are 0 to maxDevices - 1:
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb
  // So we ensure the deviceIndex is not negative.
  // We set the device because we may be called from a different thread than
  // the one that initialized the cuda context.
  cudaSetDevice(nonNegativeDeviceIndex);
  AVBufferRef* hw_device_ctx = nullptr;
  std::string deviceOrdinal = std::to_string(nonNegativeDeviceIndex);
  int err = av_hwdevice_ctx_create(
      &hw_device_ctx,
      type,
      deviceOrdinal.c_str(),
      nullptr,
      AV_CUDA_USE_CURRENT_CONTEXT);
  if (err < 0) {
    TORCH_CHECK(
        false,
        "Failed to create specified HW device",
        getFFMPEGErrorStringFromErrorCode(err));
  }
  return hw_device_ctx;
}

#else

AVBufferRef* getFFMPEGContextFromNewCudaContext(
    const torch::Device& device,
    torch::DeviceIndex nonNegativeDeviceIndex,
    enum AVHWDeviceType type) {
  AVBufferRef* hw_device_ctx = nullptr;
  std::string deviceOrdinal = std::to_string(nonNegativeDeviceIndex);
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

#endif

AVBufferRef* getCudaContext(const torch::Device& device) {
  enum AVHWDeviceType type = av_hwdevice_find_type_by_name("cuda");
  TORCH_CHECK(type != AV_HWDEVICE_TYPE_NONE, "Failed to find cuda device");
  torch::DeviceIndex nonNegativeDeviceIndex =
      getFFMPEGCompatibleDeviceIndex(device);

  AVBufferRef* hw_device_ctx = getFromCache(device);
  if (hw_device_ctx != nullptr) {
    return hw_device_ctx;
  }

  // 58.26.100 introduced the concept of reusing the existing cuda context
  // which is much faster and lower memory than creating a new cuda context.
  // So we try to use that if it is available.
  // FFMPEG 6.1.2 appears to be the earliest release that contains version
  // 58.26.100 of avutil.
  // https://github.com/FFmpeg/FFmpeg/blob/4acb9b7d1046944345ae506165fb55883d04d8a6/doc/APIchanges#L265
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(58, 26, 100)
  return getFFMPEGContextFromExistingCudaContext(
      device, nonNegativeDeviceIndex, type);
#else
  return getFFMPEGContextFromNewCudaContext(
      device, nonNegativeDeviceIndex, type);
#endif
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
  addToCacheIfCacheHasCapacity(device, codecContext);
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
    VideoDecoder::RawDecodedOutput& rawOutput,
    VideoDecoder::DecodedOutput& output,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  AVFrame* src = rawOutput.frame.get();

  TORCH_CHECK(
      src->format == AV_PIX_FMT_CUDA,
      "Expected format to be AV_PIX_FMT_CUDA, got " +
          std::string(av_get_pix_fmt_name((AVPixelFormat)src->format)));
  auto frameDims = getHeightAndWidthFromOptionsOrAVFrame(options, *src);
  int height = frameDims.height;
  int width = frameDims.width;
  torch::Tensor& dst = output.frame;
  if (preAllocatedOutputTensor.has_value()) {
    dst = preAllocatedOutputTensor.value();
    auto shape = dst.sizes();
    TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == height) && (shape[1] == width) &&
            (shape[2] == 3),
        "Expected tensor of shape ",
        height,
        "x",
        width,
        "x3, got ",
        shape);
  } else {
    dst = allocateEmptyHWCTensor(height, width, options.device);
  }

  // Use the user-requested GPU for running the NPP kernel.
  c10::cuda::CUDAGuard deviceGuard(device);

  NppiSize oSizeROI = {width, height};
  Npp8u* input[2] = {src->data[0], src->data[1]};

  auto start = std::chrono::high_resolution_clock::now();
  NppStatus status;
  if (src->colorspace == AVColorSpace::AVCOL_SPC_BT709) {
    status = nppiNV12ToRGB_709HDTV_8u_P2C3R(
        input,
        src->linesize[0],
        static_cast<Npp8u*>(dst.data_ptr()),
        dst.stride(0),
        oSizeROI);
  } else {
    status = nppiNV12ToRGB_8u_P2C3R(
        input,
        src->linesize[0],
        static_cast<Npp8u*>(dst.data_ptr()),
        dst.stride(0),
        oSizeROI);
  }
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
}

} // namespace facebook::torchcodec
