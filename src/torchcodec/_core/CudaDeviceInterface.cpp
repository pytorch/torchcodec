#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <npp.h>
#include <torch/types.h>
#include <mutex>

#include "src/torchcodec/_core/CudaDeviceInterface.h"
#include "src/torchcodec/_core/FFMPEGCommon.h"

extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {
namespace {

static bool g_cuda_default =
    registerDeviceInterface(torch::kCUDA, [](const torch::Device& device) {
      return new CudaDeviceInterface(device);
    });

// We reuse cuda contexts across VideoDeoder instances. This is because
// creating a cuda context is expensive. The cache mechanism is as follows:
// 1. There is a cache of size MAX_CONTEXTS_PER_GPU_IN_CACHE cuda contexts for
//    each GPU.
// 2. When we destroy a SingleStreamDecoder instance we release the cuda context
// to
//    the cache if the cache is not full.
// 3. When we create a SingleStreamDecoder instance we try to get a cuda context
// from
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
    AVBufferRef* hwContext) {
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
  g_cached_hw_device_ctxs[deviceIndex].push_back(av_buffer_ref(hwContext));
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
    /* clang-format off */
    TORCH_CHECK(
        false,
        "Failed to create specified HW device. This typically happens when ",
        "your installed FFmpeg doesn't support CUDA (see ",
        "https://github.com/pytorch/torchcodec#installing-cuda-enabled-torchcodec",
        "). FFmpeg error: ", getFFMPEGErrorStringFromErrorCode(err));
    /* clang-format on */
  }
  return hw_device_ctx;
}

#else

AVBufferRef* getFFMPEGContextFromNewCudaContext(
    [[maybe_unused]] const torch::Device& device,
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

NppStreamContext createNppStreamContext(int deviceIndex) {
  // From 12.9, NPP recommends using a user-created NppStreamContext and using
  // the `_Ctx()` calls:
  // https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#npp-release-12-9-update-1
  // And the nppGetStreamContext() helper is deprecated. We are explicitly
  // supposed to create the NppStreamContext manually from the CUDA device
  // properties:
  // https://github.com/NVIDIA/CUDALibrarySamples/blob/d97803a40fab83c058bb3d68b6c38bd6eebfff43/NPP/README.md?plain=1#L54-L72

  NppStreamContext nppCtx{};
  cudaDeviceProp prop{};
  cudaError_t err = cudaGetDeviceProperties(&prop, deviceIndex);
  TORCH_CHECK(
      err == cudaSuccess,
      "cudaGetDeviceProperties failed: ",
      cudaGetErrorString(err));

  nppCtx.nCudaDeviceId = deviceIndex;
  nppCtx.nMultiProcessorCount = prop.multiProcessorCount;
  nppCtx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  nppCtx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  nppCtx.nSharedMemPerBlock = prop.sharedMemPerBlock;
  nppCtx.nCudaDevAttrComputeCapabilityMajor = prop.major;
  nppCtx.nCudaDevAttrComputeCapabilityMinor = prop.minor;

  // TODO when implementing the cache logic, move these out. See other TODO
  // below.
  nppCtx.hStream = at::cuda::getCurrentCUDAStream(deviceIndex).stream();
  err = cudaStreamGetFlags(nppCtx.hStream, &nppCtx.nStreamFlags);
  TORCH_CHECK(
      err == cudaSuccess,
      "cudaStreamGetFlags failed: ",
      cudaGetErrorString(err));

  return nppCtx;
}

} // namespace

CudaDeviceInterface::CudaDeviceInterface(const torch::Device& device)
    : DeviceInterface(device) {
  TORCH_CHECK(g_cuda_default, "CudaDeviceInterface was not registered!");
  TORCH_CHECK(
      device_.type() == torch::kCUDA, "Unsupported device: ", device_.str());
}

CudaDeviceInterface::~CudaDeviceInterface() {
  if (ctx_) {
    addToCacheIfCacheHasCapacity(device_, ctx_);
    av_buffer_unref(&ctx_);
  }
}

void CudaDeviceInterface::initializeContext(AVCodecContext* codecContext) {
  TORCH_CHECK(!ctx_, "FFmpeg HW device context already initialized");

  // It is important for pytorch itself to create the cuda context. If ffmpeg
  // creates the context it may not be compatible with pytorch.
  // This is a dummy tensor to initialize the cuda context.
  torch::Tensor dummyTensorForCudaInitialization = torch::empty(
      {1}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));
  ctx_ = getCudaContext(device_);
  codecContext->hw_device_ctx = av_buffer_ref(ctx_);
  return;
}

void CudaDeviceInterface::convertAVFrameToFrameOutput(
    const VideoStreamOptions& videoStreamOptions,
    [[maybe_unused]] const AVRational& timeBase,
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  if (avFrame->format != AV_PIX_FMT_CUDA) {
    // The frame's format is AV_PIX_FMT_CUDA if and only if its content is on
    // the GPU. In this branch, the frame is on the CPU: this is what NVDEC
    // gives us if it wasn't able to decode a frame, for whatever reason.
    // Typically that happens if the video's encoder isn't supported by NVDEC.
    // Below, we choose to convert the frame's color-space using the CPU
    // codepath, and send it back to the GPU at the very end.
    // TODO: A possibly better solution would be to send the frame to the GPU
    // first, and do the color conversion there.
    auto cpuDevice = torch::Device(torch::kCPU);
    auto cpuInterface = createDeviceInterface(cpuDevice);

    FrameOutput cpuFrameOutput;
    cpuInterface->convertAVFrameToFrameOutput(
        videoStreamOptions,
        timeBase,
        avFrame,
        cpuFrameOutput,
        preAllocatedOutputTensor);

    frameOutput.data = cpuFrameOutput.data.to(device_);
    return;
  }

  // // Above we checked that the AVFrame was on GPU, but that's not enough, we
  // // also need to check that the AVFrame is in AV_PIX_FMT_NV12 format (8 bits),
  // // because this is what the NPP color conversion routines expect.
  // // TODO: we should investigate how to can perform color conversion for
  // // non-8bit videos. This is supported on CPU.
  // TORCH_CHECK(
  //     avFrame->hw_frames_ctx != nullptr,
  //     "The AVFrame does not have a hw_frames_ctx. "
  //     "That's unexpected, please report this to the TorchCodec repo.");

  // auto hwFramesCtx =
  //     reinterpret_cast<AVHWFramesContext*>(avFrame->hw_frames_ctx->data);
  // AVPixelFormat actualFormat = hwFramesCtx->sw_format;
  // TORCH_CHECK(
  //     actualFormat == AV_PIX_FMT_NV12,
  //     "The AVFrame is ",
  //     (av_get_pix_fmt_name(actualFormat) ? av_get_pix_fmt_name(actualFormat)
  //                                        : "unknown"),
  //     ", but we expected AV_PIX_FMT_NV12. This typically happens when "
  //     "the video isn't 8bit, which is not supported on CUDA at the moment. "
  //     "Try using the CPU device instead. "
  //     "If the video is 10bit, we are tracking 10bit support in "
  //     "https://github.com/pytorch/torchcodec/issues/776");

  auto frameDims =
      getHeightAndWidthFromOptionsOrAVFrame(videoStreamOptions, avFrame);
  int height = frameDims.height;
  int width = frameDims.width;
  torch::Tensor& dst = frameOutput.data;
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
    dst = allocateEmptyHWCTensor(height, width, device_);
  }

  // TODO cache the NppStreamContext! It currently gets re-recated for every
  // single frame. The cache should be per-device, similar to the existing
  // hw_device_ctx cache. When implementing the cache logic, the
  // NppStreamContext hStream and nStreamFlags should not be part of the cache
  // because they may change across calls.
  NppStreamContext nppCtx = createNppStreamContext(
      static_cast<int>(getFFMPEGCompatibleDeviceIndex(device_)));

  NppiSize oSizeROI = {width, height};
  Npp8u* input[2] = {avFrame->data[0], avFrame->data[1]};

  NppStatus status;

  if (avFrame->colorspace == AVColorSpace::AVCOL_SPC_BT709) {
    status = nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(
        input,
        avFrame->linesize[0],
        static_cast<Npp8u*>(dst.data_ptr()),
        dst.stride(0),
        oSizeROI,
        nppCtx);
  } else {
    status = nppiNV12ToRGB_8u_P2C3R_Ctx(
        input,
        avFrame->linesize[0],
        static_cast<Npp8u*>(dst.data_ptr()),
        dst.stride(0),
        oSizeROI,
        nppCtx);
  }
  TORCH_CHECK(status == NPP_SUCCESS, "Failed to convert NV12 frame.");
}

// inspired by https://github.com/FFmpeg/FFmpeg/commit/ad67ea9
// we have to do this because of an FFmpeg bug where hardware decoding is not
// appropriately set, so we just go off and find the matching codec for the CUDA
// device
std::optional<const AVCodec*> CudaDeviceInterface::findCodec(
    const AVCodecID& codecId) {
  void* i = nullptr;
  const AVCodec* codec = nullptr;
  while ((codec = av_codec_iterate(&i)) != nullptr) {
    if (codec->id != codecId || !av_codec_is_decoder(codec)) {
      continue;
    }

    const AVCodecHWConfig* config = nullptr;
    for (int j = 0; (config = avcodec_get_hw_config(codec, j)) != nullptr;
         ++j) {
      if (config->device_type == AV_HWDEVICE_TYPE_CUDA) {
        return codec;
      }
    }
  }

  return std::nullopt;
}

} // namespace facebook::torchcodec
