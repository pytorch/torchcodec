#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <npp.h>
#include <torch/types.h>
#include <mutex>

#include "src/torchcodec/_core/Cache.h"
#include "src/torchcodec/_core/CudaDeviceInterface.h"
#include "src/torchcodec/_core/FFMPEGCommon.h"

extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {
namespace {

static bool g_cuda =
    registerDeviceInterface(torch::kCUDA, [](const torch::Device& device) {
      return new CudaDeviceInterface(device);
    });

// BT.709 full range color conversion matrix for YUV to RGB conversion.
// See Note [YUV -> RGB Color Conversion, color space and color range] below.
constexpr Npp32f bt709FullRangeColorTwist[3][4] = {
    {1.0f, 0.0f, 1.5748f, 0.0f},
    {1.0f, -0.187324273f, -0.468124273f, -128.0f},
    {1.0f, 1.8556f, 0.0f, -128.0f}};

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
PerGpuCache<AVBufferRef, Deleterp<AVBufferRef, void, av_buffer_unref>>
    g_cached_hw_device_ctxs(MAX_CUDA_GPUS, MAX_CONTEXTS_PER_GPU_IN_CACHE);
PerGpuCache<NppStreamContext> g_cached_npp_ctxs(
    MAX_CUDA_GPUS,
    MAX_CONTEXTS_PER_GPU_IN_CACHE);

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

UniqueAVBufferRef getCudaContext(const torch::Device& device) {
  enum AVHWDeviceType type = av_hwdevice_find_type_by_name("cuda");
  TORCH_CHECK(type != AV_HWDEVICE_TYPE_NONE, "Failed to find cuda device");
  torch::DeviceIndex nonNegativeDeviceIndex = getNonNegativeDeviceIndex(device);

  UniqueAVBufferRef hw_device_ctx = g_cached_hw_device_ctxs.get(device);
  if (hw_device_ctx) {
    return hw_device_ctx;
  }

  // 58.26.100 introduced the concept of reusing the existing cuda context
  // which is much faster and lower memory than creating a new cuda context.
  // So we try to use that if it is available.
  // FFMPEG 6.1.2 appears to be the earliest release that contains version
  // 58.26.100 of avutil.
  // https://github.com/FFmpeg/FFmpeg/blob/4acb9b7d1046944345ae506165fb55883d04d8a6/doc/APIchanges#L265
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(58, 26, 100)
  return UniqueAVBufferRef(getFFMPEGContextFromExistingCudaContext(
      device, nonNegativeDeviceIndex, type));
#else
  return UniqueAVBufferRef(
      getFFMPEGContextFromNewCudaContext(device, nonNegativeDeviceIndex, type));
#endif
}

std::unique_ptr<NppStreamContext> getNppStreamContext(
    const torch::Device& device) {
  torch::DeviceIndex nonNegativeDeviceIndex = getNonNegativeDeviceIndex(device);

  std::unique_ptr<NppStreamContext> nppCtx = g_cached_npp_ctxs.get(device);
  if (nppCtx) {
    return nppCtx;
  }

  // From 12.9, NPP recommends using a user-created NppStreamContext and using
  // the `_Ctx()` calls:
  // https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#npp-release-12-9-update-1
  // And the nppGetStreamContext() helper is deprecated. We are explicitly
  // supposed to create the NppStreamContext manually from the CUDA device
  // properties:
  // https://github.com/NVIDIA/CUDALibrarySamples/blob/d97803a40fab83c058bb3d68b6c38bd6eebfff43/NPP/README.md?plain=1#L54-L72

  nppCtx = std::make_unique<NppStreamContext>();
  cudaDeviceProp prop{};
  cudaError_t err = cudaGetDeviceProperties(&prop, nonNegativeDeviceIndex);
  TORCH_CHECK(
      err == cudaSuccess,
      "cudaGetDeviceProperties failed: ",
      cudaGetErrorString(err));

  nppCtx->nCudaDeviceId = nonNegativeDeviceIndex;
  nppCtx->nMultiProcessorCount = prop.multiProcessorCount;
  nppCtx->nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  nppCtx->nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  nppCtx->nSharedMemPerBlock = prop.sharedMemPerBlock;
  nppCtx->nCudaDevAttrComputeCapabilityMajor = prop.major;
  nppCtx->nCudaDevAttrComputeCapabilityMinor = prop.minor;

  return nppCtx;
}

} // namespace

CudaDeviceInterface::CudaDeviceInterface(const torch::Device& device)
    : DeviceInterface(device) {
  TORCH_CHECK(g_cuda, "CudaDeviceInterface was not registered!");
  TORCH_CHECK(
      device_.type() == torch::kCUDA, "Unsupported device: ", device_.str());
}

CudaDeviceInterface::~CudaDeviceInterface() {
  if (ctx_) {
    g_cached_hw_device_ctxs.addIfCacheHasCapacity(device_, std::move(ctx_));
  }
  if (nppCtx_) {
    g_cached_npp_ctxs.addIfCacheHasCapacity(device_, std::move(nppCtx_));
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
  nppCtx_ = getNppStreamContext(device_);
  codecContext->hw_device_ctx = av_buffer_ref(ctx_.get());
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

  // Above we checked that the AVFrame was on GPU, but that's not enough, we
  // also need to check that the AVFrame is in AV_PIX_FMT_NV12 format (8 bits),
  // because this is what the NPP color conversion routines expect.
  // TODO: we should investigate how to can perform color conversion for
  // non-8bit videos. This is supported on CPU.
  TORCH_CHECK(
      avFrame->hw_frames_ctx != nullptr,
      "The AVFrame does not have a hw_frames_ctx. "
      "That's unexpected, please report this to the TorchCodec repo.");

  auto hwFramesCtx =
      reinterpret_cast<AVHWFramesContext*>(avFrame->hw_frames_ctx->data);
  AVPixelFormat actualFormat = hwFramesCtx->sw_format;
  TORCH_CHECK(
      actualFormat == AV_PIX_FMT_NV12,
      "The AVFrame is ",
      (av_get_pix_fmt_name(actualFormat) ? av_get_pix_fmt_name(actualFormat)
                                         : "unknown"),
      ", but we expected AV_PIX_FMT_NV12. This typically happens when "
      "the video isn't 8bit, which is not supported on CUDA at the moment. "
      "Try using the CPU device instead. "
      "If the video is 10bit, we are tracking 10bit support in "
      "https://github.com/pytorch/torchcodec/issues/776");

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

  torch::DeviceIndex deviceIndex = getNonNegativeDeviceIndex(device_);

  // Create a CUDA event and attach it to the AVFrame's CUDA stream. That's the
  // NVDEC stream, i.e. the CUDA stream that the frame was decoded on.
  // We will be waiting for this event to complete before calling the NPP
  // functions, to ensure NVDEC has finished decoding the frame before running
  // the NPP color-conversion.
  // Note that our code is generic and assumes that the NVDEC's stream can be
  // arbitrary, but unfortunately we know it's hardcoded to be the default
  // stream by FFmpeg:
  // https://github.com/FFmpeg/FFmpeg/blob/66e40840d15b514f275ce3ce2a4bf72ec68c7311/libavutil/hwcontext_cuda.c#L387-L388
  TORCH_CHECK(
      hwFramesCtx->device_ctx != nullptr,
      "The AVFrame's hw_frames_ctx does not have a device_ctx. ");
  auto cudaDeviceCtx =
      static_cast<AVCUDADeviceContext*>(hwFramesCtx->device_ctx->hwctx);
  at::cuda::CUDAEvent nvdecDoneEvent;
  at::cuda::CUDAStream nvdecStream = // That's always the default stream. Sad.
      c10::cuda::getStreamFromExternal(cudaDeviceCtx->stream, deviceIndex);
  nvdecDoneEvent.record(nvdecStream);

  // Don't start NPP work before NVDEC is done decoding the frame!
  at::cuda::CUDAStream nppStream = at::cuda::getCurrentCUDAStream(deviceIndex);
  nvdecDoneEvent.block(nppStream);

  // Create the NPP context if we haven't yet.
  nppCtx_->hStream = nppStream.stream();
  cudaError_t err =
      cudaStreamGetFlags(nppCtx_->hStream, &nppCtx_->nStreamFlags);
  TORCH_CHECK(
      err == cudaSuccess,
      "cudaStreamGetFlags failed: ",
      cudaGetErrorString(err));

  NppiSize oSizeROI = {width, height};
  Npp8u* yuvData[2] = {avFrame->data[0], avFrame->data[1]};

  NppStatus status;

  // For background, see
  // Note [YUV -> RGB Color Conversion, color space and color range]
  if (avFrame->colorspace == AVColorSpace::AVCOL_SPC_BT709) {
    if (avFrame->color_range == AVColorRange::AVCOL_RANGE_JPEG) {
      // NPP provides a pre-defined color conversion function for BT.709 full
      // range: nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx. But it's not closely
      // matching the results we have on CPU. So we're using a custom color
      // conversion matrix, which provides more accurate results. See the note
      // mentioned above for details, and headaches.

      int srcStep[2] = {avFrame->linesize[0], avFrame->linesize[1]};

      status = nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx(
          yuvData,
          srcStep,
          static_cast<Npp8u*>(dst.data_ptr()),
          dst.stride(0),
          oSizeROI,
          bt709FullRangeColorTwist,
          *nppCtx_);
    } else {
      // If not full range, we assume studio limited range.
      // The color conversion matrix for BT.709 limited range should be:
      // static const Npp32f bt709LimitedRangeColorTwist[3][4] = {
      //   {1.16438356f, 0.0f, 1.79274107f, -16.0f},
      //   {1.16438356f, -0.213248614f, -0.5329093290f, -128.0f},
      //   {1.16438356f, 2.11240179f, 0.0f, -128.0f}
      // };
      // We get very close results to CPU with that, but using the pre-defined
      // nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx seems to be even more accurate.
      status = nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(
          yuvData,
          avFrame->linesize[0],
          static_cast<Npp8u*>(dst.data_ptr()),
          dst.stride(0),
          oSizeROI,
          *nppCtx_);
    }
  } else {
    // TODO we're assuming BT.601 color space (and probably limited range) by
    // calling nppiNV12ToRGB_8u_P2C3R_Ctx. We should handle BT.601 full range,
    // and other color-spaces like 2020.
    status = nppiNV12ToRGB_8u_P2C3R_Ctx(
        yuvData,
        avFrame->linesize[0],
        static_cast<Npp8u*>(dst.data_ptr()),
        dst.stride(0),
        oSizeROI,
        *nppCtx_);
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

/* clang-format off */
// Note: [YUV -> RGB Color Conversion, color space and color range]
//
// The frames we get from the decoder (FFmpeg decoder, or NVCUVID) are in YUV
// format. We need to convert them to RGB. This note attempts to describe this
// process. There may be some inaccuracies and approximations that experts will
// notice, but our goal is only to provide a good enough understanding of the
// process for torchcodec developers to implement and maintain it.
// On CPU, filtergraph and swscale handle everything for us. With CUDA, we have
// to do a lot of the heavy lifting ourselves.
//
// Color space and color range
// ---------------------------
// Two main characteristics of a frame will affect the conversion process:
// 1. Color space: This basically defines what YUV values correspond to which
//    physical wavelength. No need to go into details here,the point is that
//    videos can come in different color spaces, the most common ones being
//    BT.601 and BT.709, but there are others.
//    In FFmpeg this is represented with AVColorSpace:
//    https://ffmpeg.org/doxygen/4.0/pixfmt_8h.html#aff71a069509a1ad3ff54d53a1c894c85
// 2. Color range: This defines the range of YUV values. There is:
//    - full range, also called PC range: AVCOL_RANGE_JPEG
//    - and the "limited" range, also called studio or TV range: AVCOL_RANGE_MPEG
//    https://ffmpeg.org/doxygen/4.0/pixfmt_8h.html#a3da0bf691418bc22c4bcbe6583ad589a
//
// Color space and color range are independent concepts, so we can have a BT.709
// with full range, and another one with limited range. Same for BT.601.
//
// In the first version of this note we'll focus on the full color range. It
// will later be updated to account for the limited range.
//
// Color conversion matrix
// -----------------------
// YUV -> RGB conversion is defined as the reverse process of the RGB -> YUV,
// So this is where we'll start.
// At the core of a RGB -> YUV conversion are the "luma coefficients", which are
// specific to a given color space and defined by the color space standard. In
// FFmpeg they can be found here:
// https://github.com/FFmpeg/FFmpeg/blob/7d606ef0ccf2946a4a21ab1ec23486cadc21864b/libavutil/csp.c#L46-L56
//
// For example, the BT.709 coefficients are: kr=0.2126, kg=0.7152, kb=0.0722
// Coefficients must sum to 1.
//
// Conventionally Y is in [0, 1] range, and U and V are in [-0.5, 0.5] range
// (that's mathematically, in practice they are represented in integer range).
// The conversion is defined as:
// https://en.wikipedia.org/wiki/YCbCr#R'G'B'_to_Y%E2%80%B2PbPr
// Y = kr*R + kg*G + kb*B
// U = (B - Y) * 0.5 / (1 - kb) = (B - Y) / u_scale where u_scale = 2 * (1 - kb)
// V = (R - Y) * 0.5 / (1 - kr) = (R - Y) / v_scale where v_scale = 2 * (1 - kr)
//
// Putting all this into matrix form, we get:
// [Y]   = [kr               kg            kb            ]  [R]
// [U]     [-kr/u_scale      -kg/u_scale   (1-kb)/u_scale]  [G]
// [V]     [(1-kr)/v_scale   -kg/v_scale   -kb)/v_scale  ]  [B]
//
//
// Now, to convert YUV to RGB, we just need to invert this matrix:
// ```py
// import torch
// kr, kg, kb = 0.2126, 0.7152, 0.0722  # BT.709  luma coefficients
// u_scale = 2 * (1 - kb)
// v_scale = 2 * (1 - kr)
//
// rgb_to_yuv = torch.tensor([
//     [kr, kg, kb],
//     [-kr/u_scale, -kg/u_scale, (1-kb)/u_scale],
//     [(1-kr)/v_scale, -kg/v_scale, -kb/v_scale]
// ])
//
// yuv_to_rgb_full = torch.linalg.inv(rgb_to_yuv)
// print("YUV->RGB matrix (Full Range):")
// print(yuv_to_rgb_full)
// ```
// And we get:
// tensor([[ 1.0000e+00, -3.3142e-09,  1.5748e+00],
//         [ 1.0000e+00, -1.8732e-01, -4.6812e-01],
//         [ 1.0000e+00,  1.8556e+00,  4.6231e-09]])
//
// Which matches https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.709_conversion
//
// Color conversion in NPP
// -----------------------
// https://docs.nvidia.com/cuda/npp/image_color_conversion.html.
//
// NPP provides different ways to convert YUV to RGB:
// - pre-defined color conversion functions like
//   nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx and nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx
//   which are for BT.709 limited and full range, respectively.
// - generic color conversion functions that accept a custom color conversion
//   matrix, called ColorTwist, like nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx
//
// We use the pre-defined functions or the color twist functions depending on
// which one we find to be closer to the CPU results.
//
// The color twist functionality is *partially* described in a section named
// "YUVToRGBColorTwist". Importantly:
//
// - The `nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx` function takes the YUV data
//   and the color-conversion matrix as input. The function itself and the
//   matrix assume different ranges for YUV values:
// - The **matrix coefficient** must assume that Y is in [0, 1] and U,V are in
//   [-0.5, 0.5]. That's how we defined our matrix above.
// - The function `nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx` however expects all
//   of the input Y, U, V to be in [0, 255]. That's how the data comes out of
//   the decoder.
// - But *internally*, `nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx` needs U and V to
//   be centered around 0, i.e. in [-128, 127]. So we need to apply a -128
//   offset to U and V. Y doesn't need to be offset. The offset can be applied
//   by adding a 4th column to the matrix.
//
//
// So our conversion matrix becomes the following, with new offset column:
// tensor([[ 1.0000e+00, -3.3142e-09,  1.5748e+00,     0]
//         [ 1.0000e+00, -1.8732e-01, -4.6812e-01,     -128]
//         [ 1.0000e+00,  1.8556e+00,  4.6231e-09 ,    -128]])
//
// And that's what we need to pass for BT701, full range.
/* clang-format on */
