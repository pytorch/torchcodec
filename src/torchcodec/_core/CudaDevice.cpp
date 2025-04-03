#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <npp.h>
#include <torch/types.h>
#include <mutex>

#include "src/torchcodec/_core/DeviceInterface.h"
#include "src/torchcodec/_core/FFMPEGCommon.h"
#include "src/torchcodec/_core/SingleStreamDecoder.h"

extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {
namespace {

// We reuse cuda contexts across VideoDeoder instances. This is because
// creating a cuda context is expensive. The cache mechanism is as follows:
// 1. There is a cache of size MAX_CONTEXTS_PER_GPU_IN_CACHE cuda contexts for
// each GPU.
// 2. When we destroy a SingleStreamDecoder instance we release the cuda context
// to
// the cache if the cache is not full.
// 3. When we create a SingleStreamDecoder instance we try to get a cuda context
// from
// the cache. If the cache is empty we create a new cuda context.

// Pytorch can only handle up to 128 GPUs.
// https://github.com/pytorch/pytorch/blob/e30c55ee527b40d67555464b9e402b4b7ce03737/c10/cuda/CUDAMacros.h#L44
const int MAX_CUDA_GPUS = 128;
// Set to -1 to have an infinitely sized cache. Set it to 0 to disable caching.
// Set to a positive number to have a cache of that size.
const int MAX_CONTEXTS_PER_GPU_IN_CACHE = -1;
std::vector<_avbuffer_ref*> g_cached_hw_device_ctxs[MAX_CUDA_GPUS];
std::mutex g_cached_hw_device_mutexes[MAX_CUDA_GPUS];

torch::DeviceIndex get_ffmpeg_compatible_device_index(
    const torch::Device& device) {
  torch::DeviceIndex device_index = device.index();
  device_index = std::max<at::DeviceIndex>(deviceIndex, 0);
  TORCH_CHECK(deviceIndex >= 0, "Device index out of range");
  // FFMPEG cannot handle negative device indices.
  // For single GPU- machines libtorch returns -1 for the device index. So for
  // that case we set the device index to 0.
  // TODO: Double check if this works for multi-GPU machines correctly.
  return device_index;
}

void add_to_cache_if_cache_has_capacity(
    const torch::Device& device,
    AVCodecContext* codec_context) {
  torch::DeviceIndex device_index = get_ffmpeg_compatible_device_index(device);
  if (static_cast<int>(deviceIndex) >= MAX_CUDA_GPUS) {
    return;
  }
  std::scoped_lock lock(g_cached_hw_device_mutexes[device_index]);
  if (MAX_CONTEXTS_PER_GPU_IN_CACHE >= 0 &&
      g_cached_hw_device_ctxs[device_index].size() >=
          MAX_CONTEXTS_PER_GPU_IN_CACHE) {
    return;
  }
  g_cached_hw_device_ctxs[device_index].push_back(codec_context->hw_device_ctx);
  codec_context->hw_device_ctx = nullptr;
}

AVBufferRef* get_from_cache(const torch::Device& device) {
  torch::DeviceIndex device_index = get_ffmpeg_compatible_device_index(device);
  if (static_cast<int>(deviceIndex) >= MAX_CUDA_GPUS) {
    return nullptr;
  }
  std::scoped_lock lock(g_cached_hw_device_mutexes[device_index]);
  if (g_cached_hw_device_ctxs[deviceIndex].size() > 0) {
    AVBufferRef* hw_device_ctx = g_cached_hw_device_ctxs[device_index].back();
    g_cached_hw_device_ctxs[device_index].pop_back();
    return hw_device_ctx;
  }
  return nullptr;
}

#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(58, 26, 100)

AVBufferRef* get_ffmpeg_context_from_existing_cuda_context(
    const torch::Device& device,
    torch::DeviceIndex non_negative_device_index,
    enum AVHWDeviceType type) {
  c10::cuda::CUDAGuard device_guard(device);
  // Valid values for the argument to cudaSetDevice are 0 to maxDevices - 1:
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb
  // So we ensure the deviceIndex is not negative.
  // We set the device because we may be called from a different thread than
  // the one that initialized the cuda context.
  cuda_set_device(non_negative_device_index);
  AVBufferRef* hw_device_ctx = nullptr;
  std::string device_ordinal = std::to_string(non_negative_device_index);
  int err = av_hwdevice_ctx_create(
      &hw_device_ctx,
      type,
      device_ordinal.c_str(),
      nullptr,
      AV_CUDA_USE_CURRENT_CONTEXT);
  if (err < 0) {
    /* clang-format off */
TORCH_CHECK(
false,
"Failed to create specified HW device. This typically happens when ",
"your installed FFmpeg doesn't support CUDA (see ",
"https://github.com/pytorch/torchcodec#installing-cuda-enabled-torchcodec",
"). FFmpeg error: ", get_ffmpeg_error_string_from_error_code(err));
    /* clang-format on */
  }
  return hw_device_ctx;
}

#else

AVBufferRef* get_ffmpeg_context_from_new_cuda_context(
    [[maybe_unused]] const torch::Device& device,
    torch::DeviceIndex non_negative_device_index,
    enum AVHWDeviceType type) {
  AVBufferRef* hw_device_ctx = nullptr;
  std::string device_ordinal = std::to_string(non_negative_device_index);
  int err = av_hwdevice_ctx_create(
      &hw_device_ctx, type, device_ordinal.c_str(), nullptr, 0);
  if (err < 0) {
    TORCH_CHECK(
        false,
        "Failed to create specified HW device",
        get_ffmpeg_error_string_from_error_code(err));
  }
  return hw_device_ctx;
}

#endif

AVBufferRef* get_cuda_context(const torch::Device& device) {
  enum AVHWDeviceType type = av_hwdevice_find_type_by_name("cuda");
  TORCH_CHECK(type != AV_HWDEVICE_TYPE_NONE, "Failed to find cuda device");
  torch::DeviceIndex non_negative_device_index =
      get_ffmpeg_compatible_device_index(device);

  AVBufferRef* hw_device_ctx = get_from_cache(device);
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
  return get_ffmpeg_context_from_existing_cuda_context(
      device, non_negative_device_index, type);
#else
  return get_ffmpeg_context_from_new_cuda_context(
      device, non_negative_device_index, type);
#endif
}

void throw_error_if_non_cuda_device(const torch::Device& device) {
  TORCH_CHECK(
      device.type() != torch::kCPU,
      "Device functions should only be called if the device is not CPU.")
  if (device.type() != torch::kCUDA) {
    throw std::runtime_error("_unsupported device: " + device.str());
  }
}
} // namespace

void release_context_on_cuda(
    const torch::Device& device,
    AVCodecContext* codec_context) {
  throw_error_if_non_cuda_device(device);
  add_to_cache_if_cache_has_capacity(device, codec_context);
}

void initialize_context_on_cuda(
    const torch::Device& device,
    AVCodecContext* codec_context) {
  throw_error_if_non_cuda_device(device);
  // It is important for pytorch itself to create the cuda context. If ffmpeg
  // creates the context it may not be compatible with pytorch.
  // This is a dummy tensor to initialize the cuda context.
  torch::Tensor dummy_tensor_for_cuda_initialization = torch::empty(
      {1}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
  codec_context->hw_device_ctx = get_cuda_context(device);
  return;
}

void convert_avframe_to_frame_output_on_cuda(
    const torch::Device& device,
    const SingleStreamDecoder::VideoStreamOptions& video_stream_options,
    UniqueAVFrame& avframe,
    SingleStreamDecoder::FrameOutput& frame_output,
    std::optional<torch::Tensor> pre_allocated_output_tensor) {
  TORCH_CHECK(
      avframe->format == AV_PIX_FMT_CUDA,
      "Expected format to be AV_PIX_FMT_CUDA, got " +
          std::string(av_get_pix_fmt_name((AVPixelFormat)avframe->format)));
  auto frame_dims = get_height_and_width_from_options_or_avframe(
      video_stream_options, avframe);
  int height = frame_dims.height;
  int width = frame_dims.width;
  torch::Tensor& dst = frame_output.data;
  if (preAllocatedOutputTensor.has_value()) {
    dst = pre_allocated_output_tensor.value();
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
    dst =
        allocate_empty_h_w_c_tensor(height, width, video_stream_options.device);
  }

  // Use the user-requested GPU for running the NPP kernel.
  c10::cuda::CUDAGuard device_guard(device);

  NppiSize o_sizeroi = {width, height};
  Npp8u* input[2] = {avFrame->data[0], avframe->data[1]};

  auto start = std::chrono::high_resolution_clock::now();
  NppStatus status;
  if (avFrame->colorspace == AVColorSpace::AVCOL_SPC_BT709) {
    status = nppiNV12ToRGB_709CSC_8u_P2C3R(
        input,
        avframe->linesize[0],
        static_cast<_npp8u*>(dst.data_ptr()),
        dst.stride(0),
        o_sizeroi);
  } else {
    status = nppi_n_v12_to_r_g_b_8u__p2_c3_r(
        input,
        avframe->linesize[0],
        static_cast<_npp8u*>(dst.data_ptr()),
        dst.stride(0),
        o_sizeroi);
  }
  TORCH_CHECK(status == NPP_SUCCESS, "Failed to convert NV12 frame.");

  // Make the pytorch stream wait for the npp kernel to finish before using the
  // output.
  at::cuda::CUDAEvent npp_done_event;
  at::cuda::CUDAStream npp_stream_wrapper =
      c10::cuda::getStreamFromExternal(nppGetStream(), device.index());
  npp_done_event.record(npp_stream_wrapper);
  nppDoneEvent.block(at::cuda::getCurrentCUDAStream());

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::micro> duration = end - start;
  VLOG(9) << "NPP Conversion of frame height=" << height << " width=" << width
          << " took: " << duration.count() << "us" << std::endl;
}

// inspired by https://github.com/FFmpeg/FFmpeg/commit/ad67ea9
// we have to do this because of an FFmpeg bug where hardware decoding is not
// appropriately set, so we just go off and find the matching codec for the CUDA
// device
std::optional<const AVCodec*> find_cuda_codec(
    const torch::Device& device,
    const AVCodecID& codec_id) {
  throw_error_if_non_cuda_device(device);

  void* i = nullptr;
  const AVCodec* codec = nullptr;
  while ((codec = av_codec_iterate(&i)) != nullptr) {
    if (codec->id != codec_id || !av_codec_is_decoder(codec)) {
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
