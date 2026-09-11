// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "DecodeJpegCuda.h"

#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

#include "StableABICompat.h"

// The "no GPU JPEG support" stub is defined here, but only when neither GPU
// backend is compiled in. When ROCm/rocJPEG is enabled, decode_jpegs_cuda is
// defined in DecodeJpegRocm.cpp instead, so this file compiles to nothing.
#if !TORCHCODEC_ENABLE_NVJPEG && !TORCHCODEC_ENABLE_ROCJPEG

namespace facebook::torchcodec {

std::vector<torch::stable::Tensor> decode_jpegs_cuda(
    [[maybe_unused]] std::vector<torch::stable::Tensor> encoded_images,
    [[maybe_unused]] int64_t mode,
    [[maybe_unused]] torch::stable::Device device) {
  STD_TORCH_CHECK(
      false,
      "decode_jpeg: torchcodec was not compiled with nvJPEG support, so JPEG "
      "images cannot be decoded on a CUDA device. Rebuild torchcodec with "
      "ENABLE_CUDA=1 and TORCHCODEC_BUILD_NVJPEG=1. If you see this error in a "
      "prebuilt wheel, please report it to the TorchCodec repo.");
}

} // namespace facebook::torchcodec

#elif TORCHCODEC_ENABLE_NVJPEG

#include "CUDACommon.h"
#include "Cache.h"
#include "DecodeJpegCommon.h"
#include "Exif.h"
#include "ImageCommon.h"

namespace facebook::torchcodec {

// number of cores / nvdec
//
//
// There are two main paths for decoding JPEGs with nvJPEG: the hardware path
// (which uses the built-in silicon 'fixed function' JPEG engine), and the
// software path with will use a mix of CPU and GPU kernels (the software path
// is NOT CPU-only). The software path has 'backends' which determine how to
// split the work between CPU and GPU.
//
// In our implementation, we try to use the HW path if it's available via
// nvjpegDecodeBatched(), and we fallback to the SW path with the 'default
// backend' if HW isn't available, or for those (typically progressive) JPEGs
// that the HW path doesn't support.
//
// A few things worth nothing:
// - Not super clear from the docs, but nvjpegDecodeBatched() seems to be the
// only
//   entry-point for the HW path.
// - nvjpegDecode(), which we use for the SW path, does not seem to support the
//   HW path (but we don't need it to)
// - Calling nvjpegDecodeBatched() on batch_size is much faster than calling it
//   N times on batch_size // N. This justifies why we publicly expose a batched
//   API and batched inputs.
// - Even for a single image, calling nvjpegDecodeBatched() is not slower than
//   the SW path, so there's no reason to have a smart dispatching logic based
//   on batch size. We always route through the HW path when we can.

using namespace exif_private;

namespace {

// We cache decoder objects for the same reason we cache NVDEC decoders: they're
// expensive to create and destroy. To determine the ideal cache size, I ran
// benchmarks on a A100 where each thread calls its own decode_jpeg():
// - The best throughput is achieved when num_threads ==
//   kMaxCachedDecodersPerDevice. As soon as num_threads >
//   kMaxCachedDecodersPerDevice, throughput collapses sharply, because the cost
//   of creating a new decoder is so prohibitive.
// - Provided the batch size is large enough, num_threads=1 in the HW path is
//   pretty close to the roofline already (~6k fps @ 720p) because
//   jpegDecodeBatched() will dispatch to the jpeg cores itself. So the benefits
//   of multithreading the jpeg decoding is small-ish anyway.
//
// One decoder takes <30Mb of GPU memory and RAM, so it's fairly cheap. Since
// the downside of not being able to cache a decoder is so high, we allow a
// generous amount of decoders in the cache. This isn't exposed for now, it
// probably shouldn't be.
constexpr int kMaxCachedDecodersPerDevice = 32;

PerGpuCache<CUDAJpegDecoder>& decoder_cache() {
  // Intentionally leaked (allocated with new, never freed) to avoid calling
  // into CUDA/nvJPEG during static destruction, when the CUDA runtime may
  // already be torn down (same reasoning as NVDECCache): the cached decoders'
  // destructors call nvjpegDestroy, which we must not run at process exit.
  static auto* cache = new PerGpuCache<CUDAJpegDecoder>(
      MAX_CUDA_GPUS, kMaxCachedDecodersPerDevice);
  return *cache;
}

} // namespace

std::vector<torch::stable::Tensor> decode_jpegs_cuda(
    std::vector<torch::stable::Tensor> encoded_images,
    int64_t mode,
    torch::stable::Device device) {
  STD_TORCH_CHECK(
      device.is_cuda(), "Expected the device parameter to be a cuda device");
  STD_TORCH_CHECK(
      !encoded_images.empty(), "Expected at least one image to decode");

  std::vector<torch::stable::Tensor> contig_images;
  contig_images.reserve(encoded_images.size());
  std::vector<ExifOrientation> orientations;
  orientations.reserve(encoded_images.size());

  for (const auto& encoded_image : encoded_images) {
    auto contig = validate_encoded_data(encoded_image);
    orientations.push_back(fetch_exif_orientation_from_jpeg_bytes(
        contig.const_data_ptr<uint8_t>(), contig.numel()));
    contig_images.push_back(std::move(contig));
  }

  int device_index = get_device_index(device);
  StableDeviceGuard device_guard(device_index);

  cudaStream_t current_stream = get_current_cuda_stream(device_index);

  PerGpuCache<CUDAJpegDecoder>& cache = decoder_cache();
  std::unique_ptr<CUDAJpegDecoder> decoder = cache.get(device);
  if (decoder == nullptr) {
    decoder = std::make_unique<CUDAJpegDecoder>(device);
  }

  std::vector<torch::stable::Tensor> output = decoder->decode_images(
      contig_images, static_cast<ImageReadMode>(mode), current_stream);
  cache.add_if_cache_has_capacity(device, std::move(decoder));

  for (size_t i = 0; i < output.size(); ++i) {
    output[i] = exif_orientation_transform(output[i], orientations[i]);
  }
  return output;
}

CUDAJpegDecoder::CUDAJpegDecoder(const torch::stable::Device& target_device)
    : target_device_(target_device) {
  StableDeviceGuard device_guard(target_device_.index());

  nvjpegStatus_t status;

  hw_decode_available_ = true;
  status = nvjpegCreateEx(
      NVJPEG_BACKEND_HARDWARE,
      NULL,
      NULL,
      NVJPEG_FLAGS_DEFAULT,
      &nvjpeg_handle_);
  if (status == NVJPEG_STATUS_ARCH_MISMATCH) {
    // No hardware JPEG decoder on this GPU (pre-A100); fall back to the default
    // (software) backend.
    status = nvjpegCreateEx(
        NVJPEG_BACKEND_DEFAULT,
        NULL,
        NULL,
        NVJPEG_FLAGS_DEFAULT,
        &nvjpeg_handle_);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to initialize nvjpeg with default backend: ",
        status);
    hw_decode_available_ = false;
  } else {
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to initialize nvjpeg with hardware backend: ",
        status);
  }

  if (hw_decode_available_) {
    status = nvjpegJpegStateCreate(nvjpeg_handle_, &nvjpeg_state_hw_);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to create nvjpeg state: ",
        status);

    status = nvjpegJpegStreamCreate(nvjpeg_handle_, &nvjpeg_stream_);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to create jpeg stream: ",
        status);
  }

  // Software path state, used by nvjpegDecode() for progressive JPEGs and for
  // everything when there's no HW engine.
  status = nvjpegJpegStateCreate(nvjpeg_handle_, &nvjpeg_state_sw_);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS,
      "Failed to create nvjpeg decode state: ",
      status);
}

CUDAJpegDecoder::~CUDAJpegDecoder() {
  // We properly destroy the nvjpeg stuff here. Note that this destructor is
  // only called when a decoder cannot return to the cache. This is never
  // reached during normal process teardown, because we just leak the entire
  // decoder cache, just like we leak the NVDEC cache to avoid weird CUDA
  // teardown issues.
  nvjpegJpegStateDestroy(nvjpeg_state_sw_);
  if (hw_decode_available_) {
    nvjpegJpegStreamDestroy(nvjpeg_stream_);
    nvjpegJpegStateDestroy(nvjpeg_state_hw_);
  }
  nvjpegDestroy(nvjpeg_handle_);
}

// Allocates the output tensor and creates its corresponding nvjpegImage_t for a
// single encoded image. Also figures out the output mode (nvjpegOutputFormat_t)
// based on the requested ImageReadMode and the source.
std::tuple<torch::stable::Tensor, nvjpegImage_t, nvjpegOutputFormat_t>
CUDAJpegDecoder::allocate_output(
    const torch::stable::Tensor& encoded_image,
    ImageReadMode mode) {
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];
  int source_channels = 0;
  nvjpegChromaSubsampling_t subsampling;
  nvjpegStatus_t status = nvjpegGetImageInfo(
      nvjpeg_handle_,
      encoded_image.const_data_ptr<uint8_t>(),
      encoded_image.numel(),
      &source_channels,
      &subsampling,
      widths,
      heights);
  STD_TORCH_CHECK(
      status == NVJPEG_STATUS_SUCCESS, "Failed to get image info: ", status);
  STD_TORCH_CHECK(
      subsampling != NVJPEG_CSS_UNKNOWN, "Unknown chroma subsampling");

  nvjpegOutputFormat_t output_format;
  switch (mode) {
    case ImageReadMode::GRAY:
      output_format = NVJPEG_OUTPUT_Y;
      break;
    case ImageReadMode::RGB:
      output_format = NVJPEG_OUTPUT_RGB;
      break;
    case ImageReadMode::UNCHANGED:
      output_format =
          source_channels == 1 ? NVJPEG_OUTPUT_Y : NVJPEG_OUTPUT_RGB;
      break;
    default:
      STD_TORCH_CHECK(
          false,
          "The provided mode is not supported for JPEG decoding on GPU. "
          "Supported modes are UNCHANGED, GRAY and RGB; alpha modes are "
          "emulated in Python.");
  }
  int output_channels = (output_format == NVJPEG_OUTPUT_Y) ? 1 : 3;

  auto output_tensor = torch::stable::empty(
      {int64_t(output_channels), int64_t(heights[0]), int64_t(widths[0])},
      kStableUInt8,
      std::nullopt,
      target_device_);

  nvjpegImage_t nvjpeg_image;
  for (int c = 0; c < output_channels; ++c) {
    nvjpeg_image.channel[c] =
        torch::stable::select(output_tensor, 0, c).mutable_data_ptr<uint8_t>();
    nvjpeg_image.pitch[c] = widths[0];
  }
  for (int c = output_channels; c < NVJPEG_MAX_COMPONENT; ++c) {
    nvjpeg_image.channel[c] = nullptr;
    nvjpeg_image.pitch[c] = 0;
  }
  return {output_tensor, nvjpeg_image, output_format};
}

// Split image indices into the hardware-batched group (baseline JPEGs when
// the HW engine is available) and the software group (everything else).
// Returns {hw_indices, sw_indices} into encoded_images / output_tensors.
std::pair<std::vector<size_t>, std::vector<size_t>>
CUDAJpegDecoder::split_images_by_backend(
    const std::vector<torch::stable::Tensor>& encoded_images) {
  // Baseline JPEGs (hardware-batch decodable on A100+) go to the hardware
  // group; everything else (e.g. progressive), and all images when there's no
  // HW engine, go to the software group. See
  // https://github.com/NVIDIA/CUDALibrarySamples/blob/f17940ac4e705bf47a8c39f5365925c1665f6c98/nvJPEG/nvJPEG-Decoder/nvjpegDecoder.cpp#L33
  std::vector<size_t> hw_indices, sw_indices;
  for (size_t i = 0; i < encoded_images.size(); ++i) {
    bool supports_hw = false;
    if (hw_decode_available_) {
      nvjpegJpegStreamParseHeader(
          nvjpeg_handle_,
          encoded_images[i].const_data_ptr<uint8_t>(),
          encoded_images[i].numel(),
          nvjpeg_stream_);
      int is_supported = -1;
      nvjpegDecodeBatchedSupported(
          nvjpeg_handle_, nvjpeg_stream_, &is_supported);
      supports_hw = is_supported == 0; // nvJPEG sets 0 when supported
    }
    (supports_hw ? hw_indices : sw_indices).push_back(i);
  }
  return {std::move(hw_indices), std::move(sw_indices)};
}

void CUDAJpegDecoder::decode_batched_hardware(
    const std::vector<torch::stable::Tensor>& encoded_images,
    const std::vector<size_t>& indices,
    ImageReadMode mode,
    cudaStream_t stream,
    std::vector<torch::stable::Tensor>& output_tensors) {
  std::vector<const unsigned char*> inputs;
  std::vector<size_t> sizes;
  std::vector<nvjpegImage_t> nvjpeg_images;
  std::vector<nvjpegOutputFormat_t> formats;
  inputs.reserve(indices.size());
  sizes.reserve(indices.size());
  nvjpeg_images.reserve(indices.size());
  formats.reserve(indices.size());
  for (size_t idx : indices) {
    auto [output_tensor, nvjpeg_image, output_format] =
        allocate_output(encoded_images[idx], mode);
    output_tensors[idx] = output_tensor;
    inputs.push_back(encoded_images[idx].const_data_ptr<uint8_t>());
    sizes.push_back(encoded_images[idx].numel());
    nvjpeg_images.push_back(nvjpeg_image);
    formats.push_back(output_format);
  }

  // The batch nvjpeg API only support a single output format per call, but we
  // may want both grayscale and RGB images here. So we need to split the input
  // into two groups and decode them separately. To be safe, we add  stream sync
  // point between the groups: the calls of the first group may read/write
  // scratch buffers inside the shared nvjpeg_state_hw_, and the second group's
  // nvjpegDecodeBatchedInitialize can reconfigure that same state. This is an
  // assumption, but the sync point shouldn't hurt perf.
  bool needs_sync = false;
  for (nvjpegOutputFormat_t group_format :
       {NVJPEG_OUTPUT_Y, NVJPEG_OUTPUT_RGB}) {
    std::vector<const unsigned char*> group_inputs;
    std::vector<size_t> group_sizes;
    std::vector<nvjpegImage_t> group_images;
    for (size_t i = 0; i < formats.size(); ++i) {
      if (formats[i] == group_format) {
        group_inputs.push_back(inputs[i]);
        group_sizes.push_back(sizes[i]);
        group_images.push_back(nvjpeg_images[i]);
      }
    }
    if (group_inputs.empty()) {
      continue;
    }

    if (needs_sync) {
      cudaError_t cuda_status = cudaStreamSynchronize(stream);
      STD_TORCH_CHECK(
          cuda_status == cudaSuccess,
          "Failed to synchronize CUDA stream: ",
          cuda_status);
    }

    // Should we expose max_cpu_threads????
    nvjpegStatus_t status = nvjpegDecodeBatchedInitialize(
        nvjpeg_handle_,
        nvjpeg_state_hw_,
        group_images.size(),
        /*max_cpu_threads=*/1,
        group_format);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS,
        "Failed to initialize batch decoding: ",
        status);

    status = nvjpegDecodeBatched(
        nvjpeg_handle_,
        nvjpeg_state_hw_,
        group_inputs.data(),
        group_sizes.data(),
        group_images.data(),
        stream);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS, "Failed to decode batch: ", status);

    needs_sync = true;
  }
}

void CUDAJpegDecoder::decode_software(
    const std::vector<torch::stable::Tensor>& encoded_images,
    const std::vector<size_t>& indices,
    ImageReadMode mode,
    cudaStream_t stream,
    std::vector<torch::stable::Tensor>& output_tensors) {
  for (size_t idx : indices) {
    auto [output_tensor, nvjpeg_image, output_format] =
        allocate_output(encoded_images[idx], mode);
    // Decode each image straight to its own native format (see
    // output_format_for_image), so grayscale sources stay single-channel.
    nvjpegStatus_t status = nvjpegDecode(
        nvjpeg_handle_,
        nvjpeg_state_sw_,
        encoded_images[idx].const_data_ptr<uint8_t>(),
        encoded_images[idx].numel(),
        output_format,
        &nvjpeg_image,
        stream);
    STD_TORCH_CHECK(
        status == NVJPEG_STATUS_SUCCESS, "nvjpegDecode failed: ", status);
    output_tensors[idx] = output_tensor;
  }
}

std::vector<torch::stable::Tensor> CUDAJpegDecoder::decode_images(
    const std::vector<torch::stable::Tensor>& encoded_images,
    ImageReadMode mode,
    cudaStream_t stream) {
  std::vector<torch::stable::Tensor> output_tensors(encoded_images.size());

  auto [hw_indices, sw_indices] = split_images_by_backend(encoded_images);
  if (!hw_indices.empty()) {
    decode_batched_hardware(
        encoded_images, hw_indices, mode, stream, output_tensors);
  }
  if (!sw_indices.empty()) {
    decode_software(encoded_images, sw_indices, mode, stream, output_tensors);
  }

  // Host-synchronize before returning: the decoder (and its internal nvJPEG
  // buffers) goes back to the pool and may be reused immediately by the next
  // call, so all GPU work using them must complete first.
  // TODO_IMAGE: Should we? What does the NVDEC decoder do?
  cudaError_t cuda_status = cudaStreamSynchronize(stream);
  STD_TORCH_CHECK(
      cuda_status == cudaSuccess,
      "Failed to synchronize CUDA stream: ",
      cuda_status);

  return output_tensors;
}

} // namespace facebook::torchcodec

#endif // GPU JPEG backend selection
