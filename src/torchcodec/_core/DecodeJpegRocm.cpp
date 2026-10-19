// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "DecodeJpegCuda.h" // for the shared decode_jpegs_cuda declaration

#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

#include "StableABICompat.h"

#if TORCHCODEC_ENABLE_ROCJPEG

#include "DecodeJpegCommon.h"
#include "DecodeJpegRocm.h"
#include "ImageCommon.h"

// rocJPEG is AMD's GPU JPEG decoder (part of the ROCm stack) and is the ROCm
// counterpart to nvJPEG on CUDA (see DecodeJpegCuda.cpp). PyTorch exposes AMD
// GPUs under the "cuda" device string (HIP masquerades as CUDA), so the public
// entry point and the Python dispatch are shared; only the backend library
// differs.
//
// As with nvJPEG there are two paths:
// - HARDWARE: the VCN fixed-function JPEG engine, driven by
// rocJpegDecodeBatched
//   for baseline JPEGs.
// - HYBRID: host-side entropy decode + GPU for the rest, used for progressive
//   JPEGs and whenever there's no HW engine.
// rocJPEG selects the path at rocJpegCreate() time via the backend, and (unlike
// nvJPEG) offers no per-image "is this supported by HW?" query, so we detect
// baseline vs. progressive ourselves from the SOFn marker.

namespace facebook::torchcodec {

using namespace exif_private;

namespace {

constexpr int kMemAlignment = 16;

inline uint32_t align_up(uint32_t value, uint32_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

// See kMaxCachedDecodersPerDevice in DecodeJpegCuda.cpp for the rationale.
constexpr size_t kMaxCachedDecodersPerDevice = 32;

// PyTorch supports up to 128 GPUs; mirror MAX_CUDA_GPUS from CUDACommon.h
// (which we can't include here because it pulls in <cuda_runtime.h>).
constexpr int MAX_ROCM_GPUS = 128;

int get_hip_device_index(const torch::stable::Device& device) {
  int device_index = static_cast<int>(device.index());
  STD_TORCH_CHECK(
      device_index >= -1 && device_index < MAX_ROCM_GPUS,
      "Invalid device index = ",
      device_index);
  if (device_index == -1) {
    STD_TORCH_CHECK(
        hipGetDevice(&device_index) == hipSuccess,
        "Failed to get current HIP device.");
  }
  return device_index;
}

// Whether the HARDWARE (VCN) engine can decode this JPEG. The HW engine handles
// baseline / extended-sequential DCT (SOF0/SOF1); progressive, lossless and the
// arithmetic-coded variants must go through the HYBRID path. We scan the marker
// segments for the first Start-Of-Frame and classify by its code.
bool is_hw_decodable_jpeg(const unsigned char* jpeg, size_t size) {
  constexpr unsigned char MARKER_PREFIX = 0xFF;
  constexpr unsigned char SOI = 0xD8;
  constexpr unsigned char SOS = 0xDA;
  constexpr unsigned char SOF0 = 0xC0; // baseline DCT
  constexpr unsigned char SOF1 = 0xC1; // extended sequential DCT

  if (size < 2 || jpeg[0] != MARKER_PREFIX || jpeg[1] != SOI) {
    return false;
  }
  size_t pos = 2;
  while (pos + 4 <= size && jpeg[pos] == MARKER_PREFIX) {
    unsigned char marker = jpeg[pos + 1];
    if (marker == SOS) {
      break; // reached scan data without an SOFn we recognize
    }
    if (marker == SOF0 || marker == SOF1) {
      return true;
    }
    size_t segment_length =
        (size_t(jpeg[pos + 2]) << 8) | size_t(jpeg[pos + 3]);
    if (segment_length < 2 || pos + 2 + segment_length > size) {
      break;
    }
    pos += 2 + segment_length;
  }
  return false;
}

} // namespace

RocJpegCache* RocJpegCache::get_cache_instances() {
  // Intentionally leaked, like NVJpegCache / NVDECCache: destroying rocJPEG
  // handles during static teardown can crash when the HIP runtime is already
  // gone.
  static RocJpegCache* cache_instances = new RocJpegCache[MAX_ROCM_GPUS];
  return cache_instances;
}

RocJpegCache& RocJpegCache::get_cache(const torch::stable::Device& device) {
  return get_cache_instances()[get_hip_device_index(device)];
}

std::unique_ptr<RocJpegDecoder> RocJpegCache::get_decoder(
    const torch::stable::Device& device) {
  {
    std::lock_guard<std::mutex> lock(pool_lock_);
    if (!pool_.empty()) {
      auto decoder = std::move(pool_.back());
      pool_.pop_back();
      return decoder;
    }
  }
  return std::make_unique<RocJpegDecoder>(device);
}

void RocJpegCache::return_decoder(std::unique_ptr<RocJpegDecoder> decoder) {
  STD_TORCH_CHECK(decoder != nullptr, "decoder must not be null");
  std::lock_guard<std::mutex> lock(pool_lock_);
  if (pool_.size() < kMaxCachedDecodersPerDevice) {
    pool_.push_back(std::move(decoder));
  }
  // Otherwise let `decoder` go out of scope and be destroyed.
}

std::vector<torch::stable::Tensor> decode_jpegs_cuda(
    std::vector<torch::stable::Tensor> encoded_images,
    int64_t mode,
    torch::stable::Device device) {
  STD_TORCH_CHECK(
      device.is_cuda(),
      "Expected the device parameter to be a cuda (ROCm) device");
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

  int device_index = get_hip_device_index(device);
  StableDeviceGuard device_guard(device_index);

  RocJpegCache& cache = RocJpegCache::get_cache(device);
  std::unique_ptr<RocJpegDecoder> decoder = cache.get_decoder(device);

  std::vector<torch::stable::Tensor> output =
      decoder->decode_images(contig_images, static_cast<ImageReadMode>(mode));
  cache.return_decoder(std::move(decoder));

  for (size_t i = 0; i < output.size(); ++i) {
    output[i] = exif_orientation_transform(output[i], orientations[i]);
  }
  return output;
}

RocJpegDecoder::RocJpegDecoder(const torch::stable::Device& target_device)
    : target_device_(target_device),
      device_index_(get_hip_device_index(target_device)) {
  StableDeviceGuard device_guard(device_index_);

  RocJpegStatus status =
      rocJpegCreate(ROCJPEG_BACKEND_HARDWARE, device_index_, &handle_hw_);
  if (status == ROCJPEG_STATUS_SUCCESS) {
    hw_decode_available_ = true;
    fprintf(
        stderr,
        "[torchcodec] rocJpegCreate(HARDWARE) succeeded on device %d"
        " -- VCN JPEG engine is available and will be used\n",
        device_index_);
  } else {
    fprintf(
        stderr,
        "[torchcodec] rocJpegCreate(HARDWARE) failed on device %d: %s"
        " -- falling back to HYBRID\n",
        device_index_,
        rocJpegGetErrorName(status));
    // No usable HW JPEG engine on this GPU: fall back to the hybrid backend for
    // everything. Create it eagerly here so base_handle() always has a handle.
    handle_hw_ = nullptr;
    hw_decode_available_ = false;
    status =
        rocJpegCreate(ROCJPEG_BACKEND_HYBRID, device_index_, &handle_hybrid_);
    STD_TORCH_CHECK(
        status == ROCJPEG_STATUS_SUCCESS,
        "Failed to initialize rocJPEG with the hybrid backend: ",
        rocJpegGetErrorName(status));
  }
}

RocJpegDecoder::~RocJpegDecoder() {
  // Only reached when a decoder can't return to the cache; the cache itself is
  // leaked (see get_cache_instances) to avoid HIP teardown issues.
  if (handle_hw_ != nullptr) {
    rocJpegDestroy(handle_hw_);
  }
  if (handle_hybrid_ != nullptr) {
    rocJpegDestroy(handle_hybrid_);
  }
}

RocJpegHandle RocJpegDecoder::base_handle() {
  // Any valid handle can serve rocJpegGetImageInfo (it only parses headers).
  return handle_hw_ != nullptr ? handle_hw_ : handle_hybrid_;
}

RocJpegHandle RocJpegDecoder::ensure_hybrid_handle() {
  if (handle_hybrid_ == nullptr) {
    RocJpegStatus status =
        rocJpegCreate(ROCJPEG_BACKEND_HYBRID, device_index_, &handle_hybrid_);
    STD_TORCH_CHECK(
        status == ROCJPEG_STATUS_SUCCESS,
        "Failed to initialize rocJPEG with the hybrid backend: ",
        rocJpegGetErrorName(status));
  }
  return handle_hybrid_;
}

// Parse the bitstream, query its dimensions, pick an output format from the
// requested mode, and allocate the (C, H, W) uint8 output tensor plus the
// RocJpegImage that points into it. Grayscale sources decoded as GRAY stay
// single-channel; RGB output is planar (RGB_PLANAR) so the layout matches the
// (C, H, W) tensor, exactly like nvJPEG's planar NVJPEG_OUTPUT_RGB.
RocJpegDecoder::ImagePlan RocJpegDecoder::make_plan(
    const torch::stable::Tensor& encoded_image,
    ImageReadMode mode) {
  ImagePlan plan;

  RocJpegStatus status = rocJpegStreamCreate(&plan.stream);
  STD_TORCH_CHECK(
      status == ROCJPEG_STATUS_SUCCESS,
      "Failed to create rocJPEG stream: ",
      rocJpegGetErrorName(status));
  status = rocJpegStreamParse(
      encoded_image.const_data_ptr<uint8_t>(),
      encoded_image.numel(),
      plan.stream);
  STD_TORCH_CHECK(
      status == ROCJPEG_STATUS_SUCCESS,
      "Failed to parse rocJPEG stream: ",
      rocJpegGetErrorName(status));

  uint8_t source_channels = 0;
  RocJpegChromaSubsampling subsampling = ROCJPEG_CSS_UNKNOWN;
  uint32_t widths[ROCJPEG_MAX_COMPONENT] = {0};
  uint32_t heights[ROCJPEG_MAX_COMPONENT] = {0};
  status = rocJpegGetImageInfo(
      base_handle(),
      plan.stream,
      &source_channels,
      &subsampling,
      widths,
      heights);
  STD_TORCH_CHECK(
      status == ROCJPEG_STATUS_SUCCESS,
      "Failed to get rocJPEG image info: ",
      rocJpegGetErrorName(status));
  STD_TORCH_CHECK(
      subsampling != ROCJPEG_CSS_UNKNOWN, "Unknown chroma subsampling");

  switch (mode) {
    case ImageReadMode::GRAY:
      plan.output_format = ROCJPEG_OUTPUT_Y;
      break;
    case ImageReadMode::RGB:
      plan.output_format = ROCJPEG_OUTPUT_RGB_PLANAR;
      break;
    case ImageReadMode::UNCHANGED:
      plan.output_format =
          (source_channels == 1) ? ROCJPEG_OUTPUT_Y : ROCJPEG_OUTPUT_RGB_PLANAR;
      break;
    default:
      STD_TORCH_CHECK(
          false,
          "The provided mode is not supported for JPEG decoding on GPU. "
          "Supported modes are UNCHANGED, GRAY and RGB; alpha modes are "
          "emulated in Python.");
  }
  int output_channels = (plan.output_format == ROCJPEG_OUTPUT_Y) ? 1 : 3;

  // rocJpegDecodeBatched writes rows at a 16-byte-aligned pitch, so allocate a
  // buffer padded to that alignment and narrow it back to the valid region.
  uint32_t aligned_pitch = align_up(widths[0], kMemAlignment);
  uint32_t aligned_height = align_up(heights[0], kMemAlignment);
  auto buffer = torch::stable::empty(
      {int64_t(output_channels), int64_t(aligned_height), int64_t(aligned_pitch)},
      kStableUInt8,
      std::nullopt,
      target_device_);

  for (int c = 0; c < output_channels; ++c) {
    plan.output_image.channel[c] =
        torch::stable::select(buffer, 0, c).mutable_data_ptr<uint8_t>();
    plan.output_image.pitch[c] = aligned_pitch;
  }
  for (int c = output_channels; c < ROCJPEG_MAX_COMPONENT; ++c) {
    plan.output_image.channel[c] = nullptr;
    plan.output_image.pitch[c] = 0;
  }

  // Return a view of the valid (unpadded) region.
  auto valid_height = torch::stable::narrow(buffer, 1, 0, heights[0]);
  plan.output_tensor = torch::stable::narrow(valid_height, 2, 0, widths[0]);
  return plan;
}

std::pair<std::vector<size_t>, std::vector<size_t>>
RocJpegDecoder::split_images_by_backend(
    const std::vector<torch::stable::Tensor>& encoded_images) {
  std::vector<size_t> hw_indices, hybrid_indices;
  for (size_t i = 0; i < encoded_images.size(); ++i) {
    bool supports_hw = hw_decode_available_ &&
        is_hw_decodable_jpeg(
                           encoded_images[i].const_data_ptr<uint8_t>(),
                           encoded_images[i].numel());
    (supports_hw ? hw_indices : hybrid_indices).push_back(i);
  }
  return {std::move(hw_indices), std::move(hybrid_indices)};
}

void RocJpegDecoder::decode_batched_hardware(
    std::vector<ImagePlan>& plans,
    const std::vector<size_t>& indices) {
  fprintf(
      stderr,
      "[torchcodec] decoding %zu image(s) using HARDWARE (VCN) backend\n",
      indices.size());
  // rocJpegDecodeBatched takes a single output format for the whole batch, but
  // the batch may mix grayscale (Y) and RGB images, so we split into per-format
  // sub-batches, same as the nvJPEG HW path.
  for (RocJpegOutputFormat group_format :
       {ROCJPEG_OUTPUT_Y, ROCJPEG_OUTPUT_RGB_PLANAR}) {
    std::vector<RocJpegStreamHandle> group_streams;
    std::vector<RocJpegImage> group_images;
    // rocJpegDecodeBatched indexes decode_params[i] per image, so we must
    // pass an array — not a pointer to a single struct.
    std::vector<RocJpegDecodeParams> group_params;

    for (size_t idx : indices) {
      if (plans[idx].output_format != group_format) {
        continue;
      }
      group_streams.push_back(plans[idx].stream);
      group_images.push_back(plans[idx].output_image);
      RocJpegDecodeParams params = {};
      params.output_format = group_format;
      group_params.push_back(params);
    }
    if (group_streams.empty()) {
      continue;
    }

    RocJpegStatus status = rocJpegDecodeBatched(
        handle_hw_,
        group_streams.data(),
        static_cast<int>(group_streams.size()),
        group_params.data(),
        group_images.data());
    STD_TORCH_CHECK(
        status == ROCJPEG_STATUS_SUCCESS,
        "rocJpegDecodeBatched failed: ",
        rocJpegGetErrorName(status));
  }
}

void RocJpegDecoder::decode_hybrid(
    std::vector<ImagePlan>& plans,
    const std::vector<size_t>& indices) {
  RocJpegHandle handle = ensure_hybrid_handle();
  for (size_t idx : indices) {
    RocJpegDecodeParams params = {};
    params.output_format = plans[idx].output_format;

    RocJpegStatus status = rocJpegDecode(
        handle, plans[idx].stream, &params, &plans[idx].output_image);
    STD_TORCH_CHECK(
        status == ROCJPEG_STATUS_SUCCESS,
        "rocJpegDecode failed: ",
        rocJpegGetErrorName(status));
  }
}

std::vector<torch::stable::Tensor> RocJpegDecoder::decode_images(
    const std::vector<torch::stable::Tensor>& encoded_images,
    ImageReadMode mode) {
  std::vector<ImagePlan> plans;
  plans.reserve(encoded_images.size());
  for (const auto& encoded_image : encoded_images) {
    plans.push_back(make_plan(encoded_image, mode));
  }

  auto [hw_indices, hybrid_indices] = split_images_by_backend(encoded_images);
  if (!hw_indices.empty()) {
    decode_batched_hardware(plans, hw_indices);
  }
  if (!hybrid_indices.empty()) {
    decode_hybrid(plans, hybrid_indices);
  }

  // rocJPEG decode calls take no explicit stream and run on the library's own
  // internal stream, so we device-synchronize before touching the outputs on
  // torch's stream (and before the decoder returns to the pool for reuse).
  hipError_t hip_status = hipDeviceSynchronize();
  STD_TORCH_CHECK(
      hip_status == hipSuccess,
      "Failed to synchronize HIP device: ",
      hipGetErrorString(hip_status));

  std::vector<torch::stable::Tensor> output_tensors;
  output_tensors.reserve(plans.size());
  for (auto& plan : plans) {
    rocJpegStreamDestroy(plan.stream);
    output_tensors.push_back(std::move(plan.output_tensor));
  }
  return output_tensors;
}

} // namespace facebook::torchcodec

#endif // TORCHCODEC_ENABLE_ROCJPEG
