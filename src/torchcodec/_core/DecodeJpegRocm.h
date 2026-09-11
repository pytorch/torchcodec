// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/tensor.h>

#include <memory>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

#include "ImageCommon.h"
#include "StableABICompat.h"

// The public decode_jpegs_cuda entry point is declared in DecodeJpegCuda.h and
// shared across GPU backends. This header only declares the rocJPEG-specific
// implementation classes, so it's included by DecodeJpegRocm.cpp alone.
#if TORCHCODEC_ENABLE_ROCJPEG
// Host-only HIP API (hipGetDevice, hipDeviceSynchronize, ...). We use this
// rather than <hip/hip_runtime.h> because this file is compiled by the regular
// C++ compiler, not hipcc, and makes only host-side runtime calls.
#include <hip/hip_runtime_api.h>
#include <rocjpeg/rocjpeg.h>

namespace facebook::torchcodec {

// rocJPEG counterpart of CUDAJpegDecoder. rocJPEG picks its decoding path at
// handle-creation time via the backend, rather than per-call: the HARDWARE
// backend drives the VCN fixed-function JPEG engine (baseline JPEGs), and the
// HYBRID backend does entropy decoding on the host and the rest on the GPU
// (needed for progressive JPEGs and when there's no HW engine). We create the
// HARDWARE handle up front and the HYBRID handle lazily, only if we hit a JPEG
// the HW engine can't take.
class RocJpegDecoder {
 public:
  explicit RocJpegDecoder(const torch::stable::Device& target_device);
  ~RocJpegDecoder();

  std::vector<torch::stable::Tensor> decode_images(
      const std::vector<torch::stable::Tensor>& encoded_images,
      ImageReadMode mode);

 private:
  // A parsed rocJPEG bitstream plus the output tensor we decode it into and the
  // output format we picked for it. One per input image.
  struct ImagePlan {
    RocJpegStreamHandle stream{nullptr};
    torch::stable::Tensor output_tensor;
    RocJpegImage output_image{};
    RocJpegOutputFormat output_format{ROCJPEG_OUTPUT_NATIVE};
  };

  RocJpegHandle base_handle();
  RocJpegHandle ensure_hybrid_handle();

  ImagePlan make_plan(
      const torch::stable::Tensor& encoded_image,
      ImageReadMode mode);

  std::pair<std::vector<size_t>, std::vector<size_t>> split_images_by_backend(
      const std::vector<torch::stable::Tensor>& encoded_images);

  void decode_batched_hardware(
      std::vector<ImagePlan>& plans,
      const std::vector<size_t>& indices);

  void decode_hybrid(
      std::vector<ImagePlan>& plans,
      const std::vector<size_t>& indices);

  const torch::stable::Device target_device_;
  const int device_index_;

  // HARDWARE backend handle (VCN engine); null if this GPU has no HW JPEG
  // engine, in which case everything goes through the HYBRID handle.
  RocJpegHandle handle_hw_{nullptr};
  bool hw_decode_available_{false};

  // HYBRID backend handle, created lazily the first time we need it
  // (progressive JPEGs, or all images when there's no HW engine).
  RocJpegHandle handle_hybrid_{nullptr};
};

// A per-device pool of reusable RocJpegDecoder objects. Modeled on NVJpegCache
// / NVDECCache.
class RocJpegCache {
 public:
  static RocJpegCache& get_cache(const torch::stable::Device& device);

  std::unique_ptr<RocJpegDecoder> get_decoder(
      const torch::stable::Device& device);

  void return_decoder(std::unique_ptr<RocJpegDecoder> decoder);

 private:
  static RocJpegCache* get_cache_instances();

  std::vector<std::unique_ptr<RocJpegDecoder>> pool_;
  std::mutex pool_lock_;
};

} // namespace facebook::torchcodec

#endif // TORCHCODEC_ENABLE_ROCJPEG
