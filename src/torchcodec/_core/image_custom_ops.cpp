// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// PyTorch custom-op registration for the image decoders. This lives in the
// libtorchcodec_image library, which is FFmpeg-free and loaded in its own
// RTLD_LOCAL symbol group (see load_torchcodec_shared_libraries): that keeps
// our bundled image codec libs (libjpeg/libpng/libwebp) isolated from the codec
// libs pulled in by the user's FFmpeg, so they can't collide.

#include <memory>
#include <string>

#include "DecodeAvif.h"
#include "DecodeGif.h"
#include "DecodeJpeg.h"
#include "DecodeJpegCuda.h"
#include "DecodePng.h"
#include "DecodeWebp.h"
#include "EncodeJpeg.h"
#include "EncodeJpegCuda.h"
#include "EncodePng.h"
#include "FileIO.h"
#include "FileLikeIO.h"
#include "IOInterface.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

namespace {

void encode_png_to_file(
    const torch::stable::Tensor& img,
    std::string filename,
    int64_t compression_level) {
  FileIO interface(filename, FileIO::Mode::Write);
  encode_png(img, compression_level, interface);
}

void encode_png_to_file_like(
    const torch::stable::Tensor& img,
    int64_t file_like_context,
    int64_t compression_level) {
  auto interface = adopt_file_like_context(file_like_context);
  encode_png(img, compression_level, *interface);
}

void encode_jpeg_dispatch(
    const torch::stable::Tensor& img,
    int64_t quality,
    IOInterface& interface) {
  if (img.device().is_cuda()) {
    encode_jpeg_cuda(img, quality, interface);
  } else {
    encode_jpeg(img, quality, interface);
  }
}

void encode_jpeg_to_file(
    const torch::stable::Tensor& img,
    std::string filename,
    int64_t quality) {
  FileIO interface(filename, FileIO::Mode::Write);
  encode_jpeg_dispatch(img, quality, interface);
}

void encode_jpeg_to_file_like(
    const torch::stable::Tensor& img,
    int64_t file_like_context,
    int64_t quality) {
  auto interface = adopt_file_like_context(file_like_context);
  encode_jpeg_dispatch(img, quality, *interface);
}

} // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(torchcodec_ns, m) {
  m.def("decode_jpeg(Tensor input, int mode) -> Tensor");
  m.def("decode_png(Tensor input, int mode, int output_dtype=0) -> Tensor");
  m.def("decode_webp(Tensor input, int mode) -> Tensor");
  m.def("decode_gif(Tensor input, int mode) -> Tensor");
  m.def(
      "decode_avif(Tensor input, int mode, int output_dtype=0, int num_threads=1) -> Tensor");
  m.def(
      "decode_jpegs_cuda(Tensor[] encoded_images, int mode, Device device) -> Tensor[]");
  m.def(
      "encode_png_to_file(Tensor img, str filename, int compression_level) -> ()");
  m.def(
      "encode_png_to_file_like(Tensor img, int file_like_context, int compression_level) -> ()");
  m.def("encode_jpeg_to_file(Tensor img, str filename, int quality) -> ()");
  m.def(
      "encode_jpeg_to_file_like(Tensor img, int file_like_context, int quality) -> ()");
  m.def("encode_jpeg_to_tensor_cuda(Tensor img, int quality) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(torchcodec_ns, CPU, m) {
  m.impl("decode_jpeg", TORCH_BOX(&decode_jpeg));
  m.impl("decode_png", TORCH_BOX(&decode_png));
  m.impl("decode_webp", TORCH_BOX(&decode_webp));
  m.impl("decode_gif", TORCH_BOX(&decode_gif));
  m.impl("decode_avif", TORCH_BOX(&decode_avif));
  m.impl("encode_png_to_file", TORCH_BOX(&encode_png_to_file));
  m.impl("encode_png_to_file_like", TORCH_BOX(&encode_png_to_file_like));
}

// The JPEG encoders are registered as CompositeExplicitAutograd (rather than
// CPU) because they accept both CPU and CUDA tensors and dispatch on the device
// internally (see encode_jpeg_dispatch).
STABLE_TORCH_LIBRARY_IMPL(torchcodec_ns, CompositeExplicitAutograd, m) {
  m.impl("decode_jpegs_cuda", TORCH_BOX(&decode_jpegs_cuda));
  m.impl("encode_jpeg_to_file", TORCH_BOX(&encode_jpeg_to_file));
  m.impl("encode_jpeg_to_file_like", TORCH_BOX(&encode_jpeg_to_file_like));
  m.impl("encode_jpeg_to_tensor_cuda", TORCH_BOX(&encode_jpeg_to_tensor_cuda));
}

} // namespace facebook::torchcodec
