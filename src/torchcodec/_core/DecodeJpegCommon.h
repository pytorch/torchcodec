// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstring>

#include "Exif.h"

// Helpers shared by the GPU JPEG decoders (nvJPEG on CUDA, rocJPEG on ROCm).
// The CPU JPEG decoder relies on libjpeg for EXIF handling; the GPU decoders
// don't decode metadata, so we scan the bitstream for orientation ourselves.

namespace facebook::torchcodec {

// Scan a JPEG bitstream for the APP1/EXIF segment and return its orientation.
inline exif_private::ExifOrientation fetch_exif_orientation_from_jpeg_bytes(
    const unsigned char* jpeg,
    size_t size) {
  constexpr unsigned char MARKER_PREFIX = 0xFF;
  constexpr unsigned char SOI = 0xD8;
  constexpr unsigned char SOS = 0xDA; // start of scan: no more metadata markers
  constexpr unsigned char EOI = 0xD9;
  constexpr unsigned char APP1 = 0xE1;
  constexpr size_t exif_header_size = 6; // "Exif\0\0"

  if (size < 2 || jpeg[0] != MARKER_PREFIX || jpeg[1] != SOI) {
    return exif_private::ExifOrientation::Unspecified;
  }

  size_t pos = 2;
  while (pos + 4 <= size && jpeg[pos] == MARKER_PREFIX) {
    unsigned char marker = jpeg[pos + 1];
    if (marker == SOS || marker == EOI) {
      break;
    }
    // Segment length is big-endian and includes the 2 length bytes themselves.
    size_t segment_length =
        (size_t(jpeg[pos + 2]) << 8) | size_t(jpeg[pos + 3]);
    if (segment_length < 2 || pos + 2 + segment_length > size) {
      break;
    }

    if (marker == APP1 && segment_length >= 2 + exif_header_size) {
      const unsigned char* payload = jpeg + pos + 4;
      if (std::memcmp(payload, "Exif\0\0", exif_header_size) == 0) {
        return exif_private::fetch_exif_orientation(
            payload + exif_header_size, segment_length - 2 - exif_header_size);
      }
    }
    pos += 2 + segment_length;
  }
  return exif_private::ExifOrientation::Unspecified;
}

} // namespace facebook::torchcodec
