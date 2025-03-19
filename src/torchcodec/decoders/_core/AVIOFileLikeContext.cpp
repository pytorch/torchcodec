// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/decoders/_core/AVIOFileLikeContext.h"
#include <torch/types.h>

namespace facebook::torchcodec {

AVIOFileLikeContext::AVIOFileLikeContext(py::object fileLike)
    : fileLike_{UniquePyObject(new py::object(fileLike))} {
  {
    // TODO: Is it necessary to acquire the GIL here? Is it maybe even
    // harmful? At the moment, this is only called from within a pybind
    // function, and pybind guarantees we have the GIL.
    py::gil_scoped_acquire gil;
    TORCH_CHECK(
        py::hasattr(fileLike, "read"),
        "File like object must implement a read method.");
    TORCH_CHECK(
        py::hasattr(fileLike, "seek"),
        "File like object must implement a seek method.");
  }
  createAVIOContext(&read, &seek, &fileLike_);
}

int AVIOFileLikeContext::read(void* opaque, uint8_t* buf, int buf_size) {
  auto fileLike = static_cast<UniquePyObject*>(opaque);

  // Note that we acquire the GIL outside of the loop. This is likely more
  // efficient than releasing and acquiring it each loop iteration.
  py::gil_scoped_acquire gil;
  int num_read = 0;
  while (num_read < buf_size) {
    int request = buf_size - num_read;
    auto chunk = static_cast<std::string>(
        static_cast<py::bytes>((*fileLike)->attr("read")(request)));
    int chunk_len = static_cast<int>(chunk.length());
    if (chunk_len == 0) {
      break;
    }
    TORCH_CHECK(
        chunk_len <= request,
        "Requested up to ",
        request,
        " bytes but, received ",
        chunk_len,
        " bytes. The given object does not conform to read protocol of file object.");
    memcpy(buf, chunk.data(), chunk_len);
    buf += chunk_len;
    num_read += chunk_len;
  }
  return num_read == 0 ? AVERROR_EOF : num_read;
}

int64_t AVIOFileLikeContext::seek(void* opaque, int64_t offset, int whence) {
  // We do not know the file size.
  if (whence == AVSEEK_SIZE) {
    return AVERROR(EIO);
  }
  auto fileLike = static_cast<UniquePyObject*>(opaque);
  py::gil_scoped_acquire gil;
  return py::cast<int64_t>((*fileLike)->attr("seek")(offset, whence));
}

} // namespace facebook::torchcodec
