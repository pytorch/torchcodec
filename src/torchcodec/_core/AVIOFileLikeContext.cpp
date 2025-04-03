// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/AVIOFileLikeContext.h"
#include <torch/types.h>

namespace facebook::torchcodec {

AVIOFileLikeContext::AVIOFileLikeContext(py::object file_like)
    : file_like_{UniquePyObject(new py::object(file_like))} {
  {
    // TODO: Is it necessary to acquire the GIL here? Is it maybe even
    // harmful? At the moment, this is only called from within a pybind
    // function, and pybind guarantees we have the GIL.
    py::gil_scoped_acquire gil;
    TORCH_CHECK(
        py::hasattr(file_like, "read"),
        "File like object must implement a read method.");
    TORCH_CHECK(
        py::hasattr(file_like, "seek"),
        "File like object must implement a seek method.");
  }
  create_avio_context(&read, &seek, &fileLike_);
}

int AVIOFileLikeContext::read(void* opaque, uint8_t* buf, int buf_size) {
  auto file_like = static_cast<UniquePyObject*>(opaque);

  // Note that we acquire the GIL outside of the loop. This is likely more
  // efficient than releasing and acquiring it each loop iteration.
  py::gil_scoped_acquire gil;

  int total_num_read = 0;
  while (totalNumRead < buf_size) {
    int request = buf_size - total_num_read;

    // The Python method returns the actual bytes, which we access through the
    // py::bytes wrapper. That wrapper, however, does not provide us access to
    // the underlying data pointer, which we need for the memcpy below. So we
    // convert the bytes to a string_view to get access to the data pointer.
    // Becauase it's a view and not a copy, it should be cheap.
    auto bytes_read =
        static_cast<py::bytes>((*file_like)->attr("read")(request));
    auto bytes_view = static_cast<std::string_view>(bytes_read);

    int num_bytes_read = static_cast<int>(bytes_view.size());
    if (numBytesRead == 0) {
      break;
    }

    TORCH_CHECK(
        num_bytes_read <= request,
        "Requested up to ",
        request,
        " bytes but, received ",
        num_bytes_read,
        " bytes. The given object does not conform to read protocol of file object.");

    std::memcpy(buf, bytes_view.data(), num_bytes_read);
    buf += num_bytes_read;
    total_num_read += num_bytes_read;
  }

  return total_num_read == 0 ? AVERROR_EOF : total_num_read;
}

int64_t AVIOFileLikeContext::seek(void* opaque, int64_t offset, int whence) {
  // We do not know the file size.
  if (whence == AVSEEK_SIZE) {
    return AVERROR(EIO);
  }

  auto file_like = static_cast<UniquePyObject*>(opaque);
  py::gil_scoped_acquire gil;
  return py::cast<int64_t>((*file_like)->attr("seek")(offset, whence));
}

} // namespace facebook::torchcodec
