// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/AVIOFileLikeContext.h"
#include <torch/types.h>

namespace facebook::torchcodec {

AVIOFileLikeContext::AVIOFileLikeContext(
    const py::object& fileLike,
    bool isForWriting)
    : fileLike_{UniquePyObject(new py::object(fileLike))} {
  {
    // TODO: Is it necessary to acquire the GIL here? Is it maybe even
    // harmful? At the moment, this is only called from within a pybind
    // function, and pybind guarantees we have the GIL.
    py::gil_scoped_acquire gil;

    if (isForWriting) {
      TORCH_CHECK(
          py::hasattr(fileLike, "write"),
          "File like object must implement a write method for writing.");
    } else {
      TORCH_CHECK(
          py::hasattr(fileLike, "read"),
          "File like object must implement a read method for reading.");
    }

    TORCH_CHECK(
        py::hasattr(fileLike, "seek"),
        "File like object must implement a seek method.");
  }
  createAVIOContext(&read, &write, &seek, &fileLike_, isForWriting);
}

int AVIOFileLikeContext::read(void* opaque, uint8_t* buf, int buf_size) {
  auto fileLike = static_cast<UniquePyObject*>(opaque);

  // Note that we acquire the GIL outside of the loop. This is likely more
  // efficient than releasing and acquiring it each loop iteration.
  py::gil_scoped_acquire gil;

  int totalNumRead = 0;
  while (totalNumRead < buf_size) {
    int request = buf_size - totalNumRead;

    // The Python method returns the actual bytes, which we access through the
    // py::bytes wrapper. That wrapper, however, does not provide us access to
    // the underlying data pointer, which we need for the memcpy below. So we
    // convert the bytes to a string_view to get access to the data pointer.
    // Becauase it's a view and not a copy, it should be cheap.
    auto bytesRead = static_cast<py::bytes>((*fileLike)->attr("read")(request));
    auto bytesView = static_cast<std::string_view>(bytesRead);

    int numBytesRead = static_cast<int>(bytesView.size());
    if (numBytesRead == 0) {
      break;
    }

    TORCH_CHECK(
        numBytesRead <= request,
        "Requested up to ",
        request,
        " bytes but, received ",
        numBytesRead,
        " bytes. The given object does not conform to read protocol of file object.");

    std::memcpy(buf, bytesView.data(), numBytesRead);
    buf += numBytesRead;
    totalNumRead += numBytesRead;
  }

  return totalNumRead == 0 ? AVERROR_EOF : totalNumRead;
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

int AVIOFileLikeContext::write(void* opaque, const uint8_t* buf, int buf_size) {
  auto fileLike = static_cast<UniquePyObject*>(opaque);
  py::gil_scoped_acquire gil;
  py::bytes bytes_obj(reinterpret_cast<const char*>(buf), buf_size);

  return py::cast<int>((*fileLike)->attr("write")(bytes_obj));
}

} // namespace facebook::torchcodec
