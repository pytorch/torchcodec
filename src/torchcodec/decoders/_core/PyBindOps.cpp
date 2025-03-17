// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/stl.h>
#include <cstdint>
#include <string>

#include "src/torchcodec/decoders/_core/VideoDecoder.h"

namespace py = pybind11;

namespace facebook::torchcodec {

namespace {

// Necessary to make sure that we hold the GIL when we delete a py::object.
struct PyObjectDeleter {
  inline void operator()(py::object* obj) const {
    if (obj) {
      py::gil_scoped_acquire gil;
      delete obj;
    }
  }
};

using UniquePyObject = std::unique_ptr<py::object, PyObjectDeleter>;

class AVIOFileLikeContext : public AVIOContextHolder {
 public:
  explicit AVIOFileLikeContext(py::object fileLike)
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

  static int read(void* opaque, uint8_t* buf, int buf_size) {
    auto fileLike = static_cast<UniquePyObject*>(opaque);

    int num_read = 0;
    while (num_read < buf_size) {
      int request = buf_size - num_read;
      // TODO: It is maybe more efficient to grab the lock once in the
      // surrounding scope?
      py::gil_scoped_acquire gil;
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
          " bytes. The given object does not confirm to read protocol of file object.");
      memcpy(buf, chunk.data(), chunk_len);
      buf += chunk_len;
      num_read += chunk_len;
    }
    return num_read == 0 ? AVERROR_EOF : num_read;
  }

  static int64_t seek(void* opaque, int64_t offset, int whence) {
    // We do not know the file size.
    if (whence == AVSEEK_SIZE) {
      return AVERROR(EIO);
    }
    auto fileLike = static_cast<UniquePyObject*>(opaque);
    py::gil_scoped_acquire gil;
    return py::cast<int64_t>((*fileLike)->attr("seek")(offset, whence));
  }

 private:
  // Note that we dynamically allocate the Python object because we need to
  // strictly control when its destructor is called. We must hold the GIL
  // when its destructor gets called, as it needs to update the reference
  // count. It's easiest to control that when it's dynamic memory. Otherwise,
  // we'd have to ensure whatever enclosing scope holds the object has the GIL,
  // and that's, at least, hard. For all of the common pitfalls, see:
  //
  //   https://pybind11.readthedocs.io/en/stable/advanced/misc.html#common-sources-of-global-interpreter-lock-errors
  UniquePyObject fileLike_;
};

} // namespace

// In principle, this should be able to return a tensor. But when we try that,
// we run into the bug reported here:
//
//   https://github.com/pytorch/pytorch/issues/136664
//
// So we instead launder the pointer through an int, and then use a conversion
// function on the custom ops side to launder that int into a tensor.
int64_t create_from_file_like(
    py::object file_like,
    std::optional<std::string_view> seek_mode) {
  VideoDecoder::SeekMode realSeek = VideoDecoder::SeekMode::exact;
  if (seek_mode.has_value()) {
    realSeek = seekModeFromString(seek_mode.value());
  }

  auto contextHolder = std::make_unique<AVIOFileLikeContext>(file_like);

  VideoDecoder* decoder = new VideoDecoder(std::move(contextHolder), realSeek);
  return reinterpret_cast<int64_t>(decoder);
}

#ifndef TORCHCODEC_PYBIND
#error TORCHCODEC_PYBIND must be defined.
#endif

PYBIND11_MODULE(TORCHCODEC_PYBIND, m) {
  m.def("create_from_file_like", &create_from_file_like);
}

} // namespace facebook::torchcodec
