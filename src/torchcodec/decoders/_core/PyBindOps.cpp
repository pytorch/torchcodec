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

class AVIOFileLikeContext : public AVIOContextHolder {
 public:
  AVIOFileLikeContext(py::object fileLike, int bufferSize)
      : fileLikeContext_{
            std::unique_ptr<py::object, PyObjectDeleter>(
                new py::object(fileLike)),
            bufferSize} {
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

    auto buffer = static_cast<uint8_t*>(av_malloc(bufferSize));
    TORCH_CHECK(
        buffer != nullptr,
        "Failed to allocate buffer of size " + std::to_string(bufferSize));

    avioContext_.reset(avio_alloc_context(
        buffer,
        bufferSize,
        0,
        &fileLikeContext_,
        &AVIOFileLikeContext::read,
        nullptr,
        &AVIOFileLikeContext::seek));

    if (!avioContext_) {
      av_freep(&buffer);
      TORCH_CHECK(false, "Failed to allocate AVIOContext");
    }
  }

  virtual ~AVIOFileLikeContext() {
    if (avioContext_) {
      av_freep(&avioContext_->buffer);
    }
  }

  virtual AVIOContext* getAVIOContext() const override {
    return avioContext_.get();
  }

  static int read(void* opaque, uint8_t* buf, int buf_size) {
    auto fileLikeContext = static_cast<FileLikeContext*>(opaque);
    buf_size = FFMIN(buf_size, fileLikeContext->bufferSize);

    int num_read = 0;
    while (num_read < buf_size) {
      int request = buf_size - num_read;
      py::gil_scoped_acquire gil;
      auto chunk = static_cast<std::string>(static_cast<py::bytes>(
          fileLikeContext->fileLike->attr("read")(request)));
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
    auto fileLikeContext = static_cast<FileLikeContext*>(opaque);
    py::gil_scoped_acquire gil;
    return py::cast<int64_t>(
        fileLikeContext->fileLike->attr("seek")(offset, whence));
  }

 private:
  struct FileLikeContext {
    // Note that we keep a pointer to the Python object because we need to
    // strictly control when its destructor is called. We must hold the GIL
    // when its destructor gets called, as it needs to update the reference
    // count. It's easiest to control that when it's a pointer. Otherwise, we'd
    // have to ensure whatever enclosing scope holds the object has the GIL,
    // and that's, at least, hard. For all of the common pitfalls, see:
    //
    //   https://pybind11.readthedocs.io/en/stable/advanced/misc.html#common-sources-of-global-interpreter-lock-errors
    std::unique_ptr<py::object, PyObjectDeleter> fileLike;
    int bufferSize;
  };

  UniqueAVIOContext avioContext_;
  FileLikeContext fileLikeContext_;
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

  constexpr int bufferSize = 64 * 1024;
  auto contextHolder =
      std::make_unique<AVIOFileLikeContext>(file_like, bufferSize);

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
