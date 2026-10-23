// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>

#include "IOInterface.h"

namespace facebook::torchcodec {

// C callbacks into a Python file-like object. Built on the Python side with
// ctypes (see torchcodec/_core/_file_like.py, which mirrors this layout), so
// that no torchcodec library depends on the CPython C API. ctypes callbacks
// attach the calling thread to the interpreter and take the GIL themselves, so
// they're safe to call from ops that run with the GIL released.
//
// read/write/seek return kFileLikeCallbackError if the Python call raised;
// get_error then writes the Python error message into buf (truncated to
// buf_len) and returns the number of bytes written. release() drops the Python
// side's reference to the file-like object and must be called exactly once.
struct FileLikeCallbacks {
  int64_t handle;
  int64_t (*read)(int64_t handle, uint8_t* buf, int64_t size);
  int64_t (*write)(int64_t handle, const uint8_t* buf, int64_t size);
  int64_t (*seek)(int64_t handle, int64_t offset, int64_t whence);
  void (*release)(int64_t handle);
  int64_t (*get_error)(int64_t handle, char* buf, int64_t buf_len);
};

inline constexpr int64_t kFileLikeCallbackError = INT64_MIN;

// FFmpeg-free bridge to a Python file-like object. Wrap it in an
// AVIOContextHolder to feed FFmpeg, or use it directly (e.g. the image
// encoders write straight through it).
class FileLikeIO : public IOInterface {
 public:
  explicit FileLikeIO(const FileLikeCallbacks& callbacks);
  ~FileLikeIO() override;

  FileLikeIO(const FileLikeIO&) = delete;
  FileLikeIO& operator=(const FileLikeIO&) = delete;

  int read(uint8_t* buf, int size) override;
  int write(const uint8_t* buf, int size) override;
  int64_t seek(int64_t offset, int whence) override;
  int64_t get_size() override;

 private:
  void check_callback_result(int64_t result, const char* method);

  FileLikeCallbacks callbacks_;
};

// file_like_context is the address of a FileLikeCallbacks struct, as returned
// by create_file_like_context() on the Python side. Call this before doing
// anything else in an op, so that release() is guaranteed to run even if the
// op fails.
std::unique_ptr<IOInterface> adopt_file_like_context(int64_t file_like_context);

} // namespace facebook::torchcodec
