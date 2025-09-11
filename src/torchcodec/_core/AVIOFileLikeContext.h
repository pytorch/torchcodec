// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "src/torchcodec/_core/AVIOContextHolder.h"

namespace py = pybind11;

namespace facebook::torchcodec {

// Enables uers to pass in a Python file-like object. We then forward all read
// and seek calls back up to the methods on the Python object.
class AVIOFileLikeContext : public AVIOContextHolder {
 public:
  explicit AVIOFileLikeContext(const py::object& fileLike, bool isForWriting);

 private:
  static int read(void* opaque, uint8_t* buf, int buf_size);
  static int64_t seek(void* opaque, int64_t offset, int whence);
  static int write(void* opaque, const uint8_t* buf, int buf_size);

  // Note that we dynamically allocate the Python object because we need to
  // strictly control when its destructor is called. We must hold the GIL
  // when its destructor gets called, as it needs to update the reference
  // count. It's easiest to control that when it's dynamic memory. Otherwise,
  // we'd have to ensure whatever enclosing scope holds the object has the GIL,
  // and that's, at least, hard. For all of the common pitfalls, see:
  //
  //   https://pybind11.readthedocs.io/en/stable/advanced/misc.html#common-sources-of-global-interpreter-lock-errors
  //
  // We maintain a reference to the file-like object because the file-like
  // object that was created on the Python side must live as long as our
  // potential use. That is, even if there are no more references to the object
  // on the Python side, we require that the object is still live.
  struct PyObjectDeleter {
    inline void operator()(py::object* obj) const {
      if (obj) {
        py::gil_scoped_acquire gil;
        delete obj;
      }
    }
  };

  using UniquePyObject = std::unique_ptr<py::object, PyObjectDeleter>;
  UniquePyObject fileLike_;
};

} // namespace facebook::torchcodec
