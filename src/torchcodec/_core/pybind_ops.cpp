// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>

#include "src/torchcodec/_core/AVIOFileLikeContext.h"

namespace py = pybind11;

namespace facebook::torchcodec {

// Note: It's not immediately obvous why we need both custom_ops.cpp and
//       pybind_ops.cpp. We do all other Python to C++ bridging in
//       custom_ops.cpp, and that even depends on pybind11, so why have an
//       explicit pybind-only file?
//
//       The reason is that we want to accept OWNERSHIP of a file-like object
//       from the Python side. In order to do that, we need a proper
//       py::object. For raw bytes, we can launder that through a tensor on the
//       custom_ops.cpp side, but we can't launder a proper Python object
//       through a tensor. Custom ops can't accept a proper Python object
//       through py::object, so we have to do direct pybind11 here.
//
// TODO: Investigate if we can do something better here. See:
//         https://github.com/pytorch/torchcodec/issues/896
//       Short version is that we're laundering a pointer through an int, the
//       Python side forwards that to decoder creation functions in
//       custom_ops.cpp and we do another cast on that side to get a pointer
//       again. We want to investigate if we can do something cleaner by
//       defining proper pybind objects.
int64_t create_file_like_context(py::object file_like, bool is_for_writing) {
  AVIOFileLikeContext* context =
      new AVIOFileLikeContext(file_like, is_for_writing);
  return reinterpret_cast<int64_t>(context);
}

#ifndef PYBIND_OPS_MODULE_NAME
#error PYBIND_OPS_MODULE_NAME must be defined!
#endif

PYBIND11_MODULE(PYBIND_OPS_MODULE_NAME, m) {
  m.def("create_file_like_context", &create_file_like_context);
}

} // namespace facebook::torchcodec
