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

// In principle, this should be able to return a tensor. But when we try that,
// we run into the bug reported here:
//
//   https://github.com/pytorch/pytorch/issues/136664
//
// So we instead launder the pointer through an int, and then use a conversion
// function on the custom ops side to launder that int into a tensor.
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
