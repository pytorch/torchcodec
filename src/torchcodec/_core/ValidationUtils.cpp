// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/ValidationUtils.h"
#include <limits>
#include "c10/util/Exception.h"

namespace facebook::torchcodec {

int validateInt64ToInt(int64_t value, const std::string& parameterName) {
  TORCH_CHECK(
      value >= std::numeric_limits<int>::min() &&
          value <= std::numeric_limits<int>::max(),
      parameterName,
      "=",
      value,
      " is out of range for int type.");

  return static_cast<int>(value);
}

std::optional<int> validateOptionalInt64ToInt(
    const std::optional<int64_t>& value,
    const std::string& parameterName) {
  if (value.has_value()) {
    return validateInt64ToInt(value.value(), parameterName);
  } else {
    return std::nullopt;
  }
}

} // namespace facebook::torchcodec
