// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace facebook::torchcodec {

int validateInt64ToInt(int64_t value, const std::string& parameterName);

std::optional<int> validateOptionalInt64ToInt(
    const std::optional<int64_t>& value,
    const std::string& parameterName);

} // namespace facebook::torchcodec
