// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "src/torchcodec/decoders/core/VideoDecoderOps.h"

#include <gtest/gtest.h>
#include <iostream>

#include "tools/cxx/Resources.h"

using namespace ::testing;

namespace facebook::torchcodec {
TEST(VideoDecoderOpsTest, TestCreateDecoderFromBuffer) {
  std::string filepath =
      build::getResourcePath(
          "pytorch/torchcodec/test/decoders/resources/nasa_13013.mp4")
          .string();
  std::ostringstream outputStringStream;
  std::ifstream input(filepath, std::ios::binary);
  outputStringStream << input.rdbuf();
  std::string content = outputStringStream.str();
  void* buffer = content.data();
  size_t length = outputStringStream.str().length();
  at::Tensor decoder = create_from_buffer(buffer, length);
  add_video_stream(decoder);
  at::Tensor tensor1 = get_next_frame(decoder);
  EXPECT_EQ(tensor1.sizes(), std::vector<long>({270, 480, 3}));
}

} // namespace facebook::torchcodec
