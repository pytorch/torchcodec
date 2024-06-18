// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "src/torchcodec/decoders/_core/VideoDecoderOps.h"

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <iostream>

#ifdef FBCODE_BUILD
#include "tools/cxx/Resources.h"
#endif

using namespace ::testing;

namespace facebook::torchcodec {

std::string getResourcePath(const std::string& filename) {
#ifdef FBCODE_BUILD
  std::string filepath = "pytorch/torchcodec/test/resources/" + filename;
  filepath = build::getResourcePath(filepath).string();
#else
  std::filesystem::path dirPath = std::filesystem::path(__FILE__);
  std::string filepath =
      dirPath.parent_path().string() + "/../resources/" + filename;
#endif
  return filepath;
}

TEST(VideoDecoderOpsTest, TestCreateDecoderFromBuffer) {
  std::string filepath = getResourcePath("nasa_13013.mp4");
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
