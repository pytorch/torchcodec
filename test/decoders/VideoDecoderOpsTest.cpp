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
  auto result = get_next_frame(decoder);
  at::Tensor tensor1 = std::get<0>(result);
  EXPECT_EQ(tensor1.sizes(), std::vector<long>({270, 480, 3}));
  EXPECT_EQ(std::get<1>(result).item<double>(), 0);
  EXPECT_NEAR(std::get<2>(result).item<double>(), 0.033367, 1e-6);
}

} // namespace facebook::torchcodec
