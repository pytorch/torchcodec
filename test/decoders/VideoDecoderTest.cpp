// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/decoders/_core/VideoDecoder.h"

#include <c10/util/Flags.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <filesystem>
#include <fstream>
#include <iostream>

#ifdef FBCODE_BUILD
#include "tools/cxx/Resources.h"
#endif

using namespace ::testing;

C10_DEFINE_bool(
    dump_frames_for_debugging,
    false,
    "If true, we dump frames as bmp files for debugging.");

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

class VideoDecoderTest : public testing::TestWithParam<bool> {
 protected:
  std::unique_ptr<VideoDecoder> createDecoderFromPath(
      const std::string& filepath,
      bool useMemoryBuffer) {
    if (useMemoryBuffer) {
      std::ostringstream outputStringStream;
      std::ifstream input(filepath, std::ios::binary);
      outputStringStream << input.rdbuf();
      content_ = outputStringStream.str();
      void* buffer = content_.data();
      size_t length = outputStringStream.str().length();
      return VideoDecoder::createFromBuffer(
          buffer, length, VideoDecoder::SeekMode::approximate);
    } else {
      return VideoDecoder::createFromFilePath(
          filepath, VideoDecoder::SeekMode::approximate);
    }
  }
  std::string content_;
};

TEST_P(VideoDecoderTest, ReturnsFpsAndDurationForVideoInMetadata) {
  std::string path = getResourcePath("nasa_13013.mp4");
  std::unique_ptr<VideoDecoder> decoder =
      createDecoderFromPath(path, GetParam());
  VideoDecoder::ContainerMetadata metadata = decoder->getContainerMetadata();
  EXPECT_EQ(metadata.numAudioStreams, 2);
  EXPECT_EQ(metadata.numVideoStreams, 2);
#if LIBAVFORMAT_VERSION_MAJOR >= 60
  EXPECT_NEAR(metadata.bitRate.value(), 412365, 1e-1);
#else
  EXPECT_NEAR(metadata.bitRate.value(), 324915, 1e-1);
#endif
  EXPECT_EQ(metadata.allStreamMetadata.size(), 6);
  const auto& videoStream = metadata.allStreamMetadata[3];
  EXPECT_EQ(videoStream.mediaType, AVMEDIA_TYPE_VIDEO);
  EXPECT_EQ(videoStream.codecName, "h264");
  EXPECT_NEAR(*videoStream.averageFps, 29.97f, 1e-1);
  EXPECT_NEAR(*videoStream.bitRate, 128783, 1e-1);
  EXPECT_NEAR(*videoStream.durationSeconds, 13.013, 1e-1);
  EXPECT_EQ(videoStream.numFrames, 390);
  EXPECT_FALSE(videoStream.minPtsSecondsFromScan.has_value());
  EXPECT_FALSE(videoStream.maxPtsSecondsFromScan.has_value());
  EXPECT_FALSE(videoStream.numFramesFromScan.has_value());
  decoder->scanFileAndUpdateMetadataAndIndex();
  metadata = decoder->getContainerMetadata();
  const auto& videoStream1 = metadata.allStreamMetadata[3];
  EXPECT_EQ(*videoStream1.minPtsSecondsFromScan, 0);
  EXPECT_EQ(*videoStream1.maxPtsSecondsFromScan, 13.013);
  EXPECT_EQ(*videoStream1.numFramesFromScan, 390);
}

TEST(VideoDecoderTest, MissingVideoFileThrowsException) {
  EXPECT_THROW(
      VideoDecoder::createFromFilePath("/this/file/does/not/exist"),
      std::invalid_argument);
}

void dumpTensorToDisk(
    const torch::Tensor& tensor,
    const std::string& filename) {
  std::vector<char> bytes = torch::pickle_save(tensor);
  std::ofstream fout(filename, std::ios::out | std::ios::binary);
  fout.write(bytes.data(), bytes.size());
  fout.close();
}

torch::Tensor readTensorFromDisk(const std::string& filename) {
  std::string filepath = getResourcePath(filename);
  std::ifstream file(filepath, std::ios::binary);
  std::vector<char> data(
      (std::istreambuf_iterator<char>(file)),
      (std::istreambuf_iterator<char>()));
  VLOG(3) << "Read tensor from disk: " << filepath << ": " << data.size()
          << std::endl;
  return torch::pickle_load(data).toTensor().permute({2, 0, 1});
}

torch::Tensor floatAndNormalizeFrame(const torch::Tensor& frame) {
  torch::Tensor floatFrame = frame.toType(torch::kFloat32);
  torch::Tensor normalizedFrame = floatFrame / 255.0;
  return normalizedFrame;
}

double computeAverageCosineSimilarity(
    const torch::Tensor& frame1,
    const torch::Tensor& frame2) {
  torch::Tensor frame1Norm = floatAndNormalizeFrame(frame1);
  torch::Tensor frame2Norm = floatAndNormalizeFrame(frame2);
  torch::Tensor cosineSimilarities =
      torch::cosine_similarity(frame1Norm, frame2Norm);
  double averageCosineSimilarity = cosineSimilarities.mean().item<float>();
  return averageCosineSimilarity;
}

// TEST(DecoderOptionsTest, ConvertsFromStringToOptions) {
//   std::string optionsString =
//       "ffmpeg_thread_count=3,dimension_order=NCHW,width=100,height=120";
//   VideoDecoder::DecoderOptions options =
//       VideoDecoder::DecoderOptions(optionsString);
//   EXPECT_EQ(options.ffmpegThreadCount, 3);
// }

TEST(VideoDecoderTest, RespectsWidthAndHeightFromOptions) {
  std::string path = getResourcePath("nasa_13013.mp4");
  std::unique_ptr<VideoDecoder> decoder =
      VideoDecoder::createFromFilePath(path);
  VideoDecoder::VideoStreamOptions videoStreamOptions;
  videoStreamOptions.width = 100;
  videoStreamOptions.height = 120;
  decoder->addVideoStreamDecoder(-1, videoStreamOptions);
  torch::Tensor tensor = decoder->getNextFrameNoDemux().frame;
  EXPECT_EQ(tensor.sizes(), std::vector<long>({3, 120, 100}));
}

TEST(VideoDecoderTest, RespectsOutputTensorDimensionOrderFromOptions) {
  std::string path = getResourcePath("nasa_13013.mp4");
  std::unique_ptr<VideoDecoder> decoder =
      VideoDecoder::createFromFilePath(path);
  VideoDecoder::VideoStreamOptions videoStreamOptions;
  videoStreamOptions.dimensionOrder = "NHWC";
  decoder->addVideoStreamDecoder(-1, videoStreamOptions);
  torch::Tensor tensor = decoder->getNextFrameNoDemux().frame;
  EXPECT_EQ(tensor.sizes(), std::vector<long>({270, 480, 3}));
}

TEST_P(VideoDecoderTest, ReturnsFirstTwoFramesOfVideo) {
  std::string path = getResourcePath("nasa_13013.mp4");
  std::unique_ptr<VideoDecoder> ourDecoder =
      createDecoderFromPath(path, GetParam());
  ourDecoder->addVideoStreamDecoder(-1);
  auto output = ourDecoder->getNextFrameNoDemux();
  torch::Tensor tensor0FromOurDecoder = output.data;
  EXPECT_EQ(tensor0FromOurDecoder.sizes(), std::vector<long>({3, 270, 480}));
  EXPECT_EQ(output.ptsSeconds, 0.0);
  output = ourDecoder->getNextFrameNoDemux();
  torch::Tensor tensor1FromOurDecoder = output.data;
  EXPECT_EQ(tensor1FromOurDecoder.sizes(), std::vector<long>({3, 270, 480}));
  EXPECT_EQ(output.ptsSeconds, 1'001. / 30'000);

  torch::Tensor tensor0FromFFMPEG =
      readTensorFromDisk("nasa_13013.mp4.stream3.frame000000.pt");
  torch::Tensor tensor1FromFFMPEG =
      readTensorFromDisk("nasa_13013.mp4.stream3.frame000001.pt");

  EXPECT_EQ(tensor1FromFFMPEG.sizes(), std::vector<long>({3, 270, 480}));
  EXPECT_TRUE(torch::equal(tensor0FromOurDecoder, tensor0FromFFMPEG));
  EXPECT_TRUE(torch::equal(tensor1FromOurDecoder, tensor1FromFFMPEG));
  EXPECT_TRUE(
      torch::allclose(tensor0FromOurDecoder, tensor0FromFFMPEG, 0.1, 20));
  EXPECT_EQ(tensor1FromFFMPEG.sizes(), std::vector<long>({3, 270, 480}));
  EXPECT_TRUE(
      torch::allclose(tensor1FromOurDecoder, tensor1FromFFMPEG, 0.1, 20));

  if (FLAGS_dump_frames_for_debugging) {
    dumpTensorToDisk(tensor0FromFFMPEG, "tensor0FromFFMPEG.pt");
    dumpTensorToDisk(tensor1FromFFMPEG, "tensor1FromFFMPEG.pt");
    dumpTensorToDisk(tensor0FromOurDecoder, "tensor0FromOurDecoder.pt");
    dumpTensorToDisk(tensor1FromOurDecoder, "tensor1FromOurDecoder.pt");
  }
}

TEST_P(VideoDecoderTest, DecodesFramesInABatchInNCHW) {
  std::string path = getResourcePath("nasa_13013.mp4");
  std::unique_ptr<VideoDecoder> ourDecoder =
      createDecoderFromPath(path, GetParam());
  ourDecoder->scanFileAndUpdateMetadataAndIndex();
  int bestVideoStreamIndex =
      *ourDecoder->getContainerMetadata().bestVideoStreamIndex;
  ourDecoder->addVideoStreamDecoder(bestVideoStreamIndex);
  // Frame with index 180 corresponds to timestamp 6.006.
  auto output = ourDecoder->getFramesAtIndices(bestVideoStreamIndex, {0, 180});
  auto tensor = output.data;
  EXPECT_EQ(tensor.sizes(), std::vector<long>({2, 3, 270, 480}));

  torch::Tensor tensor0FromFFMPEG =
      readTensorFromDisk("nasa_13013.mp4.stream3.frame000000.pt");
  torch::Tensor tensorTime6FromFFMPEG =
      readTensorFromDisk("nasa_13013.mp4.time6.000000.pt");

  EXPECT_TRUE(torch::equal(tensor[0], tensor0FromFFMPEG));
  EXPECT_TRUE(torch::equal(tensor[1], tensorTime6FromFFMPEG));
}

TEST_P(VideoDecoderTest, DecodesFramesInABatchInNHWC) {
  std::string path = getResourcePath("nasa_13013.mp4");
  std::unique_ptr<VideoDecoder> ourDecoder =
      createDecoderFromPath(path, GetParam());
  ourDecoder->scanFileAndUpdateMetadataAndIndex();
  int bestVideoStreamIndex =
      *ourDecoder->getContainerMetadata().bestVideoStreamIndex;
  ourDecoder->addVideoStreamDecoder(
      bestVideoStreamIndex,
      VideoDecoder::VideoStreamOptions("dimension_order=NHWC"));
  // Frame with index 180 corresponds to timestamp 6.006.
  auto output = ourDecoder->getFramesAtIndices(bestVideoStreamIndex, {0, 180});
  auto tensor = output.;
  EXPECT_EQ(tensor.sizes(), std::vector<long>({2, 270, 480, 3}));

  torch::Tensor tensor0FromFFMPEG =
      readTensorFromDisk("nasa_13013.mp4.stream3.frame000000.pt");
  torch::Tensor tensorTime6FromFFMPEG =
      readTensorFromDisk("nasa_13013.mp4.time6.000000.pt");

  tensor = tensor.permute({0, 3, 1, 2});
  EXPECT_TRUE(torch::equal(tensor[0], tensor0FromFFMPEG));
  EXPECT_TRUE(torch::equal(tensor[1], tensorTime6FromFFMPEG));
}

TEST_P(VideoDecoderTest, SeeksCloseToEof) {
  std::string path = getResourcePath("nasa_13013.mp4");
  std::unique_ptr<VideoDecoder> ourDecoder =
      createDecoderFromPath(path, GetParam());
  ourDecoder->addVideoStreamDecoder(-1);
  ourDecoder->setCursorPtsInSeconds(388388. / 30'000);
  auto output = ourDecoder->getNextFrameNoDemux();
  EXPECT_EQ(output.ptsSeconds, 388'388. / 30'000);
  output = ourDecoder->getNextFrameNoDemux();
  EXPECT_EQ(output.ptsSeconds, 389'389. / 30'000);
  EXPECT_THROW(ourDecoder->getNextFrameNoDemux(), std::exception);
}

TEST_P(VideoDecoderTest, GetsFramePlayedAtTimestamp) {
  std::string path = getResourcePath("nasa_13013.mp4");
  std::unique_ptr<VideoDecoder> ourDecoder =
      createDecoderFromPath(path, GetParam());
  ourDecoder->addVideoStreamDecoder(-1);
  auto output = ourDecoder->getFramePlayedAtTimestampNoDemux(6.006);
  EXPECT_EQ(output.ptsSeconds, 6.006);
  // The frame's duration is 0.033367 according to ffprobe,
  // so the next frame is played at timestamp=6.039367.
  const double kNextFramePts = 6.039366666666667;
  // The frame that is played a microsecond before the next frame is still
  // the previous frame.
  output = ourDecoder->getFramePlayedAtTimestampNoDemux(kNextFramePts - 1e-6);
  EXPECT_EQ(output.ptsSeconds, 6.006);
  // The frame that is played at the exact pts of the frame is the next
  // frame.
  output = ourDecoder->getFramePlayedAtTimestampNoDemux(kNextFramePts);
  EXPECT_EQ(output.ptsSeconds, kNextFramePts);

  // This is the timestamp of the last frame in this video.
  constexpr double kPtsOfLastFrameInVideoStream = 389'389. / 30'000;
  constexpr double kDurationOfLastFrameInVideoStream = 1'001. / 30'000;
  constexpr double kPtsPlusDurationOfLastFrame =
      kPtsOfLastFrameInVideoStream + kDurationOfLastFrameInVideoStream;
  // Sanity check: make sure duration is strictly positive.
  EXPECT_GT(kPtsPlusDurationOfLastFrame, kPtsOfLastFrameInVideoStream);
  output = ourDecoder->getFramePlayedAtTimestampNoDemux(
      kPtsPlusDurationOfLastFrame - 1e-6);
  EXPECT_EQ(output.ptsSeconds, kPtsOfLastFrameInVideoStream);
}

TEST_P(VideoDecoderTest, SeeksToFrameWithSpecificPts) {
  std::string path = getResourcePath("nasa_13013.mp4");
  std::unique_ptr<VideoDecoder> ourDecoder =
      createDecoderFromPath(path, GetParam());
  ourDecoder->addVideoStreamDecoder(-1);
  ourDecoder->setCursorPtsInSeconds(6.0);
  auto output = ourDecoder->getNextFrameNoDemux();
  torch::Tensor tensor6FromOurDecoder = output.data;
  EXPECT_EQ(output.ptsSeconds, 180'180. / 30'000);
  torch::Tensor tensor6FromFFMPEG =
      readTensorFromDisk("nasa_13013.mp4.time6.000000.pt");
  EXPECT_TRUE(torch::equal(tensor6FromOurDecoder, tensor6FromFFMPEG));
  EXPECT_EQ(ourDecoder->getDecodeStats().numSeeksAttempted, 1);
  // We skipped the seek since timestamp=6 and timestamp=0 share the same
  // keyframe.
  EXPECT_EQ(ourDecoder->getDecodeStats().numSeeksSkipped, 1);
  // There are about 180 packets/frames between timestamp=0 and timestamp=6 at
  // ~30 fps.
  EXPECT_GT(ourDecoder->getDecodeStats().numPacketsRead, 180);
  EXPECT_GT(ourDecoder->getDecodeStats().numPacketsSentToDecoder, 180);

  ourDecoder->setCursorPtsInSeconds(6.1);
  output = ourDecoder->getNextFrameNoDemux();
  torch::Tensor tensor61FromOurDecoder = output.data;
  EXPECT_EQ(output.ptsSeconds, 183'183. / 30'000);
  torch::Tensor tensor61FromFFMPEG =
      readTensorFromDisk("nasa_13013.mp4.time6.100000.pt");
  EXPECT_TRUE(torch::equal(tensor61FromOurDecoder, tensor61FromFFMPEG));
  EXPECT_EQ(ourDecoder->getDecodeStats().numSeeksAttempted, 1);
  // We skipped the seek since timestamp=6 and timestamp=6.1 share the same
  // keyframe.
  EXPECT_EQ(ourDecoder->getDecodeStats().numSeeksSkipped, 1);
  // If we had seeked, we would have gone to frame=0 (because that was the key
  // frame before timestamp=6.1). Because we skipped that seek the number of
  // packets we send to the decoder is minimal. This is partly why torchvision
  // is slower than decord. In fact we are more efficient than decord because we
  // rely on FFMPEG's key frame index instead of reading the entire file
  // ourselves. ^_^
  EXPECT_LT(ourDecoder->getDecodeStats().numPacketsRead, 10);
  EXPECT_LT(ourDecoder->getDecodeStats().numPacketsSentToDecoder, 10);

  ourDecoder->setCursorPtsInSeconds(10.0);
  output = ourDecoder->getNextFrameNoDemux();
  torch::Tensor tensor10FromOurDecoder = output.data;
  EXPECT_EQ(output.ptsSeconds, 300'300. / 30'000);
  torch::Tensor tensor10FromFFMPEG =
      readTensorFromDisk("nasa_13013.mp4.time10.000000.pt");
  EXPECT_TRUE(torch::equal(tensor10FromOurDecoder, tensor10FromFFMPEG));
  EXPECT_EQ(ourDecoder->getDecodeStats().numSeeksAttempted, 1);
  // We cannot skip a seek here because timestamp=10 has a different keyframe
  // than timestamp=6.
  EXPECT_EQ(ourDecoder->getDecodeStats().numSeeksSkipped, 0);
  // The keyframe is at timestamp=8. So we seek to there and decode until
  // timestamp=10. There are about 60 packets/frames between timestamp=8 and
  // timestamp=10 at ~30 fps.
  EXPECT_GT(ourDecoder->getDecodeStats().numPacketsRead, 60);
  EXPECT_GT(ourDecoder->getDecodeStats().numPacketsSentToDecoder, 60);

  ourDecoder->setCursorPtsInSeconds(6.0);
  output = ourDecoder->getNextFrameNoDemux();
  tensor6FromOurDecoder = output.data;
  EXPECT_EQ(output.ptsSeconds, 180'180. / 30'000);
  EXPECT_TRUE(torch::equal(tensor6FromOurDecoder, tensor6FromFFMPEG));
  EXPECT_EQ(ourDecoder->getDecodeStats().numSeeksAttempted, 1);
  // We cannot skip a seek here because timestamp=6 has a different keyframe
  // than timestamp=10.
  EXPECT_EQ(ourDecoder->getDecodeStats().numSeeksSkipped, 0);
  // There are about 180 packets/frames between timestamp=0 and timestamp=6 at
  // ~30 fps.
  EXPECT_GT(ourDecoder->getDecodeStats().numPacketsRead, 180);
  EXPECT_GT(ourDecoder->getDecodeStats().numPacketsSentToDecoder, 180);

  constexpr double kPtsOfLastFrameInVideoStream = 389'389. / 30'000; // ~12.9
  ourDecoder->setCursorPtsInSeconds(kPtsOfLastFrameInVideoStream);
  output = ourDecoder->getNextFrameNoDemux();
  torch::Tensor tensor7FromOurDecoder = output.data;
  EXPECT_EQ(output.ptsSeconds, 389'389. / 30'000);
  torch::Tensor tensor7FromFFMPEG =
      readTensorFromDisk("nasa_13013.mp4.time12.979633.pt");
  EXPECT_TRUE(torch::equal(tensor7FromOurDecoder, tensor7FromFFMPEG));
  EXPECT_EQ(ourDecoder->getDecodeStats().numSeeksAttempted, 1);
  // We cannot skip a seek here because timestamp=6 has a different keyframe
  // than timestamp=12.9.
  EXPECT_EQ(ourDecoder->getDecodeStats().numSeeksSkipped, 0);
  // There are about 150 packets/frames between timestamp=8 and timestamp=12.9
  // at ~30 fps.
  EXPECT_GE(ourDecoder->getDecodeStats().numPacketsRead, 150);
  EXPECT_GE(ourDecoder->getDecodeStats().numPacketsSentToDecoder, 150);

  if (FLAGS_dump_frames_for_debugging) {
    dumpTensorToDisk(tensor7FromFFMPEG, "tensor7FromFFMPEG.pt");
    dumpTensorToDisk(tensor7FromOurDecoder, "tensor7FromOurDecoder.pt");
  }
}

TEST_P(VideoDecoderTest, PreAllocatedTensorFilterGraph) {
  std::string path = getResourcePath("nasa_13013.mp4");
  auto preAllocatedOutputTensor = torch::empty({270, 480, 3}, {torch::kUInt8});

  std::unique_ptr<VideoDecoder> ourDecoder =
      VideoDecoderTest::createDecoderFromPath(path, GetParam());
  ourDecoder->scanFileAndUpdateMetadataAndIndex();
  int bestVideoStreamIndex =
      *ourDecoder->getContainerMetadata().bestVideoStreamIndex;
  ourDecoder->addVideoStreamDecoder(
      bestVideoStreamIndex,
      VideoDecoder::VideoStreamOptions("color_conversion_library=filtergraph"));
  auto output = ourDecoder->getFrameAtIndexInternal(
      bestVideoStreamIndex, 0, preAllocatedOutputTensor);
  EXPECT_EQ(output.data.data_ptr(), preAllocatedOutputTensor.data_ptr());
}

TEST_P(VideoDecoderTest, PreAllocatedTensorSwscale) {
  std::string path = getResourcePath("nasa_13013.mp4");
  auto preAllocatedOutputTensor = torch::empty({270, 480, 3}, {torch::kUInt8});

  std::unique_ptr<VideoDecoder> ourDecoder =
      VideoDecoderTest::createDecoderFromPath(path, GetParam());
  ourDecoder->scanFileAndUpdateMetadataAndIndex();
  int bestVideoStreamIndex =
      *ourDecoder->getContainerMetadata().bestVideoStreamIndex;
  ourDecoder->addVideoStreamDecoder(
      bestVideoStreamIndex,
      VideoDecoder::VideoStreamOptions("color_conversion_library=swscale"));
  auto output = ourDecoder->getFrameAtIndexInternal(
      bestVideoStreamIndex, 0, preAllocatedOutputTensor);
  EXPECT_EQ(output.data.data_ptr(), preAllocatedOutputTensor.data_ptr());
}

TEST_P(VideoDecoderTest, GetAudioMetadata) {
  std::string path = getResourcePath("nasa_13013.mp4.audio.mp3");
  std::unique_ptr<VideoDecoder> decoder =
      createDecoderFromPath(path, GetParam());
  VideoDecoder::ContainerMetadata metadata = decoder->getContainerMetadata();
  EXPECT_EQ(metadata.numAudioStreams, 1);
  EXPECT_EQ(metadata.numVideoStreams, 0);
  EXPECT_EQ(metadata.allStreamMetadata.size(), 1);

  const auto& audioStream = metadata.allStreamMetadata[0];
  EXPECT_EQ(audioStream.mediaType, AVMEDIA_TYPE_AUDIO);
  EXPECT_NEAR(*audioStream.durationSeconds, 13.25, 1e-1);
}

INSTANTIATE_TEST_SUITE_P(FromFileAndMemory, VideoDecoderTest, testing::Bool());

} // namespace facebook::torchcodec
