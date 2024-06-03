// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "src/torchcodec/decoders/_core/VideoDecoder.h"

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "tools/cxx/Resources.h"

using namespace ::testing;

DEFINE_bool(
    dump_frames_for_debugging,
    false,
    "If true, we dump frames as bmp files for debugging.");

namespace facebook::torchcodec {

std::string getResourcePath(const std::string& filename) {
  return build::getResourcePath(filename).string();
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
      return VideoDecoder::createFromBuffer(buffer, length);
    } else {
      return VideoDecoder::createFromFilePath(filepath);
    }
  }
  std::string content_;
};

TEST_P(VideoDecoderTest, ReturnsFpsAndDurationForVideoInMetadata) {
  std::string path = getResourcePath(
      "pytorch/torchcodec/test/decoders/resources/nasa_13013.mp4");
  std::unique_ptr<VideoDecoder> decoder =
      createDecoderFromPath(path, GetParam());
  VideoDecoder::ContainerMetadata metadata = decoder->getContainerMetadata();
  EXPECT_EQ(metadata.numAudioStreams, 2);
  EXPECT_EQ(metadata.numVideoStreams, 2);
  EXPECT_NEAR(metadata.bitRate.value(), 324915, 1e-1);
  EXPECT_EQ(metadata.streams.size(), 6);
  const auto& videoStream = metadata.streams[3];
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
  const auto& videoStream1 = metadata.streams[3];
  EXPECT_EQ(*videoStream1.minPtsSecondsFromScan, 0);
  EXPECT_EQ(*videoStream1.maxPtsSecondsFromScan, 13.013);
  EXPECT_EQ(*videoStream1.numFramesFromScan, 390);
}

TEST(VideoDecoderTest, MissingVideoFileThrowsException) {
  EXPECT_THROW(
      VideoDecoder::createFromFilePath("/this/file/does/not/exist"),
      std::invalid_argument);
}

void dumpTensorToBMP(const torch::Tensor& tensor, const std::string& filename) {
  if (tensor.dim() != 3 || tensor.scalar_type() != torch::kUInt8) {
    std::cerr
        << "Error: Input tensor must be 3-dimensional and of type uint8_t."
        << std::endl;
    return;
  }
  cv::Mat image(tensor.size(0), tensor.size(1), CV_8UC3, tensor.data_ptr());
  image = image.clone();
  // OpenCV saves images in BGR format, so convert from RGB to BGR
  cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
  cv::imwrite(filename, image);
}

torch::Tensor readTensorFromBMP(const std::string& filename) {
  cv::Mat image = cv::imread(filename);

  CHECK_NE(image.data, nullptr) << "could not open file: " << filename;
  // OpenCV reads images in BGR format, so convert from BGR to RGB
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  int height = image.rows;
  int width = image.cols;
  int channels = image.channels();
  auto options = torch::TensorOptions()
                     .dtype(torch::kUInt8)
                     .layout(torch::kStrided)
                     .device(torch::kCPU);
  torch::Tensor tensor = torch::empty({height, width, channels}, options);
  memcpy(tensor.data_ptr(), image.data, height * width * channels);
  std::cout << "Read tensor from " << filename << ": " << tensor.sizes()
            << std::endl;
  return tensor;
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
//       "ffmpeg_thread_count=3,shape=NCHW,width=100,height=120";
//   VideoDecoder::DecoderOptions options =
//       VideoDecoder::DecoderOptions(optionsString);
//   EXPECT_EQ(options.ffmpegThreadCount, 3);
// }

TEST(VideoDecoderTest, RespectsWidthAndHeightFromOptions) {
  std::string path = getResourcePath(
      "pytorch/torchcodec/test/decoders/resources/nasa_13013.mp4");
  std::unique_ptr<VideoDecoder> decoder =
      VideoDecoder::createFromFilePath(path);
  VideoDecoder::VideoStreamDecoderOptions streamOptions;
  streamOptions.width = 100;
  streamOptions.height = 120;
  decoder->addVideoStreamDecoder(-1, streamOptions);
  torch::Tensor tensor = decoder->getNextDecodedOutput().frame;
  EXPECT_EQ(tensor.sizes(), std::vector<long>({120, 100, 3}));
}

TEST(VideoDecoderTest, RespectsOutputTensorShapeFromOptions) {
  std::string path = getResourcePath(
      "pytorch/torchcodec/test/decoders/resources/nasa_13013.mp4");
  std::unique_ptr<VideoDecoder> decoder =
      VideoDecoder::createFromFilePath(path);
  VideoDecoder::VideoStreamDecoderOptions streamOptions;
  streamOptions.shape = "NCHW";
  decoder->addVideoStreamDecoder(-1, streamOptions);
  torch::Tensor tensor = decoder->getNextDecodedOutput().frame;
  EXPECT_EQ(tensor.sizes(), std::vector<long>({3, 270, 480}));
}

TEST_P(VideoDecoderTest, ReturnsFirstTwoFramesOfVideo) {
  std::string path = getResourcePath(
      "pytorch/torchcodec/test/decoders/resources/nasa_13013.mp4");
  std::unique_ptr<VideoDecoder> ourDecoder =
      createDecoderFromPath(path, GetParam());
  ourDecoder->addVideoStreamDecoder(-1);
  auto output = ourDecoder->getNextDecodedOutput();
  torch::Tensor tensor1FromOurDecoder = output.frame;
  EXPECT_EQ(tensor1FromOurDecoder.sizes(), std::vector<long>({270, 480, 3}));
  EXPECT_EQ(output.ptsSeconds, 0.0);
  EXPECT_EQ(output.pts, 0);
  output = ourDecoder->getNextDecodedOutput();
  torch::Tensor tensor2FromOurDecoder = output.frame;
  EXPECT_EQ(tensor2FromOurDecoder.sizes(), std::vector<long>({270, 480, 3}));
  EXPECT_EQ(output.ptsSeconds, 1'001. / 30'000);
  EXPECT_EQ(output.pts, 1001);

  torch::Tensor tensor1FromFFMPEG = readTensorFromBMP(getResourcePath(
      "pytorch/torchcodec/test/decoders/resources/nasa_13013.mp4.frame000001.bmp"));
  torch::Tensor tensor2FromFFMPEG = readTensorFromBMP(getResourcePath(
      "pytorch/torchcodec/test/decoders/resources/nasa_13013.mp4.frame000002.bmp"));

  EXPECT_EQ(tensor1FromFFMPEG.sizes(), std::vector<long>({270, 480, 3}));
  EXPECT_TRUE(torch::equal(tensor1FromOurDecoder, tensor1FromFFMPEG));
  EXPECT_TRUE(torch::equal(tensor2FromOurDecoder, tensor2FromFFMPEG));
  EXPECT_TRUE(
      torch::allclose(tensor1FromOurDecoder, tensor1FromFFMPEG, 0.1, 20));
  EXPECT_EQ(tensor2FromFFMPEG.sizes(), std::vector<long>({270, 480, 3}));
  EXPECT_TRUE(
      torch::allclose(tensor2FromOurDecoder, tensor2FromFFMPEG, 0.1, 20));

  if (FLAGS_dump_frames_for_debugging) {
    dumpTensorToBMP(
        tensor1FromFFMPEG,
        "pytorch/torchcodec/test/decoders/tensor1FromFFMPEG.bmp");
    dumpTensorToBMP(
        tensor2FromFFMPEG,
        "pytorch/torchcodec/test/decoders/tensor2FromFFMPEG.bmp");
    dumpTensorToBMP(
        tensor1FromOurDecoder,
        "pytorch/torchcodec/test/decoders/tensor1FromOurDecoder.bmp");
    dumpTensorToBMP(
        tensor2FromOurDecoder,
        "pytorch/torchcodec/test/decoders/tensor2FromOurDecoder.bmp");
  }
}

TEST_P(VideoDecoderTest, DecodesFramesInABatchInNHWC) {
  std::string path = getResourcePath(
      "pytorch/torchcodec/test/decoders/resources/nasa_13013.mp4");
  std::unique_ptr<VideoDecoder> ourDecoder =
      createDecoderFromPath(path, GetParam());
  ourDecoder->scanFileAndUpdateMetadataAndIndex();
  int bestVideoStreamIndex =
      *ourDecoder->getContainerMetadata().bestVideoStreamIndex;
  ourDecoder->addVideoStreamDecoder(bestVideoStreamIndex);
  // Frame with index 180 corresponds to timestamp 6.006.
  auto output = ourDecoder->getFramesAtIndexes(bestVideoStreamIndex, {0, 180});
  auto tensor = output.frames;
  EXPECT_EQ(tensor.sizes(), std::vector<long>({2, 270, 480, 3}));

  torch::Tensor tensor1FromFFMPEG = readTensorFromBMP(getResourcePath(
      "pytorch/torchcodec/test/decoders/resources/nasa_13013.mp4.frame000001.bmp"));
  torch::Tensor tensor2FromFFMPEG = readTensorFromBMP(getResourcePath(
      "pytorch/torchcodec/test/decoders/resources/nasa_13013.mp4.time6.000000.bmp"));

  EXPECT_TRUE(torch::equal(tensor[0], tensor1FromFFMPEG));
  EXPECT_TRUE(torch::equal(tensor[1], tensor2FromFFMPEG));
}

TEST_P(VideoDecoderTest, DecodesFramesInABatchInNCHW) {
  std::string path = getResourcePath(
      "pytorch/torchcodec/test/decoders/resources/nasa_13013.mp4");
  std::unique_ptr<VideoDecoder> ourDecoder =
      createDecoderFromPath(path, GetParam());
  ourDecoder->scanFileAndUpdateMetadataAndIndex();
  int bestVideoStreamIndex =
      *ourDecoder->getContainerMetadata().bestVideoStreamIndex;
  ourDecoder->addVideoStreamDecoder(
      bestVideoStreamIndex,
      VideoDecoder::VideoStreamDecoderOptions("shape=NCHW"));
  // Frame with index 180 corresponds to timestamp 6.006.
  auto output = ourDecoder->getFramesAtIndexes(bestVideoStreamIndex, {0, 180});
  auto tensor = output.frames;
  EXPECT_EQ(tensor.sizes(), std::vector<long>({2, 3, 270, 480}));

  torch::Tensor tensor1FromFFMPEG = readTensorFromBMP(getResourcePath(
      "pytorch/torchcodec/test/decoders/resources/nasa_13013.mp4.frame000001.bmp"));
  torch::Tensor tensor2FromFFMPEG = readTensorFromBMP(getResourcePath(
      "pytorch/torchcodec/test/decoders/resources/nasa_13013.mp4.time6.000000.bmp"));

  tensor = tensor.permute({0, 2, 3, 1});
  EXPECT_TRUE(torch::equal(tensor[0], tensor1FromFFMPEG));
  EXPECT_TRUE(torch::equal(tensor[1], tensor2FromFFMPEG));
}

TEST_P(VideoDecoderTest, SeeksCloseToEof) {
  std::string path = getResourcePath(
      "pytorch/torchcodec/test/decoders/resources/nasa_13013.mp4");
  std::unique_ptr<VideoDecoder> ourDecoder =
      createDecoderFromPath(path, GetParam());
  ourDecoder->addVideoStreamDecoder(-1);
  ourDecoder->setCursorPtsInSeconds(388388. / 30'000);
  auto output = ourDecoder->getNextDecodedOutput();
  EXPECT_EQ(output.ptsSeconds, 388'388. / 30'000);
  output = ourDecoder->getNextDecodedOutput();
  EXPECT_EQ(output.ptsSeconds, 389'389. / 30'000);
  EXPECT_THROW(ourDecoder->getNextDecodedOutput(), std::exception);
}

TEST_P(VideoDecoderTest, GetsFrameDisplayedAtTimestamp) {
  std::string path = getResourcePath(
      "pytorch/torchcodec/test/decoders/resources/nasa_13013.mp4");
  std::unique_ptr<VideoDecoder> ourDecoder =
      createDecoderFromPath(path, GetParam());
  ourDecoder->addVideoStreamDecoder(-1);
  auto output = ourDecoder->getFrameDisplayedAtTimestamp(6.006);
  EXPECT_EQ(output.ptsSeconds, 6.006);
  // The frame's duration is 0.033367 according to ffprobe,
  // so the next frame is displayed at timestamp=6.039367.
  const double kNextFramePts = 6.039366666666667;
  // The frame that is displayed a microsecond before the next frame is still
  // the previous frame.
  output = ourDecoder->getFrameDisplayedAtTimestamp(kNextFramePts - 1e-6);
  EXPECT_EQ(output.ptsSeconds, 6.006);
  // The frame that is displayed at the exact pts of the frame is the next
  // frame.
  output = ourDecoder->getFrameDisplayedAtTimestamp(kNextFramePts);
  EXPECT_EQ(output.ptsSeconds, kNextFramePts);

  // This is the timestamp of the last frame in this video.
  constexpr double kPtsOfLastFrameInVideoStream = 389'389. / 30'000;
  constexpr double kDurationOfLastFrameInVideoStream = 1'001. / 30'000;
  constexpr double kPtsPlusDurationOfLastFrame =
      kPtsOfLastFrameInVideoStream + kDurationOfLastFrameInVideoStream;
  // Sanity check: make sure duration is strictly positive.
  EXPECT_GT(kPtsPlusDurationOfLastFrame, kPtsOfLastFrameInVideoStream);
  output = ourDecoder->getFrameDisplayedAtTimestamp(
      kPtsPlusDurationOfLastFrame - 1e-6);
  EXPECT_EQ(output.ptsSeconds, kPtsOfLastFrameInVideoStream);
}

TEST_P(VideoDecoderTest, SeeksToFrameWithSpecificPts) {
  std::string path = getResourcePath(
      "pytorch/torchcodec/test/decoders/resources/nasa_13013.mp4");
  std::unique_ptr<VideoDecoder> ourDecoder =
      createDecoderFromPath(path, GetParam());
  ourDecoder->addVideoStreamDecoder(-1);
  ourDecoder->setCursorPtsInSeconds(6.0);
  auto output = ourDecoder->getNextDecodedOutput();
  torch::Tensor tensor6FromOurDecoder = output.frame;
  EXPECT_EQ(output.ptsSeconds, 180'180. / 30'000);
  torch::Tensor tensor6FromFFMPEG = readTensorFromBMP(getResourcePath(
      "pytorch/torchcodec/test/decoders/resources/nasa_13013.mp4.time6.000000.bmp"));
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
  output = ourDecoder->getNextDecodedOutput();
  torch::Tensor tensor61FromOurDecoder = output.frame;
  EXPECT_EQ(output.ptsSeconds, 183'183. / 30'000);
  torch::Tensor tensor61FromFFMPEG = readTensorFromBMP(getResourcePath(
      "pytorch/torchcodec/test/decoders/resources/nasa_13013.mp4.time6.100000.bmp"));
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
  output = ourDecoder->getNextDecodedOutput();
  torch::Tensor tensor10FromOurDecoder = output.frame;
  EXPECT_EQ(output.ptsSeconds, 300'300. / 30'000);
  torch::Tensor tensor10FromFFMPEG = readTensorFromBMP(getResourcePath(
      "pytorch/torchcodec/test/decoders/resources/nasa_13013.mp4.time10.000000.bmp"));
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
  output = ourDecoder->getNextDecodedOutput();
  tensor6FromOurDecoder = output.frame;
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
  output = ourDecoder->getNextDecodedOutput();
  torch::Tensor tensor7FromOurDecoder = output.frame;
  EXPECT_EQ(output.ptsSeconds, 389'389. / 30'000);
  torch::Tensor tensor7FromFFMPEG = readTensorFromBMP(getResourcePath(
      "pytorch/torchcodec/test/decoders/resources/nasa_13013.mp4.time12.979633.bmp"));
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
    dumpTensorToBMP(
        tensor7FromFFMPEG,
        "pytorch/torchcodec/test/decoders/tensor7FromFFMPEG.bmp");
    dumpTensorToBMP(
        tensor7FromOurDecoder,
        "pytorch/torchcodec/test/decoders/tensor7FromOurDecoder.bmp");
  }
}

TEST_P(VideoDecoderTest, GetAudioMetadata) {
  std::string path = getResourcePath(
      "pytorch/torchcodec/test/decoders/resources/nasa_13013.mp4.audio.mp3");
  std::unique_ptr<VideoDecoder> decoder =
      createDecoderFromPath(path, GetParam());
  VideoDecoder::ContainerMetadata metadata = decoder->getContainerMetadata();
  EXPECT_EQ(metadata.numAudioStreams, 1);
  EXPECT_EQ(metadata.numVideoStreams, 0);
  EXPECT_EQ(metadata.streams.size(), 1);

  const auto& audioStream = metadata.streams[0];
  EXPECT_EQ(audioStream.mediaType, AVMEDIA_TYPE_AUDIO);
  EXPECT_NEAR(*audioStream.durationSeconds, 13.25, 1e-1);
}

INSTANTIATE_TEST_SUITE_P(FromFileAndMemory, VideoDecoderTest, testing::Bool());

} // namespace facebook::torchcodec
