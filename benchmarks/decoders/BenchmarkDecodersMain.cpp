// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <chrono>
#include <iostream>
#include <vector>

#include "src/torchcodec/decoders/_core/VideoDecoder.h"
#include "src/torchcodec/decoders/_core/VideoDecoderOps.h"
#include "tools/cxx/Resources.h"

namespace facebook::torchcodec {

void printResults(
    const std::string& decoder_name,
    std::chrono::system_clock::time_point preWarmup,
    std::chrono::system_clock::time_point start,
    std::chrono::system_clock::time_point end,
    int numFrames,
    int warmupIterations,
    int totalIterations) {
  int nonWarmupIterations = totalIterations - warmupIterations;
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  auto warmupDuration =
      std::chrono::duration_cast<std::chrono::microseconds>(start - preWarmup)
          .count();
  double averageMicros =
      static_cast<double>(duration) / nonWarmupIterations / numFrames;
  double warmupAverageMicros =
      static_cast<double>(duration) / warmupIterations / numFrames;
  std::cout << "Decoder=" << decoder_name
            << " average time per frame: " << averageMicros << " us" << " ("
            << averageMicros / 1000 << "ms)"
            << " warmup average time per frame: " << warmupAverageMicros
            << " us" << " (" << warmupAverageMicros / 1000 << "ms)"
            << std::endl;
  std::cout << "Time to seek and decode " << numFrames
            << " frames: " << averageMicros * numFrames << " us" << " ("
            << averageMicros * numFrames / 1000 << " ms)" << " ("
            << nonWarmupIterations << " iterations)"
            << " warmup time: " << warmupAverageMicros * numFrames << " us"
            << " (" << warmupDuration * numFrames / 1000 << " ms)" << " ("
            << warmupIterations << " iterations)" << std::endl;
}

void runNDecodeIterations(
    const std::string& videoPath,
    std::vector<double>& ptsList,
    int totalIterations,
    int warmupIterations) {
  assert(warmupIterations <= totalIterations);
  std::chrono::system_clock::time_point preWarmup =
      std::chrono::high_resolution_clock::now();
  std::chrono::system_clock::time_point start = preWarmup;
  for (int i = 0; i < totalIterations; ++i) {
    std::unique_ptr<VideoDecoder> decoder =
        VideoDecoder::createFromFilePath(videoPath);
    decoder->addVideoStreamDecoder(-1);
    for (double pts : ptsList) {
      decoder->setCursorPtsInSeconds(pts);
      torch::Tensor tensor = decoder->getNextDecodedOutput().frame;
    }
    if (i + 1 == warmupIterations) {
      start = std::chrono::high_resolution_clock::now();
    }
  }
  std::chrono::system_clock::time_point end =
      std::chrono::high_resolution_clock::now();
  printResults(
      "Raw C++ seek+next",
      preWarmup,
      start,
      end,
      ptsList.size(),
      warmupIterations,
      totalIterations);
}

void runNdecodeIterationsGrabbingConsecutiveFrames(
    const std::string& videoPath,
    int consecutiveFrameCount,
    int totalIterations,
    int warmupIterations) {
  assert(warmupIterations <= totalIterations);
  std::chrono::system_clock::time_point preWarmup =
      std::chrono::high_resolution_clock::now();
  std::chrono::system_clock::time_point start = preWarmup;
  for (int i = 0; i < totalIterations; ++i) {
    std::unique_ptr<VideoDecoder> decoder =
        VideoDecoder::createFromFilePath(videoPath);
    decoder->addVideoStreamDecoder(-1);
    for (int j = 0; j < consecutiveFrameCount; ++j) {
      torch::Tensor tensor = decoder->getNextDecodedOutput().frame;
    }
    if (i + 1 == warmupIterations) {
      start = std::chrono::high_resolution_clock::now();
    }
  }
  std::chrono::system_clock::time_point end =
      std::chrono::high_resolution_clock::now();
  printResults(
      "Raw C++ next-only",
      preWarmup,
      start,
      end,
      consecutiveFrameCount,
      warmupIterations,
      totalIterations);
}

void runNDecodeIterationsWithCustomOps(
    const std::string& videoPath,
    std::vector<double>& ptsList,
    int totalIterations,
    int warmupIterations) {
  assert(warmupIterations <= totalIterations);
  std::chrono::system_clock::time_point preWarmup =
      std::chrono::high_resolution_clock::now();
  std::chrono::system_clock::time_point start = preWarmup;
  auto createDecoderOp =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("torchcodec_ns::create_from_file", "")
          .typed<decltype(create_from_file)>();
  auto addVideoStreamOp =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("torchcodec_ns::add_video_stream", "")
          .typed<decltype(add_video_stream)>();
  auto seekFrameOp = torch::Dispatcher::singleton()
                         .findSchemaOrThrow("torchcodec_ns::seek_to_pts", "")
                         .typed<decltype(seek_to_pts)>();
  auto getNextFrameOp =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("torchcodec_ns::get_next_frame", "")
          .typed<decltype(get_next_frame)>();
  for (int i = 0; i < totalIterations; ++i) {
    torch::Tensor decoderTensor = createDecoderOp.call(videoPath);
    addVideoStreamOp.call(
        decoderTensor,
        /*width=*/std::nullopt,
        /*height=*/std::nullopt,
        /*thread_count=*/std::nullopt,
        /*shape=*/std::nullopt,
        /*stream_index=*/std::nullopt);

    for (double pts : ptsList) {
      seekFrameOp.call(decoderTensor, pts);
      torch::Tensor tensor = getNextFrameOp.call(decoderTensor);
    }
    if (i + 1 == warmupIterations) {
      start = std::chrono::high_resolution_clock::now();
    }
  }
  std::chrono::system_clock::time_point end =
      std::chrono::high_resolution_clock::now();
  printResults(
      "CustomOps",
      preWarmup,
      start,
      end,
      ptsList.size(),
      warmupIterations,
      totalIterations);
}

void runBenchmark() {
  std::string videoPath =
      build::getResourcePath(
          "pytorch/torchcodec/benchmarks/decoders/resources/nasa_13013.mp4")
          .string();
  // TODO(T180763625): Add more test cases involving random seeks forwards and
  // backwards.
  std::vector<double> ptsList = {
      0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  runNDecodeIterations(videoPath, ptsList, 100, 5);
  runNDecodeIterationsWithCustomOps(videoPath, ptsList, 100, 5);
  runNdecodeIterationsGrabbingConsecutiveFrames(videoPath, 20, 100, 5);
}

} // namespace facebook::torchcodec

int main() {
  facebook::torchcodec::runBenchmark();
  return 0;
}
