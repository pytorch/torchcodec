#pragma once
#include <torch/types.h>
#include "src/torchcodec/_core/AVIOTensorContext.h"
#include "src/torchcodec/_core/AVIOFileLikeContext.h"
#include "src/torchcodec/_core/FFMPEGCommon.h"
#include "src/torchcodec/_core/StreamOptions.h"

namespace facebook::torchcodec {
class AudioEncoder {
 public:
  ~AudioEncoder();

  AudioEncoder(
      const torch::Tensor& samples,
      // TODO-ENCODING: update this comment when we support an output sample
      // rate. This will become the input sample rate.
      // The *output* sample rate. We can't really decide for the user what it
      // should be. Particularly, the sample rate of the input samples should
      // match this, and that's up to the user. If sample rates don't match,
      // encoding will still work but audio will be distorted.
      int sampleRate,
      std::string_view fileName,
      const AudioStreamOptions& audioStreamOptions);
  AudioEncoder(
      const torch::Tensor& samples,
      int sampleRate,
      std::string_view formatName,
      std::unique_ptr<AVIOToTensorContext> avioContextHolder,
      const AudioStreamOptions& audioStreamOptions);
  AudioEncoder(
      const torch::Tensor& samples,
      int sampleRate,
      std::string_view formatName,
      std::unique_ptr<AVIOFileLikeContext> avioContextHolder,
      const AudioStreamOptions& audioStreamOptions);
  void encode();
  torch::Tensor encodeToTensor();

 private:
  void initializeEncoder(
      int sampleRate,
      const AudioStreamOptions& audioStreamOptions);
  UniqueAVFrame maybeConvertAVFrame(const UniqueAVFrame& avFrame);
  void encodeInnerLoop(
      AutoAVPacket& autoAVPacket,
      const UniqueAVFrame& srcAVFrame);
  void flushBuffers();

  UniqueEncodingAVFormatContext avFormatContext_;
  UniqueAVCodecContext avCodecContext_;
  int streamIndex_;
  UniqueSwrContext swrContext_;
  AudioStreamOptions audioStreamOptions;

  int outNumChannels_ = -1;

  const torch::Tensor samples_;

  // Stores the AVIOContext for the output tensor buffer.
  std::unique_ptr<AVIOToTensorContext> avioTensorContextHolder_;

  // Stores the AVIOContext for file-like object output.
  std::unique_ptr<AVIOFileLikeContext> avioFileLikeContextHolder_;

  bool encodeWasCalled_ = false;
};
} // namespace facebook::torchcodec
