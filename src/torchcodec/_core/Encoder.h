#pragma once
#include <torch/types.h>
#include "src/torchcodec/_core/FFMPEGCommon.h"

namespace facebook::torchcodec {
class Encoder {
 public:
  ~Encoder();

  // TODO Are we OK passing a string_view to the constructor?
  // TODO fileName should be optional.
  // TODO doesn't make much sense to pass fileName and the wf tensor in 2
  // different calls. Same with sampleRate.
  Encoder(int sampleRate, std::string_view fileName);
  void encode(const torch::Tensor& wf);

 private:
  void encode_inner_loop(
      AutoAVPacket& autoAVPacket,
      const UniqueAVFrame& avFrame);

  UniqueAVFormatContextForEncoding avFormatContext_;
  UniqueAVCodecContext avCodecContext_;
  AVStream* avStream_;

  // The *output* sample rate. We can't really decide for the user what it
  // should be. Particularly, the sample rate of the input waveform should match
  // this, and that's up to the user. If sample rates don't match, encoding will
  // still work but audio will be distorted.
  // We technically could let the user also specify the input sample rate, and
  // resample the waveform internally to match them, but that's not in scope for
  // an initial version (if at all).
  int sampleRate_;
};
} // namespace facebook::torchcodec
