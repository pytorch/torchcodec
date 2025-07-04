#pragma once
#include <torch/types.h>
#include "src/torchcodec/_core/AVIOTensorContext.h"
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
  void encode();
  torch::Tensor encodeToTensor();

 private:
  void initializeEncoder(const AudioStreamOptions& audioStreamOptions);
  UniqueAVFrame maybeConvertAVFrame(const UniqueAVFrame& avFrame);
  void encodeFrameThroughFifo(
      AutoAVPacket& autoAVPacket,
      const UniqueAVFrame& avFrame,
      bool andFlushFifo = false);
  void encodeFrame(AutoAVPacket& autoAVPacket, const UniqueAVFrame& avFrame);
  void maybeFlushSwrBuffers(AutoAVPacket& autoAVPacket);
  void flushBuffers();

  UniqueEncodingAVFormatContext avFormatContext_;
  UniqueAVCodecContext avCodecContext_;
  int streamIndex_;
  UniqueSwrContext swrContext_;
  AudioStreamOptions audioStreamOptions;

  const torch::Tensor samples_;

  int outNumChannels_ = -1;
  int outSampleRate_ = -1;
  int inSampleRate_ = -1;

  UniqueAVAudioFifo avAudioFifo_;

  // Stores the AVIOContext for the output tensor buffer.
  std::unique_ptr<AVIOToTensorContext> avioContextHolder_;

  bool encodeWasCalled_ = false;
  int64_t lastEncodedAVFramePts_ = 0;
};
} // namespace facebook::torchcodec

/* clang-format off */
//
// Note: [Encoding loop, sample rate conversion and FIFO]
//
// The input samples are in a given format, sample rate, and number of channels.
// We may want to change these properties before encoding. The conversion is
// done in maybeConvertAVFrame() and we rely on libswresample. When sample rate
// conversion is needed, this means two things:
// - swr will be storing samples in its internal buffers, which we'll need to
//   flush at the very end of the encoding process.
// - the converted AVFrame we get back from maybeConvertAVFrame() typically
//   won't have the same number of samples as the original AVFrame. And that's
//   a problem, because some encoders expect AVFrames with a specific and
//   constant number of samples. If we were to send it as-is, we'd get an error
//   in avcodec_send_frame(). In order to feed the encoder with AVFrames
//   with the expected number of samples, we go through an intermediate FIFO
//   from which we can pull the exact number of samples that we need. Note that
//   this involves at least 2 additional copies.
//
// To be clear, the FIFO is only used if BOTH the following conditions are met:
//  - sample rate conversion is needed (inSampleRate_ != outSampleRate_)
//  - the encoder expects a specific number of samples per AVFrame (fixed frame size)
//    This is not the case for all encoders, e.g. WAV doesn't care about frame size.
//
// ┌─One─iteration─of─main─encoding─loop─(encode())───────────────────────────────────────────┐
// │                                                                                          │
// │                        Converts:                                                         │
// │                         - num channels                                                   │
// │                         - format                                                         │
// │                         - sample rate                                                    │
// │                        If sample rate,                                                   │
// │                        stores data in                                                    │
// │                        swr buffers                                                       │
// │                        which will need                                                   │
// │                        to be flushed                                                     │
// │                                                                                          │
// │                               ▲                                                          │
// │                               │                 ┌─EncodeFrameThroughFifo()──────────────┐│
// │                               │                 │                                       ││
// │    AVFrame  ──────►  MaybeConvertAVFrame()───▲──│─┬──────────────┬──▲────►encodeFrame() ││
// │    with                                      │  │ │              │  │                   ││
// │    input                                     │  │ │              │  │                   ││
// │    samples                                   │  │ │              │  │                   ││
// │                                              │  │ │              │  │                   ││
// │                                              │  │ └────► FIFO ───┘  │                   ││
// │                                              │  └───────────────────┼───────────────────┘│
// └──────────────────────────────────────────────┼──────────────────────┼────────────────────┘
//                                                │                      │
//  AVFrame from  maybeFlushSwrBuffers()       ───┘                      │
//  Only if sample rate conversion was needed
//                                                                 nullptr, to flush
//                                                                 FFmpeg buffers
/* clang-format on */
