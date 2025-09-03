// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/FFMPEGCommon.h"

#include <c10/util/Exception.h>

namespace facebook::torchcodec {

AutoAVPacket::AutoAVPacket() : avPacket_(av_packet_alloc()) {
  TORCH_CHECK(avPacket_ != nullptr, "Couldn't allocate avPacket.");
}

AutoAVPacket::~AutoAVPacket() {
  av_packet_free(&avPacket_);
}

ReferenceAVPacket::ReferenceAVPacket(AutoAVPacket& shared)
    : avPacket_(shared.avPacket_) {}

ReferenceAVPacket::~ReferenceAVPacket() {
  av_packet_unref(avPacket_);
}

AVPacket* ReferenceAVPacket::get() {
  return avPacket_;
}

AVPacket* ReferenceAVPacket::operator->() {
  return avPacket_;
}

AVCodecOnlyUseForCallingAVFindBestStream
makeAVCodecOnlyUseForCallingAVFindBestStream(const AVCodec* codec) {
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(59, 18, 100)
  return const_cast<AVCodec*>(codec);
#else
  return codec;
#endif
}

std::string getFFMPEGErrorStringFromErrorCode(int errorCode) {
  char errorBuffer[AV_ERROR_MAX_STRING_SIZE] = {0};
  av_strerror(errorCode, errorBuffer, AV_ERROR_MAX_STRING_SIZE);
  return std::string(errorBuffer);
}

int64_t getDuration(const UniqueAVFrame& avFrame) {
#if LIBAVUTIL_VERSION_MAJOR < 58
  return avFrame->pkt_duration;
#else
  return avFrame->duration;
#endif
}

int getNumChannels(const UniqueAVFrame& avFrame) {
#if LIBAVFILTER_VERSION_MAJOR > 8 || \
    (LIBAVFILTER_VERSION_MAJOR == 8 && LIBAVFILTER_VERSION_MINOR >= 44)
  return avFrame->ch_layout.nb_channels;
#else
  int numChannels = av_get_channel_layout_nb_channels(avFrame->channel_layout);
  // Handle FFmpeg 4 bug where channel_layout and numChannels are 0 or unset
  // Set values based on avFrame->channels which appears to be correct
  // to allow successful initialization of SwrContext
  if (numChannels == 0 && avFrame->channels > 0) {
    avFrame->channel_layout = av_get_default_channel_layout(avFrame->channels);
    numChannels = avFrame->channels;
  }
  return numChannels;
#endif
}

int getNumChannels(const UniqueAVCodecContext& avCodecContext) {
#if LIBAVFILTER_VERSION_MAJOR > 8 || \
    (LIBAVFILTER_VERSION_MAJOR == 8 && LIBAVFILTER_VERSION_MINOR >= 44)
  return avCodecContext->ch_layout.nb_channels;
#else
  return avCodecContext->channels;
#endif
}

void setDefaultChannelLayout(
    UniqueAVCodecContext& avCodecContext,
    int numChannels) {
#if LIBAVFILTER_VERSION_MAJOR > 7 // FFmpeg > 4
  AVChannelLayout channel_layout;
  av_channel_layout_default(&channel_layout, numChannels);
  avCodecContext->ch_layout = channel_layout;
#else
  uint64_t channel_layout = av_get_default_channel_layout(numChannels);
  avCodecContext->channel_layout = channel_layout;
  avCodecContext->channels = numChannels;
#endif
}

void setDefaultChannelLayout(UniqueAVFrame& avFrame, int numChannels) {
#if LIBAVFILTER_VERSION_MAJOR > 7 // FFmpeg > 4
  AVChannelLayout channel_layout;
  av_channel_layout_default(&channel_layout, numChannels);
  avFrame->ch_layout = channel_layout;
#else
  uint64_t channel_layout = av_get_default_channel_layout(numChannels);
  avFrame->channel_layout = channel_layout;
  avFrame->channels = numChannels;
#endif
}

void validateNumChannels(const AVCodec& avCodec, int numChannels) {
#if LIBAVFILTER_VERSION_MAJOR > 7 // FFmpeg > 4
  if (avCodec.ch_layouts == nullptr) {
    // If we can't validate, we must assume it'll be fine. If not, FFmpeg will
    // eventually raise.
    return;
  }
  // FFmpeg doc indicate that the ch_layouts array is terminated by a zeroed
  // layout, so checking for nb_channels == 0 should indicate its end.
  for (auto i = 0; avCodec.ch_layouts[i].nb_channels != 0; ++i) {
    if (numChannels == avCodec.ch_layouts[i].nb_channels) {
      return;
    }
  }
  // At this point it seems that the encoder doesn't support the requested
  // number of channels, so we error out.
  std::stringstream supportedNumChannels;
  for (auto i = 0; avCodec.ch_layouts[i].nb_channels != 0; ++i) {
    if (i > 0) {
      supportedNumChannels << ", ";
    }
    supportedNumChannels << avCodec.ch_layouts[i].nb_channels;
  }
#else
  if (avCodec.channel_layouts == nullptr) {
    // can't validate, same as above.
    return;
  }
  for (auto i = 0; avCodec.channel_layouts[i] != 0; ++i) {
    if (numChannels ==
        av_get_channel_layout_nb_channels(avCodec.channel_layouts[i])) {
      return;
    }
  }
  // At this point it seems that the encoder doesn't support the requested
  // number of channels, so we error out.
  std::stringstream supportedNumChannels;
  for (auto i = 0; avCodec.channel_layouts[i] != 0; ++i) {
    if (i > 0) {
      supportedNumChannels << ", ";
    }
    supportedNumChannels << av_get_channel_layout_nb_channels(
        avCodec.channel_layouts[i]);
  }
#endif
  TORCH_CHECK(
      false,
      "Desired number of channels (",
      numChannels,
      ") is not supported by the ",
      "encoder. Supported number of channels are: ",
      supportedNumChannels.str(),
      ".");
}

namespace {
#if LIBAVFILTER_VERSION_MAJOR > 7 // FFmpeg > 4

// Returns:
// - the srcAVFrame's channel layout if srcAVFrame has outNumChannels
// - the default channel layout with outNumChannels otherwise.
AVChannelLayout getOutputChannelLayout(
    int outNumChannels,
    const UniqueAVFrame& srcAVFrame) {
  AVChannelLayout outLayout;
  if (outNumChannels == getNumChannels(srcAVFrame)) {
    outLayout = srcAVFrame->ch_layout;
  } else {
    av_channel_layout_default(&outLayout, outNumChannels);
  }
  return outLayout;
}

#else

// Same as above
int64_t getOutputChannelLayout(
    int outNumChannels,
    const UniqueAVFrame& srcAVFrame) {
  int64_t outLayout;
  if (outNumChannels == getNumChannels(srcAVFrame)) {
    outLayout = srcAVFrame->channel_layout;
  } else {
    outLayout = av_get_default_channel_layout(outNumChannels);
  }
  return outLayout;
}
#endif
} // namespace

// Sets dstAVFrame' channel layout to getOutputChannelLayout(): see doc above
void setChannelLayout(
    UniqueAVFrame& dstAVFrame,
    const UniqueAVFrame& srcAVFrame,
    int outNumChannels) {
#if LIBAVFILTER_VERSION_MAJOR > 7 // FFmpeg > 4
  AVChannelLayout outLayout =
      getOutputChannelLayout(outNumChannels, srcAVFrame);
  auto status = av_channel_layout_copy(&dstAVFrame->ch_layout, &outLayout);
  TORCH_CHECK(
      status == AVSUCCESS,
      "Couldn't copy channel layout to avFrame: ",
      getFFMPEGErrorStringFromErrorCode(status));
#else
  dstAVFrame->channel_layout =
      getOutputChannelLayout(outNumChannels, srcAVFrame);
  dstAVFrame->channels = outNumChannels;
#endif
}

UniqueAVFrame allocateAVFrame(
    int numSamples,
    int sampleRate,
    int numChannels,
    AVSampleFormat sampleFormat) {
  auto avFrame = UniqueAVFrame(av_frame_alloc());
  TORCH_CHECK(avFrame != nullptr, "Couldn't allocate AVFrame.");

  avFrame->nb_samples = numSamples;
  avFrame->sample_rate = sampleRate;
  setDefaultChannelLayout(avFrame, numChannels);
  avFrame->format = sampleFormat;
  auto status = av_frame_get_buffer(avFrame.get(), 0);

  TORCH_CHECK(
      status == AVSUCCESS,
      "Couldn't allocate avFrame's buffers: ",
      getFFMPEGErrorStringFromErrorCode(status));

  status = av_frame_make_writable(avFrame.get());
  TORCH_CHECK(
      status == AVSUCCESS,
      "Couldn't make AVFrame writable: ",
      getFFMPEGErrorStringFromErrorCode(status));
  return avFrame;
}

SwrContext* createSwrContext(
    AVSampleFormat srcSampleFormat,
    AVSampleFormat outSampleFormat,
    int srcSampleRate,
    int outSampleRate,
    const UniqueAVFrame& srcAVFrame,
    int outNumChannels) {
  SwrContext* swrContext = nullptr;
  int status = AVSUCCESS;
#if LIBAVFILTER_VERSION_MAJOR > 7 // FFmpeg > 4
  AVChannelLayout outLayout =
      getOutputChannelLayout(outNumChannels, srcAVFrame);
  status = swr_alloc_set_opts2(
      &swrContext,
      &outLayout,
      outSampleFormat,
      outSampleRate,
      &srcAVFrame->ch_layout,
      srcSampleFormat,
      srcSampleRate,
      0,
      nullptr);

  TORCH_CHECK(
      status == AVSUCCESS,
      "Couldn't create SwrContext: ",
      getFFMPEGErrorStringFromErrorCode(status));
#else
  int64_t outLayout = getOutputChannelLayout(outNumChannels, srcAVFrame);
  swrContext = swr_alloc_set_opts(
      nullptr,
      outLayout,
      outSampleFormat,
      outSampleRate,
      srcAVFrame->channel_layout,
      srcSampleFormat,
      srcSampleRate,
      0,
      nullptr);
#endif

  TORCH_CHECK(swrContext != nullptr, "Couldn't create swrContext");
  status = swr_init(swrContext);
  TORCH_CHECK(
      status == AVSUCCESS,
      "Couldn't initialize SwrContext: ",
      getFFMPEGErrorStringFromErrorCode(status),
      ". If the error says 'Invalid argument', it's likely that you are using "
      "a buggy FFmpeg version. FFmpeg4 is known to fail here in some "
      "valid scenarios. Try to upgrade FFmpeg?");
  return swrContext;
}

UniqueAVFrame convertAudioAVFrameSamples(
    const UniqueSwrContext& swrContext,
    const UniqueAVFrame& srcAVFrame,
    AVSampleFormat outSampleFormat,
    int outSampleRate,
    int outNumChannels) {
  UniqueAVFrame convertedAVFrame(av_frame_alloc());
  TORCH_CHECK(
      convertedAVFrame,
      "Could not allocate frame for sample format conversion.");

  convertedAVFrame->pts = srcAVFrame->pts;
  convertedAVFrame->format = static_cast<int>(outSampleFormat);

  convertedAVFrame->sample_rate = outSampleRate;
  int srcSampleRate = srcAVFrame->sample_rate;
  if (srcSampleRate != outSampleRate) {
    // Note that this is an upper bound on the number of output samples.
    // `swr_convert()` will likely not fill convertedAVFrame with that many
    // samples if sample rate conversion is needed. It will buffer the last few
    // ones because those require future samples. That's also why we reset
    // nb_samples after the call to `swr_convert()`.
    // We could also use `swr_get_out_samples()` to determine the number of
    // output samples, but empirically `av_rescale_rnd()` seems to provide a
    // tighter bound.
    convertedAVFrame->nb_samples = av_rescale_rnd(
        swr_get_delay(swrContext.get(), srcSampleRate) + srcAVFrame->nb_samples,
        outSampleRate,
        srcSampleRate,
        AV_ROUND_UP);
  } else {
    convertedAVFrame->nb_samples = srcAVFrame->nb_samples;
  }

  setChannelLayout(convertedAVFrame, srcAVFrame, outNumChannels);

  auto status = av_frame_get_buffer(convertedAVFrame.get(), 0);
  TORCH_CHECK(
      status == AVSUCCESS,
      "Could not allocate frame buffers for sample format conversion: ",
      getFFMPEGErrorStringFromErrorCode(status));

  auto numConvertedSamples = swr_convert(
      swrContext.get(),
      convertedAVFrame->data,
      convertedAVFrame->nb_samples,
      static_cast<const uint8_t**>(
          const_cast<const uint8_t**>(srcAVFrame->data)),
      srcAVFrame->nb_samples);
  // numConvertedSamples can be 0 if we're downsampling by a great factor and
  // the first frame doesn't contain a lot of samples. It should be handled
  // properly by the caller.
  TORCH_CHECK(
      numConvertedSamples >= 0,
      "Error in swr_convert: ",
      getFFMPEGErrorStringFromErrorCode(numConvertedSamples));

  // See comment above about nb_samples
  convertedAVFrame->nb_samples = numConvertedSamples;

  return convertedAVFrame;
}

void setFFmpegLogLevel() {
  auto logLevel = AV_LOG_QUIET;
  const char* logLevelEnvPtr = std::getenv("TORCHCODEC_FFMPEG_LOG_LEVEL");
  if (logLevelEnvPtr != nullptr) {
    std::string logLevelEnv(logLevelEnvPtr);
    if (logLevelEnv == "QUIET") {
      logLevel = AV_LOG_QUIET;
    } else if (logLevelEnv == "PANIC") {
      logLevel = AV_LOG_PANIC;
    } else if (logLevelEnv == "FATAL") {
      logLevel = AV_LOG_FATAL;
    } else if (logLevelEnv == "ERROR") {
      logLevel = AV_LOG_ERROR;
    } else if (logLevelEnv == "WARNING") {
      logLevel = AV_LOG_WARNING;
    } else if (logLevelEnv == "INFO") {
      logLevel = AV_LOG_INFO;
    } else if (logLevelEnv == "VERBOSE") {
      logLevel = AV_LOG_VERBOSE;
    } else if (logLevelEnv == "DEBUG") {
      logLevel = AV_LOG_DEBUG;
    } else if (logLevelEnv == "TRACE") {
      logLevel = AV_LOG_TRACE;
    } else {
      TORCH_CHECK(
          false,
          "Invalid TORCHCODEC_FFMPEG_LOG_LEVEL: ",
          logLevelEnv,
          ". Use e.g. 'QUIET', 'PANIC', 'VERBOSE', etc.");
    }
  }
  av_log_set_level(logLevel);
}

AVIOContext* avioAllocContext(
    uint8_t* buffer,
    int buffer_size,
    int write_flag,
    void* opaque,
    AVIOReadFunction read_packet,
    AVIOWriteFunction write_packet,
    AVIOSeekFunction seek) {
  return avio_alloc_context(
      buffer,
      buffer_size,
      write_flag,
      opaque,
      read_packet,
// The buf parameter of the write function is not const before FFmpeg 7.
#if LIBAVFILTER_VERSION_MAJOR >= 10 // FFmpeg >= 7
      write_packet,
#else
      reinterpret_cast<AVIOWriteFunctionOld>(write_packet),
#endif
      seek);
}

} // namespace facebook::torchcodec
