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
  return av_get_channel_layout_nb_channels(avFrame->channel_layout);
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

void setChannelLayout(
    UniqueAVFrame& dstAVFrame,
    const UniqueAVCodecContext& avCodecContext) {
#if LIBAVFILTER_VERSION_MAJOR > 7 // FFmpeg > 4
  auto status = av_channel_layout_copy(
      &dstAVFrame->ch_layout, &avCodecContext->ch_layout);
  TORCH_CHECK(
      status == AVSUCCESS,
      "Couldn't copy channel layout to avFrame: ",
      getFFMPEGErrorStringFromErrorCode(status));
#else
  dstAVFrame->channel_layout = avCodecContext->channel_layout;
  dstAVFrame->channels = avCodecContext->channels;

#endif
}

void setChannelLayout(
    UniqueAVFrame& dstAVFrame,
    const UniqueAVFrame& srcAVFrame) {
#if LIBAVFILTER_VERSION_MAJOR > 7 // FFmpeg > 4
  dstAVFrame->ch_layout = srcAVFrame->ch_layout;
#else
  dstAVFrame->channel_layout = srcAVFrame->channel_layout;
#endif
}

SwrContext* createSwrContext(
    UniqueAVCodecContext& avCodecContext,
    AVSampleFormat sourceSampleFormat,
    AVSampleFormat desiredSampleFormat,
    int sourceSampleRate,
    int desiredSampleRate) {
  SwrContext* swrContext = nullptr;
  int status = AVSUCCESS;
#if LIBAVFILTER_VERSION_MAJOR > 7 // FFmpeg > 4
  AVChannelLayout layout = avCodecContext->ch_layout;
  status = swr_alloc_set_opts2(
      &swrContext,
      &layout,
      desiredSampleFormat,
      desiredSampleRate,
      &layout,
      sourceSampleFormat,
      sourceSampleRate,
      0,
      nullptr);

  TORCH_CHECK(
      status == AVSUCCESS,
      "Couldn't create SwrContext: ",
      getFFMPEGErrorStringFromErrorCode(status));
#else
  int64_t layout = static_cast<int64_t>(avCodecContext->channel_layout);
  swrContext = swr_alloc_set_opts(
      nullptr,
      layout,
      desiredSampleFormat,
      desiredSampleRate,
      layout,
      sourceSampleFormat,
      sourceSampleRate,
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

} // namespace facebook::torchcodec
