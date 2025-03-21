// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/decoders/_core/FFMPEGCommon.h"

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

void setChannelLayout(
    UniqueAVFrame& dstAVFrame,
    const UniqueAVFrame& srcAVFrame) {
#if LIBAVFILTER_VERSION_MAJOR > 7 // FFmpeg > 4
  dstAVFrame->ch_layout = srcAVFrame->ch_layout;
#else
  dstAVFrame->channel_layout = srcAVFrame->channel_layout;
#endif
}

SwrContext* allocateSwrContext(
    UniqueAVCodecContext& avCodecContext,
    AVSampleFormat sourceSampleFormat,
    AVSampleFormat desiredSampleFormat,
    int sourceSampleRate,
    int desiredSampleRate) {
  SwrContext* swrContext = nullptr;
#if LIBAVFILTER_VERSION_MAJOR > 7 // FFmpeg > 4
  AVChannelLayout layout = avCodecContext->ch_layout;
  auto status = swr_alloc_set_opts2(
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
  return swrContext;
}

AVIOBytesContext::AVIOBytesContext(
    const void* data,
    size_t dataSize,
    size_t bufferSize)
    : bufferData_{static_cast<const uint8_t*>(data), dataSize, 0} {
  auto buffer = static_cast<uint8_t*>(av_malloc(bufferSize));
  TORCH_CHECK(
      buffer != nullptr,
      "Failed to allocate buffer of size " + std::to_string(bufferSize));

  avioContext_.reset(avio_alloc_context(
      buffer,
      bufferSize,
      0,
      &bufferData_,
      &AVIOBytesContext::read,
      nullptr,
      &AVIOBytesContext::seek));

  if (!avioContext_) {
    av_freep(&buffer);
    TORCH_CHECK(false, "Failed to allocate AVIOContext");
  }
}

AVIOBytesContext::~AVIOBytesContext() {
  if (avioContext_) {
    av_freep(&avioContext_->buffer);
  }
}

AVIOContext* AVIOBytesContext::getAVIO() {
  return avioContext_.get();
}

// The signature of this function is defined by FFMPEG.
int AVIOBytesContext::read(void* opaque, uint8_t* buf, int buf_size) {
  auto bufferData = static_cast<AVIOBufferData*>(opaque);
  TORCH_CHECK(
      bufferData->current <= bufferData->size,
      "Tried to read outside of the buffer: current=",
      bufferData->current,
      ", size=",
      bufferData->size);

  buf_size =
      FFMIN(buf_size, static_cast<int>(bufferData->size - bufferData->current));
  TORCH_CHECK(
      buf_size >= 0,
      "Tried to read negative bytes: buf_size=",
      buf_size,
      ", size=",
      bufferData->size,
      ", current=",
      bufferData->current);

  if (!buf_size) {
    return AVERROR_EOF;
  }
  memcpy(buf, bufferData->data + bufferData->current, buf_size);
  bufferData->current += buf_size;
  return buf_size;
}

// The signature of this function is defined by FFMPEG.
int64_t AVIOBytesContext::seek(void* opaque, int64_t offset, int whence) {
  auto bufferData = static_cast<AVIOBufferData*>(opaque);
  int64_t ret = -1;

  switch (whence) {
    case AVSEEK_SIZE:
      ret = bufferData->size;
      break;
    case SEEK_SET:
      bufferData->current = offset;
      ret = offset;
      break;
    default:
      break;
  }

  return ret;
}

} // namespace facebook::torchcodec
