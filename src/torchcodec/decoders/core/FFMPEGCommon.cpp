// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "src/torchcodec/decoders/core/FFMPEGCommon.h"

namespace facebook::torchcodec {

std::string getFFMPEGErrorStringFromErrorCode(int errorCode) {
  char errorBuffer[AV_ERROR_MAX_STRING_SIZE] = {0};
  av_strerror(errorCode, errorBuffer, AV_ERROR_MAX_STRING_SIZE);
  return std::string(errorBuffer);
}

AVIOBytesContext::AVIOBytesContext(
    const void* data,
    size_t data_size,
    size_t tempBufferSize) {
  auto buffer = static_cast<uint8_t*>(av_malloc(tempBufferSize));
  if (!buffer) {
    throw std::runtime_error(
        "Failed to allocate buffer of size " + std::to_string(tempBufferSize));
  }
  bufferData_.data = static_cast<const uint8_t*>(data);
  bufferData_.size = data_size;
  bufferData_.current = 0;

  avioContext_.reset(avio_alloc_context(
      buffer,
      tempBufferSize,
      0,
      &bufferData_,
      &AVIOBytesContext::read,
      nullptr,
      &AVIOBytesContext::seek));
  if (!avioContext_) {
    av_freep(&buffer);
    throw std::runtime_error("Failed to allocate AVIOContext");
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
  struct AVIOBufferData* bufferData =
      static_cast<struct AVIOBufferData*>(opaque);
  buf_size = FFMIN(buf_size, bufferData->size - bufferData->current);
  if (!buf_size) {
    return AVERROR_EOF;
  }
  memcpy(buf, bufferData->data + bufferData->current, buf_size);
  bufferData->current += buf_size;
  return buf_size;
}

// The signature of this function is defined by FFMPEG.
int64_t AVIOBytesContext::seek(void* opaque, int64_t offset, int whence) {
  AVIOBufferData* bufferData = (AVIOBufferData*)opaque;
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
