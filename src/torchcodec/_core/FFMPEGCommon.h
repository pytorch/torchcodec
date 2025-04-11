// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <stdexcept>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavutil/avutil.h>
#include <libavutil/dict.h>
#include <libavutil/display.h>
#include <libavutil/file.h>
#include <libavutil/opt.h>
#include <libavutil/pixfmt.h>
#include <libavutil/version.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

namespace facebook::torchcodec {

// FFMPEG uses special delete functions for some structures. These template
// functions are used to pass into unique_ptr as custom deleters so we can
// wrap FFMPEG structs with unique_ptrs for ease of use.
template <typename T, typename R, R (*Fn)(T**)>
struct Deleterp {
  inline void operator()(T* p) const {
    if (p) {
      Fn(&p);
    }
  }
};

template <typename T, typename R, R (*Fn)(T*)>
struct Deleter {
  inline void operator()(T* p) const {
    if (p) {
      Fn(p);
    }
  }
};

// Unique pointers for FFMPEG structures.
using UniqueDecodingAVFormatContext = std::unique_ptr<
    AVFormatContext,
    Deleterp<AVFormatContext, void, avformat_close_input>>;
using UniqueEncodingAVFormatContext = std::unique_ptr<
    AVFormatContext,
    Deleter<AVFormatContext, void, avformat_free_context>>;
using UniqueAVCodecContext = std::unique_ptr<
    AVCodecContext,
    Deleterp<AVCodecContext, void, avcodec_free_context>>;
using UniqueAVFrame =
    std::unique_ptr<AVFrame, Deleterp<AVFrame, void, av_frame_free>>;
using UniqueAVFilterGraph = std::unique_ptr<
    AVFilterGraph,
    Deleterp<AVFilterGraph, void, avfilter_graph_free>>;
using UniqueAVFilterInOut = std::unique_ptr<
    AVFilterInOut,
    Deleterp<AVFilterInOut, void, avfilter_inout_free>>;
using UniqueAVIOContext = std::
    unique_ptr<AVIOContext, Deleterp<AVIOContext, void, avio_context_free>>;
using UniqueSwsContext =
    std::unique_ptr<SwsContext, Deleter<SwsContext, void, sws_freeContext>>;
using UniqueSwrContext =
    std::unique_ptr<SwrContext, Deleterp<SwrContext, void, swr_free>>;

// These 2 classes share the same underlying AVPacket object. They are meant to
// be used in tandem, like so:
//
// AutoAVPacket autoAVPacket; // <-- malloc for AVPacket happens here
// while(...){
//   ReferenceAVPacket packet(autoAVPacket);
//   av_read_frame(..., packet.get());  <-- av_packet_ref() called by FFmpeg
// } <-- av_packet_unref() called here
//
// This achieves a few desirable things:
// - Memory allocation of the underlying AVPacket happens only once, when
//   autoAVPacket is created.
// - av_packet_free() is called when autoAVPacket gets out of scope
// - av_packet_unref() is automatically called when needed, i.e. at the end of
//   each loop iteration (or when hitting break / continue). This prevents the
//   risk of us forgetting to call it.
class AutoAVPacket {
  friend class ReferenceAVPacket;

 private:
  AVPacket* avPacket_;

 public:
  AutoAVPacket();
  AutoAVPacket(const AutoAVPacket& other) = delete;
  AutoAVPacket& operator=(const AutoAVPacket& other) = delete;
  ~AutoAVPacket();
};

class ReferenceAVPacket {
 private:
  AVPacket* avPacket_;

 public:
  explicit ReferenceAVPacket(AutoAVPacket& shared);
  ReferenceAVPacket(const ReferenceAVPacket& other) = delete;
  ReferenceAVPacket& operator=(const ReferenceAVPacket& other) = delete;
  ~ReferenceAVPacket();
  AVPacket* get();
  AVPacket* operator->();
};

// av_find_best_stream is not const-correct before commit:
// https://github.com/FFmpeg/FFmpeg/commit/46dac8cf3d250184ab4247809bc03f60e14f4c0c
// which was released in FFMPEG version=5.0.3
// with libavcodec's version=59.18.100
// (https://www.ffmpeg.org/olddownload.html).
// Note that the alias is so-named so that it is only used when interacting with
// av_find_best_stream(). It is not needed elsewhere.
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(59, 18, 100)
using AVCodecOnlyUseForCallingAVFindBestStream = AVCodec*;
#else
using AVCodecOnlyUseForCallingAVFindBestStream = const AVCodec*;
#endif

AVCodecOnlyUseForCallingAVFindBestStream
makeAVCodecOnlyUseForCallingAVFindBestStream(const AVCodec* codec);

// Success code from FFMPEG is just a 0. We define it to make the code more
// readable.
const int AVSUCCESS = 0;

// Returns the FFMPEG error as a string using the provided `errorCode`.
std::string getFFMPEGErrorStringFromErrorCode(int errorCode);

// Returns duration from the frame. Abstracted into a function because the
// struct member representing duration has changed across the versions we
// support.
int64_t getDuration(const UniqueAVFrame& frame);

int getNumChannels(const UniqueAVFrame& avFrame);
int getNumChannels(const UniqueAVCodecContext& avCodecContext);

void setDefaultChannelLayout(
    UniqueAVCodecContext& avCodecContext,
    int numChannels);

void setChannelLayout(
    UniqueAVFrame& dstAVFrame,
    const UniqueAVCodecContext& avCodecContext);

void setChannelLayout(
    UniqueAVFrame& dstAVFrame,
    const UniqueAVFrame& srcAVFrame);
SwrContext* createSwrContext(
    UniqueAVCodecContext& avCodecContext,
    AVSampleFormat sourceSampleFormat,
    AVSampleFormat desiredSampleFormat,
    int sourceSampleRate,
    int desiredSampleRate);

UniqueAVFrame convertAudioAVFrameSampleFormatAndSampleRate(
    const UniqueSwrContext& swrContext,
    const UniqueAVFrame& srcAVFrame,
    AVSampleFormat desiredSampleFormat,
    int sourceSampleRate,
    int desiredSampleRate);

// Returns true if sws_scale can handle unaligned data.
bool canSwsScaleHandleUnalignedData();

void setFFmpegLogLevel();

} // namespace facebook::torchcodec
