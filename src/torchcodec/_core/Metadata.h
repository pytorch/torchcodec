// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/rational.h>
}

namespace facebook::torchcodec {

struct StreamMetadata {
  // Common (video and audio) fields derived from the AVStream.
  int streamIndex;
  // See this link for what various values are available:
  // https://ffmpeg.org/doxygen/trunk/group__lavu__misc.html#ga9a84bba4713dfced21a1a56163be1f48
  AVMediaType mediaType;
  std::optional<AVCodecID> codecId;
  std::optional<std::string> codecName;
  std::optional<double> durationSecondsFromHeader;
  std::optional<double> beginStreamSecondsFromHeader;
  std::optional<int64_t> numFramesFromHeader;
  std::optional<int64_t> numKeyFrames;
  std::optional<double> averageFpsFromHeader;
  std::optional<double> bitRate;

  // More accurate duration, obtained by scanning the file.
  // These presentation timestamps are in time base.
  std::optional<int64_t> beginStreamPtsFromContent;
  std::optional<int64_t> endStreamPtsFromContent;
  // These presentation timestamps are in seconds.
  std::optional<double> beginStreamPtsSecondsFromContent;
  std::optional<double> endStreamPtsSecondsFromContent;
  // This can be useful for index-based seeking.
  std::optional<int64_t> numFramesFromContent;

  // Video-only fields derived from the AVCodecContext.
  std::optional<int> width;
  std::optional<int> height;
  std::optional<AVRational> sampleAspectRatio;

  // Audio-only fields
  std::optional<int64_t> sampleRate;
  std::optional<int64_t> numChannels;
  std::optional<std::string> sampleFormat;
};

struct ContainerMetadata {
  std::vector<StreamMetadata> allStreamMetadata;
  int numAudioStreams = 0;
  int numVideoStreams = 0;
  // Note that this is the container-level duration, which is usually the max
  // of all stream durations available in the container.
  std::optional<double> durationSecondsFromHeader;
  // Total BitRate level information at the container level in bit/s
  std::optional<double> bitRate;
  // If set, this is the index to the default audio stream.
  std::optional<int> bestAudioStreamIndex;
  // If set, this is the index to the default video stream.
  std::optional<int> bestVideoStreamIndex;
};

} // namespace facebook::torchcodec
