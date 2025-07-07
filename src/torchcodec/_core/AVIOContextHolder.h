// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "src/torchcodec/_core/FFMPEGCommon.h"

namespace facebook::torchcodec {

// The AVIOContextHolder serves several purposes:
//
//   1. It is a smart pointer for the AVIOContext. It has the logic to create
//      a new AVIOContext and will appropriately free the AVIOContext when it
//      goes out of scope. Note that this requires more than just having a
//      UniqueAVIOContext, as the AVIOContext points to a buffer which must be
//      freed.
//   2. It is a base class for AVIOContext specializations. When specializing a
//      AVIOContext, we need to provide four things:
//        1. A read callback function, for decoding.
//        2. A seek callback function, for decoding and encoding.
//        3. A write callback function, for encoding.
//        4. A pointer to some context object that has the same lifetime as the
//           AVIOContext itself. This context object holds the custom state that
//           tracks the custom behavior of reading, seeking and writing. It is
//           provided upon AVIOContext creation and to the read, seek and
//           write callback functions.
//      The callback functions do not need to be members of the derived class,
//      but the derived class must have access to them. The context object must
//      be a member of the derived class. Derived classes need to call
//      createAVIOContext(), ideally in their constructor.
//  3. A generic handle for those that just need to manage having access to an
//     AVIOContext, but aren't necessarily concerned with how it was customized:
//     typically, the SingleStreamDecoder.
class AVIOContextHolder {
 public:
  virtual ~AVIOContextHolder();
  AVIOContext* getAVIOContext();

 protected:
  // Make constructor protected to prevent anyone from constructing
  // an AVIOContextHolder without deriving it. (Ordinarily this would be
  // enforced by having a pure virtual methods, but we don't have any.)
  AVIOContextHolder() = default;

  // Deriving classes should call this function in their constructor.
  void createAVIOContext(
      AVIOReadFunction read,
      AVIOWriteFunction write,
      AVIOSeekFunction seek,
      void* heldData,
      bool isForWriting,
      int bufferSize = defaultBufferSize);

 private:
  UniqueAVIOContext avioContext_;

  // Defaults to 64 KB
  static const int defaultBufferSize = 64 * 1024;
};

} // namespace facebook::torchcodec
