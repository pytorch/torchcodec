// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include "FFMPEGCommon.h"
#include "Frame.h"
#include "StableABICompat.h"
#include "StreamOptions.h"
#include "Transform.h"

namespace facebook::torchcodec {

// Key for device interface registration with device type + variant support
struct DeviceInterfaceKey {
  StableDeviceType device_type;
  std::string_view variant = "default"; // e.g., "default", "ffmpeg"

  bool operator<(const DeviceInterfaceKey& other) const {
    if (device_type != other.device_type) {
      return device_type < other.device_type;
    }
    return variant < other.variant;
  }

  explicit DeviceInterfaceKey(StableDeviceType type) : device_type(type) {}

  DeviceInterfaceKey(StableDeviceType type, const std::string_view& variant)
      : device_type(type), variant(variant) {}
};

class DeviceInterface {
 public:
  DeviceInterface(const StableDevice& device) : device_(device) {}

  virtual ~DeviceInterface(){};

  StableDevice& device() {
    return device_;
  };

  virtual std::optional<const AVCodec*> find_codec(
      [[maybe_unused]] const AVCodecID& codec_id,
      [[maybe_unused]] bool is_decoder = true) {
    return std::nullopt;
  };

  // Initialize the device with the codec context, which is needed for the
  // default sendPacket/receiveFrame/flush implementations.
  virtual void initialize(const SharedAVCodecContext& codec_context) = 0;

  // Initialize state needed to decode packets into raw AVFrames.
  virtual void initialize_video_decoding(
      [[maybe_unused]] const AVStream* av_stream,
      [[maybe_unused]] const UniqueDecodingAVFormatContext& av_format_ctx,
      [[maybe_unused]] const VideoStreamOptions& video_stream_options) {}

  // Initialize state needed to color-convert decoded AVFrames into output
  // tensors.
  virtual void initialize_color_conversion(
      [[maybe_unused]] const VideoStreamOptions& video_stream_options,
      [[maybe_unused]] const std::vector<std::unique_ptr<Transform>>&
          transforms = {},
      [[maybe_unused]] const std::optional<FrameDims>& resized_output_dims =
          std::nullopt) {}

  // Convenience for the combined decode + color-convert path, kept for BC as
  // it's used by SingleStreamDecoder and out-of-tree interfaces rely on it.
  void initialize_video(
      const AVStream* av_stream,
      const UniqueDecodingAVFormatContext& av_format_ctx,
      const VideoStreamOptions& video_stream_options,
      const std::vector<std::unique_ptr<Transform>>& transforms,
      const std::optional<FrameDims>& resized_output_dims) {
    initialize_video_decoding(av_stream, av_format_ctx, video_stream_options);
    initialize_color_conversion(
        video_stream_options, transforms, resized_output_dims);
  }

  // Initialize the device with parameters specific to audio decoding. There is
  // a default empty implementation.
  virtual void initialize_audio(
      [[maybe_unused]] const AudioStreamOptions& audio_stream_options,
      [[maybe_unused]] double stream_start_seconds = 0.0) {}

  // Flush any remaining samples from the audio resampler buffer.
  // When sample rate conversion is involved, some samples may be buffered
  // between frames for proper interpolation. This function flushes those
  // buffered samples.
  // Returns an optional tensor containing the flushed samples, or std::nullopt
  // if there are no buffered samples or audio is not supported.
  virtual std::optional<torch::stable::Tensor> maybe_flush_audio_buffers() {
    return std::nullopt;
  }

  // In order for decoding to actually happen on an FFmpeg managed hardware
  // device, we need to register the DeviceInterface managed
  // AVHardwareDeviceContext with the AVCodecContext. We don't need to do this
  // on the CPU and if FFmpeg is not managing the hardware device.
  virtual void register_hardware_device_with_codec(
      [[maybe_unused]] AVCodecContext* codec_context) {}

  // The dtype that should be used for the pre-allocated tensors on batch APIs.
  // Usually that will just be the user's requested dtype, but not always: on
  // CUDA with SDR videos when float32 is asked, will fallback from P016 to NV12
  // to avoid falling back all the way to the CPU. The pre-allocated tensors
  // then need to be uint8, not float32.
  virtual OutputDtype get_pre_allocation_dtype(
      OutputDtype requested_dtype) const {
    return requested_dtype;
  }

  virtual void convert_av_frame_to_frame_output(
      const AVFrame& av_frame,
      FrameOutput& frame_output,
      std::optional<torch::stable::Tensor> pre_allocated_output_tensor =
          std::nullopt) = 0;

  // ------------------------------------------
  // Extension points for custom decoding paths
  // ------------------------------------------

  // Returns AVSUCCESS on success, AVERROR(EAGAIN) if decoder queue full, or
  // other AVERROR on failure
  // Default implementation uses FFmpeg directly
  // The packet is borrowed: an implementation that hands it to something which
  // takes ownership (a bitstream filter, say) must reference it first.
  virtual int send_packet(const AVPacket& av_packet) {
    STD_TORCH_CHECK(
        codec_context_ != nullptr,
        "Codec context not available for default packet sending");
    return avcodec_send_packet(codec_context_.get(), &av_packet);
  }

  // Send an EOF packet to flush the decoder
  // Returns AVSUCCESS on success, or other AVERROR on failure
  // Default implementation uses FFmpeg directly
  virtual int send_eof_packet() {
    STD_TORCH_CHECK(
        codec_context_ != nullptr,
        "Codec context not available for default EOF packet sending");
    return avcodec_send_packet(codec_context_.get(), nullptr);
  }

  // Returns AVSUCCESS on success, AVERROR(EAGAIN) if no frame ready,
  // AVERROR_EOF if end of stream, or other AVERROR on failure
  // Default implementation uses FFmpeg directly
  virtual int receive_frame(UniqueAVFrame& av_frame) {
    STD_TORCH_CHECK(
        codec_context_ != nullptr,
        "Codec context not available for default frame receiving");
    return avcodec_receive_frame(codec_context_.get(), av_frame.get());
  }

  virtual void make_frame_standalone([[maybe_unused]] UniqueAVFrame& av_frame) {
  };

  virtual std::optional<torch::stable::Tensor> get_frame_storage(
      [[maybe_unused]] const AVFrame& av_frame) const {
    return std::nullopt;
  }

  // Flush remaining frames from decoder
  virtual void flush() {
    STD_TORCH_CHECK(
        codec_context_ != nullptr,
        "Codec context not available for default flushing");
    avcodec_flush_buffers(codec_context_.get());

    // We also manually flush any remaining frames in the decoder buffer. We
    // shouldn't have to do this, because avcodec_flush_buffers should handle
    // it, but some codecs like HEVC may still have frames buffered internally
    // in edge cases (ex. hitting EOF) as observed in
    // https://github.com/meta-pytorch/torchcodec/issues/1339.
    UniqueAVFrame tmp(av_frame_alloc());
    while (avcodec_receive_frame(codec_context_.get(), tmp.get()) ==
           AVSUCCESS) {
    }
  }

  virtual std::string get_details() {
    return "";
  }

  virtual UniqueAVFrame convert_tensor_to_av_frame_for_encoding(
      [[maybe_unused]] const torch::stable::Tensor& tensor,
      [[maybe_unused]] int frame_index,
      [[maybe_unused]] AVCodecContext* codec_context) {
    STD_TORCH_CHECK(false, "convertTensorToAVFrameForEncoding not implemented");
  }

  // Returns the pixel format the encoder should use for this device.
  virtual AVPixelFormat get_encoding_pixel_format(
      [[maybe_unused]] const AVCodec& av_codec,
      [[maybe_unused]] const std::optional<std::string>& user_pixel_format)
      const {
    STD_TORCH_CHECK(false, "get_encoding_pixel_format not implemented");
  }

  // No-op on CPU so the encoder can call it unconditionally; HW devices
  // override to attach an AVHWFramesContext.
  virtual void setup_hardware_frame_context_for_encoding(
      [[maybe_unused]] AVCodecContext* codec_context) {}

  virtual std::optional<const AVCodec*> find_hardware_encoder(
      [[maybe_unused]] const AVCodecID& codec_id) {
    STD_TORCH_CHECK(false, "findHardwareEncoder not implemented");
  }

 protected:
  StableDevice device_;
  SharedAVCodecContext codec_context_;
  AVMediaType av_media_type_;
};

using CreateDeviceInterfaceFn =
    std::function<DeviceInterface*(const StableDevice& device)>;

TORCHCODEC_THIRD_PARTY_API bool register_device_interface(
    const DeviceInterfaceKey& key,
    const CreateDeviceInterfaceFn create_interface);

FORCE_PUBLIC_VISIBILITY void validate_device_interface(
    const std::string& device,
    const std::string& variant = "default");

TORCHCODEC_THIRD_PARTY_API std::unique_ptr<DeviceInterface>
create_device_interface(
    const StableDevice& device,
    const std::string_view variant = "default");

torch::stable::Tensor rgb_av_frame_to_tensor(const AVFrame& av_frame);

} // namespace facebook::torchcodec
