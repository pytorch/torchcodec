// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "PacketDecoder.h"

#include "AudioCommon.h"

#include <algorithm>
#include <cstring>

namespace facebook::torchcodec {

SharedAVCodecContext create_and_open_codec_context(
    AVStream* stream,
    const AVCodec* av_codec,
    DeviceInterface* device_interface,
    std::optional<int> thread_count) {
  AVCodecContext* raw_codec_context = avcodec_alloc_context3(av_codec);
  STD_TORCH_CHECK(
      raw_codec_context != nullptr, "Failed to allocate codec context");
  SharedAVCodecContext codec_context =
      make_shared_av_codec_context(raw_codec_context);

  int ret =
      avcodec_parameters_to_context(codec_context.get(), stream->codecpar);
  STD_TORCH_CHECK(ret == AVSUCCESS, "avcodec_parameters_to_context failed");

  codec_context->thread_count = thread_count.value_or(0);
  codec_context->pkt_timebase = stream->time_base;

  // We must register the hardware device context with the codec context before
  // calling avcodec_open2(). Otherwise, decoding will happen on the CPU and not
  // the hardware device.
  device_interface->register_hardware_device_with_codec(codec_context.get());
  ret = avcodec_open2(codec_context.get(), av_codec, nullptr);
  STD_TORCH_CHECK(
      ret >= AVSUCCESS, get_ffmpeg_error_string_from_error_code(ret));

  codec_context->time_base = stream->time_base;
  return codec_context;
}

namespace {
const AVCodec* find_decoder(
    AVStream* stream,
    DeviceInterface* device_interface) {
  const AVCodec* av_codec = avcodec_find_decoder(stream->codecpar->codec_id);
  STD_TORCH_CHECK(av_codec != nullptr, "Codec not found");
  if (stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
    av_codec = device_interface->find_codec(stream->codecpar->codec_id)
                   .value_or(av_codec);
  }
  return av_codec;
}
} // namespace

PacketDecoder::PacketDecoder(
    const Demuxer& demuxer,
    std::optional<int> stream_index,
    const StableDevice& device,
    std::optional<int> ffmpeg_thread_count) {
  AVStream* stream = demuxer.format_context()
                         ->streams[demuxer.resolve_stream_index(stream_index)];
  media_type_ = stream->codecpar->codec_type;

  bool is_audio = media_type_ == AVMEDIA_TYPE_AUDIO;
  STD_TORCH_CHECK(
      !is_audio || device.type() == kStableCPU,
      "Audio can only be decoded on the CPU.");

  device_interface_ = create_device_interface(device);
  STD_TORCH_CHECK(
      device_interface_ != nullptr,
      "Failed to create device interface. This should never happen, please report.");

  time_base_ = stream->time_base;

  is_mpeg_ps_ =
      std::string_view(demuxer.format_context()->iformat->name) == "mpeg";

  if (is_audio) {
    // Audio codecs are hardcoded to a single FFmpeg thread, see
    // https://github.com/pytorch/torchcodec/issues/1253.
    ffmpeg_thread_count = 1;
  } else if (const int32_t* matrix = get_display_matrix_from_stream(stream)) {
    display_matrix_.emplace();
    std::copy(
        matrix, matrix + display_matrix_->size(), display_matrix_->begin());
  }

  const AVCodec* av_codec = find_decoder(stream, device_interface_.get());
  codec_context_ = create_and_open_codec_context(
      stream, av_codec, device_interface_.get(), ffmpeg_thread_count);
  device_interface_->initialize(codec_context_);

  if (is_audio) {
    // Nothing else to set up: unlike video, we hand out the samples in the
    // codec's own format, so no conversion state is needed here. Note we
    // deliberately do NOT set request_sample_fmt: what SingleStreamDecoder
    // asks for (FLTP) is an optimization for its own conversion, and here it
    // would hide what the codec natively produces.
    return;
  }

  const AVPixFmtDescriptor* stream_desc =
      av_pix_fmt_desc_get(codec_context_->pix_fmt);
  int stream_bit_depth = stream_desc ? stream_desc->comp[0].depth : 8;

  VideoStreamOptions options;
  options.device = device;
  // This is ugly: what we actually mean is "let the device interface decode
  // into the native surface", which matters for NVDEC.
  // TODO_API_BREAKDOWN CC P1: Find a cleaner way to express this?
  options.output_dtype =
      stream_bit_depth > 8 ? OutputDtype::FLOAT32 : OutputDtype::UINT8;

  device_interface_->initialize_video_decoding(
      stream, demuxer.format_context(), options);
}

int PacketDecoder::send_packet(AVPacket* packet) {
  // The decode seam expects a ReferenceAVPacket. Copy a reference of the
  // caller- owned packet into a temporary one (cheap, refcount bump); the
  // temporary is unref'd on scope exit while the caller retains ownership of
  // `packet`.
  AutoAVPacket auto_packet;
  ReferenceAVPacket ref(auto_packet);
  int status = av_packet_ref(ref.get(), packet);
  STD_TORCH_CHECK(status >= AVSUCCESS, "av_packet_ref failed");

  status = device_interface_->send_packet(ref);

  if (status == AVERROR_INVALIDDATA && packet_data_may_be_misaligned_) {
    // Seeking in an MPEG program stream lands on a container-level byte offset,
    // so the parser resumes mid-frame and the packets it rebuilds are garbage
    // until it resyncs. Report those as consumed rather than as a corrupt file:
    // dropping them is exactly what resyncing means.
    return AVSUCCESS;
  }
  if (status >= AVSUCCESS) {
    // The decoder accepted a packet, so we're aligned again: from now on
    // invalid data means the file is corrupt, and we want to report it.
    packet_data_may_be_misaligned_ = false;
  }
  return status;
}

int PacketDecoder::send_eof() {
  return device_interface_->send_eof_packet();
}

void PacketDecoder::reset() {
  device_interface_->flush();
  packet_data_may_be_misaligned_ = is_mpeg_ps_;
}

int PacketDecoder::receive_frame(UniqueAVFrame& av_frame) {
  int status = device_interface_->receive_frame(av_frame);
  if (status == AVSUCCESS) {
    device_interface_->make_frame_standalone(av_frame);
    if (media_type_ == AVMEDIA_TYPE_VIDEO) {
      // Attach a copy of the display matrix to the frame, so the ColorConverter
      // can use it.
      set_display_matrix_on_frame(
          *av_frame, display_matrix_ ? display_matrix_->data() : nullptr);
    }
  }
  return status;
}

namespace {
const AVPixFmtDescriptor* get_pix_fmt_desc(const AVFrame& av_frame) {
  const AVPixFmtDescriptor* desc =
      av_pix_fmt_desc_get(static_cast<AVPixelFormat>(av_frame.format));
  STD_TORCH_CHECK(desc != nullptr, "Unknown pixel format on decoded frame");
  return desc;
}

std::string get_pix_fmt_name(const AVFrame& av_frame) {
  const char* name =
      av_get_pix_fmt_name(static_cast<AVPixelFormat>(av_frame.format));
  return name ? name : "unknown";
}
} // namespace

FrameMetadata frame_metadata(const AVFrame& av_frame) {
  const AVPixFmtDescriptor* desc = get_pix_fmt_desc(av_frame);
  const char* colorspace_name = av_color_space_name(av_frame.colorspace);
  const char* color_range_name = av_color_range_name(av_frame.color_range);

  FrameMetadata result;
  result.pix_fmt = get_pix_fmt_name(av_frame);
  result.colorspace = colorspace_name ? colorspace_name : "unknown";
  result.color_range = color_range_name ? color_range_name : "unknown";
  result.bit_depth = desc->comp[0].depth;
  result.width = av_frame.width;
  result.height = av_frame.height;
  result.rotation_degrees = get_rotation_from_frame(av_frame).value_or(0);
  return result;
}

std::vector<torch::stable::Tensor> frame_planes(
    const AVFrame& av_frame,
    const StableDevice& device,
    const torch::stable::Tensor& tensor_handle) {
  const AVPixFmtDescriptor* desc = get_pix_fmt_desc(av_frame);
  std::string fmt_name = get_pix_fmt_name(av_frame);

  STD_TORCH_CHECK(
      !(desc->flags &
        (AV_PIX_FMT_FLAG_BITSTREAM | AV_PIX_FMT_FLAG_PAL |
         AV_PIX_FMT_FLAG_FLOAT)),
      "Cannot expose ",
      fmt_name,
      " as a view: sub-byte-packed, palettised, and float pixel formats need a copy.");
  STD_TORCH_CHECK(
      desc->nb_components >= 1 && desc->nb_components <= 4,
      fmt_name,
      " has an unsupported number of components.");

  std::vector<torch::stable::Tensor> planes;
  for (int c = 0; c < desc->nb_components; ++c) {
    const AVComponentDescriptor& comp = desc->comp[c];
    int64_t bytes_per_sample = (comp.depth > 8) ? 2 : 1;
    int64_t linesize = av_frame.linesize[comp.plane];
    STD_TORCH_CHECK(
        comp.depth <= 16 && linesize > 0 && comp.step % bytes_per_sample == 0 &&
            linesize % bytes_per_sample == 0,
        "Cannot expose component ",
        c,
        " of ",
        fmt_name,
        " as a view: its samples aren't a whole number of bytes, or the frame "
        "is stored bottom-up (negative line size).");

    // Each plan is output as a 2D tensor. Y is full size while U/V (the chroma)
    // are potentially subsampled by 2.
    // Only the chroma components are subsampled; luma and alpha are full size.
    bool is_chroma = !(desc->flags & AV_PIX_FMT_FLAG_RGB) && (c == 1 || c == 2);
    int64_t height = is_chroma
        ? AV_CEIL_RSHIFT(av_frame.height, desc->log2_chroma_h)
        : av_frame.height;
    int64_t width = is_chroma
        ? AV_CEIL_RSHIFT(av_frame.width, desc->log2_chroma_w)
        : av_frame.width;

    int64_t sizes[] = {height, width};
    int64_t strides[] = {
        linesize / bytes_per_sample, comp.step / bytes_per_sample};

    // The planes are views on the AVFrame's data. The AVFrame and its data are
    // owned by the tensor_handle. We want the planes to outlive the Python
    // tensor handle (see test_planes_outlive_frame). So for each plane, we
    // create a (shallow) copy of the tensor handle, and capture it in the
    // plane's deleter. As long as a handle [copy] lives, the AVFrame and its
    // data are alive.
    torch::stable::Tensor handle_copy = tensor_handle;
    planes.push_back(torch::stable::from_blob(
        av_frame.data[comp.plane] + comp.offset,
        {sizes, 2},
        {strides, 2},
        device,
        (comp.depth > 8) ? kStableUInt16 : kStableUInt8,
        [handle_copy](void*) {}));
  }

  return planes;
}

namespace {
// Scatters `num_channels`-interleaved samples into one contiguous row per
// channel. Templated on an integer of the right width rather than the actual
// sample type: we're only moving bytes around, so all that matters is size.
template <typename T>
void deinterleave(
    const uint8_t* src,
    uint8_t* dst,
    int num_channels,
    int num_samples) {
  const T* in = reinterpret_cast<const T*>(src);
  T* out = reinterpret_cast<T*>(dst);
  for (int channel = 0; channel < num_channels; ++channel) {
    T* row = out + static_cast<int64_t>(channel) * num_samples;
    for (int sample = 0; sample < num_samples; ++sample) {
      row[sample] = in[static_cast<int64_t>(sample) * num_channels + channel];
    }
  }
}
} // namespace

torch::stable::Tensor audio_samples(const AVFrame& av_frame) {
  auto sample_format = static_cast<AVSampleFormat>(av_frame.format);
  int num_channels = get_num_channels(av_frame);
  int64_t num_samples = av_frame.nb_samples;

  torch::stable::Tensor samples = torch::stable::empty(
      {num_channels, num_samples}, sample_format_dtype(sample_format));
  if (num_samples == 0) {
    return samples;
  }

  int bytes_per_sample = av_get_bytes_per_sample(sample_format);
  auto* dst = static_cast<uint8_t*>(samples.mutable_data_ptr());
  int64_t bytes_per_channel = num_samples * bytes_per_sample;

  if (av_sample_fmt_is_planar(sample_format)) {
    for (int channel = 0; channel < num_channels; ++channel) {
      // extended_data rather than data: the latter only holds
      // AV_NUM_DATA_POINTERS (8) pointers, and we support more channels.
      std::memcpy(
          dst + channel * bytes_per_channel,
          av_frame.extended_data[channel],
          bytes_per_channel);
    }
  } else {
    const uint8_t* src = av_frame.extended_data[0];
    int num_samples_int = static_cast<int>(num_samples);
    switch (bytes_per_sample) {
      case 1:
        deinterleave<uint8_t>(src, dst, num_channels, num_samples_int);
        break;
      case 2:
        deinterleave<uint16_t>(src, dst, num_channels, num_samples_int);
        break;
      case 4:
        deinterleave<uint32_t>(src, dst, num_channels, num_samples_int);
        break;
      case 8:
        deinterleave<uint64_t>(src, dst, num_channels, num_samples_int);
        break;
      default:
        STD_TORCH_CHECK(
            false, "Unexpected sample width: ", bytes_per_sample, " bytes.");
    }
  }
  return samples;
}

} // namespace facebook::torchcodec
