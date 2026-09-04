// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// NVDEC CUDA device interface that provides direct control over NVDEC
// while keeping FFmpeg for demuxing. A lot of the logic, particularly the use
// of a cache for the decoders, is inspired by DALI's implementation which is
// APACHE 2.0:
// https://github.com/NVIDIA/DALI/blob/c7539676a24a8e9e99a6e8665e277363c5445259/dali/operators/video/frames_decoder_gpu.cc#L1
//
// NVDEC / NVCUVID docs:
// https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-video-decoder-api-prog-guide/index.html#using-nvidia-video-decoder-nvdecode-api

#pragma once

#include "CUDACommon.h"
#include "Cache.h"
#include "DeviceInterface.h"
#include "FFMPEGCommon.h"
#include "NVDECCache.h"
#include "Transform.h"
#include "color_conversion.h"

#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nvcuvid_include/cuviddec.h"
#include "nvcuvid_include/nvcuvid.h"

namespace facebook::torchcodec {
// The buffer a frame owns its samples in, hung off the AVFrame as opaque data.
struct OwnedFrameStorage {
  // Marks the point where the copy (or upload) that filled `storage` was
  // enqueued. A consumer on another stream must wait on it.
  CudaEvent frame_ready;
  torch::stable::Tensor storage;
};

struct GpuFrameAndStorage {
  UniqueAVFrame av_frame;
  torch::stable::Tensor storage;
};

class BetaCudaDeviceInterface : public DeviceInterface {
 public:
  explicit BetaCudaDeviceInterface(const StableDevice& device);
  virtual ~BetaCudaDeviceInterface();

  void initialize(const SharedAVCodecContext& codec_context) override;

  void initialize_video_decoding(
      const AVStream* av_stream,
      const UniqueDecodingAVFormatContext& av_format_ctx,
      const VideoStreamOptions& video_stream_options) override;

  void initialize_color_conversion(
      const VideoStreamOptions& video_stream_options,
      const std::vector<std::unique_ptr<Transform>>& transforms,
      const std::optional<FrameDims>& resized_output_dims) override;

  OutputDtype get_pre_allocation_dtype(
      [[maybe_unused]] OutputDtype requested_dtype) const override;

  void convert_av_frame_to_frame_output(
      const AVFrame& av_frame,
      FrameOutput& frame_output,
      std::optional<torch::stable::Tensor> pre_allocated_output_tensor)
      override;

  int send_packet(const AVPacket& packet) override;
  int send_eof_packet() override;
  int receive_frame(UniqueAVFrame& av_frame) override;
  void flush() override;

  // NVDEC callback functions (must be public for C callbacks)
  int stream_property_change(CUVIDEOFORMAT* video_format);
  int frame_ready_for_decoding(CUVIDPICPARAMS* pic_params);
  int frame_ready_in_display_order(CUVIDPARSERDISPINFO* disp_info);

  std::string get_details() override;

 private:
  enum class Mode { Uninitialized, DecoderOnly, ColorConverterOnly, Both };
  Mode mode() const;

  int send_cuvid_packet(CUVIDSOURCEDATAPACKET& cuvid_packet);

  void send_seqhdr_packet();

  void initialize_bsf(
      const AVCodecParameters* codec_par,
      const UniqueDecodingAVFormatContext& av_format_ctx);
  // Apply bitstream filter, returns filtered packet or original if no filter
  // needed.
  const AVPacket& apply_bsf(
      const AVPacket& packet,
      ReferenceAVPacket& filtered_packet);

  CUdeviceptr previously_mapped_frame_ = 0;
  void unmap_previous_frame();

  cudaStream_t nvdec_output_stream_ = nullptr;

  // Marks the point in nvdec_output_stream_ where the mapping of the
  // currently-mapped surface was enqueued. Consumers of that surface running on
  // another stream wait on it. Re-recorded by every mapping, which is safe:
  // NVDEC has a single output surface, so a frame is always consumed before the
  // next one is mapped.
  CudaEvent nvdec_surface_ready_;

  // NVDEC gives us a single output surface, so every mapped frame lives at the
  // same address and a new mapping overwrites whatever the previous frame's
  // consumer is reading. These track that read so the next mapping, in
  // receive_frame(), can be ordered after it.
  CudaEvent surface_read_done_;
  void record_surface_read(cudaStream_t stream);

  UniqueAVFrame convert_cuda_frame_to_av_frame(
      CUdeviceptr frame_ptr,
      unsigned int pitch,
      const CUVIDPARSERDISPINFO& disp_info);

  void make_frame_standalone(UniqueAVFrame& av_frame) override;

  std::optional<torch::stable::Tensor> get_frame_storage(
      const AVFrame& av_frame) const override;

  GpuFrameAndStorage upload_cpu_frame_to_gpu(
      const AVFrame& cpu_frame,
      cudaStream_t stream);

  torch::stable::Tensor copy_nvdec_surface(
      UniqueAVFrame& av_frame,
      cudaStream_t stream);

  void apply_rotation(
      FrameOutput& frame_output,
      std::optional<torch::stable::Tensor> pre_allocated_output_tensor);

  CUvideoparser video_parser_ = nullptr;
  UniqueCUvideodecoder decoder_;
  CUVIDEOFORMAT video_format_ = {};
  CUVIDEOFORMATEX parser_ext_info_ = {};

  std::queue<CUVIDPARSERDISPINFO> ready_frames_;

  // The packets flagged AV_PKT_FLAG_DISCARD must be decoded, but their frames
  // must not be returned (that's how libavcodec does it). We track the
  // timestamps of those packets and drop the corresponding frames in
  // receive_frame(). We rely on the packet's pts to identify it: it's not
  // ideal, the pts may be non-unique or missing. But that's working so far.
  // Unfortuntely, NVCUVID doesn't give us any other way to pass down that info.
  std::unordered_set<CUvideotimestamp> discarded_timestamps_;

  bool eof_sent_ = false;

  AVRational time_base_ = {0, 1};
  AVRational frame_rate_avg_from_ffmpeg_ = {0, 1};

  UniqueAVBSFContext bitstream_filter_;

  bool decoding_initialized_ = false;
  bool color_conversion_initialized_ = false;

  std::unique_ptr<DeviceInterface> cpu_interface_;
  // Whether this instance decodes on CPU because NVDEC can't handle the stream.
  bool decoding_on_cpu_ = false;
  bool nvcuvid_available_ = false;
  UniqueSwsContext sws_context_;

  SwsConfig prev_sws_config_;
  Rotation rotation_ = Rotation::NONE;
  OutputDtype output_dtype_ = OutputDtype::UINT8;
  cudaVideoSurfaceFormat surface_format_ = cudaVideoSurfaceFormat_NV12;

  CachedColorMatrix cached_color_matrix_;
};

} // namespace facebook::torchcodec

/* clang-format off */
// Note: [General design, sendPacket, receiveFrame, frame ordering and NVCUVID callbacks]
//
// At a high level, this decoding interface mimics the FFmpeg send/receive
// architecture:
// - sendPacket(AVPacket) sends an AVPacket from the FFmpeg demuxer to the
//   NVCUVID parser.
// - receiveFrame(AVFrame) is a non-blocking call:
//   - if a frame is ready **in display order**, it must return it. By display
//   order, we mean that receiveFrame() must return frames with increasing pts
//   values when called successively.
//   - if no frame is ready, it must return AVERROR(EAGAIN) to indicate the
//   caller should send more packets.
//
// The rest of this note assumes you have a reasonable level of familiarity with
// the sendPacket/receiveFrame calling pattern. If you don't, look up the core
// decoding loop in SingleVideoDecoder.
//
// The frame re-ordering problem:
// Depending on the codec and on the encoding parameters, a packet from a video
// stream may contain exactly one frame, more than one frame, or a fraction of a
// frame. And, there may be non-linear frame dependencies because of B-frames,
// which need both past *and* future frames to be decoded. Consider the
// following stream, with frames presented in display order: I0 B1 P2 B3 P4 ...
// - I0 is an I-frame (also key frame, can be decoded independently)
// - B1 is a B-frame (bi-directional) which needs both I0 and P2 to be decoded
// - P2 is a P-frame (predicted frame) which only needs I0 to be decodec.
//
// Because B1 needs both I0 and P2 to be properly decoded, the decode order
// (packet order), defined by the encoder, must be: I0 P2 B1 P4 B3 ... which is
// different from the display order.
//
// SendPacket(AVPacket)'s job is just to pass down the packet to the NVCUVID
// parser by calling cuvidParseVideoData(packet). When
// cuvidParseVideoData(packet) is called, it may trigger callbacks,
// particularly:
// - streamPropertyChange(videoFormat): triggered once at the start of the
//   stream, and possibly later if the stream properties change (e.g.
//   resolution).
// - frameReadyForDecoding(picParams)): triggered **in decode order** when the
//   parser has accumulated enough data to decode a frame. We send that frame to
//   the NVDEC hardware for **async** decoding.
// - frameReadyInDisplayOrder(dispInfo)): triggered **in display order** when a
//   frame is ready to be "displayed" (returned). At that point, the parser also
//   gives us the pts of that frame. We store (a reference to) that frame in a
//   FIFO queue: readyFrames_.
//
// When receiveFrame(AVFrame) is called, if readyFrames_ is not empty, we pop
// the front of the queue, which is the next frame in display order, and map it
// to an AVFrame by calling cuvidMapVideoFrame(). If readyFrames_ is empty we
// return EAGAIN to indicate the caller should send more packets.
//
// There is potentially a small inefficiency due to the callback design: in
// order for us to know that a frame is ready in display order, we need the
// frameReadyInDisplayOrder callback to be triggered. This can only happen
// within cuvidParseVideoData(packet) in sendPacket(). This means there may be
// the following sequence of calls:
//
// sendPacket(relevantAVPacket)
//   cuvidParseVideoData(relevantAVPacket)
//     frameReadyForDecoding()
//       cuvidDecodePicture()            Send frame to NVDEC for async decoding
//
// receiveFrame() -> EAGAIN              Frame is potentially already decoded
//                                       and could be returned, but we don't
//                                       know because frameReadyInDisplayOrder
//                                       hasn't been triggered yet. We'll only
//                                       know after sending another,
//                                       potentially irrelevant packet.
//
// sendPacket(irrelevantAVPacket)
//   cuvidParseVideoData(irrelevantAVPacket)
//     frameReadyInDisplayOrder()       Only now do we know that our target
//                                      frame is ready.
//
// receiveFrame()                       return target frame
//
// How much this matters in practice is unclear, but probably negligible in
// general. Particularly when frames are decoded consecutively anyway, the
// "irrelevantPacket" is actually relevant for a future target frame.
//
// Note that the alternative is to *not* rely on the frameReadyInDisplayOrder
// callback. It's technically possible, but it would mean we now have to solve
// two hard, *codec-dependent* problems that the callback was solving for us:
// - we have to guess the frame's pts ourselves
// - we have to re-order the frames ourselves to preserve display order.
//
/* clang-format on */
