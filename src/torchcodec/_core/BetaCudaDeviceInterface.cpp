// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <limits>
#include <map>
#include <mutex>
#include <vector>
#include "StableABICompat.h"
#include "ValidationUtils.h"

#include "BetaCudaDeviceInterface.h"

#include "DeviceInterface.h"
#include "FFMPEGCommon.h"
#include "Logging.h"
#include "NVDECCache.h"

#include "NVCUVIDRuntimeLoader.h"
#include "color_conversion.h"
#include "nvcuvid_include/cuviddec.h"
#include "nvcuvid_include/nvcuvid.h"

extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {

namespace {

// Per-device cache for cuvidGetDecoderCaps results.
// The key is a tuple of (device index, codec type, chroma format, bit depth
// minus 8).
struct DecoderCapsCache {
  using Key =
      std::tuple<int, cudaVideoCodec, cudaVideoChromaFormat, unsigned int>;
  std::map<Key, CUVIDDECODECAPS> cache;
  std::mutex mutex;

  std::pair<CUresult, CUVIDDECODECAPS> get_decoder_caps(
      int device_index,
      cudaVideoCodec codec_type,
      cudaVideoChromaFormat chroma_format,
      unsigned int bit_depth_minus8) {
    Key key{device_index, codec_type, chroma_format, bit_depth_minus8};

    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
      return {CUDA_SUCCESS, it->second};
    }

    CUVIDDECODECAPS caps = {};
    caps.eCodecType = codec_type;
    caps.eChromaFormat = chroma_format;
    caps.nBitDepthMinus8 = bit_depth_minus8;

    CUresult result = cuvidGetDecoderCaps(&caps);
    if (result == CUDA_SUCCESS) {
      cache[key] = caps;
    }
    return {result, caps};
  }
};

static DecoderCapsCache& get_decoder_caps_cache() {
  static DecoderCapsCache cache;
  return cache;
}

// NVDEC's output surface formats come in a 4:2:0 and a 4:4:4 flavour, each with
// an 8-bit and a 16-bit variant. We decode on the surface that respects the
// source chroma, but we don't respect the source bit depth and instead try to
// honor the user's requested output dtype:
// - if the user wants uint8 output, we try to decode on a uint8 surface,
//   including for >8bit sources. It's not always supported by NVDEC, so the
//   caller must fallback to the >8bit surface in such case.
// - similarly if the user wants float32 output, we try to decode on a >8bit
//   surface, including for 8bit sources. The caller must handle a similar
//   fallback.
cudaVideoSurfaceFormat get_preferred_surface_format(
    cudaVideoChromaFormat chroma_format,
    OutputDtype output_dtype) {
  bool want_uint8 = output_dtype == OutputDtype::UINT8;
  if (chroma_format == cudaVideoChromaFormat_444) {
    return want_uint8 ? cudaVideoSurfaceFormat_YUV444
                      : cudaVideoSurfaceFormat_YUV444_16Bit;
  } else {
    return want_uint8 ? cudaVideoSurfaceFormat_NV12
                      : cudaVideoSurfaceFormat_P016;
  }
}

// The AVPixelFormat describing a given surface, for a source of that bit depth.
// nvdec_pix_fmt() takes our own enum rather than the NVDEC type, so that
// FFMPEGCommon doesn't have to include the NVDEC headers.
AVPixelFormat surface_to_pix_fmt(
    cudaVideoSurfaceFormat surface_format,
    int bit_depth) {
  NvdecSurface surface = NvdecSurface::NV12;
  switch (surface_format) {
    case cudaVideoSurfaceFormat_P016:
      surface = NvdecSurface::P016;
      break;
    case cudaVideoSurfaceFormat_YUV444:
      surface = NvdecSurface::YUV444;
      break;
    case cudaVideoSurfaceFormat_YUV444_16Bit:
      surface = NvdecSurface::YUV444_16Bit;
      break;
    default:
      break;
  }
  return nvdec_pix_fmt(surface, bit_depth);
}

bool is_444_surface_format(cudaVideoSurfaceFormat format) {
  return format == cudaVideoSurfaceFormat_YUV444 ||
      format == cudaVideoSurfaceFormat_YUV444_16Bit;
}

bool is_16bit_surface_format(cudaVideoSurfaceFormat format) {
  return format == cudaVideoSurfaceFormat_P016 ||
      format == cudaVideoSurfaceFormat_YUV444_16Bit;
}

bool is_expected_pix_fmt_from_nvdec(AVPixelFormat pix_fmt) {
  return pix_fmt == AV_PIX_FMT_NV12 || is_nvdec_16bit_pix_fmt(pix_fmt) ||
      pix_fmt == AV_PIX_FMT_YUV444P || pix_fmt == AV_PIX_FMT_YUV444P16LE;
}

static bool g_cuda_nvdec = register_device_interface(
    DeviceInterfaceKey(kStableCUDA, /*variant=*/"default"),
    [](const StableDevice& device) {
      return new BetaCudaDeviceInterface(device);
    });

static int CUDAAPI
pfn_sequence_callback(void* p_user_data, CUVIDEOFORMAT* video_format) {
  auto decoder = static_cast<BetaCudaDeviceInterface*>(p_user_data);
  return decoder->stream_property_change(video_format);
}

static int CUDAAPI
pfn_decode_picture_callback(void* p_user_data, CUVIDPICPARAMS* pic_params) {
  auto decoder = static_cast<BetaCudaDeviceInterface*>(p_user_data);
  return decoder->frame_ready_for_decoding(pic_params);
}

static int CUDAAPI pfn_display_picture_callback(
    void* p_user_data,
    CUVIDPARSERDISPINFO* disp_info) {
  auto decoder = static_cast<BetaCudaDeviceInterface*>(p_user_data);
  return decoder->frame_ready_in_display_order(disp_info);
}

static UniqueCUvideodecoder create_decoder(
    CUVIDEOFORMAT* video_format,
    cudaVideoSurfaceFormat surface_format) {
  // Decoder creation parameters, most are taken from DALI
  CUVIDDECODECREATEINFO decoder_params = {};
  decoder_params.bitDepthMinus8 = video_format->bit_depth_luma_minus8;
  decoder_params.ChromaFormat = video_format->chroma_format;
  decoder_params.OutputFormat = surface_format;
  decoder_params.ulCreationFlags = cudaVideoCreate_Default;
  decoder_params.CodecType = video_format->codec;
  decoder_params.ulHeight = video_format->coded_height;
  decoder_params.ulWidth = video_format->coded_width;
  decoder_params.ulMaxHeight = video_format->coded_height;
  decoder_params.ulMaxWidth = video_format->coded_width;
  decoder_params.ulTargetHeight =
      video_format->display_area.bottom - video_format->display_area.top;
  decoder_params.ulTargetWidth =
      video_format->display_area.right - video_format->display_area.left;
  decoder_params.ulNumDecodeSurfaces = video_format->min_num_decode_surfaces;
  // We should only ever need 1 output surface, since we process frames
  // sequentially, and we always unmap the previous frame before mapping a new
  // one.
  // TODONVDEC P3: set this to 2, allow for 2 frames to be mapped at a time, and
  // benchmark to see if this makes any difference.
  decoder_params.ulNumOutputSurfaces = 1;
  decoder_params.display_area.left = video_format->display_area.left;
  decoder_params.display_area.right = video_format->display_area.right;
  decoder_params.display_area.top = video_format->display_area.top;
  decoder_params.display_area.bottom = video_format->display_area.bottom;

  CUvideodecoder* decoder = new CUvideodecoder();
  CUresult result = cuvidCreateDecoder(decoder, &decoder_params);
  STD_TORCH_CHECK(
      result == CUDA_SUCCESS, "Failed to create NVDEC decoder: ", result);
  return UniqueCUvideodecoder(decoder, CUvideoDecoderDeleter{});
}

std::optional<cudaVideoChromaFormat> validate_chroma_support(
    const AVPixFmtDescriptor* desc) {
  // Return the corresponding cudaVideoChromaFormat if supported, std::nullopt
  // otherwise.
  STD_TORCH_CHECK(desc != nullptr, "desc can't be null");

  if (desc->nb_components == 1) {
    return cudaVideoChromaFormat_Monochrome;
  } else if (desc->nb_components >= 3 && !(desc->flags & AV_PIX_FMT_FLAG_RGB)) {
    // Make sure it's YUV: has chroma planes and isn't RGB
    if (desc->log2_chroma_w == 0 && desc->log2_chroma_h == 0) {
      return cudaVideoChromaFormat_444; // 1x1 subsampling = 4:4:4
    } else if (desc->log2_chroma_w == 1 && desc->log2_chroma_h == 1) {
      return cudaVideoChromaFormat_420; // 2x2 subsampling = 4:2:0
    } else if (desc->log2_chroma_w == 1 && desc->log2_chroma_h == 0) {
      return cudaVideoChromaFormat_422; // 2x1 subsampling = 4:2:2
    }
  }

  return std::nullopt;
}

std::optional<cudaVideoCodec> validate_codec_support(AVCodecID codec_id) {
  // Return the corresponding cudaVideoCodec if supported, std::nullopt
  // otherwise
  // Note that we currently return nullopt (and thus fallback to CPU) for some
  // codecs that are technically supported by NVDEC, see comment below.
  switch (codec_id) {
    case AV_CODEC_ID_H264:
      return cudaVideoCodec_H264;
    case AV_CODEC_ID_HEVC:
      return cudaVideoCodec_HEVC;
    case AV_CODEC_ID_AV1:
      return cudaVideoCodec_AV1;
    case AV_CODEC_ID_VP9:
      return cudaVideoCodec_VP9;
    case AV_CODEC_ID_VP8:
      return cudaVideoCodec_VP8;
    case AV_CODEC_ID_MPEG4:
      return cudaVideoCodec_MPEG4;
    // Formats below are currently not tested, but they should "mostly" work.
    // MPEG1 was briefly locally tested and it was ok-ish despite duration being
    // off. Since they're far less popular, we keep them disabled by default but
    // we can consider enabling them upon user requests.
    // case AV_CODEC_ID_MPEG1VIDEO:
    //   return cudaVideoCodec_MPEG1;
    // case AV_CODEC_ID_MPEG2VIDEO:
    //   return cudaVideoCodec_MPEG2;
    // case AV_CODEC_ID_MJPEG:
    //   return cudaVideoCodec_JPEG;
    // case AV_CODEC_ID_VC1:
    //   return cudaVideoCodec_VC1;
    default:
      return std::nullopt;
  }
}

std::optional<cudaVideoSurfaceFormat> get_nvdec_surface_format(
    const StableDevice& device,
    const SharedAVCodecContext& codec_context,
    OutputDtype output_dtype) {
  // Return the surface format to use for NVDEC decoding if the stream is
  // supported, or nullopt to fall back to CPU.

  auto codec_type = validate_codec_support(codec_context->codec_id);
  if (!codec_type.has_value()) {
    return std::nullopt;
  }

  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(codec_context->pix_fmt);
  if (!desc) {
    return std::nullopt;
  }

  auto chroma_format = validate_chroma_support(desc);
  if (!chroma_format.has_value()) {
    return std::nullopt;
  }

  auto bit_depth_minus8 = static_cast<unsigned int>(desc->comp[0].depth - 8);
  auto [result, caps] = get_decoder_caps_cache().get_decoder_caps(
      get_device_index(device),
      codec_type.value(),
      chroma_format.value(),
      bit_depth_minus8);
  if (result != CUDA_SUCCESS) {
    return std::nullopt;
  }

  if (!caps.bIsSupported) {
    return std::nullopt;
  }

  auto coded_width = static_cast<unsigned int>(codec_context->coded_width);
  auto coded_height = static_cast<unsigned int>(codec_context->coded_height);
  if (coded_width < static_cast<unsigned int>(caps.nMinWidth) ||
      coded_height < static_cast<unsigned int>(caps.nMinHeight) ||
      coded_width > caps.nMaxWidth || coded_height > caps.nMaxHeight) {
    return std::nullopt;
  }

  // See nMaxMBCount in cuviddec.h
  constexpr unsigned int macroblock_constant = 256;
  if (coded_width * coded_height / macroblock_constant > caps.nMaxMBCount) {
    return std::nullopt;
  }

  auto preferred_format =
      get_preferred_surface_format(chroma_format.value(), output_dtype);

  auto is_supported = [&](cudaVideoSurfaceFormat format) {
    return ((caps.nOutputFormatMask >> format) & 1) != 0;
  };

  if (is_supported(preferred_format)) {
    return preferred_format;
  }

  // The preferred_format heuristic tries to take a shortcut that might cause us
  // to miss valid formats. We fallabck here:
  // if source is 8bit we can try the 8bit surface.
  // if surface is 8bit we can try the 16bit surface.

  bool source_is_8_bits = bit_depth_minus8 == 0;
  if (is_16bit_surface_format(preferred_format) && source_is_8_bits) {
    auto narrower = preferred_format == cudaVideoSurfaceFormat_YUV444_16Bit
        ? cudaVideoSurfaceFormat_YUV444
        : cudaVideoSurfaceFormat_NV12;

    if (is_supported(narrower)) {
      return narrower;
    }
  }
  if (!is_16bit_surface_format(preferred_format)) {
    auto wider = preferred_format == cudaVideoSurfaceFormat_YUV444
        ? cudaVideoSurfaceFormat_YUV444_16Bit
        : cudaVideoSurfaceFormat_P016;

    if (is_supported(wider)) {
      return wider;
    }
  }

  return std::nullopt;
}

void standalone_frame_free_callback(
    [[maybe_unused]] void* opaque,
    uint8_t* data) {
  delete reinterpret_cast<OwnedFrameStorage*>(data);
}

class CudaContextGuard {
  // There's one CUDA context per process per device. But new threads aren't
  // bound to a context. The binding often happens automatically when calling
  // CUDA APIs (like cudaFree), but some APIs like the NVCUVID ones that we use
  // here aren't automatically binding.
  // So for a thread to be able to use NVCUVID APIs, it must have a context
  // bound to it, and we have to enforce that binding manually.
  // That's what this guard does: it calls cudaFree(nullptr), which is a common
  // near-free way to force the CUDA runtime to bind the context for the current
  // thread. And this call must happen within a device guard to make sure we're
  // binding the context of the device this interface is using.
  // We must call this guard in every public method of the interface that uses
  // NVCUVID APIs, because these methods can, in theory, be called from any
  // thread.
  // Note that none of this was an issue before when our only entry-point was
  // the SingleStreamDecoder: all the entry-points were called from the same
  // thread. Now that we have split the APIs in different blocks (PacketDecoder,
  // ColorConverter), each of these blocks can be on different threads - and
  // importantly, they can be created in the main thread (where the context is
  // bound by our call to initialize_cuda_context_with_pytorch()), but then used
  // in a different thread that doesn't have the context.
 public:
  explicit CudaContextGuard(int device_index) : device_guard_(device_index) {
    cudaFree(nullptr);
  }

 private:
  StableDeviceGuard device_guard_;
};

} // namespace

BetaCudaDeviceInterface::BetaCudaDeviceInterface(const StableDevice& device)
    : DeviceInterface(device) {
  STD_TORCH_CHECK(g_cuda_nvdec, "NvdecCudaDeviceInterface was not registered!");
  STD_TORCH_CHECK(
      device_.type() == kStableCUDA, "Unsupported device: must be CUDA");

  device_ = StableDevice(kStableCUDA, get_device_index(device_));

  // Note: now that we have the CudaContextGuard, we might not need to do that
  // anymore. The comment says we need pytorch to create the context - maybe
  // that's true, but that's a very old comment now.
  initialize_cuda_context_with_pytorch(device_);

  nvcuvid_available_ = load_nvcuvid_library();
}

BetaCudaDeviceInterface::Mode BetaCudaDeviceInterface::mode() const {
  if (decoding_initialized_ && color_conversion_initialized_) {
    return Mode::Both;
  } else if (decoding_initialized_) {
    return Mode::DecoderOnly;
  } else if (color_conversion_initialized_) {
    return Mode::ColorConverterOnly;
  }
  return Mode::Uninitialized;
}

void BetaCudaDeviceInterface::initialize_color_conversion(
    const VideoStreamOptions& video_stream_options,
    [[maybe_unused]] const std::vector<std::unique_ptr<Transform>>& transforms,
    [[maybe_unused]] const std::optional<FrameDims>& resized_output_dims) {
  output_dtype_ = video_stream_options.output_dtype;
  color_conversion_initialized_ = true;
}

void BetaCudaDeviceInterface::initialize_video_decoding(
    const AVStream* av_stream,
    const UniqueDecodingAVFormatContext& av_format_ctx,
    const VideoStreamOptions& video_stream_options) {
  STD_TORCH_CHECK(av_stream != nullptr, "AVStream cannot be null");
  CudaContextGuard context_guard(device_.index());
  decoding_initialized_ = true;
  rotation_ = rotation_from_degrees(get_rotation_from_stream(av_stream));
  output_dtype_ = video_stream_options.output_dtype;

  auto maybe_surface_format = nvcuvid_available_
      ? get_nvdec_surface_format(device_, codec_context_, output_dtype_)
      : std::nullopt;

  if (!maybe_surface_format.has_value()) {
    if (!nvcuvid_available_) {
      TC_LOG("NVCUVID library not available; falling back to CPU decoding.");
    } else {
      TC_LOG(
          "Video stream not supported by NVDEC; falling back to CPU decoding.");
    }
    decoding_on_cpu_ = true;
    cpu_interface_ = create_device_interface(kStableCPU);
    STD_TORCH_CHECK(
        cpu_interface_ != nullptr, "Failed to create CPU device interface");
    cpu_interface_->initialize(codec_context_);
    cpu_interface_->initialize_video_decoding(
        av_stream, av_format_ctx, video_stream_options);
    return;
  }

  surface_format_ = maybe_surface_format.value();
  time_base_ = av_stream->time_base;
  frame_rate_avg_from_ffmpeg_ = av_stream->r_frame_rate;

  const AVCodecParameters* codec_par = av_stream->codecpar;
  STD_TORCH_CHECK(codec_par != nullptr, "CodecParameters cannot be null");

  initialize_bsf(codec_par, av_format_ctx);

  // Create parser. Default values that aren't obvious are taken from DALI.
  CUVIDPARSERPARAMS parser_params = {};
  auto codec_type = validate_codec_support(codec_par->codec_id);
  STD_TORCH_CHECK(
      codec_type.has_value(),
      "This should never happen, we should be using the CPU fallback by now. "
      "Please report a bug.");
  parser_params.CodecType = codec_type.value();
  parser_params.ulMaxNumDecodeSurfaces = 8;
  parser_params.ulMaxDisplayDelay = 0;
  // Callback setup, all are triggered by the parser within a call
  // to cuvidParseVideoData
  parser_params.pUserData = this;
  parser_params.pfnSequenceCallback = pfn_sequence_callback;
  parser_params.pfnDecodePicture = pfn_decode_picture_callback;
  parser_params.pfnDisplayPicture = pfn_display_picture_callback;

  // Some containers (e.g. MP4/MOV) store codec config (H.264 SPS/PPS,
  // MPEG-4 VOS/VOL, etc.) in extradata rather than inline in the
  // bitstream. The NVCUVID parser needs this data to initialize, so we
  // pass it via pExtVideoInfo. Same approach as DALI and FFmpeg cuviddec.
  // DALI does the same thing
  // https://github.com/NVIDIA/DALI/blob/ae79f316ae9b14c464d9cb98465f7f783da9ea89/dali/operators/video/frames_decoder_gpu.cc#L402-L408
  //
  const uint8_t* seqhdr = codec_par->extradata;
  int seqhdr_data_size = codec_par->extradata_size;
  if (bitstream_filter_ && bitstream_filter_->par_out->extradata_size > 0) {
    // when a BSF is used (e.g. h264_mp4toannexb), we must pass the
    // *filtered* extradata!
    seqhdr = bitstream_filter_->par_out->extradata;
    seqhdr_data_size = bitstream_filter_->par_out->extradata_size;
  }
  if (seqhdr_data_size > 0) {
    auto seqhdr_size = std::min(
        static_cast<size_t>(seqhdr_data_size),
        sizeof(parser_ext_info_.raw_seqhdr_data));
    parser_ext_info_.format.seqhdr_data_length = seqhdr_size;
    memcpy(parser_ext_info_.raw_seqhdr_data, seqhdr, seqhdr_size);
    parser_params.pExtVideoInfo = &parser_ext_info_;
  }

  CUresult result = cuvidCreateVideoParser(&video_parser_, &parser_params);
  STD_TORCH_CHECK(
      result == CUDA_SUCCESS, "Failed to create video parser: ", result);

  send_seqhdr_packet();
}

void BetaCudaDeviceInterface::send_seqhdr_packet() {
  // This must be called at parser initialization, and after each flush.
  // See https://github.com/meta-pytorch/torchcodec/pull/1503 for details.
  // FFmpeg's nvcuviddec.c does the same thing (not the nvdec.c one, because it
  // doesn't rely on the nvcuvid parser):
  // -
  // https://github.com/FFmpeg/FFmpeg/blob/d244d438c372b76d825be4527fccfd162429010a/libavcodec/cuviddec.c#L1211
  // -
  // https://github.com/FFmpeg/FFmpeg/blob/d244d438c372b76d825be4527fccfd162429010a/libavcodec/cuviddec.c#L1157
  if (parser_ext_info_.format.seqhdr_data_length == 0) {
    return;
  }
  CUVIDSOURCEDATAPACKET seq_pkt = {};
  seq_pkt.payload = parser_ext_info_.raw_seqhdr_data;
  seq_pkt.payload_size = parser_ext_info_.format.seqhdr_data_length;
  send_cuvid_packet(seq_pkt);
}

BetaCudaDeviceInterface::~BetaCudaDeviceInterface() {
  CudaContextGuard context_guard(device_.index());
  if (decoder_) {
    // DALI doesn't seem to do any particular cleanup of the decoder before
    // sending it to the cache, so we probably don't need to do anything either.
    // Just to be safe, we flush.
    // What happens to those decode surfaces that haven't yet been mapped is
    // unclear.
    flush();
    // Whoever picks this decoder up from the cache will map its output surface
    // and overwrite it. We block the host until the last consumer is done
    // reading.
    surface_read_done_.synchronize();
    unmap_previous_frame();
    NVDECCache::get_cache(device_).return_decoder(
        &video_format_, surface_format_, std::move(decoder_));
  }

  if (video_parser_) {
    cuvidDestroyVideoParser(video_parser_);
    video_parser_ = nullptr;
  }
}

void BetaCudaDeviceInterface::initialize(
    const SharedAVCodecContext& codec_context) {
  codec_context_ = codec_context;
}

void BetaCudaDeviceInterface::initialize_bsf(
    const AVCodecParameters* codec_par,
    const UniqueDecodingAVFormatContext& av_format_ctx) {
  // Setup bit stream filters (BSF):
  // https://ffmpeg.org/doxygen/7.0/group__lavc__bsf.html
  // This is only needed for some formats, like H264 or HEVC.

  STD_TORCH_CHECK(codec_par != nullptr, "codecPar cannot be null");
  STD_TORCH_CHECK(av_format_ctx != nullptr, "AVFormatContext cannot be null");
  STD_TORCH_CHECK(
      av_format_ctx->iformat != nullptr,
      "AVFormatContext->iformat cannot be null");
  std::string filter_name;

  // Matching logic is taken from DALI
  switch (codec_par->codec_id) {
    case AV_CODEC_ID_H264: {
      const std::string format_name = av_format_ctx->iformat->long_name
          ? av_format_ctx->iformat->long_name
          : "";

      if (format_name == "QuickTime / MOV" ||
          format_name == "FLV (Flash Video)" ||
          format_name == "Matroska / WebM" ||
          format_name == "raw H.264 video") {
        filter_name = "h264_mp4toannexb";
      }
      break;
    }

    case AV_CODEC_ID_HEVC: {
      const std::string format_name = av_format_ctx->iformat->long_name
          ? av_format_ctx->iformat->long_name
          : "";

      if (format_name == "QuickTime / MOV" ||
          format_name == "FLV (Flash Video)" ||
          format_name == "Matroska / WebM" || format_name == "raw HEVC video") {
        filter_name = "hevc_mp4toannexb";
      }
      break;
    }
    case AV_CODEC_ID_MPEG4: {
      const std::string format_name =
          av_format_ctx->iformat->name ? av_format_ctx->iformat->name : "";
      if (format_name == "avi") {
        filter_name = "mpeg4_unpack_bframes";
      }
      break;
    }

    default:
      // No bitstream filter needed for other codecs
      break;
  }

  if (filter_name.empty()) {
    // Only initialize BSF if we actually need one
    return;
  }

  const AVBitStreamFilter* av_bsf = av_bsf_get_by_name(filter_name.c_str());
  STD_TORCH_CHECK(
      av_bsf != nullptr, "Failed to find bitstream filter: ", filter_name);

  AVBSFContext* av_bsf_context = nullptr;
  int ret_val = av_bsf_alloc(av_bsf, &av_bsf_context);
  STD_TORCH_CHECK(
      ret_val >= AVSUCCESS,
      "Failed to allocate bitstream filter: ",
      get_ffmpeg_error_string_from_error_code(ret_val));

  bitstream_filter_.reset(av_bsf_context);

  ret_val = avcodec_parameters_copy(bitstream_filter_->par_in, codec_par);
  STD_TORCH_CHECK(
      ret_val >= AVSUCCESS,
      "Failed to copy codec parameters: ",
      get_ffmpeg_error_string_from_error_code(ret_val));

  ret_val = av_bsf_init(bitstream_filter_.get());
  STD_TORCH_CHECK(
      ret_val == AVSUCCESS,
      "Failed to initialize bitstream filter: ",
      get_ffmpeg_error_string_from_error_code(ret_val));
}

// This callback is called by the parser within cuvidParseVideoData when there
// is a change in the stream's properties (like resolution change), as specified
// by CUVIDEOFORMAT. Particularly (but not just!), this is called at the very
// start of the stream.
// TODONVDEC P1: Code below mostly assume this is called only once at the start,
// we should handle the case of multiple calls. Probably need to flush buffers,
// etc.
int BetaCudaDeviceInterface::stream_property_change(
    CUVIDEOFORMAT* video_format) {
  STD_TORCH_CHECK(video_format != nullptr, "Invalid video format");

  video_format_ = *video_format;

  if (video_format_.min_num_decode_surfaces == 0) {
    // Same as DALI's fallback
    video_format_.min_num_decode_surfaces = 20;
  }

  if (!decoder_) {
    decoder_ = NVDECCache::get_cache(device_).get_decoder(
        video_format, surface_format_);

    if (!decoder_) {
      // TODONVDEC P2: consider re-configuring an existing decoder instead of
      // re-creating one. See docs, see DALI. Re-configuration doesn't seem to
      // be enabled in DALI by default.
      decoder_ = create_decoder(video_format, surface_format_);
    }

    STD_TORCH_CHECK(decoder_, "Failed to get or create decoder");
  }

  // DALI also returns min_num_decode_surfaces from this function. This
  // instructs the parser to reset its ulMaxNumDecodeSurfaces field to this
  // value.
  return static_cast<int>(video_format_.min_num_decode_surfaces);
}

// Moral equivalent of avcodec_send_packet(). Here, we pass the AVPacket down to
// the NVCUVID parser.
int BetaCudaDeviceInterface::send_packet(ReferenceAVPacket& packet) {
  CudaContextGuard context_guard(device_.index());
  if (decoding_on_cpu_) {
    return cpu_interface_->send_packet(packet);
  }

  STD_TORCH_CHECK(
      packet.get() && packet->data && packet->size > 0,
      "sendPacket received an empty packet, this is unexpected, please report.");

  // Apply BSF if needed. We want applyBSF to return a *new* filtered packet, or
  // the original one if no BSF is needed. This new filtered packet must be
  // allocated outside of applyBSF: if it were allocated inside applyBSF, it
  // would be destroyed at the end of the function, leaving us with a dangling
  // reference.
  AutoAVPacket filtered_auto_packet;
  ReferenceAVPacket filtered_packet(filtered_auto_packet);
  ReferenceAVPacket& packet_to_send = apply_bsf(packet, filtered_packet);

  CUVIDSOURCEDATAPACKET cuvid_packet = {};
  cuvid_packet.payload = packet_to_send->data;
  cuvid_packet.payload_size = packet_to_send->size;
  cuvid_packet.flags = CUVID_PKT_TIMESTAMP;
  cuvid_packet.timestamp = packet_to_send->pts;

  if (packet_to_send->flags & AV_PKT_FLAG_DISCARD) {
    discarded_timestamps_.insert(cuvid_packet.timestamp);
  }

  return send_cuvid_packet(cuvid_packet);
}

int BetaCudaDeviceInterface::send_eof_packet() {
  CudaContextGuard context_guard(device_.index());
  if (decoding_on_cpu_) {
    return cpu_interface_->send_eof_packet();
  }

  CUVIDSOURCEDATAPACKET cuvid_packet = {};
  cuvid_packet.flags = CUVID_PKT_ENDOFSTREAM;
  eof_sent_ = true;

  return send_cuvid_packet(cuvid_packet);
}

int BetaCudaDeviceInterface::send_cuvid_packet(
    CUVIDSOURCEDATAPACKET& cuvid_packet) {
  CUresult result = cuvidParseVideoData(video_parser_, &cuvid_packet);
  return result == CUDA_SUCCESS ? AVSUCCESS : AVERROR_EXTERNAL;
}

ReferenceAVPacket& BetaCudaDeviceInterface::apply_bsf(
    ReferenceAVPacket& packet,
    ReferenceAVPacket& filtered_packet) {
  if (!bitstream_filter_) {
    return packet;
  }

  int ret_val = av_bsf_send_packet(bitstream_filter_.get(), packet.get());
  STD_TORCH_CHECK(
      ret_val >= AVSUCCESS,
      "Failed to send packet to bitstream filter: ",
      get_ffmpeg_error_string_from_error_code(ret_val));

  // TODO P1: the docs mention there can theoretically be multiple output
  // packets for a single input, i.e. we may need to call av_bsf_receive_packet
  // more than once. We should figure out whether that applies to the BSF we're
  // using.
  ret_val =
      av_bsf_receive_packet(bitstream_filter_.get(), filtered_packet.get());
  STD_TORCH_CHECK(
      ret_val >= AVSUCCESS,
      "Failed to receive packet from bitstream filter: ",
      get_ffmpeg_error_string_from_error_code(ret_val));

  return filtered_packet;
}

// Parser triggers this callback within cuvidParseVideoData when a frame is
// ready to be decoded, i.e. the parser received all the necessary packets for a
// given frame. It means we can send that frame to be decoded by the hardware
// NVDEC decoder by calling cuvidDecodePicture.
int BetaCudaDeviceInterface::frame_ready_for_decoding(
    CUVIDPICPARAMS* pic_params) {
  STD_TORCH_CHECK(pic_params != nullptr, "Invalid picture parameters");
  STD_TORCH_CHECK(decoder_, "Decoder not initialized before picture decode");
  // Send frame to be decoded by NVDEC. This may or may not block, depending on
  // the internal state of the NVDEC. Presumably, when it blocks, it gets
  // automatically unblocked once a frame has been decoded, although how and
  // when it happens is unclear. The docs say:
  // > cuvidDecodePicture() will stall if wait queue on NVDEC inside driver is
  //   full.
  // and cuviddec.h says:
  // > cuvidDecodePicture may block the calling thread if there are too many
  //   pictures pending in the decode queue.
  CUresult result = cuvidDecodePicture(*decoder_.get(), pic_params);

  // Yes, you're reading that right, 0 means error, 1 means success
  return (result == CUDA_SUCCESS);
}

int BetaCudaDeviceInterface::frame_ready_in_display_order(
    CUVIDPARSERDISPINFO* disp_info) {
  ready_frames_.push(*disp_info);
  return 1; // success
}

// Moral equivalent of avcodec_receive_frame().
int BetaCudaDeviceInterface::receive_frame(UniqueAVFrame& av_frame) {
  CudaContextGuard context_guard(device_.index());
  if (decoding_on_cpu_) {
    return cpu_interface_->receive_frame(av_frame);
  }

  // Drop those packets that were marked as discard. We only wanted to decode
  // those, not to return them.
  while (!ready_frames_.empty() &&
         discarded_timestamps_.erase(ready_frames_.front().timestamp) > 0) {
    ready_frames_.pop();
  }

  if (ready_frames_.empty()) {
    // No frame found, instruct caller to try again later after sending more
    // packets, or to stop if EOF was already sent.
    return eof_sent_ ? AVERROR_EOF : AVERROR(EAGAIN);
  }

  CUVIDPARSERDISPINFO disp_info = ready_frames_.front();
  ready_frames_.pop();

  CUVIDPROCPARAMS proc_params = {};
  proc_params.progressive_frame = disp_info.progressive_frame;
  proc_params.top_field_first = disp_info.top_field_first;
  proc_params.unpaired_field = disp_info.repeat_first_field < 0;
  // We set the NVDEC stream to the current stream, and remember it: consumers
  // of the mapped surface run later and possibly on a different stream, so they
  // need to know which stream produces the surface's content in order to wait
  // on it.
  // Re types: we get a cudaStream_t from PyTorch but it's interchangeable with
  // CUstream
  nvdec_output_stream_ = get_current_cuda_stream(device_.index());
  proc_params.output_stream = reinterpret_cast<CUstream>(nvdec_output_stream_);

  CUdeviceptr frame_ptr = 0;
  unsigned int pitch = 0;

  // We know the frame we want was sent to the hardware decoder, but now we need
  // to "map" it to an "output surface" before we can use its data. This is a
  // blocking calls that waits until the frame is fully decoded and ready to be
  // used.
  // When a frame is mapped to an output surface, it needs to be unmapped
  // eventually, so that the decoder can re-use the output surface. Failing to
  // unmap will cause map to eventually fail. DALI unmaps frames almost
  // immediately  after mapping them: they do the color-conversion in-between,
  // which involves a copy of the data, so that works.
  // We, OTOH, will do the color-conversion later, outside of receive_frame().
  // So we unmap here: just before mapping a new frame. At that point the
  // previously-mapped frame has been consumed, or at least its consumption has
  // been enqueued:
  // - With SingleStreamDecoder, that frame was either color-converted (with a
  //   copy), or that's a frame that was discarded in SingleStreamDecoder.
  // - With the "Blocks" APIs, the PacketDecoder forces a copy in
  //   make_frame_standalone().
  // Those reads are asynchronous, so we must wait on them to finish.
  surface_read_done_.make_stream_wait(nvdec_output_stream_);
  unmap_previous_frame();
  CUresult result = cuvidMapVideoFrame(
      *decoder_.get(),
      disp_info.picture_index,
      &frame_ptr,
      &pitch,
      &proc_params);
  if (result != CUDA_SUCCESS) {
    return AVERROR_EXTERNAL;
  }
  previously_mapped_frame_ = frame_ptr;

  nvdec_surface_ready_.record(nvdec_output_stream_);

  av_frame = convert_cuda_frame_to_av_frame(frame_ptr, pitch, disp_info);

  return AVSUCCESS;
}

void BetaCudaDeviceInterface::record_surface_read(cudaStream_t stream) {
  // Called by every consumer of the mapped surface, once its read has been
  // enqueued on `stream`.
  // This sets the surface_read_done_ event that must be waited upon before
  // mapping a new frame on the surface.
  surface_read_done_.record(stream);
}

void BetaCudaDeviceInterface::unmap_previous_frame() {
  if (previously_mapped_frame_ == 0) {
    return;
  }
  CUresult result =
      cuvidUnmapVideoFrame(*decoder_.get(), previously_mapped_frame_);
  STD_TORCH_CHECK(
      result == CUDA_SUCCESS, "Failed to unmap previous frame: ", result);
  previously_mapped_frame_ = 0;
}

UniqueAVFrame BetaCudaDeviceInterface::convert_cuda_frame_to_av_frame(
    CUdeviceptr frame_ptr,
    unsigned int pitch,
    const CUVIDPARSERDISPINFO& disp_info) {
  STD_TORCH_CHECK(frame_ptr != 0, "Invalid CUDA frame pointer");

  // Get frame dimensions from video format display area (not coded dimensions)
  // This matches DALI's approach and avoids padding issues
  int width =
      video_format_.display_area.right - video_format_.display_area.left;
  int height =
      video_format_.display_area.bottom - video_format_.display_area.top;

  STD_TORCH_CHECK(width > 0 && height > 0, "Invalid frame dimensions");
  STD_TORCH_CHECK(
      pitch >= static_cast<unsigned int>(width), "Pitch must be >= width");

  UniqueAVFrame av_frame(av_frame_alloc());
  STD_TORCH_CHECK(av_frame.get() != nullptr, "Failed to allocate AVFrame");

  av_frame->width = width;
  av_frame->height = height;
  av_frame->format = surface_to_pix_fmt(
      surface_format_,
      static_cast<int>(video_format_.bit_depth_luma_minus8) + 8);
  av_frame->pts = disp_info.timestamp;

  // TODONVDEC P2: We compute the duration based on average frame rate info, so
  // so if the video has variable frame rate, the durations may be off. We
  // should try to see if we can set the duration more accurately. Unfortunately
  // it's not given by dispInfo. One option would be to set it based on the pts
  // difference between consecutive frames, if the next frame is already
  // available.
  // Note that we used to rely on videoFormat_.frame_rate for this, but that
  // proved less accurate than FFmpeg.
  set_duration(
      *av_frame,
      compute_safe_duration(frame_rate_avg_from_ffmpeg_, time_base_));

  // We need to assign the frame colorspace. This is crucial for proper color
  // conversion. NVCUVID stores that in the matrix_coefficients field, but
  // doesn't document the semantics of the values. Claude code generated this,
  // which seems to work. Reassuringly, the values seem to match the
  // corresponding indices in the FFmpeg enum for colorspace conversion
  // (ff_yuv2rgb_coeffs):
  // https://ffmpeg.org/doxygen/trunk/yuv2rgb_8c_source.html#l00047
  switch (video_format_.video_signal_description.matrix_coefficients) {
    case 1:
      av_frame->colorspace = AVCOL_SPC_BT709;
      break;
    case 6:
      av_frame->colorspace = AVCOL_SPC_SMPTE170M; // BT.601
      break;
    case 9:
      av_frame->colorspace = AVCOL_SPC_BT2020_NCL;
      break;
    case 10:
      av_frame->colorspace = AVCOL_SPC_BT2020_CL;
      break;
    default:
      // Default to BT.601
      av_frame->colorspace = AVCOL_SPC_SMPTE170M;
      break;
  }

  av_frame->color_range =
      video_format_.video_signal_description.video_full_range_flag
      ? AVCOL_RANGE_JPEG
      : AVCOL_RANGE_MPEG;

  // NVDEC stacks the planes in a single allocation, all with the same pitch,
  // and it rounds the Y plane's row count up to even. So consecutive planes
  // start plane_stride bytes apart, which is more than pitch * height for an
  // odd-height frame. NVIDIA's own NvDecoder addresses the chroma plane the
  // same way: dpSrcFrame + srcPitch * ((surface_height + 1) & ~1).
  unsigned int num_luma_plane_rows = round_up_to_even(height);
  unsigned int plane_stride = pitch * num_luma_plane_rows;
  auto plane = [&](unsigned int index) {
    return reinterpret_cast<uint8_t*>(frame_ptr + (plane_stride * index));
  };
  bool is_444 = is_444_surface_format(surface_format_);

  av_frame->data[0] = plane(0);
  av_frame->data[1] = plane(1);
  av_frame->data[2] = is_444 ? plane(2) : nullptr;
  av_frame->data[3] = nullptr;
  STD_TORCH_CHECK(
      pitch <= static_cast<unsigned int>(std::numeric_limits<int>::max()),
      "NVDEC returned a pitch of ",
      pitch,
      " bytes, which doesn't fit in an AVFrame line size. This should never "
      "happen, please report.");
  av_frame->linesize[0] = static_cast<int>(pitch);
  av_frame->linesize[1] = static_cast<int>(pitch);
  av_frame->linesize[2] = is_444 ? static_cast<int>(pitch) : 0;
  av_frame->linesize[3] = 0;

  return av_frame;
}

void BetaCudaDeviceInterface::make_frame_standalone(UniqueAVFrame& av_frame) {
  // Make the frame standalone, i.e. safely consumable by a user or by a
  // ColorConverter (potentially a different CUDA stream):
  // - GPU frames are copied: we copy the frame data so that its surface can be
  // unmapped in
  //   receive_frame() without losing the data.
  // - CPU-fallback frames are uploaded here too, so that a PacketDecoder always
  //   hands out frames that live on its own device.
  // Both are async, so we record an event in the attached data right after
  // enqueueing them: a ColorConverter on another stream must wait on it before
  // reading the frame.
  STD_TORCH_CHECK(
      mode() == Mode::DecoderOnly,
      "make_frame_standalone() is only valid in decoder-only mode: standalone "
      "frames are meant to be consumed by a separate ColorConverter.");
  CudaContextGuard context_guard(device_.index());
  cudaStream_t current_stream = get_current_cuda_stream(device_.index());

  torch::stable::Tensor storage;
  if (decoding_on_cpu_) {
    auto uploaded = upload_cpu_frame_to_gpu(*av_frame, current_stream);
    av_frame = std::move(uploaded.av_frame);
    storage = std::move(uploaded.storage);
  } else {
    storage = copy_nvdec_surface(av_frame, current_stream);
  }

  auto attached_data = new OwnedFrameStorage();
  attached_data->frame_ready.record(current_stream);
  attached_data->storage = std::move(storage);
  av_frame->opaque_ref = av_buffer_create(
      reinterpret_cast<uint8_t*>(attached_data),
      sizeof(OwnedFrameStorage),
      standalone_frame_free_callback,
      nullptr,
      0);
  STD_TORCH_CHECK(
      av_frame->opaque_ref != nullptr,
      "Failed to attach standalone frame data");
}

std::optional<torch::stable::Tensor> BetaCudaDeviceInterface::get_frame_storage(
    const AVFrame& av_frame) const {
  STD_TORCH_CHECK(
      // Only decoder-only should reach here, and this should only be called on
      // frames that went through make_frame_standalone(), which sets
      // opaque_ref.
      mode() == Mode::DecoderOnly && av_frame.opaque_ref != nullptr,
      "Unexpected call to get_frame_storage(), please report a bug ");

  // Note [Standalone Frame Storage and the need for record_stream]
  //
  // A PacketDecoder and a ColorConverter may run on different CUDA streams.
  // Consider the following:
  //
  // ```
  // with decoder_stream:
  //   frame = decoder.receive_frame()
  // with color_converter_stream:
  //   color_converter.convert(frame)
  //
  // del frame
  //
  // with decoder_stream:
  //   frame = decoder.receive_frame()
  // ```
  //
  // The call to convert(frame) is non-blocking and just enqueues the
  // color-conversion kernel. The CPU moves on immediately to `del frame` while
  // the kernel is still running (it may also not even have started depending on
  // how color_converter_stream is congested).
  //
  // When the frame is deleted, the torch CUDA allocator reclaims its memory and
  // it becomes available for reuse for any subsequent allocation on the
  // decoder_stream. If the next decoder.receive_frame() happens before the
  // color-conversion kernel has finished (specifically: the new storage
  // allocation for that next frame in make_frame_standalone()), the memory is
  // reused, overwritten, and the color-conversion kernel reads garbage (i.e.
  // the next frame's samples!).
  //
  // We're hitting exactly what
  // https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html describes
  // in the 'Streams and freeing memory' section, and the solution is to call
  // record_stream() on the frame's storage within color_conversion_stream just
  // after the kernel is enqueued: this tells the allocator that it must wait
  // until this point (on the device side) before reclaiming the memory.
  //
  // We call record_stream(color_conversion_stream) on the frame storage in the
  // ColorConverter, on behalf of the user. But we still must expose the storage
  // for those users who would like to consume the frame with their own
  // consumer, i.e. not using the ColorConverter: they need to call
  // frame.storage.record_stream(color_conversion_stream) themselves.
  return reinterpret_cast<OwnedFrameStorage*>(av_frame.opaque_ref->data)
      ->storage;
}

torch::stable::Tensor BetaCudaDeviceInterface::copy_nvdec_surface(
    UniqueAVFrame& av_frame,
    cudaStream_t current_stream) {
  // The amount of bytes an NV12 image takes is:
  // num_bytes =  len(Y) + len(UV)
  //           = num_pixels + num_pixels / 2
  //           = num_pixels * 3 / 2
  //
  // where num_pixels = pitch * num_luma_plane_rows, not width * height: the
  // pitch accounts for both the row padding and the data size (uint8 vs
  // uint16), and NVDEC rounds the Y plane's row count up to even. A 4:4:4
  // surface has two full-size chroma planes instead of one half-height one, so
  // it's num_pixels * 3.
  int64_t num_luma_plane_rows =
      static_cast<int64_t>(round_up_to_even(av_frame->height));
  int64_t pitch = static_cast<int64_t>(av_frame->linesize[0]);
  bool is_444 = is_444_surface_format(surface_format_);
  int64_t num_bytes = is_444 ? pitch * num_luma_plane_rows * 3
                             : pitch * num_luma_plane_rows * 3 / 2;

  auto storage =
      torch::stable::empty({num_bytes}, kStableUInt8, std::nullopt, device_);

  // The surface's content is produced by the mapping post-processing that
  // receive_frame() enqueued on nvdec_output_stream_, which isn't necessarily
  // the stream we're copying on.
  nvdec_surface_ready_.make_stream_wait(current_stream);

  cudaError_t err = cudaMemcpyAsync(
      storage.mutable_data_ptr(),
      av_frame->data[0],
      static_cast<size_t>(num_bytes),
      cudaMemcpyDeviceToDevice,
      current_stream);
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "Failed to copy NVDEC surface: ",
      cudaGetErrorString(err));

  // The copy is async, so the next mapping must be ordered after it.
  record_surface_read(current_stream);

  auto y_plane = static_cast<uint8_t*>(storage.mutable_data_ptr());
  int64_t plane_stride = pitch * num_luma_plane_rows;
  av_frame->data[0] = y_plane;
  av_frame->data[1] = y_plane + plane_stride;
  if (is_444) {
    av_frame->data[2] = y_plane + (2 * plane_stride);
  }

  return storage;
}

void BetaCudaDeviceInterface::flush() {
  CudaContextGuard context_guard(device_.index());
  if (decoding_on_cpu_) {
    cpu_interface_->flush();
    return;
  }

  // The NVCUVID docs mention that after seeking, i.e. when flush() is called,
  // we should send a packet with the CUVID_PKT_DISCONTINUITY flag. The docs
  // don't say whether this should be an empty packet, or whether it should be
  // a flag on the next non-empty packet. It doesn't matter: neither work :)
  // Sending an EOF packet, however, does work. So we do that. And we re-set
  // the eofSent_ flag to false because that's not a true EOF notification.
  send_eof_packet();
  eof_sent_ = false;

  std::queue<CUVIDPARSERDISPINFO> empty_queue;
  std::swap(ready_frames_, empty_queue);
  discarded_timestamps_.clear();

  send_seqhdr_packet();
}

GpuFrameAndStorage BetaCudaDeviceInterface::upload_cpu_frame_to_gpu(
    const AVFrame& cpu_frame,
    cudaStream_t stream) {
  // This is called in the context of the CPU fallback: the frame was decoded
  // on the CPU, and in this function we convert that frame into a format we
  // can color-convert on the GPU, and send it there.
  // We do that in 2 steps:
  // - First we convert the input CPU frame into an intermediate CPU frame in
  //   the target format using sws_scale.
  // - Then we allocate GPU memory and copy the CPU frame to the GPU on the
  //   given stream.
  // We return the new AVFrame and its associated GPU storage so that the caller
  // can handle the memory lifetime. The GPU storage is a torch
  // Tensor because we want to rely on the torch CUDA allocator.

  const AVPixFmtDescriptor* source_desc =
      av_pix_fmt_desc_get(static_cast<AVPixelFormat>(cpu_frame.format));
  STD_TORCH_CHECK(
      source_desc != nullptr, "Unknown pixel format on decoded frame");
  int source_bit_depth = source_desc->comp[0].depth;

  // We convert to a format our CUDA color-conversion kernels can read, keeping
  // the source's chroma and bit depth: 4:2:0 and monochrome go semi-planar,
  // everything else (4:2:2, 4:1:1, RGB, ...) goes 4:4:4, which is the only
  // other layout the kernels handle and never reduces chroma. Unlike the decode
  // path we're not choosing between what NVDEC happens to offer, so the
  // requested output dtype doesn't come into it: narrowing to uint8, if that's
  // what was asked for, happens after color conversion.
  // We go through a cudaVideoSurfaceFormat because that's what
  // surface_to_pix_fmt, and we need to call that because of the
  // FFmpeg-version-dependent P012 vs P016 distinction (sad).
  bool semi_planar_420 = source_desc->nb_components == 1 ||
      (source_desc->log2_chroma_w == 1 && source_desc->log2_chroma_h == 1);
  bool want_16bit = source_bit_depth > 8;

  cudaVideoSurfaceFormat surface_format;
  if (semi_planar_420) {
    surface_format =
        want_16bit ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
  } else {
    surface_format = want_16bit ? cudaVideoSurfaceFormat_YUV444_16Bit
                                : cudaVideoSurfaceFormat_YUV444;
  }
  AVPixelFormat target_pix_fmt =
      surface_to_pix_fmt(surface_format, source_bit_depth);

  int num_planes = semi_planar_420 ? 2 : 3;
  int bytes_per_sample = want_16bit ? 2 : 1;

  // The 4:2:0 kernel works on 2x2 blocks and never writes a trailing odd row or
  // column, so its input planes must be even-sized. We allocate the buffer with
  // even dimensions but keep the frame's real width and height, exactly like
  // the even-sized surfaces NVDEC hands us for odd-sized videos: the pad
  // row/column is only read as part of a boundary block, and the color
  // conversion crops it away. Nothing about the pixel format itself requires
  // this: FFmpeg is happy with odd-sized 4:2:0 frames. The 4:4:4 kernel is
  // per-pixel, so those need no padding.
  int width = cpu_frame.width;
  int height = cpu_frame.height;
  int padded_width = semi_planar_420 ? round_up_to_even(width) : width;
  int padded_height = semi_planar_420 ? round_up_to_even(height) : height;

  UniqueAVFrame intermediate_cpu_frame(av_frame_alloc());
  STD_TORCH_CHECK(
      intermediate_cpu_frame != nullptr,
      "Failed to allocate intermediate CPU frame");

  intermediate_cpu_frame->format = target_pix_fmt;
  intermediate_cpu_frame->width = padded_width;
  intermediate_cpu_frame->height = padded_height;

  int ret = av_frame_get_buffer(intermediate_cpu_frame.get(), 0);
  STD_TORCH_CHECK(
      ret >= 0,
      "Failed to allocate intermediate CPU frame buffer: ",
      get_ffmpeg_error_string_from_error_code(ret));

  // Source and destination dimensions are the same: this is a pixel format
  // conversion, not a rescale. sws_scale() writes into the even-sized buffer
  // allocated above but only fills the real width and height.
  SwsConfig sws_config(
      width,
      height,
      static_cast<AVPixelFormat>(cpu_frame.format),
      cpu_frame.colorspace,
      width,
      height,
      target_pix_fmt,
      // The frame keeps its range tag through the upload, so the samples must
      // keep the range that tag names. Left to itself, swscale writes limited
      // range into a YUV target whatever it read, and the color conversion
      // would then expand a full-range source a second time.
      cpu_frame.color_range);

  if (!sws_context_ || prev_sws_config_ != sws_config) {
    // Nothing is rescaled here, so the flags only pick how chroma is
    // resampled, which happens when the source is subsampled more finely than
    // the target surface (4:2:2 into 4:4:4, say). SWS_POINT replicates it,
    // which is what the CPU converter does on its way to RGB - interpolating
    // instead would invent chroma the CPU never sees, and show up as colored
    // fringes along sharp edges.
    sws_context_ = create_sws_context(sws_config, SWS_POINT);
    prev_sws_config_ = sws_config;
  }

  int converted_height = sws_scale(
      sws_context_.get(),
      cpu_frame.data,
      cpu_frame.linesize,
      0,
      height,
      intermediate_cpu_frame->data,
      intermediate_cpu_frame->linesize);
  STD_TORCH_CHECK(
      converted_height == height,
      "sws_scale failed for the CPU-fallback upload conversion");

  // The chroma plane of a semi-planar 4:2:0 frame carries interleaved UV pairs,
  // so it's as wide as the luma plane but half as tall.
  int row_bytes = padded_width * bytes_per_sample;
  int plane_heights[3] = {
      padded_height,
      semi_planar_420 ? padded_height / 2 : padded_height,
      padded_height};

  int64_t plane_offsets[3] = {0, 0, 0};
  int64_t total_bytes = 0;
  for (int p = 0; p < num_planes; ++p) {
    plane_offsets[p] = total_bytes;
    total_bytes += static_cast<int64_t>(row_bytes) * plane_heights[p];
  }

  CudaContextGuard context_guard(device_.index());
  auto storage =
      torch::stable::empty({total_bytes}, kStableUInt8, std::nullopt, device_);
  auto storage_ptr = static_cast<uint8_t*>(storage.mutable_data_ptr());

  UniqueAVFrame gpu_frame(av_frame_alloc());
  STD_TORCH_CHECK(gpu_frame != nullptr, "Failed to allocate GPU AVFrame");

  gpu_frame->format = target_pix_fmt;
  gpu_frame->width = width;
  gpu_frame->height = height;

  // One copy per plane: av_frame_get_buffer() allocates each plane with its own
  // alignment padding, so they are neither contiguous with each other nor
  // packed, and each has its own height.
  for (int p = 0; p < num_planes; ++p) {
    gpu_frame->data[p] = storage_ptr + plane_offsets[p];
    gpu_frame->linesize[p] = row_bytes;

    // Note that we use cudaMemcpy2DAsync here instead of cudaMemcpyAsync
    // because the linesizes (strides) may be different than the widths for the
    // input CPU frame. That's precisely what the 2D variants are for.
    cudaError_t err = cudaMemcpy2DAsync(
        gpu_frame->data[p],
        gpu_frame->linesize[p],
        intermediate_cpu_frame->data[p],
        intermediate_cpu_frame->linesize[p],
        row_bytes,
        plane_heights[p],
        cudaMemcpyHostToDevice,
        stream);
    STD_TORCH_CHECK(
        err == cudaSuccess,
        "Failed to copy plane ",
        p,
        " to GPU: ",
        cudaGetErrorString(err));
  }

  // intermediate_cpu_frame is freed when this function returns, so we can't
  // leave the copies in flight, we must wait for it to finish. Making the
  // upload truly async would mean allocating the intermediate frame in pinned
  // memory and keeping it alive until the copies complete. Probably not worth
  // it for this CPU fallback path that is already slow by nature.
  cudaError_t err = cudaStreamSynchronize(stream);
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "Failed to wait for the CPU-to-GPU upload: ",
      cudaGetErrorString(err));

  ret = av_frame_copy_props(gpu_frame.get(), &cpu_frame);
  STD_TORCH_CHECK(
      ret >= 0,
      "Failed to copy frame properties: ",
      get_ffmpeg_error_string_from_error_code(ret));

  // AVCOL_SPC_RGB says "these planes are RGB", which the planes we just wrote
  // aren't. Name the matrix swscale encoded them with instead: it maps
  // AVCOL_SPC_RGB, like any colorspace it doesn't know, to its BT.601 default.
  if (cpu_frame.colorspace == AVCOL_SPC_RGB) {
    gpu_frame->colorspace = AVCOL_SPC_SMPTE170M;
  }

  return {std::move(gpu_frame), std::move(storage)};
}

void BetaCudaDeviceInterface::convert_av_frame_to_frame_output(
    const AVFrame& av_frame,
    FrameOutput& frame_output,
    std::optional<torch::stable::Tensor> pre_allocated_output_tensor) {
  CudaContextGuard context_guard(device_.index());
  cudaStream_t current_stream = get_current_cuda_stream(device_.index());

  // We may need to upload a frame here in case of the CPU fallback. This is
  // only needed in Both() mode i.e. with the SingleStreamDecoder. The reason we
  // do it here and not just after decoding is because the `decode_av_frame()`
  // loop of the SingleStreamDecoder may discard frames while decoding forward
  // to a target pts - we don't want to upload these frames that will be
  // discarded anyway. So we upload as late as possible for those frame we
  // *know* we must return.
  //
  // In contrast, a PacketDecoder will always upload CPU frames before retuning
  // them because its contract is to respect its device parameter.
  bool needs_upload = mode() == Mode::Both && decoding_on_cpu_;

  // `uploaded` owns the GPU buffer for as long as it's in scope, which covers
  // the color conversion below.
  GpuFrameAndStorage uploaded;
  if (needs_upload) {
    uploaded = upload_cpu_frame_to_gpu(av_frame, current_stream);
  }
  const AVFrame& gpu_frame = needs_upload ? *uploaded.av_frame : av_frame;

  // Both NVDEC surfaces and uploaded CPU frames may be backed by even-sized
  // buffers while describing an odd-sized frame; the color conversion crops the
  // padding away.
  FrameDims output_dims(gpu_frame.height, gpu_frame.width);

  auto gpu_pix_fmt = static_cast<AVPixelFormat>(gpu_frame.format);
  STD_TORCH_CHECK(
      is_expected_pix_fmt_from_nvdec(gpu_pix_fmt),
      "Expected a pixel format we can color-convert on the GPU, got ",
      av_get_pix_fmt_name(gpu_pix_fmt));

  if (mode() == Mode::ColorConverterOnly) {
    STD_TORCH_CHECK(
        gpu_frame.opaque_ref != nullptr,
        "ColorConverter received a non-standalone frame; frames fed to a "
        "standalone ColorConverter must come from a PacketDecoder.");
    auto attached_data =
        reinterpret_cast<OwnedFrameStorage*>(gpu_frame.opaque_ref->data);
    attached_data->frame_ready.make_stream_wait(current_stream);
  } else {
    STD_TORCH_CHECK(
        mode() == Mode::Both,
        "Color conversion requires the interface to be initialized for color "
        "conversion.");
    if (!needs_upload) {
      // The frame is a mapped NVDEC surface, whose content is produced by the
      // mapping post-processing that receive_frame() enqueued on
      // nvdec_output_stream_. An uploaded frame, on the other hand, was
      // uploaded on current_stream and needs no ordering.
      nvdec_surface_ready_.make_stream_wait(current_stream);
    }
  }

  auto convert_frame = [&](std::optional<torch::stable::Tensor> pre_alloc)
      -> torch::stable::Tensor {
    return convert_yuv_frame_to_rgb(
        gpu_frame,
        device_,
        current_stream,
        pre_alloc,
        output_dims,
        static_cast<AVPixelFormat>(gpu_frame.format),
        cached_color_matrix_);
  };

  if (rotation_ == Rotation::NONE) {
    validate_pre_allocated_tensor_shape(
        pre_allocated_output_tensor, output_dims);
    frame_output.data = convert_frame(pre_allocated_output_tensor);
  } else {
    // preAllocatedOutputTensor has post-rotation dimensions, but the
    // conversion outputs pre-rotation dimensions, so we can't use it as the
    // conversion destination or validate it against the frame shape.
    // Once we support native transforms on the NVDEC CUDA interface,
    // rotation should be handled as part of the transform pipeline instead.
    frame_output.data = convert_frame(/*preAlloc=*/std::nullopt);
    apply_rotation(frame_output, pre_allocated_output_tensor);
  }

  if (mode() == Mode::Both && !needs_upload) {
    // The conversion read the mapped surface directly, and did so
    // asynchronously, so the next mapping must be ordered after it.
    // This is only needed in Both() mode because in ColorConverterOnly() mode,
    // the frame is already standalone and doesn't come from the mapped surface.
    // It's also only needed in the non-fallback mode (needs_upload is false)
    // because with the fallback, the GPU frame is a copy of the CPU frame, so
    // the mapped surface isn't read at all.
    record_surface_read(current_stream);
  }
}

void BetaCudaDeviceInterface::apply_rotation(
    FrameOutput& frame_output,
    std::optional<torch::stable::Tensor> pre_allocated_output_tensor) {
  frame_output.data = rotate_hwc_tensor(frame_output.data, rotation_);

  if (pre_allocated_output_tensor.has_value()) {
    torch::stable::copy_(
        pre_allocated_output_tensor.value(), frame_output.data);
    frame_output.data = pre_allocated_output_tensor.value();
  }
}

OutputDtype BetaCudaDeviceInterface::get_pre_allocation_dtype(
    [[maybe_unused]] OutputDtype requested_dtype) const {
  if (decoding_on_cpu_) {
    // the upload keeps the source's own depth, see upload_cpu_frame_to_gpu().
    const AVPixFmtDescriptor* desc =
        av_pix_fmt_desc_get(codec_context_->pix_fmt);
    bool is_16bit = desc != nullptr && desc->comp[0].depth > 8;
    return is_16bit ? OutputDtype::FLOAT32 : OutputDtype::UINT8;
  } else {
    return is_16bit_surface_format(surface_format_) ? OutputDtype::FLOAT32
                                                    : OutputDtype::UINT8;
  }
}

std::string BetaCudaDeviceInterface::get_details() {
  std::string details = "NVDEC CUDA Device Interface.";
  if (decoding_on_cpu_) {
    details += " Using CPU fallback.";
    if (!nvcuvid_available_) {
      details += " NVCUVID not available!";
    }
  } else {
    details += " Using NVDEC.";
  }
  return details;
}

} // namespace facebook::torchcodec
