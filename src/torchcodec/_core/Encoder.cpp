#include <sstream>

#include "src/torchcodec/_core/Encoder.h"
#include "torch/types.h"

namespace facebook::torchcodec {

namespace {

void validateSampleRate(const AVCodec& avCodec, int sampleRate) {
  if (avCodec.supported_samplerates == nullptr) {
    return;
  }

  for (auto i = 0; avCodec.supported_samplerates[i] != 0; ++i) {
    if (sampleRate == avCodec.supported_samplerates[i]) {
      return;
    }
  }
  std::stringstream supportedRates;
  for (auto i = 0; avCodec.supported_samplerates[i] != 0; ++i) {
    if (i > 0) {
      supportedRates << ", ";
    }
    supportedRates << avCodec.supported_samplerates[i];
  }

  TORCH_CHECK(
      false,
      "invalid sample rate=",
      sampleRate,
      ". Supported sample rate values are: ",
      supportedRates.str());
}

static const std::vector<AVSampleFormat> preferredFormatsOrder = {
    AV_SAMPLE_FMT_FLTP,
    AV_SAMPLE_FMT_FLT,
    AV_SAMPLE_FMT_DBLP,
    AV_SAMPLE_FMT_DBL,
    AV_SAMPLE_FMT_S64P,
    AV_SAMPLE_FMT_S64,
    AV_SAMPLE_FMT_S32P,
    AV_SAMPLE_FMT_S32,
    AV_SAMPLE_FMT_S16P,
    AV_SAMPLE_FMT_S16,
    AV_SAMPLE_FMT_U8P,
    AV_SAMPLE_FMT_U8};

AVSampleFormat findBestOutputSampleFormat(const AVCodec& avCodec) {
  // Find a sample format that the encoder supports. We prefer using FLT[P],
  // since this is the format of the input waveform. If FLTP isn't supported
  // then we'll need to convert the AVFrame's format. Our heuristic is to encode
  // into the format with the highest resolution.
  if (avCodec.sample_fmts == nullptr) {
    // Can't really validate anything in this case, best we can do is hope that
    // FLTP is supported by the encoder. If not, FFmpeg will raise.
    return AV_SAMPLE_FMT_FLTP;
  }

  for (AVSampleFormat preferredFormat : preferredFormatsOrder) {
    for (int i = 0; avCodec.sample_fmts[i] != -1; ++i) {
      if (avCodec.sample_fmts[i] == preferredFormat) {
        return preferredFormat;
      }
    }
  }
  // We should always find a match in preferredFormatsOrder, so we should always
  // return earlier. But in the event that a future FFmpeg version defines an
  // additional sample format that isn't in preferredFormatsOrder, we fallback:
  return avCodec.sample_fmts[0];
}

} // namespace

AudioEncoder::~AudioEncoder() {}

AudioEncoder::AudioEncoder(
    const torch::Tensor wf,
    int sampleRate,
    std::string_view fileName,
    std::optional<int64_t> bitRate)
    : wf_(wf) {
  TORCH_CHECK(
      wf_.dtype() == torch::kFloat32,
      "waveform must have float32 dtype, got ",
      wf_.dtype());
  // TODO-ENCODING check contiguity of the input wf to ensure that it is indeed
  // planar (fltp).
  TORCH_CHECK(
      wf_.dim() == 2, "waveform must have 2 dimensions, got ", wf_.dim());

  setFFmpegLogLevel();
  AVFormatContext* avFormatContext = nullptr;
  auto status = avformat_alloc_output_context2(
      &avFormatContext, nullptr, nullptr, fileName.data());
  TORCH_CHECK(
      avFormatContext != nullptr,
      "Couldn't allocate AVFormatContext. ",
      "Check the desired extension? ",
      getFFMPEGErrorStringFromErrorCode(status));
  avFormatContext_.reset(avFormatContext);

  // TODO-ENCODING: Should also support encoding into bytes (use
  // AVIOBytesContext)
  TORCH_CHECK(
      !(avFormatContext->oformat->flags & AVFMT_NOFILE),
      "AVFMT_NOFILE is set. We only support writing to a file.");
  status = avio_open(&avFormatContext_->pb, fileName.data(), AVIO_FLAG_WRITE);
  TORCH_CHECK(
      status >= 0,
      "avio_open failed: ",
      getFFMPEGErrorStringFromErrorCode(status));

  // We use the AVFormatContext's default codec for that
  // specific format/container.
  const AVCodec* avCodec =
      avcodec_find_encoder(avFormatContext_->oformat->audio_codec);
  TORCH_CHECK(avCodec != nullptr, "Codec not found");

  AVCodecContext* avCodecContext = avcodec_alloc_context3(avCodec);
  TORCH_CHECK(avCodecContext != nullptr, "Couldn't allocate codec context.");
  avCodecContext_.reset(avCodecContext);

  if (bitRate.has_value()) {
    TORCH_CHECK(*bitRate >= 0, "bit_rate=", *bitRate, " must be >= 0.");
  }
  // bit_rate=None defaults to 0, which is what the FFmpeg CLI seems to use as
  // well when "-b:a" isn't specified.
  avCodecContext_->bit_rate = bitRate.value_or(0);

  validateSampleRate(*avCodec, sampleRate);
  avCodecContext_->sample_rate = sampleRate;

  // Input waveform is expected to be FLTP. Not all encoders support FLTP, so we
  // may need to convert the wf into a supported output sample format, which is
  // what the `.sample_fmt` defines.
  avCodecContext_->sample_fmt = findBestOutputSampleFormat(*avCodec);

  int numChannels = static_cast<int>(wf_.sizes()[0]);
  TORCH_CHECK(
      // TODO-ENCODING is this even true / needed? We can probably support more
      // with non-planar data?
      numChannels <= AV_NUM_DATA_POINTERS,
      "Trying to encode ",
      numChannels,
      " channels, but FFmpeg only supports ",
      AV_NUM_DATA_POINTERS,
      " channels per frame.");

  setDefaultChannelLayout(avCodecContext_, numChannels);

  status = avcodec_open2(avCodecContext_.get(), avCodec, nullptr);
  TORCH_CHECK(
      status == AVSUCCESS,
      "avcodec_open2 failed: ",
      getFFMPEGErrorStringFromErrorCode(status));

  // We're allocating the stream here. Streams are meant to be freed by
  // avformat_free_context(avFormatContext), which we call in the
  // avFormatContext_'s destructor.
  AVStream* avStream = avformat_new_stream(avFormatContext_.get(), nullptr);
  TORCH_CHECK(avStream != nullptr, "Couldn't create new stream.");
  status = avcodec_parameters_from_context(
      avStream->codecpar, avCodecContext_.get());
  TORCH_CHECK(
      status == AVSUCCESS,
      "avcodec_parameters_from_context failed: ",
      getFFMPEGErrorStringFromErrorCode(status));
  streamIndex_ = avStream->index;
}

void AudioEncoder::encode() {
  UniqueAVFrame avFrame(av_frame_alloc());
  TORCH_CHECK(avFrame != nullptr, "Couldn't allocate AVFrame.");
  //  Default to 256 like in torchaudio
  int numSamplesAllocatedPerFrame =
      avCodecContext_->frame_size > 0 ? avCodecContext_->frame_size : 256;
  avFrame->nb_samples = numSamplesAllocatedPerFrame;
  avFrame->format = AV_SAMPLE_FMT_FLTP;
  avFrame->sample_rate = avCodecContext_->sample_rate;
  avFrame->pts = 0;
  setChannelLayout(avFrame, avCodecContext_);

  auto status = av_frame_get_buffer(avFrame.get(), 0);
  TORCH_CHECK(
      status == AVSUCCESS,
      "Couldn't allocate avFrame's buffers: ",
      getFFMPEGErrorStringFromErrorCode(status));

  AutoAVPacket autoAVPacket;

  uint8_t* pwf = static_cast<uint8_t*>(wf_.data_ptr());
  int numSamples = static_cast<int>(wf_.sizes()[1]); // per channel
  int numEncodedSamples = 0; // per channel
  int numBytesPerSample = static_cast<int>(wf_.element_size());
  int numBytesPerChannel = numSamples * numBytesPerSample;

  status = avformat_write_header(avFormatContext_.get(), nullptr);
  TORCH_CHECK(
      status == AVSUCCESS,
      "Error in avformat_write_header: ",
      getFFMPEGErrorStringFromErrorCode(status));

  while (numEncodedSamples < numSamples) {
    status = av_frame_make_writable(avFrame.get());
    TORCH_CHECK(
        status == AVSUCCESS,
        "Couldn't make AVFrame writable: ",
        getFFMPEGErrorStringFromErrorCode(status));

    int numSamplesToEncode =
        std::min(numSamplesAllocatedPerFrame, numSamples - numEncodedSamples);
    int numBytesToEncode = numSamplesToEncode * numBytesPerSample;

    for (int ch = 0; ch < wf_.sizes()[0]; ch++) {
      std::memcpy(
          avFrame->data[ch], pwf + ch * numBytesPerChannel, numBytesToEncode);
    }
    pwf += numBytesToEncode;

    // Above, we set the AVFrame's .nb_samples to AVCodecContext.frame_size so
    // that the frame buffers are allocated to a big enough size. Here, we reset
    // it to the exact number of samples that need to be encoded, otherwise the
    // encoded frame would contain more samples than necessary and our results
    // wouldn't match the ffmpeg CLI.
    avFrame->nb_samples = numSamplesToEncode;
    encodeInnerLoop(autoAVPacket, avFrame);

    avFrame->pts += static_cast<int64_t>(numSamplesToEncode);
    numEncodedSamples += numSamplesToEncode;
  }
  TORCH_CHECK(numEncodedSamples == numSamples, "Hmmmmmm something went wrong.");

  flushBuffers();

  status = av_write_trailer(avFormatContext_.get());
  TORCH_CHECK(
      status == AVSUCCESS,
      "Error in: av_write_trailer",
      getFFMPEGErrorStringFromErrorCode(status));
}

void AudioEncoder::encodeInnerLoop(
    AutoAVPacket& autoAVPacket,
    const UniqueAVFrame& srcAVFrame) {
  bool mustConvert =
      (avCodecContext_->sample_fmt != AV_SAMPLE_FMT_FLTP &&
       srcAVFrame != nullptr);
  UniqueAVFrame convertedAVFrame;
  if (mustConvert) {
    if (!swrContext_) {
      swrContext_.reset(createSwrContext(
          avCodecContext_,
          AV_SAMPLE_FMT_FLTP,
          avCodecContext_->sample_fmt,
          srcAVFrame->sample_rate, // No sample rate conversion
          srcAVFrame->sample_rate));
    }
    convertedAVFrame = convertAudioAVFrameSampleFormatAndSampleRate(
        swrContext_,
        srcAVFrame,
        avCodecContext_->sample_fmt,
        srcAVFrame->sample_rate, // No sample rate conversion
        srcAVFrame->sample_rate);
    TORCH_CHECK(
        convertedAVFrame->nb_samples == srcAVFrame->nb_samples,
        "convertedAVFrame->nb_samples=",
        convertedAVFrame->nb_samples,
        " differs from ",
        "srcAVFrame->nb_samples=",
        srcAVFrame->nb_samples,
        "This is unexpected, please report on the TorchCodec bug tracker.");
  }
  const UniqueAVFrame& avFrame = mustConvert ? convertedAVFrame : srcAVFrame;

  auto status = avcodec_send_frame(avCodecContext_.get(), avFrame.get());
  TORCH_CHECK(
      status == AVSUCCESS,
      "Error while sending frame: ",
      getFFMPEGErrorStringFromErrorCode(status));

  while (status >= 0) {
    ReferenceAVPacket packet(autoAVPacket);
    status = avcodec_receive_packet(avCodecContext_.get(), packet.get());
    if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) {
      // TODO-ENCODING this is from TorchAudio, probably needed, but not sure.
      //   if (status == AVERROR_EOF) {
      //     status = av_interleaved_write_frame(avFormatContext_.get(),
      //     nullptr); TORCH_CHECK(
      //         status == AVSUCCESS,
      //         "Failed to flush packet ",
      //         getFFMPEGErrorStringFromErrorCode(status));
      //   }
      return;
    }
    TORCH_CHECK(
        status >= 0,
        "Error receiving packet: ",
        getFFMPEGErrorStringFromErrorCode(status));

    packet->stream_index = streamIndex_;

    status = av_interleaved_write_frame(avFormatContext_.get(), packet.get());
    TORCH_CHECK(
        status == AVSUCCESS,
        "Error in av_interleaved_write_frame: ",
        getFFMPEGErrorStringFromErrorCode(status));
  }
}

void AudioEncoder::flushBuffers() {
  // We flush the main FFmpeg buffers, but not swresample buffers. Flushing
  // swresample is only necessary when converting sample rates, which we don't
  // do for encoding.
  AutoAVPacket autoAVPacket;
  encodeInnerLoop(autoAVPacket, UniqueAVFrame(nullptr));
}
} // namespace facebook::torchcodec
