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
  std::string supportedRates;
  for (auto i = 0; avCodec.supported_samplerates[i] != 0; ++i) {
    if (i > 0) {
      supportedRates += ", ";
    }
    supportedRates += std::to_string(avCodec.supported_samplerates[i]);
  }

  TORCH_CHECK(
      false,
      "invalid sample rate=",
      sampleRate,
      ". Supported sample rate values are: ",
      supportedRates);
}

} // namespace

AudioEncoder::~AudioEncoder() {}

// TODO-ENCODING: disable ffmpeg logs by default

AudioEncoder::AudioEncoder(
    const torch::Tensor wf,
    int sampleRate,
    std::string_view fileName)
    : wf_(wf) {
  TORCH_CHECK(
      wf_.dtype() == torch::kFloat32,
      "waveform must have float32 dtype, got ",
      wf_.dtype());
  TORCH_CHECK(
      wf_.dim() == 2, "waveform must have 2 dimensions, got ", wf_.dim());
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

  // TODO-ENCODING I think this sets the bit rate to the minimum supported.
  // That's not what the ffmpeg CLI would choose by default, so we should try to
  // do the same.
  // TODO-ENCODING Should also let user choose for compressed formats like mp3.
  avCodecContext_->bit_rate = 0;

  validateSampleRate(*avCodec, sampleRate);
  avCodecContext_->sample_rate = sampleRate;

  // Note: This is the format of the **input** waveform. This doesn't determine
  // the output.
  // TODO-ENCODING check contiguity of the input wf to ensure that it is indeed
  // planar.
  // TODO-ENCODING If the encoder doesn't support FLTP (like flac), FFmpeg will
  // raise. We need to handle this, probably converting the format with
  // libswresample.
  avCodecContext_->sample_fmt = AV_SAMPLE_FMT_FLTP;

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

  TORCH_CHECK(
      avCodecContext_->frame_size > 0,
      "frame_size is ",
      avCodecContext_->frame_size,
      ". Cannot encode. This should probably never happen?");

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
  avFrame->nb_samples = avCodecContext_->frame_size;
  avFrame->format = avCodecContext_->sample_fmt;
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
  int numSamplesPerFrame = avCodecContext_->frame_size; // per channel
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
        std::min(numSamplesPerFrame, numSamples - numEncodedSamples);
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
    const UniqueAVFrame& avFrame) {
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
  AutoAVPacket autoAVPacket;
  encodeInnerLoop(autoAVPacket, UniqueAVFrame(nullptr));
}
} // namespace facebook::torchcodec
