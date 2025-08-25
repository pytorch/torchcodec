#include <sstream>

#include "src/torchcodec/_core/AVIOTensorContext.h"
#include "src/torchcodec/_core/Encoder.h"
#include "torch/types.h"

namespace facebook::torchcodec {

namespace {

torch::Tensor validateSamples(const torch::Tensor& samples) {
  TORCH_CHECK(
      samples.dtype() == torch::kFloat32,
      "samples must have float32 dtype, got ",
      samples.dtype());
  TORCH_CHECK(
      samples.dim() == 2,
      "samples must have 2 dimensions, got ",
      samples.dim());

  // We enforce this, but if we get user reports we should investigate whether
  // that's actually needed.
  int numChannels = static_cast<int>(samples.sizes()[0]);
  TORCH_CHECK(
      numChannels <= AV_NUM_DATA_POINTERS,
      "Trying to encode ",
      numChannels,
      " channels, but FFmpeg only supports ",
      AV_NUM_DATA_POINTERS,
      " channels per frame.");

  return samples.contiguous();
}

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
  // since this is the format of the input samples. If FLTP isn't supported
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

AudioEncoder::~AudioEncoder() {
  close_avio();
}

void AudioEncoder::close_avio() {
  if (avFormatContext_ && avFormatContext_->pb) {
    if (avFormatContext_->pb->error == 0) {
      avio_flush(avFormatContext_->pb);
    }

    if (!avioContextHolder_) {
      if (avFormatContext_->pb->error == 0) {
        avio_close(avFormatContext_->pb);
      }
      // avoids closing again in destructor, which would segfault.
      avFormatContext_->pb = nullptr;
    }
  }
}

AudioEncoder::AudioEncoder(
    const torch::Tensor& samples,
    int sampleRate,
    std::string_view fileName,
    const AudioStreamOptions& audioStreamOptions)
    : samples_(validateSamples(samples)), inSampleRate_(sampleRate) {
  setFFmpegLogLevel();
  AVFormatContext* avFormatContext = nullptr;
  int status = avformat_alloc_output_context2(
      &avFormatContext, nullptr, nullptr, fileName.data());

  TORCH_CHECK(
      avFormatContext != nullptr,
      "Couldn't allocate AVFormatContext. ",
      "The destination file is ",
      fileName,
      ", check the desired extension? ",
      getFFMPEGErrorStringFromErrorCode(status));
  avFormatContext_.reset(avFormatContext);

  status = avio_open(&avFormatContext_->pb, fileName.data(), AVIO_FLAG_WRITE);
  TORCH_CHECK(
      status >= 0,
      "avio_open failed. The destination file is ",
      fileName,
      ", make sure it's a valid path? ",
      getFFMPEGErrorStringFromErrorCode(status));

  initializeEncoder(audioStreamOptions);
}

AudioEncoder::AudioEncoder(
    const torch::Tensor& samples,
    int sampleRate,
    std::string_view formatName,
    std::unique_ptr<AVIOContextHolder> avioContextHolder,
    const AudioStreamOptions& audioStreamOptions)
    : samples_(validateSamples(samples)),
      inSampleRate_(sampleRate),
      avioContextHolder_(std::move(avioContextHolder)) {
  setFFmpegLogLevel();
  AVFormatContext* avFormatContext = nullptr;
  int status = avformat_alloc_output_context2(
      &avFormatContext, nullptr, formatName.data(), nullptr);

  TORCH_CHECK(
      avFormatContext != nullptr,
      "Couldn't allocate AVFormatContext. ",
      "Check the desired format? Got format=",
      formatName,
      ". ",
      getFFMPEGErrorStringFromErrorCode(status));
  avFormatContext_.reset(avFormatContext);

  avFormatContext_->pb = avioContextHolder_->getAVIOContext();

  initializeEncoder(audioStreamOptions);
}

void AudioEncoder::initializeEncoder(
    const AudioStreamOptions& audioStreamOptions) {
  // We use the AVFormatContext's default codec for that
  // specific format/container.
  const AVCodec* avCodec =
      avcodec_find_encoder(avFormatContext_->oformat->audio_codec);
  TORCH_CHECK(avCodec != nullptr, "Codec not found");

  AVCodecContext* avCodecContext = avcodec_alloc_context3(avCodec);
  TORCH_CHECK(avCodecContext != nullptr, "Couldn't allocate codec context.");
  avCodecContext_.reset(avCodecContext);

  auto desiredBitRate = audioStreamOptions.bitRate;
  if (desiredBitRate.has_value()) {
    TORCH_CHECK(
        *desiredBitRate >= 0, "bit_rate=", *desiredBitRate, " must be >= 0.");
  }
  // bit_rate=None defaults to 0, which is what the FFmpeg CLI seems to use as
  // well when "-b:a" isn't specified.
  avCodecContext_->bit_rate = desiredBitRate.value_or(0);

  outNumChannels_ = static_cast<int>(
      audioStreamOptions.numChannels.value_or(samples_.sizes()[0]));
  validateNumChannels(*avCodec, outNumChannels_);
  // The avCodecContext layout defines the layout of the encoded output, it's
  // not related to the input sampes.
  setDefaultChannelLayout(avCodecContext_, outNumChannels_);

  outSampleRate_ = audioStreamOptions.sampleRate.value_or(inSampleRate_);
  validateSampleRate(*avCodec, outSampleRate_);
  avCodecContext_->sample_rate = outSampleRate_;

  // Input samples are expected to be FLTP. Not all encoders support FLTP, so we
  // may need to convert the samples into a supported output sample format,
  // which is what the `.sample_fmt` defines.
  avCodecContext_->sample_fmt = findBestOutputSampleFormat(*avCodec);

  int status = avcodec_open2(avCodecContext_.get(), avCodec, nullptr);
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

  // If sample rate conversion is needed and the encoder doesn't support
  // variable frame size, we need to create an intermediate FIFO. See
  // [Encoding loop, sample rate conversion and FIFO].
  if (((avCodec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE) == 0) &&
      (inSampleRate_ != outSampleRate_)) {
    // frame_size * 2 is a decent default size. FFmpeg automatically
    // re-allocates the fifo if more space is needed.
    auto avAudioFifo = av_audio_fifo_alloc(
        avCodecContext_->sample_fmt,
        outNumChannels_,
        avCodecContext_->frame_size * 2);
    TORCH_CHECK(avAudioFifo != nullptr, "Couldn't create AVAudioFifo.");
    avAudioFifo_.reset(avAudioFifo);
  }
}

torch::Tensor AudioEncoder::encodeToTensor() {
  TORCH_CHECK(
      avioContextHolder_ != nullptr,
      "Cannot encode to tensor, avio tensor context doesn't exist.");
  encode();
  auto avioToTensorContext =
      dynamic_cast<AVIOToTensorContext*>(avioContextHolder_.get());
  TORCH_CHECK(avioToTensorContext != nullptr, "Invalid AVIO context holder.");
  return avioToTensorContext->getOutputTensor();
}

void AudioEncoder::encode() {
  // To be on the safe side we enforce that encode() can only be called once on
  // an encoder object. Whether this is actually necessary is unknown, so this
  // may be relaxed if needed.
  TORCH_CHECK(!encodeWasCalled_, "Cannot call encode() twice.");
  encodeWasCalled_ = true;

  //  Default to 256 like in torchaudio
  int numSamplesAllocatedPerFrame =
      avCodecContext_->frame_size > 0 ? avCodecContext_->frame_size : 256;
  UniqueAVFrame avFrame = allocateAVFrame(
      numSamplesAllocatedPerFrame,
      inSampleRate_,
      static_cast<int>(samples_.sizes()[0]),
      AV_SAMPLE_FMT_FLTP);
  avFrame->pts = 0;

  AutoAVPacket autoAVPacket;

  uint8_t* psamples = static_cast<uint8_t*>(samples_.data_ptr());
  int numSamples = static_cast<int>(samples_.sizes()[1]); // per channel
  int numEncodedSamples = 0; // per channel
  int numBytesPerSample = static_cast<int>(samples_.element_size());
  int numBytesPerChannel = numSamples * numBytesPerSample;

  auto status = avformat_write_header(avFormatContext_.get(), nullptr);
  TORCH_CHECK(
      status == AVSUCCESS,
      "Error in avformat_write_header: ",
      getFFMPEGErrorStringFromErrorCode(status));

  while (numEncodedSamples < numSamples) {
    int numSamplesToEncode =
        std::min(numSamplesAllocatedPerFrame, numSamples - numEncodedSamples);
    int numBytesToEncode = numSamplesToEncode * numBytesPerSample;

    for (int ch = 0; ch < samples_.sizes()[0]; ch++) {
      std::memcpy(
          avFrame->data[ch],
          psamples + ch * numBytesPerChannel,
          numBytesToEncode);
    }
    psamples += numBytesToEncode;

    // Above, we set the AVFrame's .nb_samples to AVCodecContext.frame_size so
    // that the frame buffers are allocated to a big enough size. Here, we reset
    // it to the exact number of samples that need to be encoded, otherwise the
    // encoded frame would contain more samples than necessary and our results
    // wouldn't match the ffmpeg CLI.
    avFrame->nb_samples = numSamplesToEncode;

    UniqueAVFrame convertedAVFrame = maybeConvertAVFrame(avFrame);
    encodeFrameThroughFifo(autoAVPacket, convertedAVFrame);

    numEncodedSamples += numSamplesToEncode;
  }
  TORCH_CHECK(numEncodedSamples == numSamples, "Hmmmmmm something went wrong.");

  flushBuffers();

  status = av_write_trailer(avFormatContext_.get());
  TORCH_CHECK(
      status == AVSUCCESS,
      "Error in: av_write_trailer",
      getFFMPEGErrorStringFromErrorCode(status));

  close_avio();
}

UniqueAVFrame AudioEncoder::maybeConvertAVFrame(const UniqueAVFrame& avFrame) {
  if (static_cast<AVSampleFormat>(avFrame->format) ==
          avCodecContext_->sample_fmt &&
      getNumChannels(avFrame) == outNumChannels_ &&
      avFrame->sample_rate == outSampleRate_) {
    // Note: the clone references the same underlying data, it's a cheap copy.
    return UniqueAVFrame(av_frame_clone(avFrame.get()));
  }

  if (!swrContext_) {
    swrContext_.reset(createSwrContext(
        static_cast<AVSampleFormat>(avFrame->format),
        avCodecContext_->sample_fmt,
        avFrame->sample_rate,
        outSampleRate_,
        avFrame,
        outNumChannels_));
  }
  UniqueAVFrame convertedAVFrame = convertAudioAVFrameSamples(
      swrContext_,
      avFrame,
      avCodecContext_->sample_fmt,
      outSampleRate_,
      outNumChannels_);

  if (avFrame->sample_rate == outSampleRate_) {
    TORCH_CHECK(
        convertedAVFrame->nb_samples == avFrame->nb_samples,
        "convertedAVFrame->nb_samples=",
        convertedAVFrame->nb_samples,
        " differs from ",
        "avFrame->nb_samples=",
        avFrame->nb_samples,
        "This is unexpected, please report on the TorchCodec bug tracker.");
  }
  return convertedAVFrame;
}

void AudioEncoder::encodeFrameThroughFifo(
    AutoAVPacket& autoAVPacket,
    const UniqueAVFrame& avFrame,
    // flushFifo is only set to true in maybeFlushSwrBuffers(), i.e. at the very
    // end of the encoding process when we're flushing buffers. We also want to
    // flush the FIFO so as to not leave any remaining samples in it.
    bool flushFifo) {
  if (avAudioFifo_ == nullptr) {
    encodeFrame(autoAVPacket, avFrame);
    return;
  }
  int numSamplesWritten = av_audio_fifo_write(
      avAudioFifo_.get(),
      reinterpret_cast<void**>(avFrame->data),
      avFrame->nb_samples);
  TORCH_CHECK(
      numSamplesWritten == avFrame->nb_samples,
      "Tried to write ",
      avFrame->nb_samples,
      " samples, but only wrote ",
      numSamplesWritten);

  UniqueAVFrame newavFrame = allocateAVFrame(
      avCodecContext_->frame_size,
      outSampleRate_,
      outNumChannels_,
      avCodecContext_->sample_fmt);

  // Explaining the while bound:
  // - if we're not flushing the FIFO, i.e. in most cases, we want to pull
  //   exactly `frame_size` samples from the FIFO, so we have to stop before it
  //   contains less than `frame_size` samples.
  // - if we're flushing the FIFO, we want to read from the FIFO until the very
  //   last sample it contains.
  //
  // In both cases, for as long as we can, we're trying to pull exatly
  // `frame_size` samples from the FIFO and send each `frame_size`-sized avFrame
  // to encodeFrame(). Only the very last avFrame of the encoding process is
  // allowed to contained less than frame_size samples. That only happens when
  // flushFifo is true.
  while (av_audio_fifo_size(avAudioFifo_.get()) >=
         (flushFifo ? 1 : avCodecContext_->frame_size)) {
    int samplesToRead = std::min(
        av_audio_fifo_size(avAudioFifo_.get()), newavFrame->nb_samples);
    int numSamplesRead = av_audio_fifo_read(
        avAudioFifo_.get(),
        reinterpret_cast<void**>(newavFrame->data),
        samplesToRead);
    TORCH_CHECK(
        numSamplesRead == samplesToRead,
        "Tried to read ",
        samplesToRead,
        " samples, but only read ",
        numSamplesRead);

    newavFrame->nb_samples = numSamplesRead;
    encodeFrame(autoAVPacket, newavFrame);
  }
}

void AudioEncoder::encodeFrame(
    AutoAVPacket& autoAVPacket,
    const UniqueAVFrame& avFrame) {
  if (avFrame != nullptr) {
    avFrame->pts = lastEncodedAVFramePts_;
    lastEncodedAVFramePts_ += avFrame->nb_samples;
  }

  auto status = avcodec_send_frame(avCodecContext_.get(), avFrame.get());
  TORCH_CHECK(
      status == AVSUCCESS,
      "Error while sending frame: ",
      getFFMPEGErrorStringFromErrorCode(status));

  while (status >= 0) {
    ReferenceAVPacket packet(autoAVPacket);
    status = avcodec_receive_packet(avCodecContext_.get(), packet.get());
    if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) {
      if (status == AVERROR_EOF) {
        // Flush the packets that were potentially buffered by
        // av_interleaved_write_frame(). See corresponding block in
        // TorchAudio:
        // https://github.com/pytorch/audio/blob/d60ce09e2c532d5bf2e05619e700ab520543465e/src/libtorio/ffmpeg/stream_writer/encoder.cpp#L21
        status = av_interleaved_write_frame(avFormatContext_.get(), nullptr);
        TORCH_CHECK(
            status == AVSUCCESS,
            "Failed to flush packet: ",
            getFFMPEGErrorStringFromErrorCode(status));
      }
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

void AudioEncoder::maybeFlushSwrBuffers(AutoAVPacket& autoAVPacket) {
  // Similar to the decoder's method with the same name, but for encoding this
  // time. That is, when sample conversion is involved, libswresample may have
  // buffered some samples that we now need to flush and send to the encoder.
  if (swrContext_ == nullptr && inSampleRate_ == outSampleRate_) {
    return;
  }
  TORCH_CHECK(
      swrContext_ != nullptr,
      "swrContext is null, but sample rate conversion is needed. ",
      "This is unexpected, please report on the TorchCodec bug tracker.");

  int numRemainingSamples = // this is an upper bound
      swr_get_out_samples(swrContext_.get(), 0);
  if (numRemainingSamples == 0) {
    return;
  }

  UniqueAVFrame avFrame = allocateAVFrame(
      numRemainingSamples,
      outSampleRate_,
      outNumChannels_,
      avCodecContext_->sample_fmt);
  int actualNumRemainingSamples = swr_convert(
      swrContext_.get(), avFrame->data, avFrame->nb_samples, nullptr, 0);
  avFrame->nb_samples = actualNumRemainingSamples;

  // We're potentially sending avFrame through the FIFO (if it exists), in which
  // case we also want to flush the FIFO itself.
  encodeFrameThroughFifo(autoAVPacket, avFrame, /*flushFifo=*/true);
}

void AudioEncoder::flushBuffers() {
  AutoAVPacket autoAVPacket;
  maybeFlushSwrBuffers(autoAVPacket);

  encodeFrame(autoAVPacket, UniqueAVFrame(nullptr));
}
} // namespace facebook::torchcodec
