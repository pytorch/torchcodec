#include "src/torchcodec/_core/Encoder.h"
#include "torch/types.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

namespace facebook::torchcodec {

Encoder::~Encoder() {}

Encoder::Encoder(int sampleRate, std::string_view fileName)
    : sampleRate_(sampleRate) {
  AVFormatContext* avFormatContext = nullptr;
  avformat_alloc_output_context2(&avFormatContext, NULL, NULL, fileName.data());
  TORCH_CHECK(avFormatContext != nullptr, "Couldn't allocate AVFormatContext.");
  avFormatContext_.reset(avFormatContext);

  TORCH_CHECK(
      !(avFormatContext->oformat->flags & AVFMT_NOFILE),
      "AVFMT_NOFILE is set. We only support writing to a file.");
  auto ffmpegRet =
      avio_open(&avFormatContext_->pb, fileName.data(), AVIO_FLAG_WRITE);
  TORCH_CHECK(
      ffmpegRet >= 0,
      "avio_open failed: ",
      getFFMPEGErrorStringFromErrorCode(ffmpegRet));

  // We use the AVFormatContext's default codec for that
  // specificavcodec_parameters_from_context format/container.
  const AVCodec* avCodec =
      avcodec_find_encoder(avFormatContext_->oformat->audio_codec);
  TORCH_CHECK(avCodec != nullptr, "Codec not found");

  AVCodecContext* avCodecContext = avcodec_alloc_context3(avCodec);
  TORCH_CHECK(avCodecContext != nullptr, "Couldn't allocate codec context.");
  avCodecContext_.reset(avCodecContext);

  // This will use the default bit rate
  // TODO-ENCODING Should let user choose for compressed formats like mp3.
  avCodecContext_->bit_rate = 0;

  // FFmpeg will raise a reasonably informative error if the desired sample rate
  // isn't supported by the encoder.
  avCodecContext_->sample_rate = sampleRate_;

  // Note: This is the format of the **input** waveform. This doesn't determine
  // the output.
  // TODO-ENCODING check contiguity of the input wf to ensure that it is indeed
  // planar.
  // TODO-ENCODING If the encoder doesn't support FLTP (like flac), FFmpeg will
  // raise. We need to handle this, probably converting the format with
  // libswresample.
  avCodecContext_->sample_fmt = AV_SAMPLE_FMT_FLTP;

  AVChannelLayout channel_layout;
  av_channel_layout_default(&channel_layout, 2);
  avCodecContext_->ch_layout = channel_layout;

  ffmpegRet = avcodec_open2(avCodecContext_.get(), avCodec, nullptr);
  TORCH_CHECK(
      ffmpegRet == AVSUCCESS, getFFMPEGErrorStringFromErrorCode(ffmpegRet));

  TORCH_CHECK(
      avCodecContext_->frame_size > 0,
      "frame_size is ",
      avCodecContext_->frame_size,
      ". Cannot encode. This should probably never happen?");

  // We're allocating the stream here. Streams are meant to be freed by
  // avformat_free_context(avFormatContext), which we call in the
  // avFormatContext_'s destructor.
  avStream_ = avformat_new_stream(avFormatContext_.get(), NULL);
  TORCH_CHECK(avStream_ != nullptr, "Couldn't create new stream.");
  avcodec_parameters_from_context(avStream_->codecpar, avCodecContext_.get());
}

void Encoder::encode(const torch::Tensor& wf) {
  UniqueAVFrame avFrame(av_frame_alloc());
  TORCH_CHECK(avFrame != nullptr, "Couldn't allocate AVFrame.");
  avFrame->nb_samples = avCodecContext_->frame_size;
  avFrame->format = avCodecContext_->sample_fmt;
  avFrame->sample_rate = avCodecContext_->sample_rate;
  avFrame->pts = 0;
  auto ffmpegRet =
      av_channel_layout_copy(&avFrame->ch_layout, &avCodecContext_->ch_layout);
  TORCH_CHECK(
      ffmpegRet == AVSUCCESS,
      "Couldn't copy channel layout to avFrame: ",
      getFFMPEGErrorStringFromErrorCode(ffmpegRet));

  ffmpegRet = av_frame_get_buffer(avFrame.get(), 0);
  TORCH_CHECK(
      ffmpegRet == AVSUCCESS,
      "Couldn't allocate avFrame's buffers: ",
      getFFMPEGErrorStringFromErrorCode(ffmpegRet));

  AutoAVPacket autoAVPacket;

  uint8_t* pWf = static_cast<uint8_t*>(wf.data_ptr());
  auto numChannels = wf.sizes()[0];
  auto numSamples = wf.sizes()[1]; // per channel
  auto numEncodedSamples = 0; // per channel
  auto numSamplesPerFrame =
      static_cast<long>(avCodecContext_->frame_size); // per channel
  auto numBytesPerSample = wf.element_size();
  auto numBytesPerChannel = wf.sizes()[1] * numBytesPerSample;

  TORCH_CHECK(
      // TODO-ENCODING is this even true / needed? We can probably support more
      // with non-planar data?
      numChannels <= AV_NUM_DATA_POINTERS,
      "Trying to encode ",
      numChannels,
      " channels, but FFmpeg only supports ",
      AV_NUM_DATA_POINTERS,
      " channels per frame.");

  ffmpegRet = avformat_write_header(avFormatContext_.get(), NULL);
  TORCH_CHECK(
      ffmpegRet == AVSUCCESS,
      "Error in avformat_write_header: ",
      getFFMPEGErrorStringFromErrorCode(ffmpegRet));

  while (numEncodedSamples < numSamples) {
    ffmpegRet = av_frame_make_writable(avFrame.get());
    TORCH_CHECK(
        ffmpegRet == AVSUCCESS,
        "Couldn't make AVFrame writable: ",
        getFFMPEGErrorStringFromErrorCode(ffmpegRet));

    auto numSamplesToEncode =
        std::min(numSamplesPerFrame, numSamples - numEncodedSamples);
    auto numBytesToEncode = numSamplesToEncode * numBytesPerSample;

    for (int ch = 0; ch < numChannels; ch++) {
      memcpy(
          avFrame->data[ch], pWf + ch * numBytesPerChannel, numBytesToEncode);
    }
    pWf += numBytesToEncode;
    encode_inner_loop(autoAVPacket, avFrame);

    avFrame->pts += avFrame->nb_samples;
    numEncodedSamples += numSamplesToEncode;
  }
  TORCH_CHECK(numEncodedSamples == numSamples, "Hmmmmmm something went wrong.");

  encode_inner_loop(autoAVPacket, UniqueAVFrame(nullptr)); // flush

  ffmpegRet = av_write_trailer(avFormatContext_.get());
  TORCH_CHECK(
      ffmpegRet == AVSUCCESS,
      "Error in: av_write_trailer",
      getFFMPEGErrorStringFromErrorCode(ffmpegRet));
}

void Encoder::encode_inner_loop(
    AutoAVPacket& autoAVPacket,
    const UniqueAVFrame& avFrame) {
  auto ffmpegRet = avcodec_send_frame(avCodecContext_.get(), avFrame.get());
  TORCH_CHECK(
      ffmpegRet == AVSUCCESS,
      "Error while sending frame: ",
      getFFMPEGErrorStringFromErrorCode(ffmpegRet));

  while (ffmpegRet >= 0) {
    ReferenceAVPacket packet(autoAVPacket);
    ffmpegRet = avcodec_receive_packet(avCodecContext_.get(), packet.get());
    if (ffmpegRet == AVERROR(EAGAIN) || ffmpegRet == AVERROR_EOF) {
      // TODO-ENCODING this is from TorchAudio, probably needed, but not sure.
      //   if (ffmpegRet == AVERROR_EOF) {
      //     ffmpegRet = av_interleaved_write_frame(avFormatContext_.get(),
      //     nullptr); TORCH_CHECK(
      //         ffmpegRet == AVSUCCESS,
      //         "Failed to flush packet ",
      //         getFFMPEGErrorStringFromErrorCode(ffmpegRet));
      //   }
      return;
    }
    TORCH_CHECK(
        ffmpegRet >= 0,
        "Error receiving packet: ",
        getFFMPEGErrorStringFromErrorCode(ffmpegRet));

    // TODO-ENCODING why are these 2 lines needed??
    av_packet_rescale_ts(
        packet.get(), avCodecContext_->time_base, avStream_->time_base);
    packet->stream_index = avStream_->index;

    ffmpegRet =
        av_interleaved_write_frame(avFormatContext_.get(), packet.get());
    TORCH_CHECK(
        ffmpegRet == AVSUCCESS,
        "Error in av_interleaved_write_frame: ",
        getFFMPEGErrorStringFromErrorCode(ffmpegRet));
  }
}
} // namespace facebook::torchcodec
