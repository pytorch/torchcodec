#include <c10/cuda/CUDAStream.h>
#include <npp.h>
#include <torch/types.h>
#include "src/torchcodec/decoders/_core/DeviceInterface.h"
#include "src/torchcodec/decoders/_core/FFMPEGCommon.h"
#include "src/torchcodec/decoders/_core/VideoDecoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {
namespace {
AVBufferRef* getCudaContext(const torch::Device& device) {
  enum AVHWDeviceType type = av_hwdevice_find_type_by_name("cuda");
  TORCH_CHECK(type != AV_HWDEVICE_TYPE_NONE, "Failed to find cuda device");
  int err = 0;
  AVBufferRef* hw_device_ctx;
  torch::DeviceIndex deviceIndex = device.index();
  deviceIndex = std::max<at::DeviceIndex>(deviceIndex, 0);
  std::string deviceOrdinal = std::to_string(deviceIndex);
  err = av_hwdevice_ctx_create(
      &hw_device_ctx, type, deviceOrdinal.c_str(), nullptr, 0);
  if (err < 0) {
    TORCH_CHECK(
        false,
        "Failed to create specified HW device",
        getFFMPEGErrorStringFromErrorCode(err));
  }
  return hw_device_ctx;
}

torch::Tensor allocateDeviceTensor(
    at::IntArrayRef shape,
    torch::Device device,
    const torch::Dtype dtype = torch::kUInt8) {
  return torch::empty(
      shape,
      torch::TensorOptions()
          .dtype(dtype)
          .layout(torch::kStrided)
          .device(device));
}

void throwErrorIfNonCudaDevice(const torch::Device& device) {
  TORCH_CHECK(
      device.type() != torch::kCPU,
      "Device functions should only be called if the device is not CPU.")
  if (device.type() != torch::kCUDA) {
    throw std::runtime_error("Unsupported device: " + device.str());
  }
}
} // namespace

void initializeDeviceContext(
    const torch::Device& device,
    AVCodecContext* codecContext) {
  throwErrorIfNonCudaDevice(device);
  // This is a dummy tensor to initialize the cuda context.
  torch::Tensor dummyTensorForCudaInitialization = torch::empty(
      {1}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
  codecContext->hw_device_ctx = av_buffer_ref(getCudaContext(device));
  return;
}

void convertAVFrameToDecodedOutputOnDevice(
    const torch::Device& device,
    const VideoDecoder::VideoStreamDecoderOptions& options,
    AVCodecContext* codecContext,
    VideoDecoder::RawDecodedOutput& rawOutput,
    VideoDecoder::DecodedOutput& output) {
  AVFrame* src = rawOutput.frame.get();

  TORCH_CHECK(
      src->format == AV_PIX_FMT_CUDA,
      "Expected format to be AV_PIX_FMT_CUDA, got " +
          std::string(av_get_pix_fmt_name((AVPixelFormat)src->format)));
  int width = options.width.value_or(codecContext->width);
  int height = options.height.value_or(codecContext->height);
  NppStatus status;
  NppiSize oSizeROI;
  oSizeROI.width = width;
  oSizeROI.height = height;
  Npp8u* input[2];
  input[0] = (Npp8u*)src->data[0];
  input[1] = (Npp8u*)src->data[1];
  torch::Tensor& dst = output.frame;
  dst = allocateDeviceTensor({height, width, 3}, options.device);
  at::DeviceIndex deviceIndex = device.index();
  deviceIndex = std::max<at::DeviceIndex>(deviceIndex, 0);
  at::DeviceIndex originalDeviceIndex = at::cuda::current_device();
  cudaSetDevice(deviceIndex);

  auto start = std::chrono::high_resolution_clock::now();

  cudaStream_t nppStream = nppGetStream();
  cudaStream_t torchStream = at::cuda::getCurrentCUDAStream().stream();
  status = nppiNV12ToRGB_8u_P2C3R(
      input,
      src->linesize[0],
      static_cast<Npp8u*>(dst.data_ptr()),
      dst.stride(0),
      oSizeROI);
  // Make the pytorch stream wait for the npp kernel to finish before using the
  // output.
  cudaEvent_t nppDoneEvent;
  cudaEventCreate(&nppDoneEvent);
  cudaEventRecord(nppDoneEvent, nppStream);
  cudaEvent_t torchDoneEvent;
  cudaEventCreate(&torchDoneEvent);
  cudaEventRecord(torchDoneEvent, torchStream);
  cudaStreamWaitEvent(torchStream, nppDoneEvent, 0);
  TORCH_CHECK(status == NPP_SUCCESS, "Failed to convert NV12 frame.");
  cudaEventDestroy(nppDoneEvent);
  cudaEventDestroy(torchDoneEvent);

  auto end = std::chrono::high_resolution_clock::now();

  // Restore the original device_index.
  cudaSetDevice(originalDeviceIndex);

  std::chrono::duration<double, std::micro> duration = end - start;
  VLOG(9) << "NPP Conversion of frame height=" << height << " width=" << width
          << " took: " << duration.count() << "us" << std::endl;
  if (options.dimensionOrder == "NCHW") {
    // The docs guaranty this to return a view:
    // https://pytorch.org/docs/stable/generated/torch.permute.html
    dst = dst.permute({2, 0, 1});
  }
}

} // namespace facebook::torchcodec
