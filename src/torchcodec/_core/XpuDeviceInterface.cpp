#include <unistd.h>

#include <level_zero/ze_api.h>
#include <va/va_drmcommon.h>

#include <ATen/DLConvertor.h>
#include <c10/xpu/XPUStream.h>

#include "src/torchcodec/_core/XpuDeviceInterface.h"
#include "src/torchcodec/_core/FFMPEGCommon.h"

extern "C" {
#include <libavutil/hwcontext_vaapi.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {
namespace {

static bool g_xpu =
    registerDeviceInterface(torch::kXPU, [](const torch::Device& device) {
      return new XpuDeviceInterface(device);
    });

const int MAX_XPU_GPUS = 128;
// Set to -1 to have an infinitely sized cache. Set it to 0 to disable caching.
// Set to a positive number to have a cache of that size.
const int MAX_CONTEXTS_PER_GPU_IN_CACHE = -1;
std::vector<AVBufferRef*> g_cached_hw_device_ctxs[MAX_XPU_GPUS];
std::mutex g_cached_hw_device_mutexes[MAX_XPU_GPUS];

torch::DeviceIndex getFFMPEGCompatibleDeviceIndex(const torch::Device& device) {
  torch::DeviceIndex deviceIndex = device.index();
  deviceIndex = std::max<at::DeviceIndex>(deviceIndex, 0);
  TORCH_CHECK(deviceIndex >= 0, "Device index out of range");
  // For single GPU- machines libtorch returns -1 for the device index. So for
  // that case we set the device index to 0.
  return deviceIndex;
}

void addToCacheIfCacheHasCapacity(
    const torch::Device& device,
    AVBufferRef* hwContext) {
  torch::DeviceIndex deviceIndex = getFFMPEGCompatibleDeviceIndex(device);
  if (static_cast<int>(deviceIndex) >= MAX_XPU_GPUS) {
    return;
  }
  std::scoped_lock lock(g_cached_hw_device_mutexes[deviceIndex]);
  if (MAX_CONTEXTS_PER_GPU_IN_CACHE >= 0 &&
      g_cached_hw_device_ctxs[deviceIndex].size() >=
          MAX_CONTEXTS_PER_GPU_IN_CACHE) {
    return;
  }
  g_cached_hw_device_ctxs[deviceIndex].push_back(av_buffer_ref(hwContext));
}

AVBufferRef* getFromCache(const torch::Device& device) {
  torch::DeviceIndex deviceIndex = getFFMPEGCompatibleDeviceIndex(device);
  if (static_cast<int>(deviceIndex) >= MAX_XPU_GPUS) {
    return nullptr;
  }
  std::scoped_lock lock(g_cached_hw_device_mutexes[deviceIndex]);
  if (g_cached_hw_device_ctxs[deviceIndex].size() > 0) {
    AVBufferRef* hw_device_ctx = g_cached_hw_device_ctxs[deviceIndex].back();
    g_cached_hw_device_ctxs[deviceIndex].pop_back();
    return hw_device_ctx;
  }
  return nullptr;
}

AVBufferRef* getVaapiContext(const torch::Device& device) {
  enum AVHWDeviceType type = av_hwdevice_find_type_by_name("vaapi");
  TORCH_CHECK(type != AV_HWDEVICE_TYPE_NONE, "Failed to find vaapi device");
  torch::DeviceIndex nonNegativeDeviceIndex =
      getFFMPEGCompatibleDeviceIndex(device);

  AVBufferRef* hw_device_ctx = getFromCache(device);
  if (hw_device_ctx != nullptr) {
    return hw_device_ctx;
  }

  std::string renderD = "/dev/dri/renderD128";

  sycl::device syclDevice = c10::xpu::get_raw_device(nonNegativeDeviceIndex);
  if (syclDevice.has(sycl::aspect::ext_intel_pci_address)) {
    auto BDF =
        syclDevice.get_info<sycl::ext::intel::info::device::pci_address>();
    renderD = "/dev/dri/by-path/pci-" + BDF + "-render";
  }

  int err =
      av_hwdevice_ctx_create(&hw_device_ctx, type, renderD.c_str(), nullptr, 0);
  if (err < 0) {
    TORCH_CHECK(
        false,
        "Failed to create specified HW device: ",
        getFFMPEGErrorStringFromErrorCode(err));
  }
  return hw_device_ctx;
}

} // namespace

XpuDeviceInterface::XpuDeviceInterface(const torch::Device& device)
    : DeviceInterface(device) {
  TORCH_CHECK(g_xpu, "XpuDeviceInterface was not registered!");
  TORCH_CHECK(
      device_.type() == torch::kXPU, "Unsupported device: ", device_.str());
}

XpuDeviceInterface::~XpuDeviceInterface() {
  if (ctx_) {
    addToCacheIfCacheHasCapacity(device_, ctx_);
    av_buffer_unref(&ctx_);
  }
}

void XpuDeviceInterface::initializeContext(AVCodecContext* codecContext) {
  TORCH_CHECK(!ctx_, "FFmpeg HW device context already initialized");

  // It is important for pytorch itself to create the xpu context. If ffmpeg
  // creates the context it may not be compatible with pytorch.
  // This is a dummy tensor to initialize the xpu context.
  torch::Tensor dummyTensorForXpuInitialization = torch::empty(
      {1}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));
  ctx_ = getVaapiContext(device_);
  codecContext->hw_device_ctx = av_buffer_ref(ctx_);
  return;
}

struct vaapiSurface {
  vaapiSurface(VADisplay dpy, uint32_t width, uint32_t height);

  ~vaapiSurface() {
    vaDestroySurfaces(dpy_, &id_, 1);
  }

  inline VASurfaceID id() const {
    return id_;
  }

  torch::Tensor toTensor(const torch::Device& device);

 private:
  VADisplay dpy_;
  VASurfaceID id_;
};

vaapiSurface::vaapiSurface(VADisplay dpy, uint32_t width, uint32_t height)
    : dpy_(dpy) {
  VASurfaceAttrib attrib{};

  attrib.type = VASurfaceAttribPixelFormat;
  attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
  attrib.value.type = VAGenericValueTypeInteger;
  attrib.value.value.i = VA_FOURCC_RGBX;

  VAStatus res = vaCreateSurfaces(
      dpy_, VA_RT_FORMAT_RGB32, width, height, &id_, 1, &attrib, 1);
  TORCH_CHECK(
      res == VA_STATUS_SUCCESS,
      "Failed to create VAAPI surface: ",
      vaErrorStr(res));
}

void deleter(DLManagedTensor* self) {
  std::unique_ptr<DLManagedTensor> tensor(self);
  std::unique_ptr<ze_context_handle_t> context(
      (ze_context_handle_t*)self->manager_ctx);
  zeMemFree(*context, self->dl_tensor.data);
}

torch::Tensor vaapiSurface::toTensor(const torch::Device& device) {
  VADRMPRIMESurfaceDescriptor desc{};

  VAStatus sts = vaExportSurfaceHandle(
      dpy_,
      id_,
      VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
      VA_EXPORT_SURFACE_READ_ONLY,
      &desc);
  TORCH_CHECK(
      sts == VA_STATUS_SUCCESS,
      "vaExportSurfaceHandle failed: ",
      vaErrorStr(sts));

  TORCH_CHECK(desc.num_objects == 1, "Expected 1 fd, got ", desc.num_objects);
  TORCH_CHECK(desc.num_layers == 1, "Expected 1 layer, got ", desc.num_layers);
  TORCH_CHECK(
      desc.layers[0].num_planes == 1,
      "Expected 1 plane, got ",
      desc.num_layers);

  std::unique_ptr<ze_context_handle_t> ze_context =
      std::make_unique<ze_context_handle_t>();
  ze_device_handle_t ze_device{};
  sycl::queue queue = c10::xpu::getCurrentXPUStream(device.index());

  queue
      .submit([&](sycl::handler& cgh) {
        cgh.host_task([&](const sycl::interop_handle& ih) {
          *ze_context =
              ih.get_native_context<sycl::backend::ext_oneapi_level_zero>();
          ze_device =
              ih.get_native_device<sycl::backend::ext_oneapi_level_zero>();
        });
      })
      .wait();

  ze_external_memory_import_fd_t import_fd_desc{};
  import_fd_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
  import_fd_desc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
  import_fd_desc.fd = desc.objects[0].fd;

  ze_device_mem_alloc_desc_t alloc_desc{};
  alloc_desc.pNext = &import_fd_desc;
  void* usm_ptr = nullptr;

  ze_result_t res = zeMemAllocDevice(
      *ze_context, &alloc_desc, desc.objects[0].size, 0, ze_device, &usm_ptr);
  TORCH_CHECK(
      res == ZE_RESULT_SUCCESS, "Failed to import fd=", desc.objects[0].fd);

  close(desc.objects[0].fd);

  std::unique_ptr<DLManagedTensor> dl_dst = std::make_unique<DLManagedTensor>();
  int64_t shape[3] = {desc.height, desc.width, 4};

  dl_dst->manager_ctx = ze_context.release();
  dl_dst->deleter = deleter;
  dl_dst->dl_tensor.data = usm_ptr;
  dl_dst->dl_tensor.device.device_type = kDLOneAPI;
  dl_dst->dl_tensor.device.device_id = device.index();
  dl_dst->dl_tensor.ndim = 3;
  dl_dst->dl_tensor.dtype.code = kDLUInt;
  dl_dst->dl_tensor.dtype.bits = 8;
  dl_dst->dl_tensor.dtype.lanes = 1;
  dl_dst->dl_tensor.shape = shape;
  dl_dst->dl_tensor.strides = nullptr;
  dl_dst->dl_tensor.byte_offset = desc.layers[0].offset[0];

  auto dst = at::fromDLPack(dl_dst.release());

  return dst;
}

VADisplay getVaDisplayFromAV(UniqueAVFrame& avFrame) {
  AVHWFramesContext* hwfc = (AVHWFramesContext*)avFrame->hw_frames_ctx->data;
  AVHWDeviceContext* hwdc = hwfc->device_ctx;
  AVVAAPIDeviceContext* vactx = (AVVAAPIDeviceContext*)hwdc->hwctx;
  return vactx->display;
}

struct vaapiVpContext {
  VADisplay dpy_;
  VAConfigID config_id_ = VA_INVALID_ID;
  VAContextID context_id_ = VA_INVALID_ID;
  VABufferID pipeline_buf_id_ = VA_INVALID_ID;

  // These structures must be available thru all life
  // circle of the struct since they are reused by the media
  // driver internally during vaRenderPicture().
  VAProcPipelineParameterBuffer pipeline_{};
  VARectangle surface_region_{};

  vaapiVpContext() = delete;
  vaapiVpContext(
      VADisplay dpy,
      UniqueAVFrame& avFrame,
      uint16_t width,
      uint16_t height);

  ~vaapiVpContext() {
    if (pipeline_buf_id_ != VA_INVALID_ID)
      vaDestroyBuffer(dpy_, pipeline_buf_id_);
    if (context_id_ != VA_INVALID_ID)
      vaDestroyContext(dpy_, context_id_);
    if (config_id_ != VA_INVALID_ID)
      vaDestroyConfig(dpy_, config_id_);
  }

  void convertTo(VASurfaceID id);
};

vaapiVpContext::vaapiVpContext(
    VADisplay dpy,
    UniqueAVFrame& avFrame,
    uint16_t width,
    uint16_t height)
    : dpy_(dpy) {
  VAStatus res = vaCreateConfig(
      dpy_, VAProfileNone, VAEntrypointVideoProc, nullptr, 0, &config_id_);
  TORCH_CHECK(
      res == VA_STATUS_SUCCESS,
      "Failed to create VAAPI config: ",
      vaErrorStr(res));

  res = vaCreateContext(
      dpy_,
      config_id_,
      width,
      height,
      VA_PROGRESSIVE,
      nullptr,
      0,
      &context_id_);
  TORCH_CHECK(
      res == VA_STATUS_SUCCESS,
      "Failed to create VAAPI VP context: ",
      vaErrorStr(res));

  surface_region_.width = width;
  surface_region_.height = height;

  pipeline_.surface = (VASurfaceID)(uintptr_t)avFrame->data[3];
  pipeline_.surface_region = &surface_region_;
  pipeline_.output_region = &surface_region_;
  if (avFrame->colorspace == AVColorSpace::AVCOL_SPC_BT709)
    pipeline_.surface_color_standard = VAProcColorStandardBT709;

  res = vaCreateBuffer(
      dpy_,
      context_id_,
      VAProcPipelineParameterBufferType,
      sizeof(pipeline_),
      1,
      &pipeline_,
      &pipeline_buf_id_);
  TORCH_CHECK(
      res == VA_STATUS_SUCCESS, "vaCreateBuffer failed: ", vaErrorStr(res));
}

void vaapiVpContext::convertTo(VASurfaceID id) {
  VAStatus res = vaBeginPicture(dpy_, context_id_, id);
  TORCH_CHECK(
      res == VA_STATUS_SUCCESS, "vaBeginPicture failed: ", vaErrorStr(res));

  res = vaRenderPicture(dpy_, context_id_, &pipeline_buf_id_, 1);
  TORCH_CHECK(
      res == VA_STATUS_SUCCESS, "vaRenderPicture failed: ", vaErrorStr(res));

  res = vaEndPicture(dpy_, context_id_);
  TORCH_CHECK(
      res == VA_STATUS_SUCCESS, "vaEndPicture failed: ", vaErrorStr(res));

  res = vaSyncSurface(dpy_, id);
  TORCH_CHECK(
      res == VA_STATUS_SUCCESS, "vaSyncSurface failed: ", vaErrorStr(res));
}

torch::Tensor convertAVFrameToTensor(
    const torch::Device& device,
    UniqueAVFrame& avFrame,
    int width,
    int height) {
  TORCH_CHECK(height > 0, "height must be > 0, got: ", height);
  TORCH_CHECK(width > 0, "width must be > 0, got: ", width);

  // Allocating intermediate tensor we can convert input to with VAAPI.
  // This tensor should be WxHx4 since VAAPI does not support RGB24
  // and works only with RGB32.
  VADisplay va_dpy = getVaDisplayFromAV(avFrame);
  // Importing tensor to VAAPI.
  vaapiSurface va_surface(va_dpy, width, height);

  vaapiVpContext va_vp(va_dpy, avFrame, width, height);
  va_vp.convertTo(va_surface.id());

  return va_surface.toTensor(device);
}

void XpuDeviceInterface::convertAVFrameToFrameOutput(
    const VideoStreamOptions& videoStreamOptions,
    [[maybe_unused]] const AVRational& timeBase,
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  // TODO: consider to copy handling of CPU frame from CUDA
  // TODO: consider to copy NV12 format check from CUDA
  TORCH_CHECK(
      avFrame->format == AV_PIX_FMT_VAAPI,
      "Expected format to be AV_PIX_FMT_VAAPI, got " +
          std::string(av_get_pix_fmt_name((AVPixelFormat)avFrame->format)));
  auto frameDims =
      getHeightAndWidthFromOptionsOrAVFrame(videoStreamOptions, avFrame);
  int height = frameDims.height;
  int width = frameDims.width;
  torch::Tensor& dst = frameOutput.data;
  if (preAllocatedOutputTensor.has_value()) {
    dst = preAllocatedOutputTensor.value();
    auto shape = dst.sizes();
    TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == height) && (shape[1] == width) &&
            (shape[2] == 3),
        "Expected tensor of shape ",
        height,
        "x",
        width,
        "x3, got ",
        shape);
  } else {
    dst = allocateEmptyHWCTensor(height, width, device_);
  }

  auto start = std::chrono::high_resolution_clock::now();

  // We convert input to the RGBX color format with VAAPI getting WxHx4
  // tensor on the output.
  torch::Tensor dst_rgb4 =
      convertAVFrameToTensor(device_, avFrame, width, height);
  dst.copy_(dst_rgb4.narrow(2, 0, 3));

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::micro> duration = end - start;
  VLOG(9) << "NPP Conversion of frame height=" << height << " width=" << width
          << " took: " << duration.count() << "us" << std::endl;
}

// inspired by https://github.com/FFmpeg/FFmpeg/commit/ad67ea9
// we have to do this because of an FFmpeg bug where hardware decoding is not
// appropriately set, so we just go off and find the matching codec for the CUDA
// device
std::optional<const AVCodec*> XpuDeviceInterface::findCodec(
    const AVCodecID& codecId) {
  void* i = nullptr;
  const AVCodec* codec = nullptr;
  while ((codec = av_codec_iterate(&i)) != nullptr) {
    if (codec->id != codecId || !av_codec_is_decoder(codec)) {
      continue;
    }

    const AVCodecHWConfig* config = nullptr;
    for (int j = 0; (config = avcodec_get_hw_config(codec, j)) != nullptr;
         ++j) {
      if (config->device_type == AV_HWDEVICE_TYPE_VAAPI) {
        return codec;
      }
    }
  }

  return std::nullopt;
}

} // namespace facebook::torchcodec
