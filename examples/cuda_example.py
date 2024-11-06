"""
Accelerated video decoding with NVDEC
=====================================

.. _nvdec_tutorial:

**Author**: `Ahmad Sharif <ahmads@meta.com>`__

This tutorial shows how to use NVIDIA’s hardware video decoder (NVDEC)
with TorchCodec, and how it improves the performance of video decoding.
"""

######################################################################
#
# .. note::
#
#    This tutorial requires FFmpeg libraries compiled with HW
#    acceleration enabled.
#
#    Please refer to
#    :ref:`Enabling GPU video decoder/encoder <enabling_hw_decoder>`
#    for how to build FFmpeg with HW acceleration.
#

import torch

print(torch.__version__)
print(torchaudio.__version__)

######################################################################
#

import matplotlib.pyplot as plt
from torchcodec import VideoDecoder

print("Avaialbe GPU:")
print(torch.cuda.get_device_properties(0))

######################################################################
#
# We will use the following video which has the following properties;
#
# - Codec: H.264
# - Resolution: 960x540
# - FPS: 29.97
# - Pixel format: YUV420P
#
# .. raw:: html
#
#    <video style="max-width: 100%" controls>
#      <source src="https://download.pytorch.org/torchaudio/tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4" type="video/mp4">
#    </video>

######################################################################
#

src = torchaudio.utils.download_asset(
    "tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4"
)

######################################################################
# Decoding videos with NVDEC
# --------------------------
#
# To use HW video decoder, you need to specify the HW decoder when
# defining the output video stream by passing ``decoder`` option to
# :py:meth:`~torchaudio.io.StreamReader.add_video_stream` method.
#

vd = VideoDecoder(src)
vd.add_video_stream(0, device="cuda:0")
frame = vd[0]

######################################################################
#
# The video frames are decoded and returned as tensor of NCHW format.

print(frame.data.shape, frame.data.dtype)

######################################################################
#
# By default, the decoded frames are sent back to CPU memory, and
# CPU tensors are created.

print(frame.data.device)


######################################################################
# .. note::
#
#    When there are multiple of GPUs available, ``StreamReader`` by
#    default uses the first GPU. You can change this by providing
#    ``"gpu"`` option.
#
# .. code::
#
#    # Video data is sent to CUDA device 0, decoded and
#    # converted on the same device.
#    s.add_video_stream(
#        ...,
#        decoder="h264_cuvid",
#        decoder_option={"gpu": "0"},
#        hw_accel="cuda:0",
#    )
#
# .. note::
#
#    ``"gpu"`` option and ``hw_accel`` option can be specified
#    independently. If they do not match, decoded frames are
#    transfered to the device specified by ``hw_accell``
#    automatically.
#
# .. code::
#
#    # Video data is sent to CUDA device 0, and decoded there.
#    # Then it is transfered to CUDA device 1, and converted to
#    # CUDA tensor.
#    s.add_video_stream(
#        ...,
#        decoder="h264_cuvid",
#        decoder_option={"gpu": "0"},
#        hw_accel="cuda:1",
#    )

######################################################################
# Visualization
# -------------
#
# Let's look at the frames decoded by HW decoder and compare them
# against equivalent results from software decoders.
#
# The following function seeks into the given timestamp and decode one
# frame with the specificed decoder.


def test_decode(decoder: str, seek: float):
    vd = VideoDecoder(src)
    return vd.get_frame_played_at(seek)


######################################################################
#

timestamps = [12, 19, 45, 131, 180]

cpu_frames = [test_decode(decoder="h264", seek=ts) for ts in timestamps]
cuda_frames = [test_decode(decoder="h264_cuvid", seek=ts) for ts in timestamps]


######################################################################
#
# Now we visualize the resutls.
#


def plot_cpu_and_cuda():
    n_rows = len(timestamps)
    fig, axes = plt.subplots(n_rows, 2, figsize=[12.8, 16.0])
    for i in range(n_rows):
        axes[i][0].imshow(cpu_frames[i])
        axes[i][1].imshow(cuda_frames[i])

    axes[0][0].set_title("Software decoder")
    axes[0][1].set_title("HW decoder")
    plt.setp(axes, xticks=[], yticks=[])
    plt.tight_layout()


plot_cpu_and_cuda()

######################################################################
#
# They are indistinguishable to the eyes of the author.
# Feel free to let us know if you spot something. :)
#
