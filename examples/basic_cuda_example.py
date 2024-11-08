# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Accelerated video decoding with NVDEC
=====================================

.. _nvdec_tutorial:

**Author**: `Ahmad Sharif <ahmads@meta.com>`__

TorchCodec can use Nvidia hardware to speed-up video decoding. This is called "CUDA Decoding".
CUDA Decoding can be faster than CPU Decoding for the actual decoding step and for
subsequent transform steps like scaling, cropping or rotating. This is because the decode step leaves
the decoded tensor in GPU memory so the GPU doesn't have to fetch from main memory before
running the transform steps. Encoded packets are often much smaller than decoded frames so
CUDA decoding also uses less PCI-e bandwidth.

CUDA Decoding can offer speed-up over CPU Decoding in a few scenarios:

#. You are decoding a large resolution video
#. You are decoding a large batch of videos that's saturating the CPU
#. You want to do whole-image transforms like scaling or convolutions on the decoded tensors
   after decoding
#. Your CPU is saturated and you want to free it up for other work

Here are situations where CUDA Decoding may not make sense:

#. You want bit-exact results compared to CPU Decoding
#. If you have small resolution videos and the PCI-e transfer latency is large
#. If your GPU is already busy and CPU is not

It's best to experiment with CUDA Decoding to see if it improves your use-case. With
TorchCodec you can simply pass in a device parameter to the
:class:`~torchcodec.decoders.VideoDecoder` class to use CUDA Decoding.

In order use CUDA Decoding will need the following installed in your environment:

#. CUDA-enabled pytorch
#. FFmpeg binaries that support NVDEC-enabled codecs
#. libnpp and nvrtc (these are usually installed when you install the full cuda-toolkit)


FFmpeg versions 5, 6 and 7 from conda-forge are built with NVDEC support and you can
install them with conda. For example, to install FFmpeg version 7:

.. code-block:: bash

   conda install ffmpeg=7 -c conda-forge
   conda install libnpp cuda-nvrtc -c nvidia
"""

# %%
######################################################################
# Checking if Pytorch has CUDA enabled
######################################################################
#
# .. note::
#
#    This tutorial requires FFmpeg libraries compiled with CUDA support.
#
#
import torch

print(f"{torch.__version__=}")
print(f"{torch.cuda.is_available()=}")
print(f"{torch.cuda.get_device_properties(0)=}")


# %%
######################################################################
# Downloading the video
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
import urllib.request

video_file = "video.mp4"
urllib.request.urlretrieve(
    "https://download.pytorch.org/torchaudio/tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4",
    video_file,
)


# %%
######################################################################
# Decoding with CUDA
######################################################################
#
# To use CUDA decoder, you need to pass in a cuda device to the decoder.
#
from torchcodec.decoders import VideoDecoder

vd = VideoDecoder(video_file, device="cuda:0")
frame = vd[0]

# %%
#
# The video frames are decoded and returned as tensor of NCHW format.

print(frame.data.shape, frame.data.dtype)

# %%
#
# The video frames are left on the GPU memory.

print(frame.data.device)


# %%
######################################################################
# Visualizing Frames
######################################################################
#
# Let's look at the frames decoded by CUDA decoder and compare them
# against equivalent results from the CPU decoders.
import matplotlib.pyplot as plt


def get_frames(timestamps: list[float], device: str):
    decoder = VideoDecoder(video_file, device=device)
    return [decoder.get_frame_played_at(ts) for ts in timestamps]


def get_numpy_images(frames):
    numpy_images = []
    for frame in frames:
        # We transfer to the CPU so they can be visualized by matplotlib.
        numpy_image = frame.data.to("cpu").permute(1, 2, 0).numpy()
        numpy_images.append(numpy_image)
    return numpy_images


timestamps = [12, 19, 45, 131, 180]
cpu_frames = get_frames(timestamps, device="cpu")
cuda_frames = get_frames(timestamps, device="cuda:0")
cpu_numpy_images = get_numpy_images(cpu_frames)
cuda_numpy_images = get_numpy_images(cuda_frames)


def plot_cpu_and_cuda_images():
    n_rows = len(timestamps)
    fig, axes = plt.subplots(n_rows, 2, figsize=[12.8, 16.0])
    for i in range(n_rows):
        axes[i][0].imshow(cpu_numpy_images[i])
        axes[i][1].imshow(cuda_numpy_images[i])

    axes[0][0].set_title("CPU decoder")
    axes[0][1].set_title("CUDA decoder")
    plt.setp(axes, xticks=[], yticks=[])
    plt.tight_layout()


plot_cpu_and_cuda_images()

# %%
#
# They look visually similar to the human eye but there may be subtle
# differences because CUDA math is not bit-exact to CPU math.
#
first_cpu_frame = cpu_frames[0].data.to("cpu")
first_cuda_frame = cuda_frames[0].data.to("cpu")
frames_equal = torch.equal(first_cpu_frame, first_cuda_frame)
print(f"{frames_equal=}")
