# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
CUDA Decoding on Nvidia GPUs
=====================================

.. _ndecoderec_tutorial:

TorchCodec can use supported Nvidia hardware (see support matrix here
<https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new>) to speed-up
video decoding. This is called "CUDA Decoding".
CUDA Decoding can be faster than CPU Decoding for the actual decoding step and also for
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
#. You have small resolution videos and the PCI-e transfer latency is large
#. Your GPU is already busy and CPU is not

It's best to experiment with CUDA Decoding to see if it improves your use-case. With
TorchCodec you can simply pass in a device parameter to the
:class:`~torchcodec.decoders.VideoDecoder` class to use CUDA Decoding.


In order to use CUDA Decoding will need the following installed in your environment:

#. An Nvidia GPU that supports decoding the video format you want to decode. See
   the support matrix here <https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new>
#. `CUDA-enabled pytorch <https://pytorch.org/get-started/locally/>`_
#. FFmpeg binaries that support NdecoderEC-enabled codecs
#. libnpp and nvrtc (these are usually installed when you install the full cuda-toolkit)


FFmpeg versions 5, 6 and 7 from conda-forge are built with NdecoderEC support and you can
install them with conda. For example, to install FFmpeg version 7:

.. code-block:: bash
   conda install ffmpeg=7 -c conda-forge
   conda install libnpp cuda-nvrtc -c nvidia
"""

# %%
# Checking if Pytorch has CUDA enabled
# -------------------------------------
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
# Downloading the video
# -------------------------------------
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
# CUDA Decoding using VideoDecoder
# -------------------------------------
#
# To use CUDA decoder, you need to pass in a cuda device to the decoder.
#
from torchcodec.decoders import VideoDecoder

decoder = VideoDecoder(video_file, device="cuda")
frame = decoder[0]

# %%
#
# The video frames are decoded and returned as tensor of NCHW format.

print(frame.data.shape, frame.data.dtype)

# %%
#
# The video frames are left on the GPU memory.

print(frame.data.device)


# %%
# Visualizing Frames
# -------------------------------------
#
# Let's look at the frames decoded by CUDA decoder and compare them
# against equivalent results from the CPU decoders.


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
cuda_frames = get_frames(timestamps, device="cuda")
cpu_numpy_images = get_numpy_images(cpu_frames)
cuda_numpy_images = get_numpy_images(cuda_frames)


def plot(
    frames1: List[torch.Tensor],
    frames2: List[torch.Tensor],
    title1: Optional[str] = None,
    title2: Optional[str] = None,
):
    try:
        import matplotlib.pyplot as plt
        from torchvision.transforms.v2.functional import to_pil_image
        from torchvision.utils import make_grid
    except ImportError:
        print("Cannot plot, please run `pip install torchvision matplotlib`")
        return

    plt.rcParams["savefig.bbox"] = "tight"

    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(to_pil_image(make_grid(frames1)))
    ax[0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if title1 is not None:
        ax[0].set_title(title1)

    ax[1].imshow(to_pil_image(make_grid(frames2)))
    ax[1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if title2 is not None:
        ax[1].set_title(title2)

    plt.tight_layout()


plot(cpu_frames, cuda_frames, "CPU decoder", "CUDA decoder")

# %%
#
# They look visually similar to the human eye but there may be subtle
# differences because CUDA math is not bit-exact with respect to CPU math.
#
first_cpu_frame = cpu_frames[0].data.to("cpu")
first_cuda_frame = cuda_frames[0].data.to("cpu")
frames_equal = torch.equal(first_cpu_frame, first_cuda_frame)
print(f"{frames_equal=}")
