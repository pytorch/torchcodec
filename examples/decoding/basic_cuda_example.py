# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Accelerated video decoding on GPUs with CUDA and NVDEC
================================================================

TorchCodec can use supported Nvidia hardware (see support matrix
`here <https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new>`_) to speed-up
video decoding. This is called "CUDA Decoding" and it uses Nvidia's
`NVDEC hardware decoder <https://developer.nvidia.com/video-codec-sdk>`_
and CUDA kernels to respectively decompress and convert to RGB.
CUDA Decoding can be faster than CPU Decoding for the actual decoding step and also for
subsequent transform steps like scaling, cropping or rotating. This is because the decode step leaves
the decoded tensor in GPU memory so the GPU doesn't have to fetch from main memory before
running the transform steps. Encoded packets are often much smaller than decoded frames so
CUDA decoding also uses less PCI-e bandwidth.

When to and when not to use CUDA Decoding
-----------------------------------------

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

Installing TorchCodec with CUDA Enabled
---------------------------------------

Refer to the installation guide in the `README <https://github.com/pytorch/torchcodec#installing-cuda-enabled-torchcodec>`_.

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
# We will use the following video which has the following properties:
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

print(frame.shape, frame.dtype)

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
timestamps = [12, 19, 45, 131, 180]
cpu_decoder = VideoDecoder(video_file, device="cpu")
cuda_decoder = VideoDecoder(video_file, device="cuda")
cpu_frames = cpu_decoder.get_frames_played_at(timestamps).data
cuda_frames = cuda_decoder.get_frames_played_at(timestamps).data


def plot_cpu_and_cuda_frames(cpu_frames: torch.Tensor, cuda_frames: torch.Tensor):
    try:
        import matplotlib.pyplot as plt
        from torchvision.transforms.v2.functional import to_pil_image
    except ImportError:
        print("Cannot plot, please run `pip install torchvision matplotlib`")
        return
    n_rows = len(timestamps)
    fig, axes = plt.subplots(n_rows, 2, figsize=[12.8, 16.0])
    for i in range(n_rows):
        axes[i][0].imshow(to_pil_image(cpu_frames[i].to("cpu")))
        axes[i][1].imshow(to_pil_image(cuda_frames[i].to("cpu")))

    axes[0][0].set_title("CPU decoder", fontsize=24)
    axes[0][1].set_title("CUDA decoder", fontsize=24)
    plt.setp(axes, xticks=[], yticks=[])
    plt.tight_layout()


plot_cpu_and_cuda_frames(cpu_frames, cuda_frames)

# %%
#
# They look visually similar to the human eye but there may be subtle
# differences because CUDA math is not bit-exact with respect to CPU math.
#
frames_equal = torch.equal(cpu_frames.to("cuda"), cuda_frames)
mean_abs_diff = torch.mean(
    torch.abs(cpu_frames.float().to("cuda") - cuda_frames.float())
)
max_abs_diff = torch.max(torch.abs(cpu_frames.to("cuda").float() - cuda_frames.float()))
print(f"{frames_equal=}")
print(f"{mean_abs_diff=}")
print(f"{max_abs_diff=}")
