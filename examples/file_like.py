# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
===================================================================
Streaming data through file-like support
===================================================================

In this example, we will show how to decode streaming data. That is, when files
do not reside locally, we will show how to only download the data segments that
are needed to decode the frames you care about. We accomplish this capability
with Python
`file-like objects <https://docs.python.org/3/glossary.html#term-file-like-object>`_."""

# %%
# First, a bit of boilerplate. We define two functions: one to download content
# from a given URL, and another to time the execution of a given function.


import torch
import requests
from time import perf_counter_ns


def get_url_content(url):
    response = requests.get(url, headers={"User-Agent": ""})
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download video. {response.status_code = }.")
    return response.content


def bench(f, average_over=20, warmup=2):
    for _ in range(warmup):
        f()

    times = []
    for _ in range(average_over):
        start = perf_counter_ns()
        f()
        end = perf_counter_ns()
        times.append(end - start)

    times = torch.tensor(times) * 1e-6  # ns to ms
    std = times.std().item()
    med = times.median().item()
    print(f"{med = :.2f}ms +- {std:.2f}")

# %%
# Performance: downloading first versus streaming
# -----------------------------------------------
#
# We are going to investigate the cost of having to download an entire video
# before decoding any frames versus being able to stream the video's data
# while decoding. To demonsrate an extreme case, we're going to always decode
# just the first frame of the video, while we vary how we get that video's
# data.
#
# The video we're going to use in this tutorial is publicly available on the
# internet. We perform an initial download of it so that we can understand
# its size and content:

from torchcodec.decoders import VideoDecoder

nasa_url = "https://download.pytorch.org/torchaudio/tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4.mp4"

pre_downloaded_raw_video_bytes = get_url_content(nasa_url)
decoder = VideoDecoder(pre_downloaded_raw_video_bytes)

print(f"Video size in MB: {len(pre_downloaded_raw_video_bytes) / 1024 / 1024}")
print(decoder.metadata)
print()

# %%
# We can see that the video is about 253 MB, has the resolution 1920x1080, is
# about 30 frames per second and is almost 3 and a half minutes long. As we
# only want to decode the first frame, we would clearly benefit from not having
# to download the entire video!
#
# Let's first test three scenarios:
#
#   1. Decode from the *existing* video we just downloaded. This is our baseline
#      performance, as we've reduced the downloading cost to 0.
#   2. Download the entire video before decoding. This is the worst case
#      that we want to avoid.
#   3. Provde the URL directly to the :class:`~torchcodec.decoders.VideoDecoder` class, which will pass
#      the URL on to FFmpeg. Then FFmpeg will decide how much of the video to
#      download before decoding.
#
# Note that in our scenarios, we are always setting the ``seek_mode`` parameter of
# the :class:`~torchcodec.decoders.VideoDecoder` class to ``"approximate"``. We do
# this to avoid scanning the entire video during initialization, which would
# require downloading the entire video even if we only want to decode the first
# frame. See :ref:`sphx_glr_generated_examples_approximate_mode.py` for more.

def decode_from_existing_download():
    decoder = VideoDecoder(
        source=pre_downloaded_raw_video_bytes,
        seek_mode="approximate",
    )
    return decoder[0]


def download_before_decode():
    raw_video_bytes = get_url_content(nasa_url)
    decoder = VideoDecoder(
        source=raw_video_bytes,
        seek_mode="approximate",
    )
    return decoder[0]


def direct_url_to_ffmpeg():
    decoder = VideoDecoder(
        source=nasa_url,
        seek_mode="approximate",
    )
    return decoder[0]


print("Decode from existing download:")
bench(decode_from_existing_download)
print()

print("Download before decode:")
bench(download_before_decode)
print()

print("Direct url to FFmpeg:")
bench(direct_url_to_ffmpeg)
print()

# %%
# Decoding the already downloaded video is clearly the fastest. Having to
# download the entire video each time we want to decode just the first frame
# is over 4x slower than decoding an existing video. Providing a direct URL
# is much better, as its about 2.5x faster than downloding the video first.
#
# We can do better, and the way how is to use a file-like object which
# implements its own read and seek methods that only download data from a URL as
# needed. Rather than implementing our own, we can use such objects from the
# `fsspec <https://github.com/fsspec/filesystem_spec>`_ module that provides
# `Filesystem interfaces for Python <https://filesystem-spec.readthedocs.io/en/latest/?badge=latest>`_.

import fsspec

def stream_while_decode():
    # The `client_kwargs` are passed down to the aiohttp module's client
    # session; we need to indicate that we need to trust the environment
    # settings for proxy configuration. Depending on your environment, you may
    # not need this setting.
    with fsspec.open(nasa_url, client_kwargs={'trust_env': True}) as file:
        decoder = VideoDecoder(file, seek_mode="approximate")
        return decoder[0]


print("Stream while decode: ")
bench(stream_while_decode)
print()

# %%
# Streaming the data through a file-like object is about 4.3x faster than
# downloading the video first. And not only is it about 1.7x faster than
# providing a direct URL, it's more general. :class:`~torchcodec.decoders.VideoDecoder` supports
# direct URLs because the underlying FFmpeg functions support them. But the
# kinds of protocols supported are determined by what that version of FFmpeg
# supports. A file-like object can adapt any kind of resource, including ones
# that are specific to your own infrastructure and are unknown to FFmpeg.


# %%
# How it works
# ------------
#

from pathlib import Path
import tempfile

temp_dir = tempfile.mkdtemp()
nasa_video_path = Path(temp_dir) / "nasa_video.mp4"
with open(nasa_video_path, "wb") as f:
    f.write(pre_downloaded_raw_video_bytes)


class FileOpCounter:
    def __init__(self, file):
        self._file = file
        self.num_reads = 0
        self.num_seeks = 0

    def read(self, size: int) -> bytes:
        self.num_reads += 1
        return self._file.read(size)

    def seek(self, offset: int, whence: int) -> bytes:
        self.num_seeks += 1
        return self._file.seek(offset, whence)


file_op_counter = FileOpCounter(open(nasa_video_path, "rb"))
counter_decoder = VideoDecoder(file_op_counter, seek_mode="approximate")

print("Decoder initialization required "
      f"{file_op_counter.num_reads} reads and "
      f"{file_op_counter.num_seeks} seeks.")

init_reads = file_op_counter.num_reads
init_seeks = file_op_counter.num_seeks

first_frame = counter_decoder[0]

print("Decoding the first frame required "
      f"{file_op_counter.num_reads - init_reads} additional reads and "
      f"{file_op_counter.num_seeks - init_seeks} additional seeks.")
print()

# %%
# Performance: local file path versus local file-like object
# ----------------------------------------------------------
#


def decode_from_existing_file_path():
    decoder = VideoDecoder(nasa_video_path, seek_mode="approximate")
    return decoder[0]


def decode_from_existing_open_file_object():
    with open(nasa_video_path, "rb") as f:
        decoder = VideoDecoder(f, seek_mode="approximate")
        return decoder[0]


print("Decode from existing file path:")
bench(decode_from_existing_file_path)
print()

print("Decode from existing open file object:")
bench(decode_from_existing_open_file_object)
print()

# %%
import shutil
shutil.rmtree(temp_dir)
# %%
