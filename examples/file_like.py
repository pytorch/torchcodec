# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
===================================================================
Streaming data through file-like support
===================================================================

In this example, we will describe the feature with references to its docs."""

# %%
# First, a bit of boilerplate: TODO.


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


from torchcodec.decoders import VideoDecoder

nasa_url = "https://download.pytorch.org/torchaudio/tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4.mp4"

pre_downloaded_raw_video_bytes = get_url_content(nasa_url)
decoder = VideoDecoder(pre_downloaded_raw_video_bytes)

print(f"Video size in MB: {len(pre_downloaded_raw_video_bytes) / 1024 / 1024}")
print(decoder.metadata)
print()

def decode_from_existing_download():
    decoder = VideoDecoder(pre_downloaded_raw_video_bytes, seek_mode="approximate")
    return decoder[0]

def download_before_decode():
    raw_video_bytes = get_url_content(nasa_url)
    decoder = VideoDecoder(raw_video_bytes, seek_mode="approximate")
    return decoder[0]

def direct_url_to_ffmpeg():
    decoder = VideoDecoder(nasa_url, seek_mode="approximate")
    return decoder[0]

print("Decode from existing download")
bench(decode_from_existing_download)
print()

print("Download before decode: ")
bench(download_before_decode)
print()

print("Direct url to FFmpeg: ")
bench(direct_url_to_ffmpeg)
print()

import fsspec
# Note: we also need: aiohttp

def stream_while_decode():
    with fsspec.open(nasa_url, client_kwargs={'trust_env': True}) as file:
        decoder = VideoDecoder(file, seek_mode="approximate")
        return decoder[0]

print("Stream while decode: ")
bench(stream_while_decode)
print()
