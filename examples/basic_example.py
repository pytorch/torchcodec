"""
==================================================
Basic Example to use TorchCodec to decode a video.
==================================================

A simple example showing how to decode the first few frames of a video  using
the :class:`~torchcodec.decoders.SimpleVideoDecoder` class.
"""

# %%
import requests

# %%
# Video is CC0 Licensed, thanks to Coverr for uploading
# Source: https://www.pexels.com/video/dog-eating-854132/
video_url = "https://videos.pexels.com/video-files/854132/854132-sd_640_360_25fps.mp4"
response = requests.get(video_url)

if response.status_code != 200:
    raise RuntimeError(f"Failed to download video")

# %%
from torchcodec.decoders import SimpleVideoDecoder

decoder = SimpleVideoDecoder(source=response.content)
decoder[0]
