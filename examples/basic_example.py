"""
================================
Decoding a video with TorchCodec
================================

In this example, we'll learn how to decode a video using the
:class:`~torchcodec.decoders.SimpleVideoDecoder` class.
"""

# %%
# Let's start by downloading a video to decode.
import requests

# Video source: https://www.pexels.com/video/dog-eating-854132/
# License: CC0. Author: Coverr.
url = "https://videos.pexels.com/video-files/854132/854132-sd_640_360_25fps.mp4"
response = requests.get(url)
if response.status_code != 200:
    raise RuntimeError(f"Failed to download video. {response.status_code = }.")

raw_video_bytes = response.content

# %%
# We can now create a decoder from the raw (encoded) video bytes. You can of
# course use a local video file and pass the path as input, rather than download
# a video.
from torchcodec.decoders import SimpleVideoDecoder

# You can also pass a path to a local file!
decoder = SimpleVideoDecoder(raw_video_bytes)

# %%
# Accessing the video metadata
# ----------------------------
#
# The has not yet been decoded by the decoder, but we already have access to
# some metadata via the ``metadata`` attribute which is a
# :class:`~torchcodec.decoder.VideoMetadata` object.
print(decoder.metadata)

# %%
# Note that the ``num_frames`` field corresponds to ``len(decoder)``:
assert decoder.metadata.num_frames == len(decoder)

# %%
# Decoding frames by indexing the decoder
# ---------------------------------------
#
# We can now start to decode frames by simply indexing the decoder with integer
# indices, or using slices. This will return the frames as :class:`torch.Tensor`
# objects.

first_frame = decoder[-1]
every_twenty_frame = decoder[0 : -1 : 20]

print(f"{first_frame.shape = }, {every_twenty_frame.shape = }")
print(f"{first_frame.dtype = }, {every_twenty_frame.dtype = }")

# %%
# By default, the shape of the frames is ``(N, C, H, W)`` where N is the batch
# size C the number of channels, H is the height, and W is the width of the
# frames.  The batch dimension N is only present when we're decoding more than
# one frame. The dimension order can be changed to ``N, H, W, C`` using the
# ``dimension_order`` parameter of
# :class:`~torchcodec.decoders.SimpleVideoDecoder`. Frames are always of
# ``torch.uint8`` dtype.
#
# Plotting the frames
# -------------------
#
# Bare with us as we introduce a little bit of boilerpalte with this small
# plotting utility.

import torch
from typing import Optional


def plot(frames: torch.Tensor, title : Optional[str] = None):
    try:
        from torchvision.utils import make_grid
        from torchvision.transforms.v2.functional import to_pil_image
        import matplotlib.pyplot as plt
    except ImportError:
        print("Cannot plot, please run `pip install torchvision matplotlib`")
        return

    plt.rcParams["savefig.bbox"] = 'tight'
    fig, ax = plt.subplots()
    ax.imshow(to_pil_image(make_grid(frames)))
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()


plot(first_frame, "First frame")

# %%
plot(every_twenty_frame, "Every 20 frame")

# %%
# Iterating over frames
# ---------------------
#
# The decoder is a normal iterable object and can be iterated over like so:

for frame in decoder:
    assert (
        isinstance(frame, torch.Tensor)
        and frame.shape == (3, decoder.metadata.height, decoder.metadata.width)
    )

# %%
# Retrieving pts and duration of frames
# -------------------------------------
#
# Indexing the decoder returns pure :class:`torch.Tensor` objects. Sometimes, it
# can be useful to retrieve additional information about the frame, such as
# their :term:`pts` (Presentation Time Stamp), and their duration.
# This can be achieved using the
# :meth:`~torchcodec.decoders.SimpleVideoDecoder.get_frame_at` and
# :meth:`~torchcodec.decoders.SimpleVideoDecoder.get_frames_at`  methods, which
# will return a :class:`~torchcodec.decoders.Frame` and
# :class:`~torchcodec.decoders.FrameBatch` objects respectively.

last_frame = decoder.get_frame_at(-1)
print(f"{type(last_frame) = }")
print(last_frame)
