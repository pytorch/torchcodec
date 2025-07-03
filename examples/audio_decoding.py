# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
========================================
Decoding audio streams with AudioDecoder
========================================

In this example, we'll learn how to decode an audio file using the
:class:`~torchcodec.decoders.AudioDecoder` class.
"""

# %%
# First, a bit of boilerplate: we'll download an audio file from the web and
# define an audio playing utility.  You can ignore that part and jump right
# below to :ref:`creating_decoder_audio`.
import requests
from IPython.display import Audio


def play_audio(samples):
    return Audio(samples.data, rate=samples.sample_rate)


# Audio source is CC0: https://opengameart.org/content/town-theme-rpg
# Attribution: cynicmusic.com pixelsphere.org
url = "https://opengameart.org/sites/default/files/TownTheme.mp3"
response = requests.get(url, headers={"User-Agent": ""})
if response.status_code != 200:
    raise RuntimeError(f"Failed to download video. {response.status_code = }.")

raw_audio_bytes = response.content

# %%
# .. _creating_decoder_audio:
#
# Creating a decoder
# ------------------
#
# We can now create a decoder from the raw (encoded) audio bytes. You can of
# course use a local audio file and pass the path as input. You can also decode
# audio streams from videos!

from torchcodec.decoders import AudioDecoder

decoder = AudioDecoder(raw_audio_bytes)

# %%
# The has not yet been decoded by the decoder, but we already have access to
# some metadata via the ``metadata`` attribute which is an
# :class:`~torchcodec.decoders.AudioStreamMetadata` object.
print(decoder.metadata)

# %%
# Decoding samples
# ----------------
#
# To get decoded samples, we just need to call the
# :meth:`~torchcodec.decoders.AudioDecoder.get_all_samples` method,
# which returns an :class:`~torchcodec.AudioSamples` object:

samples = decoder.get_all_samples()

print(samples)
play_audio(samples)

# %%
# The ``.data`` field is a tensor of shape ``(num_channels, num_samples)`` and
# of float dtype with values in [-1, 1].
#
# The ``.pts_seconds`` field indicates the starting time of the output samples.
# Here it's 0.025 seconds, even though we asked for samples starting from 0. Not
# all streams start exactly at 0! This is not a bug in TorchCodec, this is a
# property of the file that was defined when it was encoded.
#
# Specifying a range
# ------------------
#
# If we don't need all the samples, we can use
# :meth:`~torchcodec.decoders.AudioDecoder.get_samples_played_in_range` to
# decode the samples within a custom range:

samples = decoder.get_samples_played_in_range(start_seconds=10, stop_seconds=70)

print(samples)
play_audio(samples)

# %%
# Custom sample rate
# ------------------
#
# We can also decode the samples into a desired sample rate using the
# ``sample_rate`` parameter of :class:`~torchcodec.decoders.AudioDecoder`. The
# ouput will sound similar, but note that the number of samples greatly
# decreased:

decoder = AudioDecoder(raw_audio_bytes, sample_rate=16_000)
samples = decoder.get_all_samples()

print(samples)
play_audio(samples)
