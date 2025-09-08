# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
========================================
Encoding audio samples with AudioEncoder
========================================

In this example, we'll learn how to encode audio samples to a file or to raw
bytes using the :class:`~torchcodec.encoders.AudioEncoder` class.
"""

# %%
# Let's first generate some samples to be encoded. The data to be encoded could
# also just come from an :class:`~torchcodec.decoders.AudioDecoder`!
import torch
from IPython.display import Audio as play_audio


def make_sinewave() -> tuple[torch.Tensor, int]:
    freq_A = 440  # Hz
    sample_rate = 16000  # Hz
    duration_seconds = 3  # seconds
    t = torch.linspace(0, duration_seconds, int(sample_rate * duration_seconds), dtype=torch.float32)
    return torch.sin(2 * torch.pi * freq_A * t), sample_rate


samples, sample_rate = make_sinewave()

print(f"Encoding samples with {samples.shape = } and {sample_rate = }")
play_audio(samples, rate=sample_rate)

# %%
# We first instantiate an :class:`~torchcodec.encoders.AudioEncoder`. We pass it
# the samples to be encoded. The samples must be a 2D tensors of shape
# ``(num_channels, num_samples)``, or in this case, a 1D tensor where
# ``num_channels`` is assumed to be 1. The values must be float values
# normalized in ``[-1, 1]``: this is also what the
# :class:`~torchcodec.decoders.AudioDecoder` would return.
#
# .. note::
#
#     The ``sample_rate`` parameter corresponds to the sample rate of the
#     *input*, not the desired encoded sample rate.
from torchcodec.encoders import AudioEncoder

encoder = AudioEncoder(samples=samples, sample_rate=sample_rate)


# %%
# :class:`~torchcodec.encoders.AudioEncoder` supports encoding samples into a
# file via the :meth:`~torchcodec.encoders.AudioEncoder.to_file` method, or to
# raw bytes via :meth:`~torchcodec.encoders.AudioEncoder.to_tensor`.  For the
# purpose of this tutorial we'll use
# :meth:`~torchcodec.encoders.AudioEncoder.to_tensor`, so that we can easily
# re-decode the encoded samples and check their properies. The
# :meth:`~torchcodec.encoders.AudioEncoder.to_file` method works very similarly.

encoded_samples = encoder.to_tensor(format="mp3")
print(f"{encoded_samples.shape = }, {encoded_samples.dtype = }")


# %%
# That's it!
#
# Now that we have our encoded data, we can decode it back, to make sure it
# looks and sounds as expected:
from torchcodec.decoders import AudioDecoder

samples_back = AudioDecoder(encoded_samples).get_all_samples()

print(samples_back)
play_audio(samples_back.data, rate=samples_back.sample_rate)

# %%
# The encoder supports some encoding options that allow you to change how to
# data is encoded. For example, we can decide to encode our mono data (1
# channel) into stereo data (2 channels), and to specify an output sample rate:

desired_sample_rate = 32000
encoded_samples = encoder.to_tensor(format="wav", num_channels=2, sample_rate=desired_sample_rate)

stereo_samples_back = AudioDecoder(encoded_samples).get_all_samples()

print(stereo_samples_back)
play_audio(stereo_samples_back.data, rate=desired_sample_rate)

# %%
# Check the docstring of the encoding methods to learn about the different
# encoding options.
