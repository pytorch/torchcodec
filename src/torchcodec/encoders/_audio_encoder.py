from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor

from torchcodec import _core


class AudioEncoder:
    def __init__(self, samples: Tensor, *, sample_rate: int):
        # Some of these checks are also done in C++: it's OK, they're cheap, and
        # doing them here allows to surface them when the AudioEncoder is
        # instantiated, rather than later when the encoding methods are called.
        if not isinstance(samples, Tensor):
            raise ValueError(
                f"Expected samples to be a Tensor, got {type(samples) = }."
            )
        if samples.ndim != 2:
            raise ValueError(f"Expected 2D samples, got {samples.shape = }.")
        if samples.dtype != torch.float32:
            raise ValueError(f"Expected float32 samples, got {samples.dtype = }.")
        if sample_rate <= 0:
            raise ValueError(f"{sample_rate = } must be > 0.")

        self._samples = samples
        self._sample_rate = sample_rate

    def to_file(
        self,
        dest: Union[str, Path],
        *,
        bit_rate: Optional[int] = None,
        num_channels: Optional[int] = None,
        sample_rate: Optional[int] = None,
    ) -> None:
        _core.encode_audio_to_file(
            samples=self._samples,
            sample_rate=self._sample_rate,
            filename=dest,
            bit_rate=bit_rate,
            num_channels=num_channels,
            desired_sample_rate=sample_rate,
        )

    def to_tensor(
        self,
        format: str,
        *,
        bit_rate: Optional[int] = None,
        num_channels: Optional[int] = None,
        sample_rate: Optional[int] = None,
    ) -> Tensor:
        return _core.encode_audio_to_tensor(
            samples=self._samples,
            sample_rate=self._sample_rate,
            format=format,
            bit_rate=bit_rate,
            num_channels=num_channels,
            desired_sample_rate=sample_rate,
        )
