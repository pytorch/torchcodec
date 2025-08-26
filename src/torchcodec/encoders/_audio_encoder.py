from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor

from torchcodec import _core


class AudioEncoder:
    """An audio encoder.

    Args:
        samples (``torch.Tensor``): The samples to encode. This must be a 2D
            tensor of shape ``(num_channels, num_samples)``, or a 1D tensor in
            which case ``num_channels = 1`` is assumed. Values must be float
            values in ``[-1, 1]``.
        sample_rate (int): The sample rate of the **input** ``samples``. The
            sample rate of the encoded output can be specified using the
            encoding methods (``to_file``, etc.).
    """

    def __init__(self, samples: Tensor, *, sample_rate: int):
        torch._C._log_api_usage_once("torchcodec.encoders.AudioEncoder")
        # Some of these checks are also done in C++: it's OK, they're cheap, and
        # doing them here allows to surface them when the AudioEncoder is
        # instantiated, rather than later when the encoding methods are called.
        if not isinstance(samples, Tensor):
            raise ValueError(
                f"Expected samples to be a Tensor, got {type(samples) = }."
            )
        if samples.ndim == 1:
            # make it 2D and assume 1 channel
            samples = torch.unsqueeze(samples, 0)
        if samples.ndim != 2:
            raise ValueError(f"Expected 1D or 2D samples, got {samples.shape = }.")
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
        """Encode samples into a file.

        Args:
            dest (str or ``pathlib.Path``): The path to the output file, e.g.
                ``audio.mp3``. The extension of the file determines the audio
                format and container.
            bit_rate (int, optional): The output bit rate. Encoders typically
                support a finite set of bit rate values, so ``bit_rate`` will be
                matched to one of those supported values. The default is chosen
                by FFmpeg.
            num_channels (int, optional): The number of channels of the encoded
                output samples. By default, the number of channels of the input
                ``samples`` is used.
            sample_rate (int, optional): The sample rate of the encoded output.
                By default, the sample rate of the input ``samples`` is used.
        """
        _core.encode_audio_to_file(
            samples=self._samples,
            sample_rate=self._sample_rate,
            filename=str(dest),
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
        """Encode samples into raw bytes, as a 1D uint8 Tensor.

        Args:
            format (str): The format of the encoded samples, e.g. "mp3", "wav"
                or "flac".
            bit_rate (int, optional): The output bit rate. Encoders typically
                support a finite set of bit rate values, so ``bit_rate`` will be
                matched to one of those supported values. The default is chosen
                by FFmpeg.
            num_channels (int, optional): The number of channels of the encoded
                output samples. By default, the number of channels of the input
                ``samples`` is used.
            sample_rate (int, optional): The sample rate of the encoded output.
                By default, the sample rate of the input ``samples`` is used.

        Returns:
            Tensor: The raw encoded bytes as 1D uint8 Tensor.
        """
        return _core.encode_audio_to_tensor(
            samples=self._samples,
            sample_rate=self._sample_rate,
            format=format,
            bit_rate=bit_rate,
            num_channels=num_channels,
            desired_sample_rate=sample_rate,
        )

    def to_file_like(
        self,
        file_like,
        format: str,
        *,
        bit_rate: Optional[int] = None,
        num_channels: Optional[int] = None,
        sample_rate: Optional[int] = None,
    ) -> None:
        """Encode samples into a file-like object.

        Args:
            file_like: A file-like object that supports ``write()`` and
                ``seek()`` methods, such as io.BytesIO(), an open file in binary
                write mode, etc. Methods must have the following signature:
                ``write(data: bytes) -> int`` and ``seek(offset: int, whence:
                int = 0) -> int``.
            format (str): The format of the encoded samples, e.g. "mp3", "wav"
                or "flac".
            bit_rate (int, optional): The output bit rate. Encoders typically
                support a finite set of bit rate values, so ``bit_rate`` will be
                matched to one of those supported values. The default is chosen
                by FFmpeg.
            num_channels (int, optional): The number of channels of the encoded
                output samples. By default, the number of channels of the input
                ``samples`` is used.
            sample_rate (int, optional): The sample rate of the encoded output.
                By default, the sample rate of the input ``samples`` is used.
        """
        _core.encode_audio_to_file_like(
            samples=self._samples,
            sample_rate=self._sample_rate,
            format=format,
            file_like=file_like,
            bit_rate=bit_rate,
            num_channels=num_channels,
            desired_sample_rate=sample_rate,
        )
