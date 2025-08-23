# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor

from torchcodec import _core as core, AudioSamples
from torchcodec.decoders._decoder_utils import (
    create_decoder,
    ERROR_REPORTING_INSTRUCTIONS,
)


class AudioDecoder:
    """A single-stream audio decoder.

    This can be used to decode audio from pure audio files (e.g. mp3, wav,
    etc.), or from videos that contain audio streams (e.g. mp4 videos).

    Returned samples are float samples normalized in [-1, 1]

    Args:
        source (str, ``Pathlib.path``, bytes, ``torch.Tensor`` or file-like
            object): The source of the video or audio:

            - If ``str``: a local path or a URL to a video or audio file.
            - If ``Pathlib.path``: a path to a local video or audio file.
            - If ``bytes`` object or ``torch.Tensor``: the raw encoded audio data.
            - If file-like object: we read video data from the object on demand. The object must
              expose the methods `read(self, size: int) -> bytes` and
              `seek(self, offset: int, whence: int) -> int`. Read more in:
              :ref:`sphx_glr_generated_examples_decoding_file_like.py`.
        stream_index (int, optional): Specifies which stream in the file to decode samples from.
            Note that this index is absolute across all media types. If left unspecified, then
            the :term:`best stream` is used.
        sample_rate (int, optional): The desired output sample rate of the decoded samples.
            By default, the sample rate of the source is used.
        num_channels (int, optional): The desired number of channels of the decoded samples.
            By default, the number of channels of the source is used.

    Attributes:
        metadata (AudioStreamMetadata): Metadata of the audio stream.
        stream_index (int): The stream index that this decoder is retrieving samples from. If a
            stream index was provided at initialization, this is the same value. If it was left
            unspecified, this is the :term:`best stream`.
    """

    def __init__(
        self,
        source: Union[str, Path, io.RawIOBase, io.BufferedReader, bytes, Tensor],
        *,
        stream_index: Optional[int] = None,
        sample_rate: Optional[int] = None,
        num_channels: Optional[int] = None,
    ):
        torch._C._log_api_usage_once("torchcodec.decoders.AudioDecoder")
        self._decoder = create_decoder(source=source, seek_mode="approximate")

        core.add_audio_stream(
            self._decoder,
            stream_index=stream_index,
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        container_metadata = core.get_container_metadata(self._decoder)
        self.stream_index = (
            container_metadata.best_audio_stream_index
            if stream_index is None
            else stream_index
        )
        if self.stream_index is None:
            raise ValueError(
                "The best audio stream is unknown and there is no specified stream. "
                + ERROR_REPORTING_INSTRUCTIONS
            )
        self.metadata = container_metadata.streams[self.stream_index]
        assert isinstance(self.metadata, core.AudioStreamMetadata)  # mypy

        self._desired_sample_rate = (
            sample_rate if sample_rate is not None else self.metadata.sample_rate
        )

    def get_all_samples(self) -> AudioSamples:
        """Returns all the audio samples from the source.

        To decode samples in a specific range, use
        :meth:`~torchcodec.decoders.AudioDecoder.get_samples_played_in_range`.

        Returns:
            AudioSamples: The samples within the file.
        """
        return self.get_samples_played_in_range()

    def get_samples_played_in_range(
        self, start_seconds: float = 0.0, stop_seconds: Optional[float] = None
    ) -> AudioSamples:
        """Returns audio samples in the given range.

        Samples are in the half open range [start_seconds, stop_seconds).

        To decode all the samples from beginning to end, you can call this
        method while leaving ``start_seconds`` and ``stop_seconds`` to their
        default values, or use
        :meth:`~torchcodec.decoders.AudioDecoder.get_all_samples` as a more
        convenient alias.

        Args:
            start_seconds (float): Time, in seconds, of the start of the
                range. Default: 0.
            stop_seconds (float or None): Time, in seconds, of the end of the
                range. As a half open range, the end is excluded. Default: None,
                which decodes samples until the end.

        Returns:
            AudioSamples: The samples within the specified range.
        """
        if stop_seconds is not None and not start_seconds <= stop_seconds:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. It must be less than or equal to stop seconds ({stop_seconds})."
            )
        frames, first_pts = core.get_frames_by_pts_in_range_audio(
            self._decoder,
            start_seconds=start_seconds,
            stop_seconds=stop_seconds,
        )
        first_pts = first_pts.item()

        # x = frame boundaries
        #
        #            first_pts                                    last_pts
        #                v                                            v
        # ....x..........x..........x...........x..........x..........x.....
        #                    ^                                 ^
        #               start_seconds                      stop_seconds
        #
        # We want to return the samples in [start_seconds, stop_seconds). But
        # because the core API is based on frames, the `frames` tensor contains
        # the samples in [first_pts, last_pts)
        # So we do some basic math to figure out the position of the view that
        # we'll return.

        sample_rate = self._desired_sample_rate
        # TODO: metadata's sample_rate should probably not be Optional
        assert sample_rate is not None  # mypy.

        if first_pts < start_seconds:
            offset_beginning = round((start_seconds - first_pts) * sample_rate)
            output_pts_seconds = start_seconds
        else:
            # In normal cases we'll have first_pts <= start_pts, but in some
            # edge cases it's possible to have first_pts > start_seconds,
            # typically if the stream's first frame's pts isn't exactly 0.
            offset_beginning = 0
            output_pts_seconds = first_pts

        num_samples = frames.shape[1]
        last_pts = first_pts + num_samples / sample_rate
        if stop_seconds is not None and stop_seconds < last_pts:
            offset_end = num_samples - round((last_pts - stop_seconds) * sample_rate)
        else:
            offset_end = num_samples

        data = frames[:, offset_beginning:offset_end]
        return AudioSamples(
            data=data,
            pts_seconds=output_pts_seconds,
            duration_seconds=data.shape[1] / sample_rate,
            sample_rate=sample_rate,
        )
