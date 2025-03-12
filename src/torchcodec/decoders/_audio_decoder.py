# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Optional, Union

from torch import Tensor

from torchcodec.decoders import _core as core
from torchcodec.decoders._decoder_utils import (
    create_decoder,
    get_and_validate_stream_metadata,
)


class AudioDecoder:
    """TODO-AUDIO docs"""

    def __init__(
        self,
        source: Union[str, Path, bytes, Tensor],
        *,
        stream_index: Optional[int] = None,
    ):
        self._decoder = create_decoder(source=source, seek_mode="approximate")

        core.add_audio_stream(self._decoder, stream_index=stream_index)

        (
            self.metadata,
            self.stream_index,
            self._begin_stream_seconds,
            self._end_stream_seconds,
        ) = get_and_validate_stream_metadata(
            decoder=self._decoder, stream_index=stream_index, media_type="audio"
        )

    def get_samples_played_in_range(
        self, start_seconds: float = 0, stop_seconds: Optional[float] = None
    ) -> Tensor:
        """TODO-AUDIO docs"""
        if stop_seconds is not None and not start_seconds <= stop_seconds:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. It must be less than or equal to stop seconds ({stop_seconds})."
            )
        if not self._begin_stream_seconds <= start_seconds < self._end_stream_seconds:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. "
                f"It must be greater than or equal to {self._begin_stream_seconds} "
                f"and less than or equal to {self._end_stream_seconds}."
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
        # ....x..........x..........x...........x..........x..........x..........x.....
        #                    ^                                 ^
        #               start_seconds                      stop_seconds
        #
        # We want to return the samples in [start_seconds, stop_seconds). But
        # because the core API is based on frames, the `frames` tensor contains
        # the samples in [first_pts, last_pts).pts
        #
        # So we return a view on that tensor and do some basic math to figure
        # out where to chunk it.

        offset_beginning = round(
            (max(0, start_seconds - first_pts)) * self.metadata.sample_rate
        )

        num_samples = frames.shape[1]
        offset_end = num_samples
        last_pts = first_pts + num_samples / self.metadata.sample_rate
        if stop_seconds is not None and stop_seconds < last_pts:
            offset_end -= round((last_pts - stop_seconds) * self.metadata.sample_rate)

        return frames[:, offset_beginning:offset_end]
        # return frames[:, offset_beginning:offset_end]
