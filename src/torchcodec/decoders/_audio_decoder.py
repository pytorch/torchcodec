# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Literal, Optional, Tuple, Union

from torch import Tensor

from torchcodec.decoders import _core as core

_ERROR_REPORTING_INSTRUCTIONS = """
This should never happen. Please report an issue following the steps in
https://github.com/pytorch/torchcodec/issues/new?assignees=&labels=&projects=&template=bug-report.yml.
"""


class AudioDecoder:
    """A single-stream audio decoder.

    TODO docs
    """

    def __init__(
        self,
        source: Union[str, Path, bytes, Tensor],
        *,
        sample_rate: Optional[int] = None,
        stream_index: Optional[int] = None,
        seek_mode: Literal["exact", "approximate"] = "exact",
    ):
        if sample_rate is not None:
            raise ValueError("TODO implement this")

        # TODO unify validation with VideoDecoder?
        allowed_seek_modes = ("exact", "approximate")
        if seek_mode not in allowed_seek_modes:
            raise ValueError(
                f"Invalid seek mode ({seek_mode}). "
                f"Supported values are {', '.join(allowed_seek_modes)}."
            )

        if isinstance(source, str):
            self._decoder = core.create_from_file(source, seek_mode)
        elif isinstance(source, Path):
            self._decoder = core.create_from_file(str(source), seek_mode)
        elif isinstance(source, bytes):
            self._decoder = core.create_from_bytes(source, seek_mode)
        elif isinstance(source, Tensor):
            self._decoder = core.create_from_tensor(source, seek_mode)
        else:
            raise TypeError(
                f"Unknown source type: {type(source)}. "
                "Supported types are str, Path, bytes and Tensor."
            )

        core.add_audio_stream(self._decoder, stream_index=stream_index)

        self.metadata, self.stream_index = _get_and_validate_stream_metadata(
            self._decoder, stream_index
        )

        # if self.metadata.num_frames is None:
        #     raise ValueError(
        #         "The number of frames is unknown. " + _ERROR_REPORTING_INSTRUCTIONS
        #     )
        # self._num_frames = self.metadata.num_frames

        # if self.metadata.begin_stream_seconds is None:
        #     raise ValueError(
        #         "The minimum pts value in seconds is unknown. "
        #         + _ERROR_REPORTING_INSTRUCTIONS
        #     )
        # self._begin_stream_seconds = self.metadata.begin_stream_seconds

        # if self.metadata.end_stream_seconds is None:
        #     raise ValueError(
        #         "The maximum pts value in seconds is unknown. "
        #         + _ERROR_REPORTING_INSTRUCTIONS
        #     )
        # self._end_stream_seconds = self.metadata.end_stream_seconds

    # TODO we need to have a default for stop_seconds.
    def get_samples_played_in_range(
        self, start_seconds: float, stop_seconds: float
    ) -> Tensor:
        """
        TODO DOCS
        """
        # if not start_seconds <= stop_seconds:
        #     raise ValueError(
        #         f"Invalid start seconds: {start_seconds}. It must be less than or equal to stop seconds ({stop_seconds})."
        #     )
        # if not self._begin_stream_seconds <= start_seconds < self._end_stream_seconds:
        #     raise ValueError(
        #         f"Invalid start seconds: {start_seconds}. "
        #         f"It must be greater than or equal to {self._begin_stream_seconds} "
        #         f"and less than or equal to {self._end_stream_seconds}."
        #     )
        # if not stop_seconds <= self._end_stream_seconds:
        #     raise ValueError(
        #         f"Invalid stop seconds: {stop_seconds}. "
        #         f"It must be less than or equal to {self._end_stream_seconds}."
        #     )

        frames, *_ = core.get_frames_by_pts_in_range(
            self._decoder,
            start_seconds=start_seconds,
            stop_seconds=stop_seconds,
        )
        # TODO need to return view on this to account for samples instead of
        # frames
        return frames


def _get_and_validate_stream_metadata(
    decoder: Tensor,
    stream_index: Optional[int] = None,
) -> Tuple[core.AudioStreamMetadata, int]:

    # TODO should this still be called `get_video_metadata`?
    container_metadata = core.get_video_metadata(decoder)

    if stream_index is None:
        best_stream_index = container_metadata.best_audio_stream_index
        if best_stream_index is None:
            raise ValueError(
                "The best audio stream is unknown and there is no specified stream. "
                + _ERROR_REPORTING_INSTRUCTIONS
            )
        stream_index = best_stream_index

    # This should be logically true because of the above conditions, but type checker
    # is not clever enough.
    assert stream_index is not None

    stream_metadata = container_metadata.streams[stream_index]
    return (stream_metadata, stream_index)
