# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from typing import Optional, Tuple, Union

from torch import Tensor
from torchcodec.decoders import _core as core

ERROR_REPORTING_INSTRUCTIONS = """
This should never happen. Please report an issue following the steps in
https://github.com/pytorch/torchcodec/issues/new?assignees=&labels=&projects=&template=bug-report.yml.
"""


def create_decoder(
    *, source: Union[str, Path, bytes, Tensor], seek_mode: str
) -> Tensor:
    if isinstance(source, str):
        return core.create_from_file(source, seek_mode)
    elif isinstance(source, Path):
        return core.create_from_file(str(source), seek_mode)
    elif isinstance(source, bytes):
        return core.create_from_bytes(source, seek_mode)
    elif isinstance(source, Tensor):
        return core.create_from_tensor(source, seek_mode)

    raise TypeError(
        f"Unknown source type: {type(source)}. "
        "Supported types are str, Path, bytes and Tensor."
    )


def get_and_validate_stream_metadata(
    *,
    decoder: Tensor,
    stream_index: Optional[int] = None,
    media_type: str,
) -> Tuple[core._metadata.StreamMetadata, int, float, float]:

    if media_type not in ("video", "audio"):
        raise ValueError(f"Bad {media_type = }, should be audio or video")

    container_metadata = core.get_container_metadata(decoder)

    if stream_index is None:
        best_stream_index = (
            container_metadata.best_video_stream_index
            if media_type == "video"
            else container_metadata.best_audio_stream_index
        )
        if best_stream_index is None:
            raise ValueError(
                f"The best {media_type} stream is unknown and there is no specified stream. "
                + ERROR_REPORTING_INSTRUCTIONS
            )
        stream_index = best_stream_index

    # This should be logically true because of the above conditions, but type checker
    # is not clever enough.
    assert stream_index is not None

    metadata = container_metadata.streams[stream_index]

    if metadata.begin_stream_seconds is None:
        raise ValueError(
            "The minimum pts value in seconds is unknown. "
            + ERROR_REPORTING_INSTRUCTIONS
        )
    begin_stream_seconds = metadata.begin_stream_seconds

    if metadata.end_stream_seconds is None:
        raise ValueError(
            "The maximum pts value in seconds is unknown. "
            + ERROR_REPORTING_INSTRUCTIONS
        )
    end_stream_seconds = metadata.end_stream_seconds
    return (
        metadata,
        stream_index,
        begin_stream_seconds,
        end_stream_seconds,
    )
