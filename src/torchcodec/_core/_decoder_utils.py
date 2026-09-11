# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import io
from collections.abc import Sequence
from pathlib import Path

from torch import nn, Tensor
from torchcodec._core._metadata import (
    AudioStreamMetadata,
    get_container_metadata,
    VideoStreamMetadata,
)
from torchcodec._core.ops import (
    _blocks_create_demuxer_from_bytes,
    _blocks_create_demuxer_from_file,
    _blocks_create_demuxer_from_file_like,
    _blocks_create_demuxer_from_tensor,
    add_audio_stream,
    add_video_stream,
    create_from_bytes,
    create_from_file,
    create_from_file_like,
    create_from_tensor,
    create_wav_decoder_from_bytes,
    create_wav_decoder_from_file,
    create_wav_decoder_from_file_like,
    create_wav_decoder_from_tensor,
)
from torchcodec._internally_replaced_utils import load_core_libraries
from torchcodec.transforms import DecoderTransform
from torchcodec.transforms._decoder_transforms import _make_transform_specs


_ERROR_REPORTING_INSTRUCTIONS = """
This should never happen. Please report an issue following the steps in
https://github.com/pytorch/torchcodec/issues/new?assignees=&labels=&projects=&template=bug-report.yml.
"""


def create_decoder(
    *,
    source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
    seek_mode: str,
) -> Tensor:
    load_core_libraries()
    if isinstance(source, str):
        return create_from_file(source, seek_mode)
    elif isinstance(source, Path):
        return create_from_file(str(source), seek_mode)
    elif isinstance(source, io.RawIOBase) or isinstance(source, io.BufferedReader):
        return create_from_file_like(source, seek_mode)
    elif isinstance(source, bytes):
        return create_from_bytes(source, seek_mode)
    elif isinstance(source, Tensor):
        return create_from_tensor(source, seek_mode)
    elif isinstance(source, io.TextIOBase):
        raise TypeError(
            "source is for reading text, likely from open(..., 'r'). Try with 'rb' for binary reading?"
        )
    elif hasattr(source, "read") and hasattr(source, "seek"):
        return create_from_file_like(source, seek_mode)

    raise TypeError(
        f"Unknown source type: {type(source)}. "
        "Supported types are str, Path, bytes, Tensor and file-like objects with "
        "read(self, size: int) -> bytes and "
        "seek(self, offset: int, whence: int) -> int methods."
    )


def create_demuxer(
    *,
    source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
) -> Tensor:
    """Open a container and probe its header. No stream is followed yet, and no
    packet is read: call ``_blocks_demuxer_add_stream`` for each stream to
    follow, before demuxing anything."""
    load_core_libraries()
    if isinstance(source, str):
        return _blocks_create_demuxer_from_file(source)
    elif isinstance(source, Path):
        return _blocks_create_demuxer_from_file(str(source))
    elif isinstance(source, bytes):
        return _blocks_create_demuxer_from_bytes(source)
    elif isinstance(source, Tensor):
        return _blocks_create_demuxer_from_tensor(source)
    elif isinstance(source, (io.RawIOBase, io.BufferedReader)):
        return _blocks_create_demuxer_from_file_like(source)
    elif isinstance(source, io.TextIOBase):
        raise TypeError(
            "source is for reading text, likely from open(..., 'r'). Try with 'rb' for binary reading?"
        )
    elif hasattr(source, "read") and hasattr(source, "seek"):
        return _blocks_create_demuxer_from_file_like(source)

    raise TypeError(
        f"Unknown source type: {type(source)}. "
        "Supported types are str, Path, bytes, Tensor and file-like objects with "
        "read(self, size: int) -> bytes and "
        "seek(self, offset: int, whence: int) -> int methods."
    )


def create_audio_decoder(
    *,
    source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
    seek_mode: str,
    stream_index: int | None = None,
    sample_rate: int | None = None,
    num_channels: int | None = None,
) -> tuple[Tensor, int, AudioStreamMetadata]:

    decoder = create_decoder(source=source, seek_mode=seek_mode)

    container_metadata = get_container_metadata(decoder)

    if stream_index is None:
        stream_index = container_metadata.best_audio_stream_index
        if stream_index is None:
            raise ValueError(
                "The best audio stream is unknown and there is no specified stream. "
                + _ERROR_REPORTING_INSTRUCTIONS
            )

    if stream_index >= len(container_metadata.streams):
        raise ValueError(f"The stream at index {stream_index} is not a valid stream.")

    metadata = container_metadata.streams[stream_index]
    if not isinstance(metadata, AudioStreamMetadata):
        raise ValueError(f"The stream at index {stream_index} is not an audio stream.")

    add_audio_stream(
        decoder,
        stream_index=stream_index,
        sample_rate=sample_rate,
        num_channels=num_channels,
    )

    return (decoder, stream_index, metadata)


def create_wav_decoder(
    source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
) -> Tensor:
    # WavDecoder currently lives in the FFmpeg-linked core library (it uses the
    # FFmpeg-based AVIO I/O layer), so it needs FFmpeg too, for now.
    load_core_libraries()
    if isinstance(source, str):
        return create_wav_decoder_from_file(source)
    elif isinstance(source, Path):
        return create_wav_decoder_from_file(str(source))
    elif isinstance(source, bytes):
        return create_wav_decoder_from_bytes(source)
    elif isinstance(source, Tensor):
        return create_wav_decoder_from_tensor(source)
    elif isinstance(source, (io.RawIOBase, io.BufferedReader)):
        return create_wav_decoder_from_file_like(source)
    elif isinstance(source, io.TextIOBase):
        raise TypeError(
            "source is for reading text, likely from open(..., 'r'). "
            "Try with 'rb' for binary reading?"
        )
    elif hasattr(source, "read") and hasattr(source, "seek"):
        return create_wav_decoder_from_file_like(source)

    raise TypeError(
        f"Unknown source type: {type(source)}. "
        "Supported types are str, Path, bytes, Tensor and file-like objects."
    )


def _get_and_validate_video_stream_metadata(
    *,
    decoder: Tensor,
    stream_index: int | None = None,
) -> tuple[VideoStreamMetadata, int]:
    container_metadata = get_container_metadata(decoder)

    if stream_index is None:
        if (stream_index := container_metadata.best_video_stream_index) is None:
            raise ValueError(
                "The best video stream is unknown and there is no specified stream. "
                + _ERROR_REPORTING_INSTRUCTIONS
            )

    if stream_index >= len(container_metadata.streams):
        raise ValueError(f"The stream index {stream_index} is not a valid stream.")

    metadata = container_metadata.streams[stream_index]
    if not isinstance(metadata, VideoStreamMetadata):
        raise ValueError(f"The stream at index {stream_index} is not a video stream. ")

    if metadata.begin_stream_seconds is None:
        raise ValueError(
            "The minimum pts value in seconds is unknown. "
            + _ERROR_REPORTING_INSTRUCTIONS
        )

    if metadata.end_stream_seconds is None:
        raise ValueError(
            "The maximum pts value in seconds is unknown. "
            + _ERROR_REPORTING_INSTRUCTIONS
        )

    if metadata.num_frames is None:
        raise ValueError(
            "The number of frames is unknown. " + _ERROR_REPORTING_INSTRUCTIONS
        )

    return (metadata, stream_index)


def create_video_decoder(
    *,
    source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
    seek_mode: str,
    stream_index: int | None = None,
    dimension_order: str = "NCHW",
    num_ffmpeg_threads: int = 1,
    device: str,
    device_variant: str = "default",
    transforms: Sequence[DecoderTransform | nn.Module] | None = None,
    custom_frame_mappings: tuple[Tensor, Tensor, Tensor] | None = None,
    output_dtype: str = "uint8",
) -> tuple[Tensor, int, VideoStreamMetadata]:

    decoder = create_decoder(source=source, seek_mode=seek_mode)

    (
        metadata,
        stream_index,
    ) = _get_and_validate_video_stream_metadata(
        decoder=decoder, stream_index=stream_index
    )

    transform_specs = _make_transform_specs(
        transforms,
        input_dims=(metadata.height, metadata.width),
    )

    add_video_stream(
        decoder,
        stream_index=stream_index,
        dimension_order=dimension_order,
        num_threads=num_ffmpeg_threads,
        device=device,
        device_variant=device_variant,
        transform_specs=transform_specs,
        custom_frame_mappings=custom_frame_mappings,
        output_dtype=output_dtype,
    )

    return (decoder, stream_index, metadata)
