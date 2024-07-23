import json
import pathlib

from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from torchcodec.decoders._core.video_decoder_ops import (
    _get_container_json_metadata,
    _get_stream_json_metadata,
    create_from_file,
)


@dataclass
class VideoStreamMetadata:
    """Metadata of a single video stream."""

    duration_seconds: Optional[float]
    """Duration of the stream, in seconds (float or None)."""
    bit_rate: Optional[float]
    """Bit rate of the stream, in seconds (float or None)."""
    num_frames_from_header: Optional[int]
    """Number of frames, from the stream's metadata. This is potentially
    inaccurate. We recommend using the ``num_frames`` attribute instead.
    (int or None)."""
    num_frames_from_content: Optional[int]
    """Number of frames computed by TorchCodec by scanning the stream's
    content (the scan doesn't involve decoding). This is more accurate
    than ``num_frames_from_header``. We recommend using the
    ``num_frames`` attribute instead. (int or None)."""
    min_pts_seconds: Optional[float]
    """Minimum :term:`pts` of any frame in the stream (float or None)."""
    max_pts_seconds: Optional[float]
    """Maximum :term:`pts` of any frame in the stream (float or None)."""
    codec: Optional[str]
    """Codec (str or None)."""
    width: Optional[int]
    """Width of the frames (int or None)."""
    height: Optional[int]
    """Height of the frames (int or None)."""
    average_fps: Optional[float]
    """Averate fps of the stream (float or None)."""
    stream_index: int
    """Index of the stream within the video (int)."""

    @property
    def num_frames(self) -> Optional[int]:
        """Number of frames in the stream. This corresponds to
        ``num_frames_from_content`` if it's not None, otherwise it corresponds
        to ``num_frames_from_header``.
        """
        if self.num_frames_from_content is not None:
            return self.num_frames_from_content
        else:
            return self.num_frames_from_header


@dataclass
class VideoMetadata:
    duration_seconds_from_header: Optional[float]
    bit_rate_from_header: Optional[float]
    best_video_stream_index: Optional[int]
    best_audio_stream_index: Optional[int]

    streams: List[VideoStreamMetadata]

    @property
    def duration_seconds(self) -> Optional[float]:
        raise NotImplementedError("Decide on logic and implement this!")

    @property
    def bit_rate(self) -> Optional[float]:
        raise NotImplementedError("Decide on logic and implement this!")

    @property
    def best_video_stream(self) -> VideoStreamMetadata:
        if self.best_video_stream_index is None:
            raise ValueError("The best video stream is unknown.")
        return self.streams[self.best_video_stream_index]


def get_video_metadata(decoder: torch.Tensor) -> VideoMetadata:
    """Return video metadata from a video decoder.

    The accuracy of the metadata and the availability of some returned fields
    depends on whether a full scan was performed by the decoder.
    """

    container_dict = json.loads(_get_container_json_metadata(decoder))
    streams_metadata = []
    for stream_index in range(container_dict["numStreams"]):
        stream_dict = json.loads(_get_stream_json_metadata(decoder, stream_index))
        streams_metadata.append(
            VideoStreamMetadata(
                duration_seconds=stream_dict.get("durationSeconds"),
                bit_rate=stream_dict.get("bitRate"),
                # TODO_OPEN_ISSUE: We should align the C++ names and the json
                # keys with the Python names
                num_frames_from_header=stream_dict.get("numFrames"),
                num_frames_from_content=stream_dict.get("numFramesFromScan"),
                min_pts_seconds=stream_dict.get("minPtsSecondsFromScan"),
                max_pts_seconds=stream_dict.get("maxPtsSecondsFromScan"),
                codec=stream_dict.get("codec"),
                width=stream_dict.get("width"),
                height=stream_dict.get("height"),
                average_fps=stream_dict.get("averageFps"),
                stream_index=stream_index,
            )
        )

    return VideoMetadata(
        duration_seconds_from_header=container_dict.get("durationSeconds"),
        bit_rate_from_header=container_dict.get("bitRate"),
        best_video_stream_index=container_dict.get("bestVideoStreamIndex"),
        best_audio_stream_index=container_dict.get("bestAudioStreamIndex"),
        streams=streams_metadata,
    )


def get_video_metadata_from_header(filename: Union[str, pathlib.Path]) -> VideoMetadata:
    return get_video_metadata(create_from_file(str(filename)))
