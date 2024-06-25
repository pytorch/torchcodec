import json

from dataclasses import dataclass
from typing import List, Optional

import torch

from torchcodec.decoders._core.video_decoder_ops import (
    _get_container_json_metadata,
    _get_stream_json_metadata,
)


@dataclass
class StreamMetadata:
    duration_seconds: Optional[float]
    bit_rate: Optional[float]
    # TODO Comment from Nicolas:
    # Looking at this, it's not immediately obvious to me that "retrieved" means
    # "less accurate than 'computed'".
    # Are we open to different names? E.g. "num_frames_from_header" and "num_frames_accurate"?
    num_frames_retrieved: Optional[int]
    num_frames_computed: Optional[int]
    min_pts_seconds: Optional[float]
    max_pts_seconds: Optional[float]
    codec: Optional[str]
    width: Optional[int]
    height: Optional[int]
    average_fps: Optional[float]
    stream_index: int

    @property
    def num_frames(self) -> Optional[int]:
        if self.num_frames_computed is not None:
            return self.num_frames_computed
        else:
            return self.num_frames_retrieved


@dataclass
class VideoMetadata:
    # TODO: Is 'container' an FFmpeg term?
    container_duration_seconds: Optional[float]
    container_bit_rate: Optional[float]
    best_video_stream_index: Optional[int]
    best_audio_stream_index: Optional[int]

    streams: List[StreamMetadata]

    @property
    def duration_seconds(self) -> Optional[float]:
        if (
            self.best_video_stream_index is not None
            and self.streams[self.best_video_stream_index].duration_seconds is not None
        ):
            return self.streams[self.best_video_stream_index].duration_seconds
        else:
            return self.container_duration_seconds

    @property
    def bit_rate(self) -> Optional[float]:
        if (
            self.best_video_stream_index is not None
            and self.streams[self.best_video_stream_index].bit_rate is not None
        ):
            return self.streams[self.best_video_stream_index].bit_rate
        else:
            return self.container_bit_rate

    @property
    def best_video_stream(self) -> StreamMetadata:
        assert self.container.best_video_stream_index is not None
        return self.container.streams[self.container.best_video_stream_index]


def get_video_metadata(decoder: torch.tensor) -> VideoMetadata:

    container_dict = json.loads(_get_container_json_metadata(decoder))
    streams_metadata = []
    for stream_index in range(container_dict["numStreams"]):
        stream_dict = json.loads(_get_stream_json_metadata(decoder, stream_index))
        streams_metadata.append(
            StreamMetadata(
                duration_seconds=stream_dict.get("durationSeconds"),
                bit_rate=stream_dict.get("bitRate"),
                # TODO: We should align the C++ names and the json keys with the Python names
                num_frames_retrieved=stream_dict.get("numFrames"),
                num_frames_computed=stream_dict.get("numFramesFromScan"),
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
        container_duration_seconds=container_dict.get("durationSeconds"),
        container_bit_rate=container_dict.get("bitRate"),
        best_video_stream_index=container_dict.get("bestVideoStreamIndex"),
        best_audio_stream_index=container_dict.get("bestAudioStreamIndex"),
        streams=streams_metadata,
    )
