import json

from dataclasses import dataclass
from typing import List, Optional

import torch

from torchcodec.decoders._core.video_decoder_ops import (
    _get_container_json_metadata,
    _get_stream_json_metadata,
)


@dataclass
class ContainerMetadata:
    duration_seconds: Optional[float]
    bit_rate: Optional[float]
    best_video_stream_index: Optional[int]
    best_audio_stream_index: Optional[int]


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

    @property
    def num_frames(self) -> Optional[int]:
        if self.num_frames_computed is not None:
            return self.num_frames_computed
        else:
            return self.num_frames_retrieved


@dataclass
class VideoMetadata:
    container: ContainerMetadata
    streams: List[StreamMetadata]

    @property
    def duration_seconds(self) -> Optional[float]:
        if (
            self.container.best_video_stream_index is not None
            and self.streams[self.container.best_video_stream_index].duration_seconds
            is not None
        ):
            return self.streams[self.container.best_video_stream_index].duration_seconds
        else:
            return self.container.duration_seconds

    @property
    def bit_rate(self) -> Optional[float]:
        if (
            self.container.best_video_stream_index is not None
            and self.streams[self.container.best_video_stream_index].bit_rate
            is not None
        ):
            return self.streams[self.container.best_video_stream_index].bit_rate
        else:
            return self.contain.bit_rate

    @property
    def best_video_stream(self) -> StreamMetadata:
        assert self.container.best_video_stream_index is not None
        return self.container.streams[self.container.best_video_stream_index]


def get_video_metadata(decoder: torch.tensor) -> VideoMetadata:

    container_dict = json.loads(_get_container_json_metadata(decoder))
    container_metadata = ContainerMetadata(
        duration_seconds=container_dict.get("durationSeconds"),
        bit_rate=container_dict.get("bitRate"),
        best_video_stream_index=container_dict.get("bestVideoStreamIndex"),
        best_audio_stream_index=container_dict.get("bestAudioStreamIndex"),
    )

    streams_metadata = []
    for stream_index in range(container_dict["numStreams"]):
        stream_dict = json.loads(_get_stream_json_metadata(decoder, stream_index))
        streams_metadata.append(
            StreamMetadata(
                duration_seconds=stream_dict.get("durationSeconds"),
                bit_rate=stream_dict.get("bitRate"),
                num_frames_retrieved=stream_dict.get("numFrames"),
                num_frames_computed=stream_dict.get("numFramesFromScan"),
                min_pts_seconds=stream_dict.get("minPtsSecondsFromScan"),
                max_pts_seconds=stream_dict.get("maxPtsSecondsFromScan"),
                codec=stream_dict.get("codec"),
                width=stream_dict.get("width"),
                height=stream_dict.get("height"),
                average_fps=stream_dict.get("averageFps"),
            )
        )

    return VideoMetadata(container=container_metadata, streams=streams_metadata)
