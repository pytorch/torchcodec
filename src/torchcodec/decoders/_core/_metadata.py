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
    # TODO: Before release, we should come up with names that better convey the
    # " 'fast and potentially inaccurate' vs 'slower but accurate' " tradeoff.
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
    duration_seconds_container: Optional[float]
    bit_rate_container: Optional[float]
    best_video_stream_index: Optional[int]
    best_audio_stream_index: Optional[int]

    streams: List[StreamMetadata]

    @property
    def duration_seconds(self) -> Optional[float]:
        raise NotImplementedError("TODO: decide on logic and implement this!")

    @property
    def bit_rate(self) -> Optional[float]:
        raise NotImplementedError("TODO: decide on logic and implement this!")

    @property
    def best_video_stream(self) -> StreamMetadata:
        if self.best_video_stream_index is None:
            raise ValueError("The best video stream is unknown.")
        return self.streams[self.best_video_stream_index]


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
        duration_seconds_container=container_dict.get("durationSeconds"),
        bit_rate_container=container_dict.get("bitRate"),
        best_video_stream_index=container_dict.get("bestVideoStreamIndex"),
        best_audio_stream_index=container_dict.get("bestAudioStreamIndex"),
        streams=streams_metadata,
    )
