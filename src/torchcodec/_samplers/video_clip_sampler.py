# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import nn, Tensor

from torchcodec.decoders._core import (
    add_video_stream,
    create_from_tensor,
    get_frames_at_indices,
    get_json_metadata,
    get_next_frame,
    scan_all_streams_to_update_metadata,
    seek_to_pts,
)


class VideoTooShortException(Exception):
    pass


@dataclass
class DecoderArgs:
    num_threads: int = 0


@dataclass
class VideoArgs:
    """
    VideoArgs contains video related information. Video width/heigh can't be co-exist with video min/max dimension.
    Args:
        desired_width (`int`): Target width of the video
        desired_height (`int`): Target height of the video
        desired_max_dimension (`int`): Target maximum dimension of the video
        desired_min_dimension (`int`): Target minimum dimension of the video
    """

    desired_width: int = 0
    desired_height: int = 0
    desired_max_dimension: int = 0
    desired_min_dimension: int = 0


@dataclass
class SamplerArgs(abc.ABC):
    """
    Abstract class of sampler args, extended by TimeBasedSamplerArgs and IndexBasedSamplerArgs.
    Frame refers to a video/audio frame, and clip is a list of frames which may be non-consecutive.
    Args:
        sampler_type (`str`): Sampler type, can be random, uniform, periodic, target
        clips_per_video (`int`): Number of clips per video, this applys to random and uniform sampling
        frames_per_clip (`int`): Number of frames per clip
    """

    sampler_type: str
    clips_per_video: int
    frames_per_clip: int


@dataclass
class TimeBasedSamplerArgs(SamplerArgs):
    """
    TimeBasedSamplerArgs inherits from SamplerArgs and describe the time based sampling behavior.
    Args:
        video_frame_dilation (`int`): Frame dilation of the video, if frame dilation is 2, we will sample every other frame within a clip.
        sample_start_second (`float`): Start second of the sampler range, applies to all sampler types
        sample_end_second (`float`): End second of the sampler range, applies to all sampler types
        sample_per_second (`float`): Sample per second of the sampler range, applies to periodic sampling
        target_sample_start_second (`float`): Start second of the target sampling range, applies to target sampling
    """

    video_frame_dilation: int = 1
    sample_start_second: float = 0.0
    sample_end_second: float = float("inf")
    sample_per_second: float = 0.0
    target_sample_start_second: List[float] = field(default_factory=lambda: [])


@dataclass
class IndexBasedSamplerArgs(SamplerArgs):
    """
    IndexBasedSamplerArgs inherits from SamplerArgs and describe the index based sampling behavior.
    sample_start_index and sample_end_index together decide the range of the sampling.
    sample_step decides step between each clip.
    video_frame_dilation decides step between each frame within a clip.
    Args:
        video_frame_dilation (`int`): Frame dilation of the video, if frame dilation is 2, we will sample every other frame within a clip, applies to all sampler types
        sample_start_index (`int`): Start index of the sampler range, applies to all sampler types
        sample_end_index (`int`): End index of the sampler range, this is last possile frame you want to sample, applies to all sampler types
        sample_step (`int`): Step of the sampler range, if step is 10, the interval between start frames of each clip will be 10, applies to periodic sampling only.
    """

    video_frame_dilation: int = 1
    sample_start_index: int = 0
    sample_end_index: int = sys.maxsize
    sample_step: int = 1


class VideoClipSampler(nn.Module):
    """
    VideoClipSampler will do video clip sampling with given video args and sampler args.
    The video args contains video related information, frames_per_clip, dimensions etc.
    The sampler args can be either time-based or index-based, it will be used to decide clip start time pts or index.
    ClipSampling support, random, uniform, periodic, target, keyframe sampling etc.

    Args:
        video_args (`VideoArgs`): The video args
        sampler_args (`SamplerArgs`): The sampler args. Can be TimeBasedSamplerArgs or IndexBasedSamplerArgs
        decoder_args (`DecoderArgs`): Decoder args contain value needs for decoder, for example, thread count

    Example:
        >>> video_args = VideoArgs(desired_width=224, desired_height=224)
        >>> time_based_sampler_args = TimeBasedSamplerArgs(sampler_type="random", clips_per_video=1, frames_per_clip=4)
        >>> video_decoder_args = DecoderArgs(num_threads=1)
        >>> video_clip_sampler = VideoClipSampler(video_args, time_based_sampler_args, decoder_args)
        >>> clips = video_clip_sampler(video_data)
        clips now contains a list of clip, where clip is a list of frame tensors, each tensor represents a frame image.
    """

    def __init__(
        self,
        video_args: VideoArgs,
        sampler_args: SamplerArgs,
        decoder_args: Union[None, DecoderArgs] = None,
    ) -> None:
        super().__init__()
        self.video_args = video_args
        self.sampler_args = sampler_args
        self.decoder_args = DecoderArgs() if decoder_args is None else decoder_args

    def forward(self, video_data: Tensor) -> Union[List[Any]]:
        """Sample video clips from the video data

        Args:
            video_data (`Tensor`): The video data

        Return
            clips (` List[List[Tensor]]`): List of clips, where each clip is a list of Tensors, each tensor represents a frame image.

        """

        video_decoder = create_from_tensor(video_data)
        scan_all_streams_to_update_metadata(video_decoder)
        add_video_stream(video_decoder)
        metadata_json = json.loads(get_json_metadata(video_decoder))
        target_width, target_height = self._compute_frame_width_height(
            metadata_json["width"], metadata_json["height"]
        )

        video_decoder = create_from_tensor(video_data)
        scan_all_streams_to_update_metadata(video_decoder)
        add_video_stream(
            video_decoder,
            width=target_width,
            height=target_height,
            num_threads=self.decoder_args.num_threads,
        )

        clips: List[Any] = []
        # Cast sampler args to be time based or index based
        if isinstance(self.sampler_args, TimeBasedSamplerArgs):
            time_based_sampler_args = self.sampler_args
            clip_starts_in_seconds = self._get_start_seconds(
                metadata_json, time_based_sampler_args
            )
            for start_ts in clip_starts_in_seconds:
                clip = self._get_clip_with_start_second(
                    start_ts,
                    video_decoder,
                    time_based_sampler_args.video_frame_dilation,
                )
                clips.append(clip)
        elif isinstance(self.sampler_args, IndexBasedSamplerArgs):
            index_based_sampler_args = self.sampler_args
            clips = self._get_clips_for_index_based_sampling(
                video_decoder,
                index_based_sampler_args,
                metadata_json,
            )

        return clips

    def _get_clips_for_index_based_sampling(
        self,
        video_decoder: Tensor,
        index_based_sampler_args: IndexBasedSamplerArgs,
        metadata_json: Dict[str, Any],
    ) -> List[Tensor]:
        """Get clips for index based sampling, the sampling is done in 3 steps:
            1. Compute clip_start_idxs based on the sampler type and the sampler args;
            2. For each clip, given clip_start_idx, video_frame_dilation, frames_per_clip, get indexes for all frames
            3. With given index, fetch the frame and group into clip and then clips

        Args:
            video_decoder (`Tensor`): The video decoder
            index_based_sampler_args (`IndexBasedSamplerArgs`): The index based sampler args
            metadata_json (`Dict[str, Any]`): The metadata of the video in json format

        Returns:
            clips (` List[Tensor]`): List of clips, where each clip is a Tensor represents list of frames, Tensor shape default is NCHW.
        """

        sample_start_index = max(0, index_based_sampler_args.sample_start_index)
        sample_end_index = (
            min(
                index_based_sampler_args.sample_end_index,
                metadata_json["numFrames"],
            )
            - index_based_sampler_args.video_frame_dilation
            * index_based_sampler_args.frames_per_clip
        )
        sampler_type = index_based_sampler_args.sampler_type

        if sampler_type == "random":
            clip_start_idxs = torch.randint(
                sample_start_index,
                sample_end_index,
                (index_based_sampler_args.clips_per_video,),
            )
        elif sampler_type == "uniform":
            clip_start_idxs = torch.linspace(
                sample_start_index,
                sample_end_index,
                index_based_sampler_args.clips_per_video,
                dtype=torch.int32,
            )

        clips = []
        for clip_start_idx in clip_start_idxs:
            batch_indexes = [
                clip_start_idx + i * index_based_sampler_args.video_frame_dilation
                for i in range(index_based_sampler_args.frames_per_clip)
            ]
            frames, *_ = get_frames_at_indices(
                video_decoder,
                stream_index=metadata_json["bestVideoStreamIndex"],
                frame_indices=batch_indexes,
            )
            clips.append(frames)

        return clips

    def _get_start_seconds(
        self,
        metadata_json: Dict[str, Any],
        time_based_sampler_args: TimeBasedSamplerArgs,
    ) -> List[float]:
        """Get start seconds for each clip.
        Given different sampler type, the API returns different clip start seconds.

        Args:
            metadata_json (`Dict[str, Any]`): The metadata of the video in json format
            time_based_sampler_args: (`TimeBasedSamplerArgs`): The time based sampler args

        Returns:
            (`List[float]`): List of the sampled clip start position in seconds
        """
        video_duration_in_seconds = metadata_json["durationSeconds"]

        clip_duration_in_seconds = (
            time_based_sampler_args.frames_per_clip
            * time_based_sampler_args.video_frame_dilation
            + 1
        ) / metadata_json["averageFps"]

        minPtsSecondsFromScan = (
            metadata_json["minPtsSecondsFromScan"]
            if metadata_json["minPtsSecondsFromScan"]
            else 0
        )
        maxPtsSecondsFromScan = (
            metadata_json["maxPtsSecondsFromScan"]
            if metadata_json["maxPtsSecondsFromScan"] > 0
            else video_duration_in_seconds
        )
        last_possible_clip_start_in_seconds = (
            maxPtsSecondsFromScan - clip_duration_in_seconds
        )
        if last_possible_clip_start_in_seconds < 0:
            raise VideoTooShortException(
                "Cannot get clips because video duration is shorter than the clip duration!"
            )
        sampler_type = time_based_sampler_args.sampler_type
        clip_starts_in_seconds: List[float] = []
        sample_start_second = max(
            time_based_sampler_args.sample_start_second,
            minPtsSecondsFromScan,
        )
        sample_end_second = min(
            last_possible_clip_start_in_seconds,
            time_based_sampler_args.sample_end_second,
        )
        if sampler_type == "random":
            clip_starts_in_seconds = (
                torch.rand(time_based_sampler_args.clips_per_video)
                * (sample_end_second - sample_start_second)
                + sample_start_second
            ).tolist()
            clip_starts_in_seconds.sort()
        elif sampler_type == "uniform":
            clip_starts_in_seconds = torch.linspace(
                sample_start_second,
                sample_end_second,
                time_based_sampler_args.clips_per_video,
            ).tolist()
        else:
            raise NotImplementedError

        return clip_starts_in_seconds

    def _get_clip_with_start_second(
        self, start_second: float, video_decoder: Tensor, video_frame_dilation: int
    ) -> List[Tensor]:
        """Get clip with start second.

        Args:
            `start_second` (`float`): The start second of the clip
            `video_decoder` (`Tensor`): The video decoder
            `video_frame_dilation` (`int`): The video frame dilation, by default it's 1.

        Returns:
            `clip` (`List[Tensor]`): clip is list of frame tensor. Dimension of each frame tensor is user specified, by default it's HWC.
        """
        seek_to_pts(video_decoder, start_second)
        frames_needed_per_clip = (
            self.sampler_args.frames_per_clip - 1
        ) * video_frame_dilation + 1
        clip = []
        for _ in range(frames_needed_per_clip):
            frame, _, _ = get_next_frame(video_decoder)
            clip.append(frame)

        # slice the list of tensor with frame_dilation and stack to tensor
        clip = clip[::video_frame_dilation]
        return clip

    def _compute_frame_width_height(
        self, ori_width: int, ori_height: int
    ) -> Tuple[int, int]:
        """Compute output frame width and height
        desired_width, desired_height, desired_min_dimension, desired_max_dimension, (`int`): Together decide the size of the decoded video clips. (Default: `0`).
                Note that the desired_width/desired_height parameters are mutually exclusive with desired_min_dimension/desired_max_dimension parameters.
                - When desired_width = 0, desired_height = 0, desired_min_dimension = 0,
                    and desired_max_dimension = 0, keep the original frame resolution
                - When desired_width = 0, desired_height != 0, desired_min_dimension = 0,
                    and desired_max_dimension = 0, keep the aspect ratio and resize
                    the frame so that frame target_height is $desired_height
                - When desired_width != 0, desired_height == 0, desired_min_dimension = 0,
                    and desired_max_dimension = 0, keep the aspect ratio and resize
                    the frame so that frame target_width is $desired_width
                - When desired_width != 0, desired_height != 0, video_min_dimension = 0,
                    and desired_max_dimension = 0, resize the frame so that frame
                    target_width and target_height are set to $desired_width and
                    $desired_height, respectively
                - When desired_width = 0, desired_height = 0, desired_min_dimension != 0,
                    and desired_max_dimension = 0, keep the aspect ratio and resize the
                    frame so that shorter edge size is desired_min_dimension
                - When desired_width = 0, desired_height = 0, desired_min_dimension = 0,
                    and desired_max_dimension != 0, keep the aspect ratio and resize
                    the frame so that longer edge size is desired_max_dimension
                - When desired_width = 0, desired_height = 0, desired_min_dimension != 0,
                    and desired_max_dimension != 0, resize the frame so that shorter
                    edge size is desired_min_dimension, and longer edge size is
                    desired_max_dimension. The aspect ratio may not be preserved

        Args:
            ori_width (`int`): Original width of the video
            ori_height (`int`): Original height of the video

        Returns:
            (`Tuple[int, int]`): output frame width and height
        """
        width_height_ratio = ori_width / ori_height
        height_width_ratio = ori_height / ori_width

        target_width, target_height = ori_width, ori_height

        # video_height and/or video_width is non zero
        if self.video_args.desired_width == 0 and self.video_args.desired_height != 0:
            target_height = self.video_args.desired_height
            target_width = int(width_height_ratio * target_height)
        elif self.video_args.desired_width != 0 and self.video_args.desired_height == 0:
            target_width = self.video_args.desired_width
            target_height = int(height_width_ratio * target_width)
        elif self.video_args.desired_width != 0 and self.video_args.desired_height != 0:
            target_width, target_height = (
                self.video_args.desired_width,
                self.video_args.desired_height,
            )
        # video_min_dimension and/or video_max_dimension is non zero
        elif (
            self.video_args.desired_min_dimension != 0
            and self.video_args.desired_max_dimension == 0
        ):
            if ori_width > ori_height:
                target_height = self.video_args.desired_min_dimension
                target_width = int(width_height_ratio * target_height)
            else:
                target_width = self.video_args.desired_min_dimension
                target_height = int(height_width_ratio * target_width)
        elif (
            self.video_args.desired_min_dimension == 0
            and self.video_args.desired_max_dimension != 0
        ):
            if ori_width > ori_height:
                target_width = self.video_args.desired_max_dimension
                target_height = int(height_width_ratio * target_width)
            else:
                target_height = self.video_args.desired_max_dimension
                target_width = int(width_height_ratio * target_height)
        elif (
            self.video_args.desired_min_dimension != 0
            and self.video_args.desired_max_dimension != 0
        ):
            if ori_width > ori_height:
                target_width = self.video_args.desired_max_dimension
                target_height = self.video_args.desired_min_dimension
            else:
                target_height = self.video_args.desired_max_dimension
                target_width = self.video_args.desired_min_dimension

        return target_width, target_height
