from typing import List

import torch

from torchcodec.decoders import (  # TODO: move FrameBatch to torchcodec.FrameBatch?
    FrameBatch,
    SimpleVideoDecoder,
)


def clips_at_random_indices(
    decoder: SimpleVideoDecoder,
    *,
    num_clips: int = 1,
    num_frames_per_clip: int = 1,
    num_indices_between_frames: int = 1,
) -> List[FrameBatch]:
    if num_clips <= 0:
        raise ValueError(f"num_clips ({num_clips}) must be strictly positive")
    if num_frames_per_clip <= 0:
        raise ValueError(
            f"num_frames_per_clip ({num_frames_per_clip}) must be strictly positive"
        )
    if num_indices_between_frames <= 0:
        raise ValueError(
            f"num_indices_between_frames ({num_indices_between_frames}) must be strictly positive"
        )

    # Determine the span of a clip, i.e. the number of frames (or indices)
    # between the first and last frame in the clip, both included. This isn't
    # the same as the number of frames in a clip!
    # Example: f means a frame in the clip, x means a frame excluded from the clip
    # num_frames_per_clip = 4
    # num_indices_between_frames = 1, clip = ffff      , span = 4
    # num_indices_between_frames = 2, clip = fxfxfxf   , span = 7
    # num_indices_between_frames = 3, clip = fxxfxxfxxf, span = 10
    clip_span = num_indices_between_frames * (num_frames_per_clip - 1) + 1

    # TODO: We should probably not error.
    if clip_span > len(decoder):
        raise ValueError(
            f"Clip span ({clip_span}) is larger than the number of frames ({len(decoder)})"
        )

    last_clip_start_index = len(decoder) - clip_span
    clip_start_indices = torch.randint(
        low=0, high=last_clip_start_index + 1, size=(num_clips,)
    )

    # TODO: This is inefficient as we are potentially seeking backwards.
    # We should sort by clip start before querying, and re-shuffle.
    # Note: we may still have to seek backwards if we have overlapping clips.
    clips = [
        decoder.get_frames_at(
            start=clip_start_index,
            stop=clip_start_index + clip_span,
            step=num_indices_between_frames,
        )
        for clip_start_index in clip_start_indices
    ]

    return clips
