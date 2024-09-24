import random
from typing import List, Optional

import torch

from torchcodec.decoders import (  # TODO: move FrameBatch to torchcodec.FrameBatch?
    FrameBatch,
    SimpleVideoDecoder,
)


def _validate_sampling_range(
    *, sampling_range_start, sampling_range_end, num_frames, clip_span
):
    if sampling_range_start < 0:
        raise ValueError(
            f"sampling_range_start ({sampling_range_start}) must be non-negative."
        )

    # TODO: or max(sampling_range_start, num_frames - 1)?
    sampling_range_start = sampling_range_start % num_frames

    if sampling_range_end is None:
        sampling_range_end = num_frames - clip_span + 1
        if sampling_range_start > sampling_range_end:
            raise ValueError(
                f"We determined that sampling_range_end should be {sampling_range_end}, "
                f"but it is smaller than sampling_range_start ({sampling_range_start})."
            )
    else:
        if sampling_range_end < 0:
            # Support negative values so that -1 means last frame.
            # TODO: do we want to wrap around if sampling_range_end < -num_frames ?
            sampling_range_end = num_frames + sampling_range_end + 1
        if sampling_range_start > sampling_range_end:
            raise ValueError(
                f"sampling_range_start ({sampling_range_start}) must be smaller than "
                f"sampling_range_end ({sampling_range_end})."
            )

    return sampling_range_start, sampling_range_end


def clips_at_random_indices(
    decoder: SimpleVideoDecoder,
    *,
    num_clips: int = 1,
    num_frames_per_clip: int = 1,
    num_indices_between_frames: int = 1,
    sampling_range_start: int = 0,
    sampling_range_end: Optional[int] = None,  # interval is [start, end).
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

    if len(decoder) < 1:
        raise ValueError(
            f"Decoder must have at least one frame, found {len(decoder)} frames."
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

    sampling_range_start, sampling_range_end = _validate_sampling_range(
        sampling_range_start=sampling_range_start,
        sampling_range_end=sampling_range_end,
        num_frames=len(decoder),
        clip_span=clip_span,
    )

    clip_start_indices = torch.randint(
        low=sampling_range_start, high=sampling_range_end, size=(num_clips,)
    )

    # We want to avoid seeking backwards, so we sort the clip start indices
    # before decoding the frames, and then re-shuffle the clips afterwards.
    # Backward seeks may still happen if there are overlapping clips, i.e. if a
    # clip ends after the next one starts.
    # TODO: We should use a different strategy to avoid backward seeks:
    # - flatten all frames indices, irrespective of their clip
    # - sort the indices and dedup
    # - decode all frames in index order
    # - re-arrange the frames back into their original clips
    clip_start_indices = torch.sort(clip_start_indices).values
    clips = [
        decoder.get_frames_at(
            start=clip_start_index,
            stop=clip_start_index + clip_span,
            step=num_indices_between_frames,
        )
        for clip_start_index in clip_start_indices
    ]

    # This an ugly way to shuffle the clips using pytorch RNG *without*
    # affecting the python builtin RNG.
    builtin_random_state = random.getstate()
    random.seed(torch.randint(0, 2**32, (1,)).item())
    random.shuffle(clips)
    random.setstate(builtin_random_state)

    return clips
