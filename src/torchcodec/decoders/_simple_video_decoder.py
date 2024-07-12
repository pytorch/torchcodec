import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Union

from torch import Tensor

from torchcodec.decoders import _core as core


# TODO: we want to add index as well, but we need
#       the core operations to return it.
@dataclass
class Frame(Iterable):
    data: Tensor
    pts_seconds: float
    duration_seconds: float

    def __iter__(self) -> Iterator[Union[Tensor, float]]:
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)


class SimpleVideoDecoder:
    """TODO: Add docstring."""

    def __init__(self, source: Union[str, Path, bytes, Tensor]):
        if isinstance(source, str):
            self._decoder = core.create_from_file(source)
        elif isinstance(source, Path):
            self._decoder = core.create_from_file(str(source))
        elif isinstance(source, bytes):
            self._decoder = core.create_from_bytes(source)
        elif isinstance(source, Tensor):
            self._decoder = core.create_from_tensor(source)
        else:
            raise TypeError(
                f"Unknown source type: {type(source)}. "
                "Supported types are str, Path, bytes and Tensor."
            )
        core.scan_all_streams_to_update_metadata(self._decoder)
        core.add_video_stream(self._decoder)

        self.stream_metadata = _get_and_validate_stream_metadata(self._decoder)
        self._num_frames: int = self.stream_metadata.num_frames_computed  # type: ignore[assignment]
        self._stream_index = self.stream_metadata.stream_index
        self._min_pts_seconds: float = self.stream_metadata.min_pts_seconds  # type: ignore[assignment]
        self._max_pts_seconds: float = self.stream_metadata.max_pts_seconds  # type: ignore[assignment]

    def __len__(self) -> int:
        return self._num_frames

    def _getitem_int(self, key: int) -> Tensor:
        assert isinstance(key, int)

        if key < 0:
            key += self._num_frames
        if key >= self._num_frames or key < 0:
            raise IndexError(
                f"Index {key} is out of bounds; length is {self._num_frames}"
            )

        frame_data, *_ = core.get_frame_at_index(
            self._decoder, frame_index=key, stream_index=self._stream_index
        )
        return frame_data

    def _getitem_slice(self, key: slice) -> Tensor:
        assert isinstance(key, slice)

        start, stop, step = key.indices(len(self))
        return core.get_frames_in_range(
            self._decoder,
            stream_index=self._stream_index,
            start=start,
            stop=stop,
            step=step,
        )

    def __getitem__(self, key: Union[int, slice]) -> Tensor:
        if isinstance(key, int):
            return self._getitem_int(key)
        elif isinstance(key, slice):
            return self._getitem_slice(key)

        raise TypeError(
            f"Unsupported key type: {type(key)}. Supported types are int and slice."
        )

    def get_frame_at(self, index: int) -> Frame:
        if not 0 <= index < self._num_frames:
            raise IndexError(
                f"Index {index} is out of bounds; must be in the range [0, {self._num_frames})."
            )
        frame = core.get_frame_at_index(
            self._decoder, frame_index=index, stream_index=self._stream_index
        )
        return Frame(*frame)

    def get_frame_displayed_at(self, pts_seconds: float) -> Frame:
        if not self._min_pts_seconds <= pts_seconds < self._max_pts_seconds:
            raise IndexError(
                f"Invalid pts in seconds: {pts_seconds}. "
                f"It must be greater than or equal to {self._min_pts_seconds} "
                f"and less than or equal to {self._max_pts_seconds}."
            )
        frame = core.get_frame_at_pts(self._decoder, pts_seconds)
        return Frame(*frame)


def _get_and_validate_stream_metadata(decoder: Tensor) -> core.StreamMetadata:
    video_metadata = core.get_video_metadata(decoder)

    if video_metadata.best_video_stream_index is None:
        raise ValueError(
            "The best video stream is unknown. This should never happen. "
            "Please report an issue following the steps in <TODO>"
        )

    best_stream_metadata = video_metadata.streams[
        video_metadata.best_video_stream_index
    ]
    if best_stream_metadata.num_frames_computed is None:
        raise ValueError(
            "The number of frames is unknown. This should never happen. "
            "Please report an issue following the steps in <TODO>"
        )

    if best_stream_metadata.min_pts_seconds is None:
        raise ValueError(
            "The minimum pts value in seconds is unknown. This should never happen. "
            "Please report an issue following the steps in <TODO>"
        )

    if best_stream_metadata.max_pts_seconds is None:
        raise ValueError(
            "The maximum pts value in seconds is unknown. This should never happen. "
            "Please report an issue following the steps in <TODO>"
        )

    return best_stream_metadata
