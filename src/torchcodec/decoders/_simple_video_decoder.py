from typing import Union

import torch

from torchcodec.decoders import _core as core


class SimpleVideoDecoder:
    """TODO: Add docstring."""

    def __init__(self, source: Union[str, bytes, torch.Tensor]):
        # TODO: support Path objects.
        if isinstance(source, str):
            self._decoder = core.create_from_file(source)
        elif isinstance(source, bytes):
            self._decoder = core.create_from_bytes(source)
        elif isinstance(source, torch.Tensor):
            self._decoder = core.create_from_tensor(source)
        else:
            raise TypeError(
                f"Unknown source type: {type(source)}. "
                "Supported types are str, bytes and Tensor."
            )

        core.add_video_stream(self._decoder)

        self.stream_metadata = _get_and_validate_stream_metadata(self._decoder)
        # Note: these fields exist and are not None, as validated in _get_and_validate_simple_video_metadata().
        self._num_frames = self.stream_metadata.num_frames_computed
        self._stream_index = self.stream_metadata.stream_index

    def __len__(self) -> int:
        return self._num_frames

    def __getitem__(self, key: int) -> torch.Tensor:
        if not isinstance(key, int):
            raise TypeError(
                f"Unsupported key type: {type(key)}. Supported type is int."
            )

        if key < 0:
            key += self._num_frames
        if key >= self._num_frames or key < 0:
            raise IndexError(
                f"Index {key} is out of bounds; length is {self._num_frames}"
            )

        return core.get_frame_at_index(
            self._decoder, frame_index=key, stream_index=self._stream_index
        )

    def __iter__(self) -> "SimpleVideoDecoder":
        return self

    def __next__(self) -> torch.Tensor:
        # TODO: We should distinguish between expected end-of-file and unexpected
        # runtime error.
        try:
            return core.get_next_frame(self._decoder)
        except RuntimeError:
            raise StopIteration()


def _get_and_validate_stream_metadata(decoder: torch.Tensor) -> core.StreamMetadata:
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

    return best_stream_metadata
