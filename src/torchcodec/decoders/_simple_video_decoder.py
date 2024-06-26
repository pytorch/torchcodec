import json
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

        # TODO: We should either implement specific core library function to
        # retrieve these values, or replace this with a non-JSON metadata
        # retrieval.
        metadata_json = json.loads(core.get_json_metadata(self._decoder))
        self._num_frames = metadata_json["numFrames"]
        self._stream_index = metadata_json["bestVideoStreamIndex"]

    def __len__(self) -> int:
        return self._num_frames

    def _getitem_int(self, key: int) -> torch.Tensor:
        assert isinstance(key, int)

        if key < 0:
            key += self._num_frames
        if key >= self._num_frames or key < 0:
            raise IndexError(
                f"Index {key} is out of bounds; length is {self._num_frames}"
            )

        return core.get_frame_at_index(
            self._decoder, frame_index=key, stream_index=self._stream_index
        )

    def _getitem_slice(self, key: slice) -> torch.Tensor:
        assert isinstance(key, slice)

        start, stop, step = key.indices(len(self))
        return core.get_frames_in_range(
            self._decoder,
            stream_index=self._stream_index,
            start=start,
            stop=stop,
            step=step,
        )

    def __getitem__(self, key: Union[int, slice]) -> torch.Tensor:
        if isinstance(key, int):
            return self._getitem_int(key)
        elif isinstance(key, slice):
            return self._getitem_slice(key)

        raise TypeError(
            f"Unsupported key type: {type(key)}. Supported types are int and slice."
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
