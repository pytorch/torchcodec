# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from torchcodec._core.ops import _blocks_frame_metadata, _blocks_frame_planes


class _Metadata(NamedTuple):
    """The fields of the `_blocks_frame_metadata` op, in order."""

    pix_fmt: str
    colorspace: str
    color_range: str
    bit_depth: int
    width: int
    height: int
    rotation_degrees: float


class Packet:
    """Opaque, thread-movable handle to a demuxed (compressed) packet.

    Produced by :class:`Demuxer`, consumed by :class:`VideoPacketDecoder`. It wraps a raw
    pointer, so it is only valid within the process that created it (it cannot
    cross a process boundary).

    Attributes:
        stream_index (int): The index of the stream this packet belongs to,
            absolute across all media types. This is what routes a packet to
            its decoder when it comes out of a :class:`Demuxer` following more
            than one stream.
    """

    def __init__(
        self,
        handle: torch.Tensor,
        stream_index: int,
        *,
        generation: int = 0,
    ):
        self._handle = handle
        self.stream_index = stream_index
        # Which side of the demuxer's last seek this packet came from. Private:
        # it exists so a decoder can catch a missing reset(), not for callers to
        # reason about. See _BasePacketDecoder.decode().
        self._generation = generation


# TODO_API_BREAKDOWN DESIGN P1: API design - the public fields, the class name,
# whether planes should be 2D (they are), whether they should be named, etc.
class RawFrame:
    """A decoded (YUV) frame, as the decoder produced it: an opaque,
    thread-movable handle to the frame plus everything describing it.

    Produced by :class:`VideoPacketDecoder`, consumed by :class:`ColorConverter`. The
    handle wraps a raw pointer and is process-local. ``pts_seconds`` and
    ``duration_seconds`` are stamped by the decoder (which knows the stream time
    base) and carried here so the :class:`ColorConverter` need not be bound to
    any stream.

    Every field describes the frame as it was decoded, with no conversion
    applied. In particular ``width``, ``height`` and :attr:`planes` are all
    pre-rotation: :attr:`rotation_degrees` is what ``ColorConverter`` applies
    for you, and what you have to apply yourself if you color-convert the planes
    on your own.

    The samples live on the device of the decoder that produced the frame. A
    CUDA decoder falls back to CPU decoding for streams NVDEC can't handle, but
    it uploads those frames before handing them out, so they are
    indistinguishable from NVDEC ones here.

    All the fields but ``pts_seconds``, ``duration_seconds`` and ``storage`` are
    computed on first access and then cached, so nothing is paid for by a
    pipeline that only color-converts.
    """

    def __init__(
        self,
        handle: torch.Tensor,
        pts_seconds: float,
        duration_seconds: float,
        device: torch.device,
        storage: torch.Tensor | None = None,
    ):
        self._handle = handle
        # TODO_API_BREAKDOWN DESIGN P2: do we need this at all? It is always the
        # device of the decoder that produced the frame, and users can get it
        # from `storage.device` or from `planes[0].device`. We need *something*
        # here because the planes op needs a device to build the views on.
        self._device = device
        # A CUDA consumer reading the frame on a different stream than the
        # decoder's must call storage.record_stream() on it.
        # See [Standalone Frame Storage and the need for record_stream]
        self.storage = storage
        self.pts_seconds = pts_seconds
        self.duration_seconds = duration_seconds
        self._metadata: _Metadata | None = None
        self._planes: tuple[torch.Tensor, ...] | None = None

    def _get_metadata(self) -> _Metadata:
        if self._metadata is None:
            self._metadata = _Metadata(*_blocks_frame_metadata(self._handle))
        return self._metadata

    @property
    def pix_fmt(self) -> str:
        """FFmpeg pixel-format name. On CPU this is the source's own format,
        e.g. ``"yuv420p"``. On CUDA it is always an NVDEC surface format:
        ``"nv12"``, ``"p010le"``, ``"p012le"``, ``"p016le"``, ``"yuv444p"`` or
        ``"yuv444p16le"``.
        """
        return self._get_metadata().pix_fmt

    @property
    def colorspace(self) -> str:
        """e.g. ``"bt709"``."""
        return self._get_metadata().colorspace

    @property
    def color_range(self) -> str:
        """``"tv"`` (limited) or ``"pc"`` (full)."""
        return self._get_metadata().color_range

    @property
    def bit_depth(self) -> int:
        """The depth of :attr:`pix_fmt` (always), which is the source's bit
        depth except where a CUDA surface format's container is wider than the
        source samples: a 10-bit 4:4:4 source is uploaded as ``yuv444p16le``,
        and a 12-bit source is tagged ``p016le`` on FFmpeg < 6 (which lacks
        ``p012le``). Both report 16 here (on CPU they'd report 10 and 12).

        Everything downstream still reads right, because those samples are
        msb-aligned and are therefore genuinely valid 16-bit ones, with zeroed
        low bits.
        """
        # TODO_API_BREAKDOWN DESIGN P2: should this report the depth of the
        # source instead of the depth of the pixel format?
        return self._get_metadata().bit_depth

    @property
    def width(self) -> int:
        """Width of the decoded samples, before rotation."""
        return self._get_metadata().width

    @property
    def height(self) -> int:
        """Height of the decoded samples, before rotation."""
        return self._get_metadata().height

    @property
    def rotation_degrees(self) -> float:
        """Degrees counter-clockwise the frame needs to be rotated by to be
        upright, or 0 if the container asks for no rotation. It is *not*
        applied to :attr:`planes`; :class:`ColorConverter` applies it (rounded
        to the nearest multiple of 90) to its output.
        """
        return self._get_metadata().rotation_degrees

    @property
    def planes(self) -> tuple[torch.Tensor, ...]:
        """The decoder's own samples as 2D tensor views, with no copy and no
        conversion: one per component, in the order :attr:`pix_fmt` describes.

        Raises for the pixel formats that can't be viewed without a copy
        (sub-byte-packed, palettised and float ones) - check :attr:`pix_fmt`
        first if you're decoding something exotic.
        """
        if self._planes is None:
            planes = _blocks_frame_planes(self._handle, self._device)
            # Absent components come back as empty tensors; real ones are 2D
            # views.
            self._planes = tuple(p for p in planes if p.numel() > 0)
        return self._planes


# TODO_API_BREAKDOWN DESIGN P1: API design - the class name, and whether
# sample_format is worth carrying now that the layout it describes has been
# normalized away.
@dataclass
class RawAudioSamples:
    """One decoded audio frame's samples, as the decoder produced them.

    Produced by :class:`AudioPacketDecoder`, consumed by
    :class:`AudioConverter`. This is the audio counterpart of
    :class:`RawFrame`, and like it, nothing has been converted: the samples are
    in the codec's own sample type.

    It is not a handle, unlike :class:`RawFrame`. Audio frames are a few kB, so
    the samples are copied out of the ``AVFrame`` rather than viewed, which
    also lets ``[num_channels, num_samples]`` be the layout for every format:
    planar ones store each channel in its own allocation and packed ones
    interleave them, so neither is that shape as it stands.

    Attributes:
        data (torch.Tensor): ``[num_channels, num_samples]``, in the dtype that
            holds the source's samples exactly: ``uint8`` for ``u8``, ``int16``
            for ``s16``, ``int32`` for ``s32``, ``float32`` for ``flt``,
            ``float64`` for ``dbl``. Note the integer ones are *not* normalized
            to ``[-1, 1]``; :class:`AudioConverter` is what does that.
        sample_rate (int): The source's sample rate, in Hz.
        sample_format (str): FFmpeg sample-format name, e.g. ``"s16p"`` or
            ``"fltp"``. This is the format the samples were decoded in, kept
            for provenance: the trailing ``p`` (planar) no longer describes
            :attr:`data`, whose layout is always the same.
        pts_seconds (float): Presentation timestamp of the first sample.
        duration_seconds (float): How long these samples last.
    """

    data: torch.Tensor
    sample_rate: int
    sample_format: str
    pts_seconds: float
    duration_seconds: float
    # See Packet._generation.
    _generation: int = 0

    @property
    def num_channels(self) -> int:
        return self.data.shape[0]

    @property
    def num_samples(self) -> int:
        return self.data.shape[1]
