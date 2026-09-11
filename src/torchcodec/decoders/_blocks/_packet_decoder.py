# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Generic, TYPE_CHECKING, TypeVar

import torch

from torchcodec._core.ops import (
    _blocks_audio_packet_decoder_receive_frame,
    _blocks_create_packet_decoder,
    _blocks_packet_decoder_receive_frame,
    _blocks_packet_decoder_reset,
    _blocks_packet_decoder_send_eof,
    _blocks_packet_decoder_send_packet,
)

from ._frame import Packet, RawAudioSamples, RawFrame

if TYPE_CHECKING:
    # Only for the annotation: _demuxer imports this module to build decoders,
    # so importing it back at runtime would be circular.
    from ._demuxer import _Stream


# TODO_API_BREAKDOWN DOC P1 revisit every single docstring / comments at some point.

_Decoded = TypeVar("_Decoded", RawFrame, RawAudioSamples)
_Self = TypeVar("_Self", bound="_BasePacketDecoder")


class _BasePacketDecoder(Generic[_Decoded]):
    """Shared machinery for :class:`VideoPacketDecoder` and
    :class:`AudioPacketDecoder`.

    Decoding is the same ``avcodec_send_packet`` / ``avcodec_receive_frame``
    pair for both, so the C++ side is a single class; what differs is only what
    a decoded frame is turned into, which is :meth:`_receive_ready_frames`.
    """

    _handle: torch.Tensor
    _drained: bool
    _generation: int | None

    # *args so that a call with arguments gets the message below rather than a
    # TypeError about the argument count.
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{type(self).__name__} cannot be instantiated directly. Build one "
            "from the stream whose packets it decodes, with "
            "stream.make_decoder()."
        )

    @classmethod
    def _from_stream(cls: type[_Self], stream: _Stream, device_str: str) -> _Self:
        decoder = cls.__new__(cls)
        decoder._handle = _blocks_create_packet_decoder(
            stream._demuxer._handle,
            stream_index=stream.index,
            num_threads=1,
            device=device_str,
        )
        decoder._drained = False
        # The demuxer position these packets come from. None until the first
        # packet, and again after every reset(), so it is adopted rather than
        # tracked: the decoder never needs a reference back to the demuxer.
        decoder._generation = None
        return decoder

    def _receive_ready_frames(self) -> list[_Decoded]:
        raise NotImplementedError

    def decode(self, packet: Packet) -> list[_Decoded]:
        """Send one packet and return whatever is now ready (possibly empty,
        e.g. while the codec buffers B-frames)."""
        if self._drained:
            raise RuntimeError(
                "This decoder has been drained, and a codec that has been told "
                "the stream ended ignores any further packet. Create a new "
                "decoder to decode another stream."
            )
        if self._generation is None:
            self._generation = packet._generation
        elif self._generation != packet._generation:
            raise RuntimeError(
                "The demuxer seeked since this decoder was last reset(), so "
                "this packet is from a position the codec knows nothing about "
                "- decoding it would produce plausible-looking garbage. Call "
                "reset() on every decoder fed by that demuxer after a seek."
            )
        status = _blocks_packet_decoder_send_packet(self._handle, packet._handle)
        if status < 0:
            raise RuntimeError(f"Failed to send packet to decoder (status {status})")
        return self._receive_ready_frames()

    def drain(self) -> list[_Decoded]:
        """Tell the codec the stream ended, and return the frames it was still
        holding on to."""
        _blocks_packet_decoder_send_eof(self._handle)
        frames = self._receive_ready_frames()
        self._drained = True
        return frames

    def reset(self) -> None:
        """Drop the codec's buffered state and start over. Needed after the
        demuxer seeked, and after ``drain()``."""
        _blocks_packet_decoder_reset(self._handle)
        self._drained = False
        self._generation = None


class VideoPacketDecoder(_BasePacketDecoder[RawFrame]):
    """Decode building block: turns compressed :class:`Packet`\\ s into decoded
    (YUV) :class:`RawFrame`\\ s.

    Not built directly: it comes from
    :meth:`~torchcodec.decoders._blocks.VideoStream.make_decoder`, which is what
    binds it to the stream whose codec parameters it decodes with. It is
    stateful (it holds the codec's reference-frame buffer), passive, and *not*
    thread-safe: use one ``VideoPacketDecoder`` per thread. FFmpeg's internal
    codec thread count is kept at 1 for now (not exposed); parallelism comes
    from composing blocks on your own threads.
    """

    def _receive_ready_frames(self) -> list[RawFrame]:
        frames = []
        while True:
            handle, status, pts_seconds, duration_seconds, device, storage = (
                _blocks_packet_decoder_receive_frame(self._handle)
            )
            if status != 0:  # EAGAIN (need more packets) or EOF: nothing ready
                break
            frames.append(
                RawFrame(
                    handle,
                    pts_seconds,
                    duration_seconds,
                    device=device,
                    storage=storage if storage.numel() > 0 else None,
                )
            )
        return frames


class AudioPacketDecoder(_BasePacketDecoder[RawAudioSamples]):
    """Decode building block: turns compressed :class:`Packet`\\ s into
    :class:`RawAudioSamples`.

    Not built directly: it comes from
    :meth:`~torchcodec.decoders._blocks.AudioStream.make_decoder`. It is
    stateful: a lossy codec's overlap-add state means the frames decoded right
    after a seek are subtly wrong until it re-primes, so ``reset()`` is
    necessary but not by itself sufficient - see
    :meth:`~torchcodec.decoders._blocks.Demuxer.seek`. Passive and *not*
    thread-safe: use one ``AudioPacketDecoder`` per thread.

    Audio is always decoded on the CPU, and that doesn't change with
    ``torch.set_default_device``.
    """

    def _receive_ready_frames(self) -> list[RawAudioSamples]:
        samples = []
        while True:
            data, status, pts_seconds, duration_seconds, sample_rate, sample_format = (
                _blocks_audio_packet_decoder_receive_frame(self._handle)
            )
            if status != 0:  # EAGAIN (need more packets) or EOF: nothing ready
                break
            samples.append(
                RawAudioSamples(
                    data=data,
                    sample_rate=sample_rate,
                    sample_format=sample_format,
                    pts_seconds=pts_seconds,
                    duration_seconds=duration_seconds,
                    # Carried onward so AudioConverter can make the same check:
                    # a seek invalidates the resampler's state too.
                    _generation=self._generation or 0,
                )
            )
        return samples
