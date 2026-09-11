# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import io
import json
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import torch
from torch import Tensor

from torchcodec._core._decoder_utils import create_demuxer
from torchcodec._core._metadata import (
    _stream_metadata_from_dict,
    ContainerMetadata,
    DemuxerMetadata,
)
from torchcodec._core.ops import (
    _blocks_demuxer_add_stream,
    _blocks_demuxer_container_json_metadata,
    _blocks_demuxer_get_audio_video_stream_indices,
    _blocks_demuxer_next_packet,
    _blocks_demuxer_scan,
    _blocks_demuxer_seek,
    _blocks_demuxer_stream_json_metadata,
)

from .._decoder_utils import convert_device_to_str

from ._frame import Packet
from ._packet_decoder import AudioPacketDecoder, VideoPacketDecoder

# TODO_API_BREAKDOWN FEAT PERF Do we want / need to support 'batch-like' APIs
# were containers are pre-allocated for perf? Like if a user wants to decode
# specific timestamps for sampling?


@dataclass
class FrameIndex:
    """The content of a video stream, as returned by :meth:`VideoStream.scan`.

    One entry per frame, in presentation order. Everything here is derived from
    the packets of the stream rather than from the container header, so the
    values are exact.

    The scalars that the header also claims to know carry a ``_from_content``
    suffix, so that a call site says which source it trusts:
    ``num_frames_from_content`` here against ``num_frames_from_header`` on the
    stream's metadata. The per-frame arrays have no header counterpart and are
    left unsuffixed.

    Timestamps are stored in the stream's own time base and converted to
    seconds on access, which is what makes those conversions exact: a frame's
    end time is ``(pts + duration)`` converted once, not ``pts_seconds +
    duration_seconds``, which rounds twice and can land on the wrong side of a
    frame boundary.

    Attributes:
        is_key_frame (torch.Tensor): bool ``[N]``, whether each frame is a
            :term:`keyframe`.
    """

    is_key_frame: Tensor
    _pts: Tensor
    _duration: Tensor
    _time_base_num: int
    _time_base_den: int

    def __len__(self) -> int:
        """The number of frames in the stream."""
        return self.is_key_frame.shape[0]

    @property
    def num_frames_from_content(self) -> int:
        """The number of frames in the stream."""
        return len(self)

    @cached_property
    def pts_seconds(self) -> Tensor:
        """float64 ``[N]``, the presentation timestamp of each frame."""
        return self._to_seconds(self._pts)

    @cached_property
    def duration_seconds(self) -> Tensor:
        """float64 ``[N]``, how long each frame is displayed for."""
        return self._to_seconds(self._duration)

    @cached_property
    def key_frame_indices(self) -> Tensor:
        """int64 ``[K]``, the indices of the :term:`keyframe`\\ s."""
        return self.is_key_frame.nonzero().squeeze(1)

    @cached_property
    def begin_stream_seconds_from_content(self) -> float:
        """Presentation timestamp of the first frame."""
        return float(self.pts_seconds[0])

    @cached_property
    def end_stream_seconds_from_content(self) -> float:
        """Timestamp at which the last frame stops being displayed."""
        # max(), not [-1]: durations vary, so the frame that finishes last
        # isn't necessarily the one that starts last. This is how
        # end_stream_pts_from_content is accumulated in SingleStreamDecoder.
        return float(self._end_seconds.max())

    @property
    def average_fps_from_content(self) -> float:
        """Average number of frames per second over the stream."""
        return len(self) / (
            self.end_stream_seconds_from_content
            - self.begin_stream_seconds_from_content
        )

    def index_at(self, seconds: float) -> int:
        """Index of the frame that is being displayed at ``seconds``.

        A timestamp outside of the stream gives the frame closest to it, i.e.
        the first or the last one.
        """
        # First frame that hasn't finished playing by `seconds`, which is
        # get_frame_played_at()'s criterion (frame_start <= t < frame_end,
        # SingleStreamDecoder.cpp) expressed as a search rather than a scan of
        # decoded frames. Note it is *not* seconds_to_index_lower_bound(), which
        # compares against next_pts and so answers differently for a timestamp
        # falling in a gap between two frames.
        index = int(torch.searchsorted(self._end_seconds, seconds, right=True))
        return min(index, len(self) - 1)

    # TODO_API_BREAKDOWN DESIGN P1: Still kinda hate this name
    def key_frame_seconds_for(self, seconds: float) -> float:
        """Timestamp to :meth:`Demuxer.seek` to in order to reach ``seconds``:
        that of the last :term:`keyframe` which isn't after it.

        This is what makes a seek *exact*. FFmpeg resolves a seek against decode
        timestamps, so on a file whose keyframes are reordered, seeking to the
        target itself can land on a keyframe that is displayed after it, leaving
        the frames in between unreachable
        (https://trac.ffmpeg.org/ticket/11137). Seeking to a keyframe's own
        timestamp lands on that keyframe, and the target is then reached by
        decoding forward from there.

        To reach a frame *index* rather than a timestamp, pass that frame's
        timestamp: ``key_frame_seconds_for(pts_seconds[i])``.
        """
        # Same search as get_key_frame_index_for_pts_using_scanned_index()
        # (SingleStreamDecoder.cpp): upper_bound minus one, i.e. the last
        # keyframe at or before the target, and -1 when there is none.
        position = (
            int(torch.searchsorted(self._key_frame_seconds, seconds, right=True)) - 1
        )
        if position < 0:
            # No keyframe at or before the target: the one it decodes from was
            # trimmed away by an edit list, so it isn't in this index. Aim at
            # the start of the stream and let FFmpeg find it, which is what
            # exact mode does when that search returns -1.
            return float(self.pts_seconds[0])
        return float(self._key_frame_seconds[position])

    def _to_seconds(self, value: Tensor) -> Tensor:
        # pts_to_seconds() (FFMPEGCommon.cpp), and it has to stay bit-for-bit
        # identical to it: float64 is C++ `double`, and the multiplication comes
        # before the division on both sides. Timestamps from a FrameIndex are
        # compared against, and fed back into, values the C++ produced.
        return value.to(torch.float64) * self._time_base_num / self._time_base_den

    @cached_property
    def _end_seconds(self) -> Tensor:
        return self._to_seconds(self._pts + self._duration)

    @cached_property
    def _key_frame_seconds(self) -> Tensor:
        return self.pts_seconds[self.key_frame_indices]


class _Stream:
    """One of the streams a :class:`Demuxer` follows.

    This is what a packet decoder is built from. It identifies a stream *within*
    a container that is already open, so no second container is opened, and it
    keeps that container alive for as long as you hold on to it.
    """

    _media_type: str

    # TODO_API_BREAKDOWN DESIGN P1: the index field is public and it's the index
    # in the container, not the index in the demuxer's .streams field. Maybe
    # that's OK. We should find a way to make that clear.
    def __init__(self, demuxer: Demuxer, index: int):
        self._demuxer = demuxer
        self.index = index

    @cached_property
    def metadata(self):
        """What the container header says about this stream, and nothing more.

        Content-derived values never appear here: they only exist after an
        explicit :meth:`VideoStream.scan`, and they live on the
        :class:`FrameIndex` it returns. So ``metadata.num_frames_from_header``
        is the header's claim, which may be wrong, and
        ``scan().num_frames_from_content`` is the exact answer - the name says
        which one you are reading.
        """
        return _stream_metadata_from_dict(
            json.loads(
                _blocks_demuxer_stream_json_metadata(self._demuxer._handle, self.index)
            ),
            self.index,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(index={self.index})"


class VideoStream(_Stream):
    """A video stream of a :class:`Demuxer`. Build a
    :class:`VideoPacketDecoder` from it with :meth:`make_decoder` to decode its
    packets.

    Attributes:
        index (int): The stream's index within the container, absolute across
            all media types. This is what a :class:`Packet`'s ``stream_index``
            is compared against.
    """

    _media_type = "video"

    def __init__(self, demuxer: Demuxer, index: int):
        super().__init__(demuxer, index)
        self._frame_index: FrameIndex | None = None

    def scan(self) -> FrameIndex:
        """Demux this stream entirely, without decoding it, and return its
        :class:`FrameIndex`.

        This reads the whole container, and it is the only way to know the
        stream's exact frame count, timestamps and :term:`keyframe` positions:
        the container header can be wrong about all of them. The result is
        cached, and the read is shared: the first scan indexes every video
        stream the demuxer follows, so scanning a second one costs no I/O.

        **A scan has to happen before any packet is demuxed**, and calling it
        later raises. That restriction is what keeps it cheap to use: the scan
        rewinds the demuxer, and since nothing has been fed to a decoder yet,
        there is nothing to ``reset()`` afterwards.
        """
        if self._frame_index is None:
            pts, duration, is_key_frame, time_base_num, time_base_den = (
                _blocks_demuxer_scan(self._demuxer._handle, self.index)
            )
            self._frame_index = FrameIndex(
                is_key_frame=is_key_frame,
                _pts=pts,
                _duration=duration,
                _time_base_num=time_base_num,
                _time_base_den=time_base_den,
            )
        return self._frame_index

    def make_decoder(
        self, device: str | torch.device | None = None
    ) -> VideoPacketDecoder:
        """Build the :class:`VideoPacketDecoder` for this stream. This is the
        only way to build one.

        ``device`` accepts a string or a ``torch.device``. It defaults to
        ``None``, which means the current default device (see
        ``torch.set_default_device``).
        """
        return VideoPacketDecoder._from_stream(self, convert_device_to_str(device))


class AudioStream(_Stream):
    """An audio stream of a :class:`Demuxer`. Build an
    :class:`AudioPacketDecoder` from it with :meth:`make_decoder` to decode its
    packets.

    There is no ``scan()``: a :class:`FrameIndex` describes keyframes and frame
    positions, and audio has neither.

    Attributes:
        index (int): The stream's index within the container, absolute across
            all media types. This is what a :class:`Packet`'s ``stream_index``
            is compared against.
    """

    _media_type = "audio"

    def make_decoder(self) -> AudioPacketDecoder:
        """Build the :class:`AudioPacketDecoder` for this stream. This is the
        only way to build one. Audio is always decoded on the CPU, so there is
        no ``device`` parameter."""
        return AudioPacketDecoder._from_stream(self, "cpu")


class Demuxer:
    """Demux building block: opens a container and yields the compressed
    :class:`Packet`\\ s of one or more of its streams. Does no decoding.

    This block is passive (it does no threading of its own) and is *not*
    thread-safe: use one ``Demuxer`` per thread. It streams from the start of
    the container, or from wherever :meth:`seek` left it.

    Following several streams at once is the point: one demuxer per stream means
    one open container per stream, so decoding both the video and the audio of a
    file reads it twice. Here they come out of a single pass, interleaved in the
    order the container stores them, and :attr:`Packet.stream_index` says which
    stream each one belongs to::

        demuxer = Demuxer("video.mp4", streams=("video", "audio"))
        video, audio = demuxer.streams
        decoders = {s.index: s.make_decoder() for s in demuxer.streams}

        for packet in demuxer:
            for output in decoders[packet.stream_index].decode(packet):
                ...
        for decoder in decoders.values():
            for output in decoder.drain():
                ...

    Nothing is buffered per stream: a packet is handed to you once, and a caller
    that wants to consume one stream at a time has to hold on to the other
    stream's packets itself. How much to keep in flight is a pipeline decision,
    and it is deliberately left to you.

    Args:
        source (str, ``Pathlib.path``, bytes, ``torch.Tensor`` or file-like object): The source of the media:

            - If ``str``: a local path or a URL to a media file.
            - If ``Pathlib.path``: a path to a local media file.
            - If ``bytes`` object or ``torch.Tensor``: the raw encoded data.
            - If file-like object: we read data from the object on demand. The object must
              expose the methods `read(self, size: int) -> bytes` and
              `seek(self, offset: int, whence: int) -> int`. Note that every
              read has to re-acquire the GIL, so a file-like source doesn't
              parallelize with the rest of a pipeline as well as the others do.
        streams: Which streams to follow. Either one selector or a tuple of
            them, where a selector is:

            - ``"video"``: the :term:`best stream` of that type.
            - ``"audio"``: likewise.
            - an ``int``: that stream, by its index within the container,
              absolute across all media types.

            Plus one standalone form, ``"all"``, which follows every audio and
            video stream in container order and skips everything else. Defaults
            to ``"video"``.

            The order is kept: it is the order of :attr:`streams`. Selecting the
            same stream twice is an error, as is selecting a stream that is
            neither audio nor video - use ``"all"`` if you want those skipped
            rather than reported.

            Streams are chosen once, here, and a demuxer never changes which
            ones it follows.

    Attributes:
        streams (tuple): The :class:`VideoStream` and :class:`AudioStream`
            objects being followed, in the order ``streams`` named them.
    """

    def __init__(
        self,
        source: str | Path | bytes | Tensor | io.RawIOBase | io.BufferedReader,
        *,
        streams: str | int | tuple[str | int, ...] = "video",
    ):
        self._handle = create_demuxer(source=source)
        # Bumped on every seek and stamped on every packet, so that a decoder
        # can tell it is being fed packets from a position it was never reset
        # for. See _BasePacketDecoder.decode().
        self._generation = 0
        self.streams = tuple(
            self._add_stream(selector) for selector in self._parse_streams(streams)
        )

    @cached_property
    def metadata(self) -> DemuxerMetadata:
        """What the container header says about the container itself.

        Not about its streams: those are described by
        ``demuxer.streams[i].metadata``, and there is deliberately only one way
        to reach each fact. What is here is what no stream can tell you - most
        usefully ``duration_seconds_from_header``, which some containers carry
        only at this level, with their streams reporting no duration at all.

        To see every stream in a file, including the ones a demuxer cannot
        follow, use :func:`get_container_metadata`.
        """
        return DemuxerMetadata(**_container_fields(self._handle))

    def _parse_streams(self, streams) -> list[int | str]:
        """Normalise the ``streams`` argument to a list of selectors, in the
        order they were given: either a container stream index, or ``"video"`` /
        ``"audio"`` meaning the best stream of that type.

        Only the *shape* of the argument is checked here. Whether an index
        exists, whether that stream can be decoded, and whether it was named
        twice all belong to the demuxer, which checks them as each stream is
        added.
        """
        if streams == "all":
            return [
                int(index)
                for index in _blocks_demuxer_get_audio_video_stream_indices(
                    self._handle
                )
            ]

        if isinstance(streams, (str, int)) or not isinstance(streams, Iterable):
            streams = (streams,)
        streams = tuple(streams)
        if not streams:
            raise ValueError(
                "streams is empty, so this demuxer would have nothing to demux."
            )

        for selector in streams:
            if selector in ("video", "audio"):
                continue
            if selector == "all":
                raise ValueError(
                    "streams='all' can only be used on its own, not alongside "
                    f"other selectors: got {streams!r}."
                )
            if not isinstance(selector, int) or isinstance(selector, bool):
                raise ValueError(
                    f"Invalid stream selector {selector!r}. Expected 'video', "
                    "'audio', or an int stream index."
                )
        return list(streams)

    def _add_stream(self, selector: int | str) -> _Stream:
        """Follow the stream a selector names, and wrap it in the class for
        whichever media type it turns out to be."""
        index, media_type = _blocks_demuxer_add_stream(
            self._handle,
            selector if isinstance(selector, int) else None,
            selector if isinstance(selector, str) else None,
        )
        stream_class = VideoStream if media_type == "video" else AudioStream
        return stream_class(self, index)

    def next_packet(self) -> Packet | None:
        """Return the next :class:`Packet`, or ``None`` at end of stream.

        Packets come out interleaved across the streams being followed, in the
        order the container stores them; :attr:`Packet.stream_index` says which
        stream each belongs to.
        """
        handle, is_eof, stream_index = _blocks_demuxer_next_packet(self._handle)
        if is_eof:
            return None
        return Packet(handle, stream_index, generation=self._generation)

    def seek(self, seconds: float, *, stream: _Stream | None = None) -> None:
        """Move the demuxer to ``seconds``.

        There is one container and one read position, so this moves *every*
        followed stream, and every decoder fed by this demuxer must be
        ``reset()`` afterwards - as must any :class:`AudioConverter`, whose
        resampler state a seek invalidates too.

        ``stream`` says which stream the target is resolved against; it
        defaults to the first one in ``streams``. This matters because FFmpeg
        resolves a seek in one stream's time base and lands on *that* stream's
        :term:`keyframe`\\ s, with the other streams simply resuming from the
        resulting position - so the seek is exact only for the stream it was
        resolved against, and another video stream may land mid-GOP and decode
        garbage until its next keyframe. Name the stream you need to be exact.

        The default is your own ordering rather than, say, the first video
        stream, so that it doesn't depend on what the container happens to
        hold.

        Where you land, and what comes out first, depends on the medium. For
        video, a decoder can only start on a :term:`keyframe`, so this lands on
        the keyframe at or before the target and the first frames that come out
        usually precede it. Audio has no keyframe to land on: the codec's
        overlap-add state is lost, so a lossy stream's first few frames after a
        seek are subtly wrong - plausible, but not the samples that whole-file
        decoding would give - until it re-primes. Decoding a margin before the
        target and discarding it is the caller's responsibility.
        """
        _blocks_demuxer_seek(
            self._handle,
            float(seconds),
            None if stream is None else stream.index,
        )
        self._generation += 1

    def __iter__(self):
        while True:
            packet = self.next_packet()
            if packet is None:
                return
            yield packet


def _container_fields(handle: Tensor) -> dict:
    container_dict = json.loads(_blocks_demuxer_container_json_metadata(handle))
    return dict(
        duration_seconds_from_header=container_dict.get("durationSecondsFromHeader"),
        bit_rate_from_header=container_dict.get("bitRate"),
        best_video_stream_index=container_dict.get("bestVideoStreamIndex"),
        best_audio_stream_index=container_dict.get("bestAudioStreamIndex"),
    )


def get_container_metadata(
    source: str | Path | bytes | Tensor | io.RawIOBase | io.BufferedReader,
) -> ContainerMetadata:
    """Describe a container and every stream in it, without decoding anything.

    This opens the source and reads its header - no packet is demuxed - so it is
    what you reach for before you know what a file contains and therefore which
    streams to ask a :class:`Demuxer` for. It reports streams that a demuxer
    cannot follow, such as subtitles, as plain
    :class:`~torchcodec._core._metadata.StreamMetadata`.

    Note this opens the source, and constructing a :class:`Demuxer` afterwards
    opens it again. That second open is a header probe, not a read of the file,
    and it is the price of a demuxer whose streams are fixed at construction and
    never change afterwards.
    """
    handle = create_demuxer(source=source)
    container_dict = json.loads(_blocks_demuxer_container_json_metadata(handle))
    streams = [
        _stream_metadata_from_dict(
            json.loads(_blocks_demuxer_stream_json_metadata(handle, stream_index)),
            stream_index,
        )
        for stream_index in range(int(container_dict["numStreams"]))
    ]
    return ContainerMetadata(**_container_fields(handle), streams=streams)
