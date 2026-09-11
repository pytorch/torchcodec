# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Private, experimental building-block decode API.

Exposes the three decode stages -- :class:`Demuxer`,
:class:`VideoPacketDecoder`, :class:`ColorConverter` -- as passive, composable,
GIL-releasing units, so a caller can build its own (threaded) decode pipeline
and tune how the stages overlap. The blocks do no threading themselves.

A :class:`Demuxer` follows one or more streams of a container, so the video and
the audio of a file come out of a single pass over it. Metadata is exposed in
tiers that are never merged: :attr:`Demuxer.metadata` for the container,
``stream.metadata`` for what the header says about a stream, and
:meth:`VideoStream.scan` for what its packets actually say. That is the whole
point of this layer -- it shows you the machinery instead of deciding for you.

This is experimental and private; the API may change. See
API_breakdown_claude_plan.md and multi_stream_demuxer_design.md for the design
and rationale.
"""

from ._audio_converter import AudioConverter
from ._color_converter import ColorConverter
from ._demuxer import (
    # TODO_API_BREAKDOWN DESIGN P1: AudioStream and VideoStream may conflict
    # with the encoder-side classes of the same name. Maybe that's OK?
    AudioStream,
    Demuxer,
    FrameIndex,
    get_container_metadata,
    VideoStream,
)
from ._frame import Packet, RawAudioSamples, RawFrame
from ._packet_decoder import AudioPacketDecoder, VideoPacketDecoder

__all__ = [
    "Demuxer",
    "get_container_metadata",
    "VideoStream",
    "AudioStream",
    "VideoPacketDecoder",
    "AudioPacketDecoder",
    "ColorConverter",
    "AudioConverter",
    "Packet",
    "RawFrame",
    "RawAudioSamples",
    "FrameIndex",
]
