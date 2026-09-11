# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchcodec._core.ops import (
    _blocks_audio_converter_convert,
    _blocks_audio_converter_drain,
    _blocks_audio_converter_reset,
    _blocks_create_audio_converter,
)
from torchcodec._frame import AudioSamples

from ._frame import RawAudioSamples


class AudioConverter:
    """Audio conversion building block: turns a decoded
    :class:`RawAudioSamples` into normalized float32
    :class:`~torchcodec._frame.AudioSamples`, optionally resampling and
    changing the number of channels.

    Not bound to anything: the input's sample type, rate and channel count all
    come from the :class:`RawAudioSamples` itself, so one converter can process
    any source. Passive and *not* thread-safe: use one ``AudioConverter`` per
    thread.

    Unlike :class:`ColorConverter` this block is a *stream processor*, not a
    function of its input, and that difference is worth understanding:

    - Resampling is an interpolation filter, so the sample it emits at a given
      instant is a weighted sum of input samples on both sides of it. The tail
      of each frame is therefore held back until the next one arrives, which
      means :meth:`convert` returns fewer samples than it was given, sometimes
      none at all, and you must call :meth:`drain` at the end or lose the end
      of the stream.
    - Frames must be fed in order, from one stream. Feeding two sources through
      one converter is not the harmless thing it is for ``ColorConverter``.
    - After a seek, call :meth:`reset`.

    None of that applies when you leave ``sample_rate`` unset: converting the
    sample type and remixing channels are both frame-local, so the converter
    holds nothing back, :meth:`drain` returns an empty result, and it is only
    the rate change that makes any of the above true. :meth:`drain` and
    :meth:`reset` are still the right thing to call, so that adding
    ``sample_rate`` later doesn't silently change what your loop produces.

    .. warning::

        These blocks do no pre-roll. A lossy codec's first frames after a seek
        are subtly wrong until it re-primes, and a resampler started mid-stream
        emits samples on a grid of its own. So samples produced from a seek do
        not line up, bit for bit, with those from decoding the whole file.
        Decoding a margin before your target and discarding it is up to you.

    Args:
        sample_rate (int, optional): The desired output sample rate. By
            default, the source's own rate is used, i.e. no resampling.
        num_channels (int, optional): The desired output number of channels. By
            default, the source's own count is used.
    """

    def __init__(self, sample_rate: int | None = None, num_channels: int | None = None):
        self._handle = _blocks_create_audio_converter(
            sample_rate=sample_rate, num_channels=num_channels
        )
        self._requested_sample_rate = sample_rate
        self._drained = False

        self._first_frame_pts_seconds: float | None = None
        self._out_sample_rate: int | None = None
        self._num_emitted_samples = 0
        # The demuxer position these samples come from. See Packet._generation:
        # a resampler carries state across calls, so a seek invalidates it just
        # as it invalidates the decoder's.
        self._generation: int | None = None

    def _wrap(self, data) -> AudioSamples:
        assert self._out_sample_rate is not None  # mypy
        assert self._first_frame_pts_seconds is not None  # mypy
        # We manually compute the pts of all frames but the first one: when
        # resampling happens, libswresample buffers samples and emits them
        # later. Not all the samples of the first frame are necessarily emitted
        # immediately, they may be emitted with the next frame. Without this,
        # we'd fail the test_audio_converter_pts_is_contiguous() test.
        pts_seconds = (
            self._first_frame_pts_seconds
            + self._num_emitted_samples / self._out_sample_rate
        )
        self._num_emitted_samples += data.shape[1]
        return AudioSamples(
            data=data,
            pts_seconds=pts_seconds,
            duration_seconds=data.shape[1] / self._out_sample_rate,
            sample_rate=self._out_sample_rate,
        )

    def convert(self, raw_samples: RawAudioSamples) -> AudioSamples:
        """Convert one :class:`RawAudioSamples`.

        The result may be empty: when resampling, the converter needs the
        following frame before it can emit the tail of this one.
        """
        if self._generation is None:
            self._generation = raw_samples._generation
        elif self._generation != raw_samples._generation:
            raise RuntimeError(
                "The demuxer seeked since this converter was last reset(), and "
                "a resampler carries state across calls, so these samples "
                "would be resampled against the wrong history. Call reset() on "
                "every converter fed by that demuxer after a seek."
            )
        if self._drained:
            raise RuntimeError(
                "This AudioConverter has been drained. Call reset() to convert "
                "more samples."
            )
        if self._first_frame_pts_seconds is None:
            self._first_frame_pts_seconds = raw_samples.pts_seconds
            self._out_sample_rate = (
                self._requested_sample_rate
                if self._requested_sample_rate is not None
                else raw_samples.sample_rate
            )
        data = _blocks_audio_converter_convert(
            self._handle, raw_samples.data, raw_samples.sample_rate
        )
        return self._wrap(data)

    def drain(self) -> AudioSamples:
        """The samples the resampler was still holding on to.

        Skipping this loses the end of the stream. It returns an empty result
        when no resampling is being done, since nothing is held back then.
        """
        if self._first_frame_pts_seconds is None:
            raise RuntimeError(
                "This AudioConverter hasn't converted any samples, so there is "
                "nothing to drain."
            )
        data = _blocks_audio_converter_drain(self._handle)
        self._drained = True
        return self._wrap(data)

    def reset(self) -> None:
        """Drop the resampler's buffered state and start over. Needed after the
        demuxer seeked, and after ``drain()``."""
        _blocks_audio_converter_reset(self._handle)
        self._drained = False
        self._first_frame_pts_seconds = None
        self._out_sample_rate = None
        self._generation = None
        self._num_emitted_samples = 0
