# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy
import pytest
import torch
from torchcodec import FrameBatch

from torchcodec.decoders import _core, VideoDecoder

from ..utils import assert_tensor_close, assert_tensor_equal, H265_VIDEO, NASA_VIDEO


class TestVideoDecoder:
    @pytest.mark.parametrize("source_kind", ("str", "path", "tensor", "bytes"))
    def test_create(self, source_kind):
        if source_kind == "str":
            source = str(NASA_VIDEO.path)
        elif source_kind == "path":
            source = NASA_VIDEO.path
        elif source_kind == "tensor":
            source = NASA_VIDEO.to_tensor()
        elif source_kind == "bytes":
            path = str(NASA_VIDEO.path)
            with open(path, "rb") as f:
                source = f.read()
        else:
            raise ValueError("Oops, double check the parametrization of this test!")

        decoder = VideoDecoder(source)
        assert isinstance(decoder.metadata, _core.VideoStreamMetadata)
        assert (
            len(decoder)
            == decoder._num_frames
            == decoder.metadata.num_frames_from_content
            == 390
        )
        assert decoder.stream_index == decoder.metadata.stream_index == 3
        assert decoder.metadata.duration_seconds == pytest.approx(13.013)
        assert decoder.metadata.average_fps == pytest.approx(29.970029)
        assert decoder.metadata.num_frames == 390

    def test_create_fails(self):
        with pytest.raises(TypeError, match="Unknown source type"):
            decoder = VideoDecoder(123)  # noqa

        # stream index that does not exist
        with pytest.raises(ValueError, match="No valid stream found"):
            decoder = VideoDecoder(NASA_VIDEO.path, stream_index=40)  # noqa

        # stream index that does exist, but it's not video
        with pytest.raises(ValueError, match="No valid stream found"):
            decoder = VideoDecoder(NASA_VIDEO.path, stream_index=1)  # noqa

    @pytest.mark.parametrize("num_ffmpeg_threads", (1, 4))
    def test_getitem_int(self, num_ffmpeg_threads):
        decoder = VideoDecoder(NASA_VIDEO.path, num_ffmpeg_threads=num_ffmpeg_threads)

        ref_frame0 = NASA_VIDEO.get_frame_data_by_index(0)
        ref_frame1 = NASA_VIDEO.get_frame_data_by_index(1)
        ref_frame180 = NASA_VIDEO.get_frame_by_name("time6.000000")
        ref_frame_last = NASA_VIDEO.get_frame_by_name("time12.979633")

        assert_tensor_equal(ref_frame0, decoder[0])
        assert_tensor_equal(ref_frame1, decoder[1])
        assert_tensor_equal(ref_frame180, decoder[180])
        assert_tensor_equal(ref_frame_last, decoder[-1])

    def test_getitem_numpy_int(self):
        decoder = VideoDecoder(NASA_VIDEO.path)

        ref_frame0 = NASA_VIDEO.get_frame_data_by_index(0)
        ref_frame1 = NASA_VIDEO.get_frame_data_by_index(1)
        ref_frame180 = NASA_VIDEO.get_frame_by_name("time6.000000")
        ref_frame_last = NASA_VIDEO.get_frame_by_name("time12.979633")

        # test against numpy.int64
        assert_tensor_equal(ref_frame0, decoder[numpy.int64(0)])
        assert_tensor_equal(ref_frame1, decoder[numpy.int64(1)])
        assert_tensor_equal(ref_frame180, decoder[numpy.int64(180)])
        assert_tensor_equal(ref_frame_last, decoder[numpy.int64(-1)])

        # test against numpy.int32
        assert_tensor_equal(ref_frame0, decoder[numpy.int32(0)])
        assert_tensor_equal(ref_frame1, decoder[numpy.int32(1)])
        assert_tensor_equal(ref_frame180, decoder[numpy.int32(180)])
        assert_tensor_equal(ref_frame_last, decoder[numpy.int32(-1)])

        # test against numpy.uint64
        assert_tensor_equal(ref_frame0, decoder[numpy.uint64(0)])
        assert_tensor_equal(ref_frame1, decoder[numpy.uint64(1)])
        assert_tensor_equal(ref_frame180, decoder[numpy.uint64(180)])

        # test against numpy.uint32
        assert_tensor_equal(ref_frame0, decoder[numpy.uint32(0)])
        assert_tensor_equal(ref_frame1, decoder[numpy.uint32(1)])
        assert_tensor_equal(ref_frame180, decoder[numpy.uint32(180)])

    def test_getitem_slice(self):
        decoder = VideoDecoder(NASA_VIDEO.path)

        # ensure that the degenerate case of a range of size 1 works

        ref0 = NASA_VIDEO.get_frame_data_by_range(0, 1)
        slice0 = decoder[0:1]
        assert slice0.shape == torch.Size(
            [
                1,
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        assert_tensor_equal(ref0, slice0)

        ref4 = NASA_VIDEO.get_frame_data_by_range(4, 5)
        slice4 = decoder[4:5]
        assert slice4.shape == torch.Size(
            [
                1,
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        assert_tensor_equal(ref4, slice4)

        ref8 = NASA_VIDEO.get_frame_data_by_range(8, 9)
        slice8 = decoder[8:9]
        assert slice8.shape == torch.Size(
            [
                1,
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        assert_tensor_equal(ref8, slice8)

        ref180 = NASA_VIDEO.get_frame_by_name("time6.000000")
        slice180 = decoder[180:181]
        assert slice180.shape == torch.Size(
            [
                1,
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        assert_tensor_equal(ref180, slice180[0])

        # contiguous ranges
        ref0_9 = NASA_VIDEO.get_frame_data_by_range(0, 9)
        slice0_9 = decoder[0:9]
        assert slice0_9.shape == torch.Size(
            [
                9,
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        assert_tensor_equal(ref0_9, slice0_9)

        ref4_8 = NASA_VIDEO.get_frame_data_by_range(4, 8)
        slice4_8 = decoder[4:8]
        assert slice4_8.shape == torch.Size(
            [
                4,
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        assert_tensor_equal(ref4_8, slice4_8)

        # ranges with a stride
        ref15_35 = NASA_VIDEO.get_frame_data_by_range(15, 36, 5)
        slice15_35 = decoder[15:36:5]
        assert slice15_35.shape == torch.Size(
            [
                5,
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        assert_tensor_equal(ref15_35, slice15_35)

        ref0_9_2 = NASA_VIDEO.get_frame_data_by_range(0, 9, 2)
        slice0_9_2 = decoder[0:9:2]
        assert slice0_9_2.shape == torch.Size(
            [
                5,
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        assert_tensor_equal(ref0_9_2, slice0_9_2)

        # negative numbers in the slice
        ref386_389 = NASA_VIDEO.get_frame_data_by_range(386, 390)
        slice386_389 = decoder[-4:]
        assert slice386_389.shape == torch.Size(
            [
                4,
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        assert_tensor_equal(ref386_389, slice386_389)

        # an empty range is valid!
        empty_frame = decoder[5:5]
        assert_tensor_equal(empty_frame, NASA_VIDEO.empty_chw_tensor)

        # slices that are out-of-range are also valid - they return an empty tensor
        also_empty = decoder[10000:]
        assert_tensor_equal(also_empty, NASA_VIDEO.empty_chw_tensor)

        # should be just a copy
        all_frames = decoder[:]
        assert all_frames.shape == torch.Size(
            [
                len(decoder),
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        for sliced, ref in zip(all_frames, decoder):
            assert_tensor_equal(sliced, ref)

    def test_getitem_fails(self):
        decoder = VideoDecoder(NASA_VIDEO.path)

        with pytest.raises(IndexError, match="out of bounds"):
            frame = decoder[1000]  # noqa

        with pytest.raises(IndexError, match="out of bounds"):
            frame = decoder[-1000]  # noqa

        with pytest.raises(TypeError, match="Unsupported key type"):
            frame = decoder["0"]  # noqa

        with pytest.raises(TypeError, match="Unsupported key type"):
            frame = decoder[2.3]  # noqa

    def test_iteration(self):
        decoder = VideoDecoder(NASA_VIDEO.path)

        ref_frame0 = NASA_VIDEO.get_frame_data_by_index(0)
        ref_frame1 = NASA_VIDEO.get_frame_data_by_index(1)
        ref_frame9 = NASA_VIDEO.get_frame_data_by_index(9)
        ref_frame35 = NASA_VIDEO.get_frame_data_by_index(35)
        ref_frame180 = NASA_VIDEO.get_frame_by_name("time6.000000")
        ref_frame_last = NASA_VIDEO.get_frame_by_name("time12.979633")

        # Access an arbitrary frame to make sure that the later iteration
        # still works as expected. The underlying C++ decoder object is
        # actually stateful, and accessing a frame will move its internal
        # cursor.
        assert_tensor_equal(ref_frame35, decoder[35])

        for i, frame in enumerate(decoder):
            if i == 0:
                assert_tensor_equal(ref_frame0, frame)
            elif i == 1:
                assert_tensor_equal(ref_frame1, frame)
            elif i == 9:
                assert_tensor_equal(ref_frame9, frame)
            elif i == 35:
                assert_tensor_equal(ref_frame35, frame)
            elif i == 180:
                assert_tensor_equal(ref_frame180, frame)
            elif i == 389:
                assert_tensor_equal(ref_frame_last, frame)

    def test_iteration_slow(self):
        decoder = VideoDecoder(NASA_VIDEO.path)
        ref_frame_last = NASA_VIDEO.get_frame_data_by_index(389)

        # Force the decoder to seek around a lot while iterating; this will
        # slow down decoding, but we should still only iterate the exact number
        # of total frames.
        iterations = 0
        for frame in decoder:
            assert_tensor_equal(ref_frame_last, decoder[-1])
            iterations += 1

        assert iterations == len(decoder) == 390

    def test_get_frame_at(self):
        decoder = VideoDecoder(NASA_VIDEO.path)

        ref_frame9 = NASA_VIDEO.get_frame_data_by_index(9)
        frame9 = decoder.get_frame_at(9)

        assert_tensor_equal(ref_frame9, frame9.data)
        assert isinstance(frame9.pts_seconds, float)
        expected_frame_info = NASA_VIDEO.get_frame_info(9)
        assert frame9.pts_seconds == pytest.approx(expected_frame_info.pts_seconds)
        assert isinstance(frame9.duration_seconds, float)
        assert frame9.duration_seconds == pytest.approx(
            expected_frame_info.duration_seconds, rel=1e-3
        )

        # test numpy.int64
        frame9 = decoder.get_frame_at(numpy.int64(9))
        assert_tensor_equal(ref_frame9, frame9.data)

        # test numpy.int32
        frame9 = decoder.get_frame_at(numpy.int32(9))
        assert_tensor_equal(ref_frame9, frame9.data)

        # test numpy.uint64
        frame9 = decoder.get_frame_at(numpy.uint64(9))
        assert_tensor_equal(ref_frame9, frame9.data)

        # test numpy.uint32
        frame9 = decoder.get_frame_at(numpy.uint32(9))
        assert_tensor_equal(ref_frame9, frame9.data)

    def test_get_frame_at_tuple_unpacking(self):
        decoder = VideoDecoder(NASA_VIDEO.path)

        frame = decoder.get_frame_at(50)
        data, pts, duration = decoder.get_frame_at(50)

        assert_tensor_equal(frame.data, data)
        assert frame.pts_seconds == pts
        assert frame.duration_seconds == duration

    def test_get_frame_at_fails(self):
        decoder = VideoDecoder(NASA_VIDEO.path)

        with pytest.raises(IndexError, match="out of bounds"):
            frame = decoder.get_frame_at(-1)  # noqa

        with pytest.raises(IndexError, match="out of bounds"):
            frame = decoder.get_frame_at(10000)  # noqa

    def test_get_frames_at(self):
        decoder = VideoDecoder(NASA_VIDEO.path)

        frames = decoder.get_frames_at([35, 25])

        assert isinstance(frames, FrameBatch)

        assert_tensor_equal(frames[0].data, NASA_VIDEO.get_frame_data_by_index(35))
        assert_tensor_equal(frames[1].data, NASA_VIDEO.get_frame_data_by_index(25))

        expected_pts_seconds = torch.tensor(
            [
                NASA_VIDEO.get_frame_info(35).pts_seconds,
                NASA_VIDEO.get_frame_info(25).pts_seconds,
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            frames.pts_seconds, expected_pts_seconds, atol=1e-4, rtol=0
        )

        expected_duration_seconds = torch.tensor(
            [
                NASA_VIDEO.get_frame_info(35).duration_seconds,
                NASA_VIDEO.get_frame_info(25).duration_seconds,
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            frames.duration_seconds, expected_duration_seconds, atol=1e-4, rtol=0
        )

    def test_get_frames_at_fails(self):
        decoder = VideoDecoder(NASA_VIDEO.path)

        with pytest.raises(RuntimeError, match="Invalid frame index=-1"):
            decoder.get_frames_at([-1])

        with pytest.raises(RuntimeError, match="Invalid frame index=390"):
            decoder.get_frames_at([390])

        with pytest.raises(RuntimeError, match="Expected a value of type"):
            decoder.get_frames_at([0.3])

    def test_get_frame_played_at(self):
        decoder = VideoDecoder(NASA_VIDEO.path)

        ref_frame6 = NASA_VIDEO.get_frame_by_name("time6.000000")
        assert_tensor_equal(ref_frame6, decoder.get_frame_played_at(6.006).data)
        assert_tensor_equal(ref_frame6, decoder.get_frame_played_at(6.02).data)
        assert_tensor_equal(ref_frame6, decoder.get_frame_played_at(6.039366).data)
        assert isinstance(decoder.get_frame_played_at(6.02).pts_seconds, float)
        assert isinstance(decoder.get_frame_played_at(6.02).duration_seconds, float)

    def test_get_frame_played_at_h265(self):
        # Non-regression test for https://github.com/pytorch/torchcodec/issues/179
        decoder = VideoDecoder(H265_VIDEO.path)
        ref_frame6 = H265_VIDEO.get_frame_data_by_index(5)
        assert_tensor_equal(ref_frame6, decoder.get_frame_played_at(0.5).data)

    def test_get_frame_played_at_fails(self):
        decoder = VideoDecoder(NASA_VIDEO.path)

        with pytest.raises(IndexError, match="Invalid pts in seconds"):
            frame = decoder.get_frame_played_at(-1.0)  # noqa

        with pytest.raises(IndexError, match="Invalid pts in seconds"):
            frame = decoder.get_frame_played_at(100.0)  # noqa

    def test_get_frames_played_at(self):

        decoder = VideoDecoder(NASA_VIDEO.path)

        # Note: We know the frame at ~0.84s has index 25, the one at 1.16s has
        # index 35. We use those indices as reference to test against.
        seconds = [0.84, 1.17, 0.85]
        reference_indices = [25, 35, 25]
        frames = decoder.get_frames_played_at(seconds)

        assert isinstance(frames, FrameBatch)

        for i in range(len(reference_indices)):
            assert_tensor_equal(
                frames.data[i], NASA_VIDEO.get_frame_data_by_index(reference_indices[i])
            )

        expected_pts_seconds = torch.tensor(
            [NASA_VIDEO.get_frame_info(i).pts_seconds for i in reference_indices],
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            frames.pts_seconds, expected_pts_seconds, atol=1e-4, rtol=0
        )

        expected_duration_seconds = torch.tensor(
            [NASA_VIDEO.get_frame_info(i).duration_seconds for i in reference_indices],
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            frames.duration_seconds, expected_duration_seconds, atol=1e-4, rtol=0
        )

    def test_get_frames_played_at_fails(self):
        decoder = VideoDecoder(NASA_VIDEO.path)

        with pytest.raises(RuntimeError, match="must be in range"):
            decoder.get_frames_played_at([-1])

        with pytest.raises(RuntimeError, match="must be in range"):
            decoder.get_frames_played_at([14])

        with pytest.raises(RuntimeError, match="Expected a value of type"):
            decoder.get_frames_played_at(["bad"])

    @pytest.mark.parametrize("stream_index", [0, 3, None])
    def test_get_frames_in_range(self, stream_index):
        decoder = VideoDecoder(NASA_VIDEO.path, stream_index=stream_index)

        # test degenerate case where we only actually get 1 frame
        ref_frames9 = NASA_VIDEO.get_frame_data_by_range(
            start=9, stop=10, stream_index=stream_index
        )
        frames9 = decoder.get_frames_in_range(start=9, stop=10)

        assert_tensor_equal(ref_frames9, frames9.data)
        assert frames9.pts_seconds[0].item() == pytest.approx(
            NASA_VIDEO.get_frame_info(9, stream_index=stream_index).pts_seconds,
            rel=1e-3,
        )
        assert frames9.duration_seconds[0].item() == pytest.approx(
            NASA_VIDEO.get_frame_info(9, stream_index=stream_index).duration_seconds,
            rel=1e-3,
        )

        # test simple ranges
        ref_frames0_9 = NASA_VIDEO.get_frame_data_by_range(
            start=0, stop=10, stream_index=stream_index
        )
        frames0_9 = decoder.get_frames_in_range(start=0, stop=10)
        assert frames0_9.data.shape == torch.Size(
            [
                10,
                NASA_VIDEO.get_num_color_channels(stream_index=stream_index),
                NASA_VIDEO.get_height(stream_index=stream_index),
                NASA_VIDEO.get_width(stream_index=stream_index),
            ]
        )
        assert_tensor_equal(ref_frames0_9, frames0_9.data)
        assert_tensor_close(
            NASA_VIDEO.get_pts_seconds_by_range(0, 10, stream_index=stream_index),
            frames0_9.pts_seconds,
        )
        assert_tensor_close(
            NASA_VIDEO.get_duration_seconds_by_range(0, 10, stream_index=stream_index),
            frames0_9.duration_seconds,
        )

        # test steps
        ref_frames0_8_2 = NASA_VIDEO.get_frame_data_by_range(
            start=0, stop=10, step=2, stream_index=stream_index
        )
        frames0_8_2 = decoder.get_frames_in_range(start=0, stop=10, step=2)
        assert frames0_8_2.data.shape == torch.Size(
            [
                5,
                NASA_VIDEO.get_num_color_channels(stream_index=stream_index),
                NASA_VIDEO.get_height(stream_index=stream_index),
                NASA_VIDEO.get_width(stream_index=stream_index),
            ]
        )
        assert_tensor_equal(ref_frames0_8_2, frames0_8_2.data)
        assert_tensor_close(
            NASA_VIDEO.get_pts_seconds_by_range(0, 10, 2, stream_index=stream_index),
            frames0_8_2.pts_seconds,
        )
        assert_tensor_close(
            NASA_VIDEO.get_duration_seconds_by_range(
                0, 10, 2, stream_index=stream_index
            ),
            frames0_8_2.duration_seconds,
        )

        # test numpy.int64 for indices
        frames0_8_2 = decoder.get_frames_in_range(
            start=numpy.int64(0), stop=numpy.int64(10), step=numpy.int64(2)
        )
        assert_tensor_equal(ref_frames0_8_2, frames0_8_2.data)

        # an empty range is valid!
        empty_frames = decoder.get_frames_in_range(5, 5)
        assert_tensor_equal(
            empty_frames.data,
            NASA_VIDEO.get_empty_chw_tensor(stream_index=stream_index),
        )
        assert_tensor_equal(empty_frames.pts_seconds, NASA_VIDEO.empty_pts_seconds)
        assert_tensor_equal(
            empty_frames.duration_seconds, NASA_VIDEO.empty_duration_seconds
        )

    @pytest.mark.parametrize("dimension_order", ["NCHW", "NHWC"])
    @pytest.mark.parametrize(
        "frame_getter",
        (
            lambda decoder: decoder[0],
            lambda decoder: decoder.get_frame_at(0).data,
            lambda decoder: decoder.get_frames_at([0, 1]).data,
            lambda decoder: decoder.get_frames_in_range(0, 4).data,
            lambda decoder: decoder.get_frame_played_at(0).data,
            lambda decoder: decoder.get_frames_played_at([0, 1]).data,
            lambda decoder: decoder.get_frames_played_in_range(0, 1).data,
        ),
    )
    def test_dimension_order(self, dimension_order, frame_getter):
        decoder = VideoDecoder(NASA_VIDEO.path, dimension_order=dimension_order)
        frame = frame_getter(decoder)

        C, H, W = NASA_VIDEO.num_color_channels, NASA_VIDEO.height, NASA_VIDEO.width
        assert frame.shape[-3:] == (C, H, W) if dimension_order == "NCHW" else (H, W, C)

        if frame.ndim == 3:
            frame = frame[None]  # Add fake batch dim to check contiguity
        expected_memory_format = (
            torch.channels_last
            if dimension_order == "NCHW"
            else torch.contiguous_format
        )
        assert frame.is_contiguous(memory_format=expected_memory_format)

    def test_dimension_order_fails(self):
        with pytest.raises(ValueError, match="Invalid dimension order"):
            VideoDecoder(NASA_VIDEO.path, dimension_order="NCDHW")

    @pytest.mark.parametrize("stream_index", [0, 3, None])
    def test_get_frames_by_pts_in_range(self, stream_index):
        decoder = VideoDecoder(NASA_VIDEO.path, stream_index=stream_index)

        # Note that we are comparing the results of VideoDecoder's method:
        #   get_frames_played_in_range()
        # With the testing framework's method:
        #   get_frame_data_by_range()
        # That is, we are testing the correctness of a pts-based range against an index-
        # based range. We are doing this because we are primarily testing the range logic
        # in the pts-based method. We ensure it is correct by making sure it returns the
        # frames at the indices we know the pts-values map to.

        # This value is rougly half of the duration of a frame in seconds in the test
        # stream. We use it to obtain values that fall rougly halfway between the pts
        # values for two back-to-back frames.
        HALF_DURATION = (1 / decoder.metadata.average_fps) / 2

        # The intention here is that the stop and start are exactly specified. In practice, the pts
        # value for frame 5 that we have access to on the Python side is slightly less than the pts
        # value on the C++ side. This test still produces the correct result because a slightly
        # less value still falls into the correct window.
        frames0_4 = decoder.get_frames_played_in_range(
            decoder.get_frame_at(0).pts_seconds, decoder.get_frame_at(5).pts_seconds
        )
        assert_tensor_equal(
            frames0_4.data,
            NASA_VIDEO.get_frame_data_by_range(0, 5, stream_index=stream_index),
        )

        # Range where the stop seconds is about halfway between pts values for two frames.
        also_frames0_4 = decoder.get_frames_played_in_range(
            decoder.get_frame_at(0).pts_seconds,
            decoder.get_frame_at(4).pts_seconds + HALF_DURATION,
        )
        assert_tensor_equal(also_frames0_4.data, frames0_4.data)

        # Again, the intention here is to provide the exact values we care about. In practice, our
        # pts values are slightly smaller, so we nudge the start upwards.
        frames5_9 = decoder.get_frames_played_in_range(
            decoder.get_frame_at(5).pts_seconds,
            decoder.get_frame_at(10).pts_seconds,
        )
        assert_tensor_equal(
            frames5_9.data,
            NASA_VIDEO.get_frame_data_by_range(5, 10, stream_index=stream_index),
        )

        # Range where we provide start_seconds and stop_seconds that are different, but
        # also should land in the same window of time between two frame's pts values. As
        # a result, we should only get back one frame.
        frame6 = decoder.get_frames_played_in_range(
            decoder.get_frame_at(6).pts_seconds,
            decoder.get_frame_at(6).pts_seconds + HALF_DURATION,
        )
        assert_tensor_equal(
            frame6.data,
            NASA_VIDEO.get_frame_data_by_range(6, 7, stream_index=stream_index),
        )

        # Very small range that falls in the same frame.
        frame35 = decoder.get_frames_played_in_range(
            decoder.get_frame_at(35).pts_seconds,
            decoder.get_frame_at(35).pts_seconds + 1e-10,
        )
        assert_tensor_equal(
            frame35.data,
            NASA_VIDEO.get_frame_data_by_range(35, 36, stream_index=stream_index),
        )

        # Single frame where the start seconds is before frame i's pts, and the stop is
        # after frame i's pts, but before frame i+1's pts. In that scenario, we expect
        # to see frames i-1 and i.
        frames7_8 = decoder.get_frames_played_in_range(
            NASA_VIDEO.get_frame_info(8, stream_index=stream_index).pts_seconds
            - HALF_DURATION,
            NASA_VIDEO.get_frame_info(8, stream_index=stream_index).pts_seconds
            + HALF_DURATION,
        )
        assert_tensor_equal(
            frames7_8.data,
            NASA_VIDEO.get_frame_data_by_range(7, 9, stream_index=stream_index),
        )

        # Start and stop seconds are the same value, which should not return a frame.
        empty_frame = decoder.get_frames_played_in_range(
            NASA_VIDEO.get_frame_info(4, stream_index=stream_index).pts_seconds,
            NASA_VIDEO.get_frame_info(4, stream_index=stream_index).pts_seconds,
        )
        assert_tensor_equal(
            empty_frame.data, NASA_VIDEO.get_empty_chw_tensor(stream_index=stream_index)
        )
        assert_tensor_equal(
            empty_frame.pts_seconds,
            NASA_VIDEO.empty_pts_seconds,
        )
        assert_tensor_equal(
            empty_frame.duration_seconds, NASA_VIDEO.empty_duration_seconds
        )

        # Start and stop seconds land within the first frame.
        frame0 = decoder.get_frames_played_in_range(
            NASA_VIDEO.get_frame_info(0, stream_index=stream_index).pts_seconds,
            NASA_VIDEO.get_frame_info(0, stream_index=stream_index).pts_seconds
            + HALF_DURATION,
        )
        assert_tensor_equal(
            frame0.data,
            NASA_VIDEO.get_frame_data_by_range(0, 1, stream_index=stream_index),
        )

        # We should be able to get all frames by giving the beginning and ending time
        # for the stream.
        all_frames = decoder.get_frames_played_in_range(
            decoder.metadata.begin_stream_seconds, decoder.metadata.end_stream_seconds
        )
        assert_tensor_equal(all_frames.data, decoder[:])

    def test_get_frames_by_pts_in_range_fails(self):
        decoder = VideoDecoder(NASA_VIDEO.path)

        with pytest.raises(ValueError, match="Invalid start seconds"):
            frame = decoder.get_frames_played_in_range(100.0, 1.0)  # noqa

        with pytest.raises(ValueError, match="Invalid start seconds"):
            frame = decoder.get_frames_played_in_range(20, 23)  # noqa

        with pytest.raises(ValueError, match="Invalid stop seconds"):
            frame = decoder.get_frames_played_in_range(0, 23)  # noqa


if __name__ == "__main__":
    pytest.main()
