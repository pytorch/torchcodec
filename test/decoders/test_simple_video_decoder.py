# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchcodec.decoders import _core, SimpleVideoDecoder

from ..utils import assert_tensor_close, assert_tensor_equal, NASA_VIDEO


class TestSimpleDecoder:
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

        decoder = SimpleVideoDecoder(source)
        assert isinstance(decoder.metadata, _core.VideoStreamMetadata)
        assert (
            len(decoder)
            == decoder._num_frames
            == decoder.metadata.num_frames_from_content
            == 390
        )
        assert decoder._stream_index == decoder.metadata.stream_index == 3
        assert decoder.metadata.duration_seconds == pytest.approx(13.013)
        assert decoder.metadata.average_fps == pytest.approx(29.970029)
        assert decoder.metadata.num_frames == 390

    def test_create_fails(self):
        with pytest.raises(TypeError, match="Unknown source type"):
            decoder = SimpleVideoDecoder(123)  # noqa

    def test_getitem_int(self):
        decoder = SimpleVideoDecoder(NASA_VIDEO.path)

        ref_frame0 = NASA_VIDEO.get_frame_data_by_index(0)
        ref_frame1 = NASA_VIDEO.get_frame_data_by_index(1)
        ref_frame180 = NASA_VIDEO.get_frame_by_name("time6.000000")
        ref_frame_last = NASA_VIDEO.get_frame_by_name("time12.979633")

        assert_tensor_equal(ref_frame0, decoder[0])
        assert_tensor_equal(ref_frame1, decoder[1])
        assert_tensor_equal(ref_frame180, decoder[180])
        assert_tensor_equal(ref_frame_last, decoder[-1])

    def test_getitem_slice(self):
        decoder = SimpleVideoDecoder(NASA_VIDEO.path)

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
        decoder = SimpleVideoDecoder(NASA_VIDEO.path)

        with pytest.raises(IndexError, match="out of bounds"):
            frame = decoder[1000]  # noqa

        with pytest.raises(IndexError, match="out of bounds"):
            frame = decoder[-1000]  # noqa

        with pytest.raises(TypeError, match="Unsupported key type"):
            frame = decoder["0"]  # noqa

    def test_iteration(self):
        decoder = SimpleVideoDecoder(NASA_VIDEO.path)

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
        decoder = SimpleVideoDecoder(NASA_VIDEO.path)
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
        decoder = SimpleVideoDecoder(NASA_VIDEO.path)

        ref_frame9 = NASA_VIDEO.get_frame_data_by_index(9)
        frame9 = decoder.get_frame_at(9)

        assert_tensor_equal(ref_frame9, frame9.data)
        assert isinstance(frame9.pts_seconds, float)
        assert frame9.pts_seconds == pytest.approx(0.3003)
        assert isinstance(frame9.duration_seconds, float)
        assert frame9.duration_seconds == pytest.approx(0.03337, rel=1e-3)

    def test_get_frame_at_tuple_unpacking(self):
        decoder = SimpleVideoDecoder(NASA_VIDEO.path)

        frame = decoder.get_frame_at(50)
        data, pts, duration = decoder.get_frame_at(50)

        assert_tensor_equal(frame.data, data)
        assert frame.pts_seconds == pts
        assert frame.duration_seconds == duration

    def test_get_frame_at_fails(self):
        decoder = SimpleVideoDecoder(NASA_VIDEO.path)

        with pytest.raises(IndexError, match="out of bounds"):
            frame = decoder.get_frame_at(-1)  # noqa

        with pytest.raises(IndexError, match="out of bounds"):
            frame = decoder.get_frame_at(10000)  # noqa

    def test_get_frame_displayed_at(self):
        decoder = SimpleVideoDecoder(NASA_VIDEO.path)

        ref_frame6 = NASA_VIDEO.get_frame_by_name("time6.000000")
        assert_tensor_equal(ref_frame6, decoder.get_frame_displayed_at(6.006).data)
        assert_tensor_equal(ref_frame6, decoder.get_frame_displayed_at(6.02).data)
        assert_tensor_equal(ref_frame6, decoder.get_frame_displayed_at(6.039366).data)
        assert isinstance(decoder.get_frame_displayed_at(6.02).pts_seconds, float)
        assert isinstance(decoder.get_frame_displayed_at(6.02).duration_seconds, float)

    def test_get_frame_displayed_at_fails(self):
        decoder = SimpleVideoDecoder(NASA_VIDEO.path)

        with pytest.raises(IndexError, match="Invalid pts in seconds"):
            frame = decoder.get_frame_displayed_at(-1.0)  # noqa

        with pytest.raises(IndexError, match="Invalid pts in seconds"):
            frame = decoder.get_frame_displayed_at(100.0)  # noqa

    def test_get_frames_at(self):
        decoder = SimpleVideoDecoder(NASA_VIDEO.path)

        # test degenerate case where we only actually get 1 frame
        ref_frames9 = NASA_VIDEO.get_frame_data_by_range(start=9, stop=10)
        frames9 = decoder.get_frames_at(start=9, stop=10)

        assert_tensor_equal(ref_frames9, frames9.data)
        assert frames9.pts_seconds[0].item() == pytest.approx(0.3003, rel=1e-3)
        assert frames9.duration_seconds[0].item() == pytest.approx(0.03337, rel=1e-3)

        # test simple ranges
        ref_frames0_9 = NASA_VIDEO.get_frame_data_by_range(start=0, stop=10)
        frames0_9 = decoder.get_frames_at(start=0, stop=10)
        assert frames0_9.data.shape == torch.Size(
            [
                10,
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        assert_tensor_equal(ref_frames0_9, frames0_9.data)
        assert_tensor_close(
            NASA_VIDEO.get_pts_seconds_by_range(0, 10),
            frames0_9.pts_seconds,
        )
        assert_tensor_close(
            NASA_VIDEO.get_duration_seconds_by_range(0, 10),
            frames0_9.duration_seconds,
        )

        # test steps
        ref_frames0_8_2 = NASA_VIDEO.get_frame_data_by_range(start=0, stop=10, step=2)
        frames0_8_2 = decoder.get_frames_at(start=0, stop=10, step=2)
        assert frames0_8_2.data.shape == torch.Size(
            [
                5,
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        assert_tensor_equal(ref_frames0_8_2, frames0_8_2.data)
        assert_tensor_close(
            NASA_VIDEO.get_pts_seconds_by_range(0, 10, 2),
            frames0_8_2.pts_seconds,
        )
        assert_tensor_close(
            NASA_VIDEO.get_duration_seconds_by_range(0, 10, 2),
            frames0_8_2.duration_seconds,
        )

        # an empty range is valid!
        empty_frames = decoder.get_frames_at(5, 5)
        assert_tensor_equal(empty_frames.data, NASA_VIDEO.empty_chw_tensor)
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
            lambda decoder: decoder.get_frames_at(0, 4).data,
            lambda decoder: decoder.get_frame_displayed_at(0).data,
            # TODO: uncomment once D60001893 lands
            # lambda decoder: decoder.get_frames_displayed_at(0, 1).data,
        ),
    )
    def test_dimension_order(self, dimension_order, frame_getter):
        decoder = SimpleVideoDecoder(NASA_VIDEO.path, dimension_order=dimension_order)
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
            SimpleVideoDecoder(NASA_VIDEO.path, dimension_order="NCDHW")


if __name__ == "__main__":
    pytest.main()
