import pytest
import torch

from torchcodec.decoders import _core, SimpleVideoDecoder

from ..test_utils import assert_tensor_equal, NASA_VIDEO


class TestSimpleDecoder:
    @pytest.mark.parametrize("source_kind", ("path", "tensor", "bytes"))
    def test_create(self, source_kind):
        if source_kind == "path":
            source = str(NASA_VIDEO.path)
        elif source_kind == "tensor":
            source = NASA_VIDEO.to_tensor()
        elif source_kind == "bytes":
            path = str(NASA_VIDEO.path)
            with open(path, "rb") as f:
                source = f.read()
        else:
            raise ValueError("Oops, double check the parametrization of this test!")

        decoder = SimpleVideoDecoder(source)
        assert isinstance(decoder.stream_metadata, _core.StreamMetadata)
        assert (
            len(decoder)
            == decoder._num_frames
            == decoder.stream_metadata.num_frames_computed
            == 390
        )
        assert decoder._stream_index == decoder.stream_metadata.stream_index == 3

    def test_create_fails(self):
        with pytest.raises(TypeError, match="Unknown source type"):
            decoder = SimpleVideoDecoder(123)  # noqa

    def test_getitem_int(self):
        decoder = SimpleVideoDecoder(str(NASA_VIDEO.path))

        ref_frame0 = NASA_VIDEO.get_tensor_by_index(0)
        ref_frame1 = NASA_VIDEO.get_tensor_by_index(1)
        ref_frame180 = NASA_VIDEO.get_tensor_by_name("time6.000000")
        ref_frame_last = NASA_VIDEO.get_tensor_by_name("time12.979633")

        assert_tensor_equal(ref_frame0, decoder[0])
        assert_tensor_equal(ref_frame1, decoder[1])
        assert_tensor_equal(ref_frame180, decoder[180])
        assert_tensor_equal(ref_frame_last, decoder[-1])

    def test_getitem_slice(self):
        decoder = SimpleVideoDecoder(str(NASA_VIDEO.path))

        # ensure that the degenerate case of a range of size 1 works

        ref0 = NASA_VIDEO.get_stacked_tensor_by_range(0, 1)
        slice0 = decoder[0:1]
        assert slice0.shape == torch.Size(
            [
                1,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
                NASA_VIDEO.num_color_channels,
            ]
        )
        assert_tensor_equal(ref0, slice0)

        ref4 = NASA_VIDEO.get_stacked_tensor_by_range(4, 5)
        slice4 = decoder[4:5]
        assert slice4.shape == torch.Size(
            [
                1,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
                NASA_VIDEO.num_color_channels,
            ]
        )
        assert_tensor_equal(ref4, slice4)

        ref8 = NASA_VIDEO.get_stacked_tensor_by_range(8, 9)
        slice8 = decoder[8:9]
        assert slice8.shape == torch.Size(
            [
                1,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
                NASA_VIDEO.num_color_channels,
            ]
        )
        assert_tensor_equal(ref8, slice8)

        ref180 = NASA_VIDEO.get_tensor_by_name("time6.000000")
        slice180 = decoder[180:181]
        assert slice180.shape == torch.Size(
            [
                1,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
                NASA_VIDEO.num_color_channels,
            ]
        )
        assert_tensor_equal(ref180, slice180[0])

        # contiguous ranges
        ref0_9 = NASA_VIDEO.get_stacked_tensor_by_range(0, 9)
        slice0_9 = decoder[0:9]
        assert slice0_9.shape == torch.Size(
            [
                9,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
                NASA_VIDEO.num_color_channels,
            ]
        )
        assert_tensor_equal(ref0_9, slice0_9)

        ref4_8 = NASA_VIDEO.get_stacked_tensor_by_range(4, 8)
        slice4_8 = decoder[4:8]
        assert slice4_8.shape == torch.Size(
            [
                4,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
                NASA_VIDEO.num_color_channels,
            ]
        )
        assert_tensor_equal(ref4_8, slice4_8)

        # ranges with a stride
        ref15_35 = NASA_VIDEO.get_stacked_tensor_by_range(15, 36, 5)
        slice15_35 = decoder[15:36:5]
        assert slice15_35.shape == torch.Size(
            [
                5,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
                NASA_VIDEO.num_color_channels,
            ]
        )
        assert_tensor_equal(ref15_35, slice15_35)

        ref0_9_2 = NASA_VIDEO.get_stacked_tensor_by_range(0, 9, 2)
        slice0_9_2 = decoder[0:9:2]
        assert slice0_9_2.shape == torch.Size(
            [
                5,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
                NASA_VIDEO.num_color_channels,
            ]
        )
        assert_tensor_equal(ref0_9_2, slice0_9_2)

        # negative numbers in the slice
        ref386_389 = NASA_VIDEO.get_stacked_tensor_by_range(386, 390)
        slice386_389 = decoder[-4:]
        assert slice386_389.shape == torch.Size(
            [
                4,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
                NASA_VIDEO.num_color_channels,
            ]
        )
        assert_tensor_equal(ref386_389, slice386_389)

        # an empty range is valid!
        empty_frame = decoder[5:5]
        assert_tensor_equal(empty_frame, NASA_VIDEO.empty_hwc_tensor)

        # slices that are out-of-range are also valid - they return an empty tensor
        also_empty = decoder[10000:]
        assert_tensor_equal(also_empty, NASA_VIDEO.empty_hwc_tensor)

        # should be just a copy
        all_frames = decoder[:]
        assert all_frames.shape == torch.Size(
            [
                len(decoder),
                NASA_VIDEO.height,
                NASA_VIDEO.width,
                NASA_VIDEO.num_color_channels,
            ]
        )
        for sliced, ref in zip(all_frames, decoder):
            assert_tensor_equal(sliced, ref)

    def test_getitem_fails(self):
        decoder = SimpleVideoDecoder(str(NASA_VIDEO.path))

        with pytest.raises(IndexError, match="out of bounds"):
            frame = decoder[1000]  # noqa

        with pytest.raises(IndexError, match="out of bounds"):
            frame = decoder[-1000]  # noqa

        with pytest.raises(TypeError, match="Unsupported key type"):
            frame = decoder["0"]  # noqa

    def test_iteration(self):
        decoder = SimpleVideoDecoder(str(NASA_VIDEO.path))

        ref_frame0 = NASA_VIDEO.get_tensor_by_index(0)
        ref_frame1 = NASA_VIDEO.get_tensor_by_index(1)
        ref_frame9 = NASA_VIDEO.get_tensor_by_index(9)
        ref_frame35 = NASA_VIDEO.get_tensor_by_index(35)
        ref_frame180 = NASA_VIDEO.get_tensor_by_name("time6.000000")
        ref_frame_last = NASA_VIDEO.get_tensor_by_name("time12.979633")

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
        decoder = SimpleVideoDecoder(str(NASA_VIDEO.path))
        ref_frame_last = NASA_VIDEO.get_tensor_by_index(389)

        # Force the decoder to seek around a lot while iterating; this will
        # slow down decoding, but we should still only iterate the exact number
        # of total frames.
        iterations = 0
        for frame in decoder:
            assert_tensor_equal(ref_frame_last, decoder[-1])
            iterations += 1

        assert iterations == len(decoder) == 390


if __name__ == "__main__":
    pytest.main()
