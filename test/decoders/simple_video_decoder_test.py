import pytest
import torch

from torchcodec.decoders import SimpleVideoDecoder

from ..test_utils import (
    assert_equal,
    EMPTY_REF_TENSOR,
    get_reference_video_path,
    get_reference_video_tensor,
    load_tensor_from_file,
    REF_DIMS,
)


class TestSimpleDecoder:

    def test_create_from_file(self):
        decoder = SimpleVideoDecoder(str(get_reference_video_path()))
        assert len(decoder) == 390
        assert decoder._stream_index == 3

    def test_create_from_tensor(self):
        decoder = SimpleVideoDecoder(get_reference_video_tensor())
        assert len(decoder) == 390
        assert decoder._stream_index == 3

    def test_create_from_bytes(self):
        path = str(get_reference_video_path())
        with open(path, "rb") as f:
            video_bytes = f.read()

        decoder = SimpleVideoDecoder(video_bytes)
        assert len(decoder) == 390
        assert decoder._stream_index == 3

    def test_create_fails(self):
        with pytest.raises(TypeError, match="Unknown source type"):
            decoder = SimpleVideoDecoder(123)  # noqa

    def test_getitem_int(self):
        decoder = SimpleVideoDecoder(str(get_reference_video_path()))

        ref_frame0 = load_tensor_from_file("nasa_13013.mp4.frame000001.pt")
        ref_frame1 = load_tensor_from_file("nasa_13013.mp4.frame000002.pt")
        ref_frame180 = load_tensor_from_file("nasa_13013.mp4.time6.000000.pt")
        ref_frame_last = load_tensor_from_file("nasa_13013.mp4.time12.979633.pt")

        assert_equal(ref_frame0, decoder[0])
        assert_equal(ref_frame1, decoder[1])
        assert_equal(ref_frame180, decoder[180])
        assert_equal(ref_frame_last, decoder[-1])

    def test_getitem_slice(self):
        decoder = SimpleVideoDecoder(str(get_reference_video_path()))

        ref_frames0_9 = [
            load_tensor_from_file(f"nasa_13013.mp4.frame{i + 1:06d}.pt")
            for i in range(0, 9)
        ]

        # Ensure that the degenerate case of a range of size 1 works; note that we get
        # a tensor which CONTAINS a single frame, rather than a tensor that itself IS a
        # single frame. Hence we have to access the 0th element of the return tensor.
        slice_0 = decoder[0:1]
        assert slice_0.shape == torch.Size([1, *REF_DIMS])
        assert_equal(ref_frames0_9[0], slice_0[0])

        slice_4 = decoder[4:5]
        assert slice_4.shape == torch.Size([1, *REF_DIMS])
        assert_equal(ref_frames0_9[4], slice_4[0])

        slice_8 = decoder[8:9]
        assert slice_8.shape == torch.Size([1, *REF_DIMS])
        assert_equal(ref_frames0_9[8], slice_8[0])

        ref_frame180 = load_tensor_from_file("nasa_13013.mp4.time6.000000.pt")
        slice_180 = decoder[180:181]
        assert slice_180.shape == torch.Size([1, *REF_DIMS])
        assert_equal(ref_frame180, slice_180[0])

        # contiguous ranges
        slice_frames0_9 = decoder[0:9]
        assert slice_frames0_9.shape == torch.Size([9, *REF_DIMS])
        for ref_frame, slice_frame in zip(ref_frames0_9, slice_frames0_9):
            assert_equal(ref_frame, slice_frame)

        slice_frames4_8 = decoder[4:8]
        assert slice_frames4_8.shape == torch.Size([4, *REF_DIMS])
        for ref_frame, slice_frame in zip(ref_frames0_9[4:8], slice_frames4_8):
            assert_equal(ref_frame, slice_frame)

        # ranges with a stride
        ref_frames15_35 = [
            load_tensor_from_file(f"nasa_13013.mp4.frame{i:06d}.pt")
            for i in range(15, 36, 5)
        ]
        slice_frames15_35 = decoder[15:36:5]
        assert slice_frames15_35.shape == torch.Size([5, *REF_DIMS])
        for ref_frame, slice_frame in zip(ref_frames15_35, slice_frames15_35):
            assert_equal(ref_frame, slice_frame)

        slice_frames0_9_2 = decoder[0:9:2]
        assert slice_frames0_9_2.shape == torch.Size([5, *REF_DIMS])
        for ref_frame, slice_frame in zip(ref_frames0_9[0:0:2], slice_frames0_9_2):
            assert_equal(ref_frame, slice_frame)

        # negative numbers in the slice
        ref_frames386_389 = [
            load_tensor_from_file(f"nasa_13013.mp4.frame{i:06d}.pt")
            for i in range(386, 390)
        ]

        slice_frames386_389 = decoder[-4:]
        assert slice_frames386_389.shape == torch.Size([4, *REF_DIMS])
        for ref_frame, slice_frame in zip(ref_frames386_389[-4:], slice_frames386_389):
            assert_equal(ref_frame, slice_frame)

        # an empty range is valid!
        empty_frame = decoder[5:5]
        assert_equal(empty_frame, EMPTY_REF_TENSOR)

        # slices that are out-of-range are also valid - they return an empty tensor
        also_empty = decoder[10000:]
        assert_equal(also_empty, EMPTY_REF_TENSOR)

        # should be just a copy
        all_frames = decoder[:]
        assert all_frames.shape == torch.Size([len(decoder), *REF_DIMS])
        for sliced, ref in zip(all_frames, decoder):
            assert_equal(sliced, ref)

    def test_getitem_fails(self):
        decoder = SimpleVideoDecoder(str(get_reference_video_path()))

        with pytest.raises(IndexError, match="out of bounds"):
            frame = decoder[1000]  # noqa

        with pytest.raises(IndexError, match="out of bounds"):
            frame = decoder[-1000]  # noqa

        with pytest.raises(TypeError, match="Unsupported key type"):
            frame = decoder["0"]  # noqa

    def test_next(self):
        decoder = SimpleVideoDecoder(str(get_reference_video_path()))

        ref_frame0 = load_tensor_from_file("nasa_13013.mp4.frame000001.pt")
        ref_frame1 = load_tensor_from_file("nasa_13013.mp4.frame000002.pt")
        ref_frame180 = load_tensor_from_file("nasa_13013.mp4.time6.000000.pt")
        ref_frame_last = load_tensor_from_file("nasa_13013.mp4.time12.979633.pt")

        for i, frame in enumerate(decoder):
            if i == 0:
                assert_equal(ref_frame0, frame)
            elif i == 1:
                assert_equal(ref_frame1, frame)
            elif i == 180:
                assert_equal(ref_frame180, frame)
            elif i == 389:
                assert_equal(ref_frame_last, frame)


if __name__ == "__main__":
    pytest.main()
