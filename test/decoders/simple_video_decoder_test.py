import pytest

from torchcodec.decoders import _core, SimpleVideoDecoder

from ..test_utils import (
    assert_equal,
    get_reference_video_path,
    get_reference_video_tensor,
    load_tensor_from_file,
)


class TestSimpleDecoder:
    @pytest.mark.parametrize("source_kind", ("path", "tensor", "bytes"))
    def test_create(self, source_kind):
        if source_kind == "path":
            source = str(get_reference_video_path())
        elif source_kind == "tensor":
            source = get_reference_video_tensor()
        elif source_kind == "bytes":
            path = str(get_reference_video_path())
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
        decoder = SimpleVideoDecoder(str(get_reference_video_path()))

        ref_frame0 = load_tensor_from_file("nasa_13013.mp4.frame000001.pt")
        ref_frame1 = load_tensor_from_file("nasa_13013.mp4.frame000002.pt")
        ref_frame180 = load_tensor_from_file("nasa_13013.mp4.time6.000000.pt")
        ref_frame_last = load_tensor_from_file("nasa_13013.mp4.time12.979633.pt")

        assert_equal(ref_frame0, decoder[0])
        assert_equal(ref_frame1, decoder[1])
        assert_equal(ref_frame180, decoder[180])
        assert_equal(ref_frame_last, decoder[-1])

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
