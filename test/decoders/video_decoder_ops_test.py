import os

os.environ["TORCH_LOGS"] = "output_code"
import json
from typing import Tuple

import numpy as np
import pytest

import torch

from torchcodec.decoders._core import (
    add_video_stream,
    create_from_bytes,
    create_from_file,
    create_from_tensor,
    get_frame_at_index,
    get_frame_at_pts,
    get_frames_at_indices,
    get_json_metadata,
    get_next_frame,
    seek_to_pts,
)

from ..test_utils import (
    assert_equal,
    get_reference_audio_path,
    get_reference_video_path,
    load_tensor_from_file,
)

torch._dynamo.config.capture_dynamic_output_shape_ops = True


class ReferenceDecoder:
    def __init__(self):
        self.decoder: torch.Tensor = create_from_file(str(get_reference_video_path()))
        add_video_stream(self.decoder)

    def get_next_frame(self) -> torch.Tensor:
        assert self.decoder is not None
        return get_next_frame(self.decoder)

    def seek(self, pts: float):
        assert self.decoder is not None
        seek_to_pts(self.decoder, pts)


# TODO: Some of these tests could probably be unified and parametrized?
class TestOps:
    def test_seek_and_next(self):
        decoder = create_from_file(str(get_reference_video_path()))
        add_video_stream(decoder)
        frame1 = get_next_frame(decoder)
        reference_frame1 = load_tensor_from_file("nasa_13013.mp4.frame000001.pt")
        assert_equal(frame1, reference_frame1)
        reference_frame2 = load_tensor_from_file("nasa_13013.mp4.frame000002.pt")
        img2 = get_next_frame(decoder)
        assert_equal(img2, reference_frame2)
        seek_to_pts(decoder, 6.0)
        frame_time6 = get_next_frame(decoder)
        reference_frame_time6 = load_tensor_from_file("nasa_13013.mp4.time6.000000.pt")
        assert_equal(frame_time6, reference_frame_time6)

    def test_get_frame_at_pts(self):
        decoder = create_from_file(str(get_reference_video_path()))
        add_video_stream(decoder)
        # This frame has pts=6.006 and duration=0.033367, so it should be visible
        # at timestamps in the range [6.006, 6.039367) (not including the last timestamp).
        frame6 = get_frame_at_pts(decoder, 6.006)
        reference_frame6 = load_tensor_from_file("nasa_13013.mp4.time6.000000.pt")
        assert_equal(frame6, reference_frame6)
        frame6 = get_frame_at_pts(decoder, 6.02)
        assert_equal(frame6, reference_frame6)
        frame6 = get_frame_at_pts(decoder, 6.039366)
        assert_equal(frame6, reference_frame6)
        # Note that this timestamp is exactly on a frame boundary, so it should
        # return the next frame since the right boundary of the interval is
        # open.
        next_frame = get_frame_at_pts(decoder, 6.039367)
        with pytest.raises(AssertionError):
            assert_equal(next_frame, reference_frame6)

    def test_get_frame_at_index(self):
        decoder = create_from_file(str(get_reference_video_path()))
        add_video_stream(decoder)
        frame1 = get_frame_at_index(decoder, frame_index=0, stream_index=3)
        reference_frame1 = load_tensor_from_file("nasa_13013.mp4.frame000001.pt")
        assert_equal(frame1, reference_frame1)
        # The frame that is displayed at 6 seconds is frame 180 from a 0-based index.
        frame6 = get_frame_at_index(decoder, frame_index=180, stream_index=3)
        reference_frame6 = load_tensor_from_file("nasa_13013.mp4.time6.000000.pt")
        assert_equal(frame6, reference_frame6)

    def test_get_frames_at_indices(self):
        decoder = create_from_file(str(get_reference_video_path()))
        add_video_stream(decoder)
        frames1and6 = get_frames_at_indices(
            decoder, frame_indices=[0, 180], stream_index=3
        )
        reference_frame1 = load_tensor_from_file("nasa_13013.mp4.frame000001.pt")
        reference_frame6 = load_tensor_from_file("nasa_13013.mp4.time6.000000.pt")
        assert_equal(frames1and6[0], reference_frame1)
        assert_equal(frames1and6[1], reference_frame6)

    def test_throws_exception_at_eof(self):
        decoder = create_from_file(str(get_reference_video_path()))
        add_video_stream(decoder)
        seek_to_pts(decoder, 12.979633)
        last_frame = get_next_frame(decoder)
        reference_last_frame = load_tensor_from_file("nasa_13013.mp4.time12.979633.pt")
        assert_equal(last_frame, reference_last_frame)
        with pytest.raises(RuntimeError, match="End of file"):
            get_next_frame(decoder)

    def test_throws_exception_if_seek_too_far(self):
        decoder = create_from_file(str(get_reference_video_path()))
        add_video_stream(decoder)
        # pts=12.979633 is the last frame in the video.
        seek_to_pts(decoder, 12.979633 + 1.0e-4)
        with pytest.raises(RuntimeError, match="End of file"):
            get_next_frame(decoder)

    def test_compile_seek_and_next(self):
        # TODO(T180277797): Get this to work with the inductor stack. Right now
        # compilation fails because it can't handle tensors of size unknown at
        # compile-time.
        @torch.compile(fullgraph=True, backend="eager")
        def get_frame1_and_frame_time6(decoder):
            add_video_stream(decoder)
            frame1 = get_next_frame(decoder)
            seek_to_pts(decoder, 6.0)
            frame_time6 = get_next_frame(decoder)
            return frame1, frame_time6

        # NB: create needs to happen outside the torch.compile region,
        # for now. Otherwise torch.compile constant-props it.
        decoder = create_from_file(str(get_reference_video_path()))
        frame1, frame_time6 = get_frame1_and_frame_time6(decoder)
        reference_frame1 = load_tensor_from_file("nasa_13013.mp4.frame000001.pt")
        reference_frame_time6 = load_tensor_from_file("nasa_13013.mp4.time6.000000.pt")
        assert_equal(frame1, reference_frame1)
        assert_equal(frame_time6, reference_frame_time6)

    def test_class_based_compile_seek_and_next(self):
        # TODO(T180277797): Ditto as above.
        @torch.compile(fullgraph=True, backend="eager")
        def class_based_get_frame1_and_frame_time6(
            decoder: ReferenceDecoder,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            frame1 = decoder.get_next_frame()
            decoder.seek(6.0)
            frame_time6 = decoder.get_next_frame()
            return frame1, frame_time6

        decoder = ReferenceDecoder()
        frame1, frame_time6 = class_based_get_frame1_and_frame_time6(decoder)
        reference_frame1 = load_tensor_from_file("nasa_13013.mp4.frame000001.pt")
        reference_frame_time6 = load_tensor_from_file("nasa_13013.mp4.time6.000000.pt")
        assert_equal(frame1, reference_frame1)
        assert_equal(frame_time6, reference_frame_time6)

    @pytest.mark.parametrize("create_from", ("file", "tensor", "bytes"))
    def test_create_decoder(self, create_from):
        path = str(get_reference_video_path())
        if create_from == "file":
            decoder = create_from_file(path)
        elif create_from == "tensor":
            arr = np.fromfile(path, dtype=np.uint8)
            video_tensor = torch.from_numpy(arr)
            decoder = create_from_tensor(video_tensor)
        else:  # bytes
            with open(path, "rb") as f:
                video_bytes = f.read()
            decoder = create_from_bytes(video_bytes)

        add_video_stream(decoder)
        frame1 = get_next_frame(decoder)
        reference_frame1 = load_tensor_from_file("nasa_13013.mp4.frame000001.pt")
        assert_equal(frame1, reference_frame1)
        reference_frame2 = load_tensor_from_file("nasa_13013.mp4.frame000002.pt")
        img2 = get_next_frame(decoder)
        assert_equal(img2, reference_frame2)
        seek_to_pts(decoder, 6.0)
        frame_time6 = get_next_frame(decoder)
        reference_frame_time6 = load_tensor_from_file("nasa_13013.mp4.time6.000000.pt")
        assert_equal(frame_time6, reference_frame_time6)

    # TODO: Keeping the metadata tests below for now, but we should remove them
    # once we remove get_json_metadata().
    # Note that the distinction made between test_video_get_json_metadata and
    # test_video_get_json_metadata_with_stream is misleading: all of the stream
    # metadata are available even without adding a video stream, because we
    # always call scanFileAndUpdateMetadataAndIndex() when creating a decoder
    # from the core API.
    def test_video_get_json_metadata(self):
        decoder = create_from_file(str(get_reference_video_path()))
        metadata = get_json_metadata(decoder)
        metadata_dict = json.loads(metadata)

        # We should be able to see all of this metadata without adding a video stream
        assert metadata_dict["durationSeconds"] == pytest.approx(13.013, abs=0.001)
        assert metadata_dict["numFrames"] == 390
        assert metadata_dict["averageFps"] == pytest.approx(29.97, abs=0.001)
        assert metadata_dict["codec"] == "h264"
        assert metadata_dict["bitRate"] == 128783.0

    def test_video_get_json_metadata_with_stream(self):
        decoder = create_from_file(str(get_reference_video_path()))
        add_video_stream(decoder)
        metadata = get_json_metadata(decoder)
        metadata_dict = json.loads(metadata)
        assert metadata_dict["width"] == 480
        assert metadata_dict["height"] == 270
        assert metadata_dict["minPtsSecondsFromScan"] == 0
        assert metadata_dict["maxPtsSecondsFromScan"] == 13.013

    def test_audio_get_json_metadata(self):
        decoder = create_from_file(str(get_reference_audio_path()))
        metadata = get_json_metadata(decoder)
        metadata_dict = json.loads(metadata)
        assert metadata_dict["durationSeconds"] == pytest.approx(13.25, abs=0.01)


if __name__ == "__main__":
    pytest.main()
