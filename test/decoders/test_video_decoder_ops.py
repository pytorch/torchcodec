# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ["TORCH_LOGS"] = "output_code"
import json
import subprocess
from typing import Tuple

import numpy as np
import pytest

import torch

from torchcodec.decoders._core import (
    _add_video_stream,
    _test_frame_pts_equality,
    add_audio_stream,
    add_video_stream,
    create_from_bytes,
    create_from_file,
    create_from_tensor,
    get_ffmpeg_library_versions,
    get_frame_at_index,
    get_frame_at_pts,
    get_frames_at_indices,
    get_frames_by_pts,
    get_frames_by_pts_in_range,
    get_frames_in_range,
    get_json_metadata,
    get_next_frame,
    seek_to_pts,
)

from ..utils import (
    assert_frames_equal,
    cpu_and_cuda,
    NASA_AUDIO,
    NASA_VIDEO,
    needs_cuda,
)

torch._dynamo.config.capture_dynamic_output_shape_ops = True

# this is the index of the frame that gets decoded after we
# seek to pts_seconds=6. This isn't the frame "played at" pts_seconds=6
INDEX_OF_VIDEO_FRAME_AFTER_SEEKING_AT_6 = 180
INDEX_OF_AUDIO_FRAME_AFTER_SEEKING_AT_6 = 94


def _add_stream(*, decoder, test_ref, device="cpu"):
    if test_ref is NASA_VIDEO:
        add_video_stream(decoder, device=device)
    elif test_ref is NASA_AUDIO:
        add_audio_stream(decoder)
    else:
        raise ValueError("Can't add a stream for this test reference.")


class ReferenceDecoder:
    def __init__(self, device="cpu"):
        self.decoder: torch.Tensor = create_from_file(str(NASA_VIDEO.path))
        add_video_stream(self.decoder, device=device)

    def get_next_frame(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.decoder is not None
        return get_next_frame(self.decoder)

    def seek(self, pts: float):
        assert self.decoder is not None
        seek_to_pts(self.decoder, pts)


class TestOps:
    def test_add_stream(self):
        valid_video_stream, valid_audio_stream = 0, 1

        decoder = create_from_file(str(NASA_VIDEO.path))
        add_video_stream(decoder, stream_index=valid_video_stream)
        with pytest.raises(RuntimeError, match="Can only add one single stream"):
            add_video_stream(decoder, stream_index=valid_video_stream)

        decoder = create_from_file(str(NASA_VIDEO.path))
        add_audio_stream(decoder, stream_index=valid_audio_stream)
        with pytest.raises(RuntimeError, match="Can only add one single stream"):
            add_audio_stream(decoder, stream_index=valid_audio_stream)

        decoder = create_from_file(str(NASA_VIDEO.path))
        with pytest.raises(
            ValueError, match=f"Is {valid_audio_stream} of the desired media type"
        ):
            add_video_stream(decoder, stream_index=valid_audio_stream)

        decoder = create_from_file(str(NASA_VIDEO.path))
        with pytest.raises(
            ValueError, match=f"Is {valid_video_stream} of the desired media type"
        ):
            add_audio_stream(decoder, stream_index=valid_video_stream)

    @pytest.mark.parametrize(
        "test_ref, index_of_frame_after_seeking_at_6",
        (
            (NASA_VIDEO, INDEX_OF_VIDEO_FRAME_AFTER_SEEKING_AT_6),
            (NASA_AUDIO, INDEX_OF_AUDIO_FRAME_AFTER_SEEKING_AT_6),
        ),
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_seek_and_next(
        self, test_ref, index_of_frame_after_seeking_at_6, device, seek_mode
    ):
        if device == "cuda" and test_ref is NASA_AUDIO:
            pytest.skip(reason="CUDA decoding not supported for audio")
        decoder = create_from_file(str(test_ref.path), seek_mode=seek_mode)
        _add_stream(decoder=decoder, test_ref=test_ref, device=device)
        frame0, _, _ = get_next_frame(decoder)
        reference_frame0 = test_ref.get_frame_data_by_index(0)
        assert_frames_equal(frame0, reference_frame0.to(device))
        reference_frame1 = test_ref.get_frame_data_by_index(1)
        frame1, _, _ = get_next_frame(decoder)
        assert_frames_equal(frame1, reference_frame1.to(device))
        seek_to_pts(decoder, 6.0)
        frame_time6, _, _ = get_next_frame(decoder)
        reference_frame_time6 = test_ref.get_frame_data_by_index(
            index_of_frame_after_seeking_at_6
        )
        assert_frames_equal(frame_time6, reference_frame_time6.to(device))

    @pytest.mark.parametrize("test_ref", (NASA_VIDEO, NASA_AUDIO))
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_seek_to_negative_pts(self, test_ref, device, seek_mode):
        if device == "cuda" and test_ref is NASA_AUDIO:
            pytest.skip(reason="CUDA decoding not supported for audio")

        decoder = create_from_file(str(test_ref.path), seek_mode=seek_mode)
        _add_stream(decoder=decoder, test_ref=test_ref, device=device)
        frame0, _, _ = get_next_frame(decoder)
        reference_frame0 = test_ref.get_frame_data_by_index(0)
        assert_frames_equal(frame0, reference_frame0.to(device))

        seek_to_pts(decoder, -1e-4)
        frame0, _, _ = get_next_frame(decoder)
        assert_frames_equal(frame0, reference_frame0.to(device))

    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frame_at_pts_video(self, device, seek_mode):

        decoder = create_from_file(str(NASA_VIDEO.path), seek_mode=seek_mode)
        add_video_stream(decoder=decoder, device=device)
        # This frame has pts=6.006 and duration=0.033367, so it should be visible
        # at timestamps in the range [6.006, 6.039367) (not including the last timestamp).
        frame6, _, _ = get_frame_at_pts(decoder, 6.006)
        reference_frame6 = NASA_VIDEO.get_frame_data_by_index(
            INDEX_OF_VIDEO_FRAME_AFTER_SEEKING_AT_6
        )
        assert_frames_equal(frame6, reference_frame6.to(device))
        frame6, _, _ = get_frame_at_pts(decoder, 6.02)
        assert_frames_equal(frame6, reference_frame6.to(device))
        frame6, _, _ = get_frame_at_pts(decoder, 6.039366)
        assert_frames_equal(frame6, reference_frame6.to(device))
        # Note that this timestamp is exactly on a frame boundary, so it should
        # return the next frame since the right boundary of the interval is
        # open.
        next_frame, _, _ = get_frame_at_pts(decoder, 6.039367)
        if device == "cpu":
            # We can only compare exact equality on CPU.
            with pytest.raises(AssertionError):
                assert_frames_equal(next_frame, reference_frame6.to(device))

    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frame_at_pts_audio(self, seek_mode):
        decoder = create_from_file(str(NASA_AUDIO.path), seek_mode=seek_mode)
        add_audio_stream(decoder=decoder)
        # This frame has pts=6.016 and duration=0.064 , so it should be played
        # at timestamps in the range [6.016, 6.08) (not including the last timestamp).
        frame6, _, _ = get_frame_at_pts(decoder, 6.016)
        reference_frame6 = NASA_AUDIO.get_frame_data_by_index(
            INDEX_OF_AUDIO_FRAME_AFTER_SEEKING_AT_6
        )
        assert_frames_equal(frame6, reference_frame6)
        frame6, _, _ = get_frame_at_pts(decoder, 6.05)
        assert_frames_equal(frame6, reference_frame6)
        frame6, _, _ = get_frame_at_pts(decoder, 6.07999)
        assert_frames_equal(frame6, reference_frame6)
        # Note that this timestamp is exactly on a frame boundary, so it should
        # return the next frame since the right boundary of the interval is
        # open.
        next_frame, _, _ = get_frame_at_pts(decoder, 6.08)
        with pytest.raises(AssertionError):
            assert_frames_equal(next_frame, reference_frame6)

    def test_get_frame_at_pts_audio_bad(self):
        decoder = create_from_file(str(NASA_AUDIO.path))
        add_audio_stream(decoder=decoder)

        reference_frame6 = NASA_AUDIO.get_frame_data_by_index(
            INDEX_OF_AUDIO_FRAME_AFTER_SEEKING_AT_6
        )
        frame6, _, _ = get_frame_at_pts(decoder, 6.05)
        # See Note [Seek offset for audio].
        # The frame played at 6.05 should be the reference frame, but because
        # 6.05 isn't exactly the beginning of that frame, the samples are
        # decoded incorrectly.
        # TODO Fix this.
        with pytest.raises(AssertionError):
            assert_frames_equal(frame6, reference_frame6)

        # And yet another quirk: if we try to decode it again, we actually end
        # up with the samples being correctly decoded. This is because we have a
        # custom logic within getFramePlayedAt() that resets desiredPts to the
        # pts of the beginning of the frame in some very specific cases.
        frame6, _, _ = get_frame_at_pts(decoder, 6.05)
        assert_frames_equal(frame6, reference_frame6)

    @pytest.mark.parametrize("test_ref", (NASA_VIDEO, NASA_AUDIO))
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frame_at_index(self, test_ref, device, seek_mode):
        if device == "cuda" and test_ref is NASA_AUDIO:
            pytest.skip(reason="CUDA decoding not supported for audio")

        decoder = create_from_file(str(test_ref.path), seek_mode=seek_mode)
        _add_stream(decoder=decoder, test_ref=test_ref, device=device)
        frame0, _, _ = get_frame_at_index(decoder, frame_index=0)
        reference_frame0 = test_ref.get_frame_data_by_index(0)
        assert_frames_equal(frame0, reference_frame0.to(device))
        # The frame that is played at 6 seconds is frame 180 from a 0-based index.
        frame6, _, _ = get_frame_at_index(decoder, frame_index=180)
        reference_frame6 = test_ref.get_frame_data_by_index(180)
        assert_frames_equal(frame6, reference_frame6.to(device))

    @pytest.mark.parametrize(
        "test_ref, expected_pts, expected_duration",
        (
            (NASA_VIDEO, 6.006, 0.03337),
            (NASA_AUDIO, 11.52, 0.064),
        ),
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frame_with_info_at_index(
        self, test_ref, expected_pts, expected_duration, device, seek_mode
    ):
        if device == "cuda" and test_ref is NASA_AUDIO:
            pytest.skip(reason="CUDA decoding not supported for audio")
        decoder = create_from_file(str(test_ref.path), seek_mode=seek_mode)
        _add_stream(decoder=decoder, test_ref=test_ref, device=device)
        frame6, pts, duration = get_frame_at_index(decoder, frame_index=180)
        reference_frame6 = test_ref.get_frame_data_by_index(180)
        assert_frames_equal(frame6, reference_frame6.to(device))
        assert pts.item() == pytest.approx(expected_pts, rel=1e-3)
        assert duration.item() == pytest.approx(expected_duration, rel=1e-3)

    @pytest.mark.parametrize("test_ref", (NASA_VIDEO, NASA_AUDIO))
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_at_indices(self, test_ref, device, seek_mode):
        if device == "cuda" and test_ref is NASA_AUDIO:
            pytest.skip(reason="CUDA decoding not supported for audio")
        decoder = create_from_file(str(test_ref.path), seek_mode=seek_mode)
        _add_stream(decoder=decoder, test_ref=test_ref, device=device)
        frames0and180, *_ = get_frames_at_indices(decoder, frame_indices=[0, 180])
        reference_frame0 = test_ref.get_frame_data_by_index(0)
        reference_frame180 = test_ref.get_frame_data_by_index(180)

        assert_frames_equal(frames0and180[0], reference_frame0.to(device))
        assert_frames_equal(frames0and180[1], reference_frame180.to(device))

    @pytest.mark.parametrize("test_ref", (NASA_VIDEO, NASA_AUDIO))
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_at_indices_unsorted_indices(self, test_ref, device, seek_mode):
        if device == "cuda" and test_ref is NASA_AUDIO:
            pytest.skip(reason="CUDA decoding not supported for audio")

        decoder = create_from_file(str(test_ref.path), seek_mode=seek_mode)
        _add_stream(decoder=decoder, test_ref=test_ref, device=device)

        frame_indices = [2, 0, 1, 0, 2]

        expected_frames = [
            get_frame_at_index(decoder, frame_index=frame_index)[0]
            for frame_index in frame_indices
        ]

        frames, *_ = get_frames_at_indices(
            decoder,
            frame_indices=frame_indices,
        )
        for frame, expected_frame in zip(frames, expected_frames):
            assert_frames_equal(frame, expected_frame)

        # first and last frame should be equal, at index 2. We then modify the
        # first frame and assert that it's now different from the last frame.
        # This ensures a copy was properly made during the de-duplication logic.
        assert_frames_equal(frames[0], frames[-1])
        frames[0] += 20
        with pytest.raises(AssertionError):
            assert_frames_equal(frames[0], frames[-1])

    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_by_pts(self, device, seek_mode):
        decoder = create_from_file(str(NASA_VIDEO.path), seek_mode=seek_mode)
        _add_video_stream(decoder=decoder, device=device)

        # Note: 13.01 should give the last video frame for the NASA video
        timestamps = [2, 0, 1, 0 + 1e-3, 13.01, 2 + 1e-3]

        expected_frames = [
            get_frame_at_pts(decoder, seconds=pts)[0] for pts in timestamps
        ]
        frames, *_ = get_frames_by_pts(
            decoder,
            timestamps=timestamps,
        )

        for frame, expected_frame in zip(frames, expected_frames):
            assert_frames_equal(frame, expected_frame)

        # first and last frame should be equal, at pts=2 [+ eps]. We then modify
        # the first frame and assert that it's now different from the last
        # frame. This ensures a copy was properly made during the de-duplication
        # logic.
        assert_frames_equal(frames[0], frames[-1])
        frames[0] += 20
        with pytest.raises(AssertionError):
            assert_frames_equal(frames[0], frames[-1])

    @pytest.mark.parametrize("test_ref", (NASA_VIDEO, NASA_AUDIO))
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_pts_apis_against_index_ref(self, test_ref, device):
        # Non-regression test for https://github.com/pytorch/torchcodec/pull/287
        # Get all frames in the video, then query all frames with all time-based
        # APIs exactly where those frames are supposed to start. We assert that
        # we get the expected frame.
        if device == "cuda" and test_ref is NASA_AUDIO:
            pytest.skip(reason="CUDA decoding not supported for audio")
        decoder = create_from_file(str(test_ref.path))
        _add_stream(decoder=decoder, test_ref=test_ref, device=device)

        metadata = get_json_metadata(decoder)
        metadata_dict = json.loads(metadata)
        num_frames = metadata_dict["numFrames"]
        assert num_frames == 390
        if test_ref is NASA_AUDIO:
            num_frames = 204

        _, all_pts_seconds_ref, _ = zip(
            *[
                get_frame_at_index(decoder, frame_index=frame_index)
                for frame_index in range(num_frames)
            ]
        )
        all_pts_seconds_ref = torch.tensor(all_pts_seconds_ref)

        assert len(all_pts_seconds_ref.unique() == len(all_pts_seconds_ref))

        _, pts_seconds, _ = zip(
            *[get_frame_at_pts(decoder, seconds=pts) for pts in all_pts_seconds_ref]
        )
        pts_seconds = torch.tensor(pts_seconds)
        torch.testing.assert_close(pts_seconds, all_pts_seconds_ref, atol=0, rtol=0)

        _, pts_seconds, _ = get_frames_by_pts_in_range(
            decoder,
            start_seconds=0,
            stop_seconds=all_pts_seconds_ref[-1] + 1e-4,
        )
        torch.testing.assert_close(pts_seconds, all_pts_seconds_ref, atol=0, rtol=0)

        _, pts_seconds, _ = zip(
            *[
                get_frames_by_pts_in_range(
                    decoder,
                    start_seconds=pts,
                    stop_seconds=pts + 1e-4,
                )
                for pts in all_pts_seconds_ref
            ]
        )
        pts_seconds = torch.tensor(pts_seconds)
        torch.testing.assert_close(pts_seconds, all_pts_seconds_ref, atol=0, rtol=0)

        _, pts_seconds, _ = get_frames_by_pts(
            decoder, timestamps=all_pts_seconds_ref.tolist()
        )
        torch.testing.assert_close(pts_seconds, all_pts_seconds_ref, atol=0, rtol=0)

    @pytest.mark.parametrize("test_ref", (NASA_VIDEO, NASA_AUDIO))
    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_in_range(self, test_ref, device, seek_mode):
        if device == "cuda" and test_ref is NASA_AUDIO:
            pytest.skip(reason="CUDA decoding not supported for audio")
        decoder = create_from_file(str(test_ref.path), seek_mode=seek_mode)
        _add_stream(decoder=decoder, test_ref=test_ref, device=device)

        # ensure that the degenerate case of a range of size 1 works
        ref_frame0 = test_ref.get_frame_data_by_range(0, 1)
        bulk_frame0, *_ = get_frames_in_range(decoder, start=0, stop=1)
        assert_frames_equal(bulk_frame0, ref_frame0.to(device))

        ref_frame1 = test_ref.get_frame_data_by_range(1, 2)
        bulk_frame1, *_ = get_frames_in_range(decoder, start=1, stop=2)
        assert_frames_equal(bulk_frame1, ref_frame1.to(device))

        last_index = 389 if test_ref is NASA_VIDEO else 203  # TODO ew
        ref_frame389 = test_ref.get_frame_data_by_range(last_index, last_index + 1)
        bulk_frame389, *_ = get_frames_in_range(
            decoder, start=last_index, stop=last_index + 1
        )
        assert_frames_equal(bulk_frame389, ref_frame389.to(device))

        # contiguous ranges
        ref_frames0_9 = test_ref.get_frame_data_by_range(0, 9)
        bulk_frames0_9, *_ = get_frames_in_range(decoder, start=0, stop=9)
        assert_frames_equal(bulk_frames0_9, ref_frames0_9.to(device))

        ref_frames4_8 = test_ref.get_frame_data_by_range(4, 8)
        bulk_frames4_8, *_ = get_frames_in_range(decoder, start=4, stop=8)
        assert_frames_equal(bulk_frames4_8, ref_frames4_8.to(device))

        # ranges with a stride
        ref_frames15_35 = test_ref.get_frame_data_by_range(15, 36, 5)
        bulk_frames15_35, *_ = get_frames_in_range(decoder, start=15, stop=36, step=5)
        assert_frames_equal(bulk_frames15_35, ref_frames15_35.to(device))

        ref_frames0_9_2 = test_ref.get_frame_data_by_range(0, 9, 2)
        bulk_frames0_9_2, *_ = get_frames_in_range(decoder, start=0, stop=9, step=2)
        assert_frames_equal(bulk_frames0_9_2, ref_frames0_9_2.to(device))

        # an empty range is valid!
        empty_frame, *_ = get_frames_in_range(decoder, start=5, stop=5)
        assert_frames_equal(empty_frame, test_ref.empty_chw_tensor.to(device))

    @pytest.mark.parametrize(
        "test_ref, last_frame_index", ((NASA_VIDEO, 289), (NASA_AUDIO, 203))
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_throws_exception_at_eof(self, test_ref, last_frame_index, device):
        if device == "cuda" and test_ref is NASA_AUDIO:
            pytest.skip(reason="CUDA decoding not supported for audio")

        decoder = create_from_file(str(test_ref.path))
        _add_stream(decoder=decoder, test_ref=test_ref, device=device)
        seek_to_pts(decoder, 12.979633)
        last_frame, _, _ = get_next_frame(decoder)
        reference_last_frame = test_ref.get_frame_data_by_index(last_frame_index)
        assert_frames_equal(last_frame, reference_last_frame.to(device))
        with pytest.raises(IndexError, match="no more frames"):
            get_next_frame(decoder)

    @pytest.mark.parametrize(
        "test_ref, seek_offset", ((NASA_VIDEO, 1e-4), (NASA_AUDIO, 1e-1))
    )
    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_throws_exception_if_seek_too_far(self, test_ref, seek_offset, device):
        if device == "cuda" and test_ref is NASA_AUDIO:
            pytest.skip(reason="CUDA decoding not supported for audio")
        decoder = create_from_file(str(test_ref.path))
        _add_stream(decoder=decoder, test_ref=test_ref, device=device)
        # pts=12.979633 is the last frame in the video.
        seek_to_pts(decoder, 12.979633 + seek_offset)
        with pytest.raises(IndexError, match="no more frames"):
            get_next_frame(decoder)

    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_compile_seek_and_next(self, device):
        # TODO_OPEN_ISSUE Scott (T180277797): Get this to work with the inductor stack. Right now
        # compilation fails because it can't handle tensors of size unknown at
        # compile-time.
        @torch.compile(fullgraph=True, backend="eager")
        def get_frame1_and_frame_time6(decoder):
            add_video_stream(decoder, device=device)
            frame0, _, _ = get_next_frame(decoder)
            seek_to_pts(decoder, 6.0)
            frame_time6, _, _ = get_next_frame(decoder)
            return frame0, frame_time6

        # NB: create needs to happen outside the torch.compile region,
        # for now. Otherwise torch.compile constant-props it.
        decoder = create_from_file(str(NASA_VIDEO.path))
        frame0, frame_time6 = get_frame1_and_frame_time6(decoder)
        reference_frame0 = NASA_VIDEO.get_frame_data_by_index(0)
        reference_frame_time6 = NASA_VIDEO.get_frame_data_by_index(
            INDEX_OF_VIDEO_FRAME_AFTER_SEEKING_AT_6
        )
        assert_frames_equal(frame0, reference_frame0.to(device))
        assert_frames_equal(frame_time6, reference_frame_time6.to(device))

    @pytest.mark.parametrize("device", cpu_and_cuda())
    def test_class_based_compile_seek_and_next(self, device):
        # TODO_OPEN_ISSUE Scott (T180277797): Ditto as above.
        @torch.compile(fullgraph=True, backend="eager")
        def class_based_get_frame1_and_frame_time6(
            decoder: ReferenceDecoder,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            frame0, _, _ = decoder.get_next_frame()
            decoder.seek(6.0)
            frame_time6, _, _ = decoder.get_next_frame()
            return frame0, frame_time6

        decoder = ReferenceDecoder(device=device)
        frame0, frame_time6 = class_based_get_frame1_and_frame_time6(decoder)
        reference_frame0 = NASA_VIDEO.get_frame_data_by_index(0)
        reference_frame_time6 = NASA_VIDEO.get_frame_data_by_index(
            INDEX_OF_VIDEO_FRAME_AFTER_SEEKING_AT_6
        )
        assert_frames_equal(frame0, reference_frame0.to(device))
        assert_frames_equal(frame_time6, reference_frame_time6.to(device))

    @pytest.mark.parametrize("device", cpu_and_cuda())
    @pytest.mark.parametrize("create_from", ("file", "tensor", "bytes"))
    def test_create_decoder(self, create_from, device):
        path = str(NASA_VIDEO.path)
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

        add_video_stream(decoder, device=device)
        frame0, _, _ = get_next_frame(decoder)
        reference_frame0 = NASA_VIDEO.get_frame_data_by_index(0)
        assert_frames_equal(frame0, reference_frame0.to(device))
        reference_frame1 = NASA_VIDEO.get_frame_data_by_index(1)
        frame1, _, _ = get_next_frame(decoder)
        assert_frames_equal(frame1, reference_frame1.to(device))
        seek_to_pts(decoder, 6.0)
        frame_time6, _, _ = get_next_frame(decoder)
        reference_frame_time6 = NASA_VIDEO.get_frame_data_by_index(
            INDEX_OF_VIDEO_FRAME_AFTER_SEEKING_AT_6
        )
        assert_frames_equal(frame_time6, reference_frame_time6.to(device))

    # Keeping the metadata tests below for now, but we should remove them
    # once we remove get_json_metadata().
    def test_video_get_json_metadata(self):
        decoder = create_from_file(str(NASA_VIDEO.path))
        metadata = get_json_metadata(decoder)
        metadata_dict = json.loads(metadata)

        # We should be able to see all of this metadata without adding a video stream
        assert metadata_dict["durationSeconds"] == pytest.approx(13.013, abs=0.001)
        assert metadata_dict["numFrames"] == 390
        assert metadata_dict["averageFps"] == pytest.approx(29.97, abs=0.001)
        assert metadata_dict["codec"] == "h264"
        ffmpeg_dict = get_ffmpeg_library_versions()
        if ffmpeg_dict["libavformat"][0] >= 60:
            assert metadata_dict["bitRate"] == 412365.0
        else:
            assert metadata_dict["bitRate"] == 324915.0

    def test_video_get_json_metadata_with_stream(self):
        decoder = create_from_file(str(NASA_VIDEO.path))
        add_video_stream(decoder)
        metadata = get_json_metadata(decoder)
        metadata_dict = json.loads(metadata)
        assert metadata_dict["width"] == 480
        assert metadata_dict["height"] == 270
        assert metadata_dict["minPtsSecondsFromScan"] == 0
        assert metadata_dict["maxPtsSecondsFromScan"] == 13.013

    # TODO: Not sure whether this test still makes a lot of sense
    def test_audio_get_json_metadata(self):
        decoder = create_from_file(str(NASA_AUDIO.path))
        metadata = get_json_metadata(decoder)
        metadata_dict = json.loads(metadata)
        assert metadata_dict["durationSeconds"] == pytest.approx(13.013, abs=0.01)

    def test_get_ffmpeg_version(self):
        ffmpeg_dict = get_ffmpeg_library_versions()
        assert len(ffmpeg_dict["libavcodec"]) == 3
        assert len(ffmpeg_dict["libavfilter"]) == 3
        assert len(ffmpeg_dict["libavformat"]) == 3
        assert len(ffmpeg_dict["libavutil"]) == 3
        # The earliest libavutil version is 50 as per:
        # https://www.ffmpeg.org/olddownload.html
        assert ffmpeg_dict["libavutil"][0] > 50
        assert "ffmpeg_version" in ffmpeg_dict

    @pytest.mark.parametrize(
        "test_ref, num_frames", ((NASA_VIDEO, 390), (NASA_AUDIO, 204))
    )
    def test_frame_pts_equality(self, test_ref, num_frames):
        decoder = create_from_file(str(test_ref.path))
        _add_stream(decoder=decoder, test_ref=test_ref)

        # Note that for all of these tests, we store the return value of
        # _test_frame_pts_equality() into a boolean variable, and then do the assertion
        # on that variable. This indirection is necessary because if we do the assertion
        # directly on the call to _test_frame_pts_equality(), and it returns False making
        # the assertion fail, we get a PyTorch segfault.

        # If this fails, there's a good chance that we accidentally truncated a 64-bit
        # floating point value to a 32-bit floating value.
        for i in range(num_frames):
            _, pts, _ = get_frame_at_index(decoder, frame_index=i)
            pts_is_equal = _test_frame_pts_equality(
                decoder, frame_index=i, pts_seconds_to_test=pts.item()
            )
            assert pts_is_equal

    @pytest.mark.parametrize("color_conversion_library", ("filtergraph", "swscale"))
    def test_color_conversion_library(self, color_conversion_library):
        decoder = create_from_file(str(NASA_VIDEO.path))
        _add_video_stream(decoder, color_conversion_library=color_conversion_library)
        frame0, *_ = get_next_frame(decoder)
        reference_frame0 = NASA_VIDEO.get_frame_data_by_index(0)
        assert_frames_equal(frame0, reference_frame0)
        reference_frame1 = NASA_VIDEO.get_frame_data_by_index(1)
        frame1, *_ = get_next_frame(decoder)
        assert_frames_equal(frame1, reference_frame1)
        seek_to_pts(decoder, 6.0)
        frame_time6, *_ = get_next_frame(decoder)
        reference_frame_time6 = NASA_VIDEO.get_frame_data_by_index(
            INDEX_OF_VIDEO_FRAME_AFTER_SEEKING_AT_6
        )
        assert_frames_equal(frame_time6, reference_frame_time6)

    # We choose arbitrary values for width and height scaling to get better
    # test coverage. Some pairs upscale the image while others downscale it.
    @pytest.mark.parametrize(
        "width_scaling_factor,height_scaling_factor",
        ((1.31, 1.5), (0.71, 0.5), (1.31, 0.7), (0.71, 1.5), (1.0, 1.0)),
    )
    @pytest.mark.parametrize("input_video", [NASA_VIDEO])
    def test_color_conversion_library_with_scaling(
        self, input_video, width_scaling_factor, height_scaling_factor
    ):
        decoder = create_from_file(str(input_video.path))
        add_video_stream(decoder)
        metadata = get_json_metadata(decoder)
        metadata_dict = json.loads(metadata)
        assert metadata_dict["width"] == input_video.width
        assert metadata_dict["height"] == input_video.height

        target_height = int(input_video.height * height_scaling_factor)
        target_width = int(input_video.width * width_scaling_factor)
        if width_scaling_factor != 1.0:
            assert target_width != input_video.width
        if height_scaling_factor != 1.0:
            assert target_height != input_video.height

        filtergraph_decoder = create_from_file(str(input_video.path))
        _add_video_stream(
            filtergraph_decoder,
            width=target_width,
            height=target_height,
            color_conversion_library="filtergraph",
        )
        filtergraph_frame0, _, _ = get_next_frame(filtergraph_decoder)

        swscale_decoder = create_from_file(str(input_video.path))
        _add_video_stream(
            swscale_decoder,
            width=target_width,
            height=target_height,
            color_conversion_library="swscale",
        )
        swscale_frame0, _, _ = get_next_frame(swscale_decoder)
        assert_frames_equal(filtergraph_frame0, swscale_frame0)

    @pytest.mark.parametrize("dimension_order", ("NHWC", "NCHW"))
    @pytest.mark.parametrize("color_conversion_library", ("filtergraph", "swscale"))
    def test_color_conversion_library_with_dimension_order(
        self, dimension_order, color_conversion_library
    ):
        decoder = create_from_file(str(NASA_VIDEO.path))
        _add_video_stream(
            decoder,
            color_conversion_library=color_conversion_library,
            dimension_order=dimension_order,
        )

        frame0_ref = NASA_VIDEO.get_frame_data_by_index(0)
        if dimension_order == "NHWC":
            frame0_ref = frame0_ref.permute(1, 2, 0)
        expected_shape = frame0_ref.shape

        frame0, *_ = get_frame_at_index(decoder, frame_index=0)
        assert frame0.shape == expected_shape
        assert_frames_equal(frame0, frame0_ref)

        frame0, *_ = get_frame_at_pts(decoder, seconds=0.0)
        assert frame0.shape == expected_shape
        assert_frames_equal(frame0, frame0_ref)

        frames, *_ = get_frames_in_range(decoder, start=0, stop=3)
        assert frames.shape[1:] == expected_shape
        assert_frames_equal(frames[0], frame0_ref)

        frames, *_ = get_frames_by_pts_in_range(
            decoder, start_seconds=0, stop_seconds=1
        )
        assert frames.shape[1:] == expected_shape
        assert_frames_equal(frames[0], frame0_ref)

        frames, *_ = get_frames_at_indices(decoder, frame_indices=[0, 1, 3, 4])
        assert frames.shape[1:] == expected_shape
        assert_frames_equal(frames[0], frame0_ref)

    @pytest.mark.parametrize(
        "width_scaling_factor,height_scaling_factor",
        ((1.31, 1.5), (0.71, 0.5), (1.31, 0.7), (0.71, 1.5), (1.0, 1.0)),
    )
    @pytest.mark.parametrize("width", [30, 32, 300])
    @pytest.mark.parametrize("height", [128])
    def test_color_conversion_library_with_generated_videos(
        self, tmp_path, width, height, width_scaling_factor, height_scaling_factor
    ):
        ffmpeg_cli = "ffmpeg"
        if os.environ.get("IN_FBCODE_TORCHCODEC") == "1":
            import importlib.resources

            ffmpeg_cli = importlib.resources.path(__package__, "ffmpeg")
        # We consider filtergraph to be the reference color conversion library.
        # However the video decoder sometimes uses swscale as that is faster.
        # The exact color conversion library used is an implementation detail
        # of the video decoder and depends on the video's width.
        #
        # In this test we compare the output of filtergraph (which is the
        # reference) with the output of the video decoder (which may use
        # swscale if it chooses for certain video widths) to make sure they are
        # always the same.
        video_path = f"{tmp_path}/frame_numbers_{width}x{height}.mp4"
        # We don't specify a particular encoder because the ffmpeg binary could
        # be configured with different encoders. For the purposes of this test,
        # the actual encoder is irrelevant.
        command = [
            ffmpeg_cli,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=blue",
            "-pix_fmt",
            "yuv420p",
            "-s",
            f"{width}x{height}",
            "-frames:v",
            "1",
            video_path,
        ]
        subprocess.check_call(command)

        decoder = create_from_file(str(video_path))
        add_video_stream(decoder)
        metadata = get_json_metadata(decoder)
        metadata_dict = json.loads(metadata)
        assert metadata_dict["width"] == width
        assert metadata_dict["height"] == height

        target_height = int(height * height_scaling_factor)
        target_width = int(width * width_scaling_factor)
        if width_scaling_factor != 1.0:
            assert target_width != width
        if height_scaling_factor != 1.0:
            assert target_height != height

        filtergraph_decoder = create_from_file(str(video_path))
        _add_video_stream(
            filtergraph_decoder,
            width=target_width,
            height=target_height,
            color_conversion_library="filtergraph",
        )
        filtergraph_frame0, _, _ = get_next_frame(filtergraph_decoder)

        auto_decoder = create_from_file(str(video_path))
        add_video_stream(
            auto_decoder,
            width=target_width,
            height=target_height,
        )
        auto_frame0, _, _ = get_next_frame(auto_decoder)
        assert_frames_equal(filtergraph_frame0, auto_frame0)

    @needs_cuda
    def test_cuda_decoder(self):
        decoder = create_from_file(str(NASA_VIDEO.path))
        add_video_stream(decoder, device="cuda")
        frame0, pts, duration = get_next_frame(decoder)
        assert frame0.device.type == "cuda"
        reference_frame0 = NASA_VIDEO.get_frame_data_by_index(0)
        assert_frames_equal(frame0, reference_frame0.to("cuda"))
        assert pts == torch.tensor([0])
        torch.testing.assert_close(
            duration, torch.tensor(0.0334).double(), atol=0, rtol=1e-3
        )

    def test_get_same_frame_twice(self):
        # Non-regression tests that were useful while developing audio support.
        def make_decoder():
            decoder = create_from_file(str(NASA_AUDIO.path))
            add_audio_stream(decoder)
            return decoder

        for frame_index in (0, 10, 15):
            ref = NASA_AUDIO.get_frame_data_by_index(frame_index)

            decoder = make_decoder()
            a = get_frame_at_index(decoder, frame_index=frame_index)
            b = get_frame_at_index(decoder, frame_index=frame_index)
            torch.testing.assert_close(a, b)
            torch.testing.assert_close(a[0], ref)

            decoder = make_decoder()
            a = get_frames_at_indices(decoder, frame_indices=[frame_index])
            b = get_frames_at_indices(decoder, frame_indices=[frame_index])
            torch.testing.assert_close(a, b)
            torch.testing.assert_close(a[0][0], ref)

            decoder = make_decoder()
            a = get_frames_in_range(decoder, start=frame_index, stop=frame_index + 1)
            b = get_frames_in_range(decoder, start=frame_index, stop=frame_index + 1)
            torch.testing.assert_close(a, b)
            torch.testing.assert_close(a[0][0], ref)

        pts_at_frame_start = 0  # 0 corresponds exactly to a frame start
        index_of_frame_at_0 = 0
        pts_not_at_frame_start = 2  # second 2 is in the middle of a frame
        index_of_frame_at_2 = 31
        for pts, frame_index in (
            (pts_at_frame_start, index_of_frame_at_0),
            (pts_not_at_frame_start, index_of_frame_at_2),
        ):
            ref = NASA_AUDIO.get_frame_data_by_index(frame_index)

            decoder = make_decoder()
            a = get_frames_by_pts(decoder, timestamps=[pts])
            b = get_frames_by_pts(decoder, timestamps=[pts])
            torch.testing.assert_close(a, b)
            torch.testing.assert_close(a[0][0], ref)

            decoder = make_decoder()
            a = get_frames_by_pts_in_range(
                decoder, start_seconds=pts, stop_seconds=pts + 1e-4
            )
            b = get_frames_by_pts_in_range(
                decoder, start_seconds=pts, stop_seconds=pts + 1e-4
            )
            torch.testing.assert_close(a, b)
            torch.testing.assert_close(a[0][0], ref)

        decoder = make_decoder()
        a = get_frame_at_pts(decoder, seconds=pts_at_frame_start)
        b = get_frame_at_pts(decoder, seconds=pts_at_frame_start)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(
            a[0], NASA_AUDIO.get_frame_data_by_index(index_of_frame_at_0)
        )

        decoder = make_decoder()
        a_frame, a_pts, a_duration = get_frame_at_pts(
            decoder, seconds=pts_not_at_frame_start
        )
        b_frame, b_pts, b_duration = get_frame_at_pts(
            decoder, seconds=pts_not_at_frame_start
        )
        torch.testing.assert_close(a_pts, b_pts)
        torch.testing.assert_close(a_duration, b_duration)
        # TODO fix this. These checks should pass
        with pytest.raises(AssertionError):
            torch.testing.assert_close(a_frame, b_frame)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(
                a_frame, NASA_AUDIO.get_frame_data_by_index(index_of_frame_at_2)
            )
        # But second time works ¯\_(ツ)_/¯A (see also test_get_frame_at_pts_audio_bad())
        torch.testing.assert_close(
            b_frame, NASA_AUDIO.get_frame_data_by_index(index_of_frame_at_2)
        )

        decoder = make_decoder()
        seek_to_pts(decoder, pts_at_frame_start)
        a = get_next_frame(decoder)
        seek_to_pts(decoder, pts_at_frame_start)
        b = get_next_frame(decoder)
        torch.testing.assert_close(a, b)

        decoder = make_decoder()
        seek_to_pts(decoder, seconds=pts_not_at_frame_start)
        a = get_next_frame(decoder)
        seek_to_pts(decoder, seconds=pts_not_at_frame_start)
        b = get_next_frame(decoder)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(
            a[0], NASA_AUDIO.get_frame_data_by_index(index_of_frame_at_2 + 1)
        )


if __name__ == "__main__":
    pytest.main()
