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
    scan_all_streams_to_update_metadata,
    seek_to_pts,
)

from ..utils import assert_tensor_equal, NASA_AUDIO, NASA_VIDEO, needs_cuda

torch._dynamo.config.capture_dynamic_output_shape_ops = True


class ReferenceDecoder:
    def __init__(self):
        self.decoder: torch.Tensor = create_from_file(str(NASA_VIDEO.path))
        add_video_stream(self.decoder)

    def get_next_frame(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.decoder is not None
        return get_next_frame(self.decoder)

    def seek(self, pts: float):
        assert self.decoder is not None
        seek_to_pts(self.decoder, pts)


class TestOps:
    def test_seek_and_next(self):
        decoder = create_from_file(str(NASA_VIDEO.path))
        add_video_stream(decoder)
        frame0, _, _ = get_next_frame(decoder)
        reference_frame0 = NASA_VIDEO.get_frame_data_by_index(0)
        assert_tensor_equal(frame0, reference_frame0)
        reference_frame1 = NASA_VIDEO.get_frame_data_by_index(1)
        frame1, _, _ = get_next_frame(decoder)
        assert_tensor_equal(frame1, reference_frame1)
        seek_to_pts(decoder, 6.0)
        frame_time6, _, _ = get_next_frame(decoder)
        reference_frame_time6 = NASA_VIDEO.get_frame_by_name("time6.000000")
        assert_tensor_equal(frame_time6, reference_frame_time6)

    def test_get_frame_at_pts(self):
        decoder = create_from_file(str(NASA_VIDEO.path))
        add_video_stream(decoder)
        # This frame has pts=6.006 and duration=0.033367, so it should be visible
        # at timestamps in the range [6.006, 6.039367) (not including the last timestamp).
        frame6, _, _ = get_frame_at_pts(decoder, 6.006)
        reference_frame6 = NASA_VIDEO.get_frame_by_name("time6.000000")
        assert_tensor_equal(frame6, reference_frame6)
        frame6, _, _ = get_frame_at_pts(decoder, 6.02)
        assert_tensor_equal(frame6, reference_frame6)
        frame6, _, _ = get_frame_at_pts(decoder, 6.039366)
        assert_tensor_equal(frame6, reference_frame6)
        # Note that this timestamp is exactly on a frame boundary, so it should
        # return the next frame since the right boundary of the interval is
        # open.
        next_frame, _, _ = get_frame_at_pts(decoder, 6.039367)
        with pytest.raises(AssertionError):
            assert_tensor_equal(next_frame, reference_frame6)

    def test_get_frame_at_index(self):
        decoder = create_from_file(str(NASA_VIDEO.path))
        scan_all_streams_to_update_metadata(decoder)
        add_video_stream(decoder)
        frame0, _, _ = get_frame_at_index(decoder, stream_index=3, frame_index=0)
        reference_frame0 = NASA_VIDEO.get_frame_data_by_index(0)
        assert_tensor_equal(frame0, reference_frame0)
        # The frame that is displayed at 6 seconds is frame 180 from a 0-based index.
        frame6, _, _ = get_frame_at_index(decoder, stream_index=3, frame_index=180)
        reference_frame6 = NASA_VIDEO.get_frame_by_name("time6.000000")
        assert_tensor_equal(frame6, reference_frame6)

    def test_get_frame_with_info_at_index(self):
        decoder = create_from_file(str(NASA_VIDEO.path))
        scan_all_streams_to_update_metadata(decoder)
        add_video_stream(decoder)
        frame6, pts, duration = get_frame_at_index(
            decoder, stream_index=3, frame_index=180
        )
        reference_frame6 = NASA_VIDEO.get_frame_by_name("time6.000000")
        assert_tensor_equal(frame6, reference_frame6)
        assert pts.item() == pytest.approx(6.006, rel=1e-3)
        assert duration.item() == pytest.approx(0.03337, rel=1e-3)

    def test_get_frames_at_indices(self):
        decoder = create_from_file(str(NASA_VIDEO.path))
        scan_all_streams_to_update_metadata(decoder)
        add_video_stream(decoder)
        frames0and180, *_ = get_frames_at_indices(
            decoder, stream_index=3, frame_indices=[0, 180]
        )
        reference_frame0 = NASA_VIDEO.get_frame_data_by_index(0)
        reference_frame180 = NASA_VIDEO.get_frame_by_name("time6.000000")
        assert_tensor_equal(frames0and180[0], reference_frame0)
        assert_tensor_equal(frames0and180[1], reference_frame180)

    def test_get_frames_at_indices_unsorted_indices(self):
        decoder = create_from_file(str(NASA_VIDEO.path))
        _add_video_stream(decoder)
        scan_all_streams_to_update_metadata(decoder)
        stream_index = 3

        frame_indices = [2, 0, 1, 0, 2]

        expected_frames = [
            get_frame_at_index(
                decoder, stream_index=stream_index, frame_index=frame_index
            )[0]
            for frame_index in frame_indices
        ]

        frames, *_ = get_frames_at_indices(
            decoder,
            stream_index=stream_index,
            frame_indices=frame_indices,
        )
        for frame, expected_frame in zip(frames, expected_frames):
            assert_tensor_equal(frame, expected_frame)

        # first and last frame should be equal, at index 2. We then modify the
        # first frame and assert that it's now different from the last frame.
        # This ensures a copy was properly made during the de-duplication logic.
        assert_tensor_equal(frames[0], frames[-1])
        frames[0] += 20
        with pytest.raises(AssertionError):
            assert_tensor_equal(frames[0], frames[-1])

    def test_get_frames_by_pts(self):
        decoder = create_from_file(str(NASA_VIDEO.path))
        _add_video_stream(decoder)
        scan_all_streams_to_update_metadata(decoder)
        stream_index = 3

        # Note: 13.01 should give the last video frame for the NASA video
        timestamps = [2, 0, 1, 0 + 1e-3, 13.01, 2 + 1e-3]

        expected_frames = [
            get_frame_at_pts(decoder, seconds=pts)[0] for pts in timestamps
        ]

        frames, *_ = get_frames_by_pts(
            decoder,
            stream_index=stream_index,
            timestamps=timestamps,
        )
        for frame, expected_frame in zip(frames, expected_frames):
            assert_tensor_equal(frame, expected_frame)

        # first and last frame should be equal, at pts=2 [+ eps]. We then modify
        # the first frame and assert that it's now different from the last
        # frame. This ensures a copy was properly made during the de-duplication
        # logic.
        assert_tensor_equal(frames[0], frames[-1])
        frames[0] += 20
        with pytest.raises(AssertionError):
            assert_tensor_equal(frames[0], frames[-1])

    def test_pts_apis_against_index_ref(self):
        # Non-regression test for https://github.com/pytorch/torchcodec/pull/286
        # Get all frames in the video, then query all frames with all time-based
        # APIs exactly where those frames are supposed to start. We assert that
        # we get the expected frame.
        decoder = create_from_file(str(NASA_VIDEO.path))
        scan_all_streams_to_update_metadata(decoder)
        add_video_stream(decoder)

        metadata = get_json_metadata(decoder)
        metadata_dict = json.loads(metadata)
        num_frames = metadata_dict["numFrames"]
        assert num_frames == 390

        stream_index = 3
        _, all_pts_seconds_ref, _ = zip(
            *[
                get_frame_at_index(
                    decoder, stream_index=stream_index, frame_index=frame_index
                )
                for frame_index in range(num_frames)
            ]
        )
        all_pts_seconds_ref = torch.tensor(all_pts_seconds_ref)

        assert len(all_pts_seconds_ref.unique() == len(all_pts_seconds_ref))

        _, pts_seconds, _ = zip(
            *[get_frame_at_pts(decoder, seconds=pts) for pts in all_pts_seconds_ref]
        )
        pts_seconds = torch.tensor(pts_seconds)
        assert_tensor_equal(pts_seconds, all_pts_seconds_ref)

        _, pts_seconds, _ = get_frames_by_pts_in_range(
            decoder,
            stream_index=stream_index,
            start_seconds=0,
            stop_seconds=all_pts_seconds_ref[-1] + 1e-4,
        )
        assert_tensor_equal(pts_seconds, all_pts_seconds_ref)

        _, pts_seconds, _ = zip(
            *[
                get_frames_by_pts_in_range(
                    decoder,
                    stream_index=stream_index,
                    start_seconds=pts,
                    stop_seconds=pts + 1e-4,
                )
                for pts in all_pts_seconds_ref
            ]
        )
        pts_seconds = torch.tensor(pts_seconds)
        assert_tensor_equal(pts_seconds, all_pts_seconds_ref)

        _, pts_seconds, _ = get_frames_by_pts(
            decoder, stream_index=stream_index, timestamps=all_pts_seconds_ref.tolist()
        )
        assert_tensor_equal(pts_seconds, all_pts_seconds_ref)

    def test_get_frames_in_range(self):
        decoder = create_from_file(str(NASA_VIDEO.path))
        scan_all_streams_to_update_metadata(decoder)
        add_video_stream(decoder)

        # ensure that the degenerate case of a range of size 1 works
        ref_frame0 = NASA_VIDEO.get_frame_data_by_range(0, 1)
        bulk_frame0, *_ = get_frames_in_range(decoder, stream_index=3, start=0, stop=1)
        assert_tensor_equal(ref_frame0, bulk_frame0)

        ref_frame1 = NASA_VIDEO.get_frame_data_by_range(1, 2)
        bulk_frame1, *_ = get_frames_in_range(decoder, stream_index=3, start=1, stop=2)
        assert_tensor_equal(ref_frame1, bulk_frame1)

        ref_frame389 = NASA_VIDEO.get_frame_data_by_range(389, 390)
        bulk_frame389, *_ = get_frames_in_range(
            decoder, stream_index=3, start=389, stop=390
        )
        assert_tensor_equal(ref_frame389, bulk_frame389)

        # contiguous ranges
        ref_frames0_9 = NASA_VIDEO.get_frame_data_by_range(0, 9)
        bulk_frames0_9, *_ = get_frames_in_range(
            decoder, stream_index=3, start=0, stop=9
        )
        assert_tensor_equal(ref_frames0_9, bulk_frames0_9)

        ref_frames4_8 = NASA_VIDEO.get_frame_data_by_range(4, 8)
        bulk_frames4_8, *_ = get_frames_in_range(
            decoder, stream_index=3, start=4, stop=8
        )
        assert_tensor_equal(ref_frames4_8, bulk_frames4_8)

        # ranges with a stride
        ref_frames15_35 = NASA_VIDEO.get_frame_data_by_range(15, 36, 5)
        bulk_frames15_35, *_ = get_frames_in_range(
            decoder, stream_index=3, start=15, stop=36, step=5
        )
        assert_tensor_equal(ref_frames15_35, bulk_frames15_35)

        ref_frames0_9_2 = NASA_VIDEO.get_frame_data_by_range(0, 9, 2)
        bulk_frames0_9_2, *_ = get_frames_in_range(
            decoder, stream_index=3, start=0, stop=9, step=2
        )
        assert_tensor_equal(ref_frames0_9_2, bulk_frames0_9_2)

        # an empty range is valid!
        empty_frame, *_ = get_frames_in_range(decoder, stream_index=3, start=5, stop=5)
        assert_tensor_equal(empty_frame, NASA_VIDEO.empty_chw_tensor)

    def test_throws_exception_at_eof(self):
        decoder = create_from_file(str(NASA_VIDEO.path))
        add_video_stream(decoder)
        seek_to_pts(decoder, 12.979633)
        last_frame, _, _ = get_next_frame(decoder)
        reference_last_frame = NASA_VIDEO.get_frame_by_name("time12.979633")
        assert_tensor_equal(last_frame, reference_last_frame)
        with pytest.raises(IndexError, match="no more frames"):
            get_next_frame(decoder)

    def test_throws_exception_if_seek_too_far(self):
        decoder = create_from_file(str(NASA_VIDEO.path))
        add_video_stream(decoder)
        # pts=12.979633 is the last frame in the video.
        seek_to_pts(decoder, 12.979633 + 1.0e-4)
        with pytest.raises(IndexError, match="no more frames"):
            get_next_frame(decoder)

    def test_compile_seek_and_next(self):
        # TODO_OPEN_ISSUE Scott (T180277797): Get this to work with the inductor stack. Right now
        # compilation fails because it can't handle tensors of size unknown at
        # compile-time.
        @torch.compile(fullgraph=True, backend="eager")
        def get_frame1_and_frame_time6(decoder):
            add_video_stream(decoder)
            frame0, _, _ = get_next_frame(decoder)
            seek_to_pts(decoder, 6.0)
            frame_time6, _, _ = get_next_frame(decoder)
            return frame0, frame_time6

        # NB: create needs to happen outside the torch.compile region,
        # for now. Otherwise torch.compile constant-props it.
        decoder = create_from_file(str(NASA_VIDEO.path))
        frame0, frame_time6 = get_frame1_and_frame_time6(decoder)
        reference_frame0 = NASA_VIDEO.get_frame_data_by_index(0)
        reference_frame_time6 = NASA_VIDEO.get_frame_by_name("time6.000000")
        assert_tensor_equal(frame0, reference_frame0)
        assert_tensor_equal(frame_time6, reference_frame_time6)

    def test_class_based_compile_seek_and_next(self):
        # TODO_OPEN_ISSUE Scott (T180277797): Ditto as above.
        @torch.compile(fullgraph=True, backend="eager")
        def class_based_get_frame1_and_frame_time6(
            decoder: ReferenceDecoder,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            frame0, _, _ = decoder.get_next_frame()
            decoder.seek(6.0)
            frame_time6, _, _ = decoder.get_next_frame()
            return frame0, frame_time6

        decoder = ReferenceDecoder()
        frame0, frame_time6 = class_based_get_frame1_and_frame_time6(decoder)
        reference_frame0 = NASA_VIDEO.get_frame_data_by_index(0)
        reference_frame_time6 = NASA_VIDEO.get_frame_by_name("time6.000000")
        assert_tensor_equal(frame0, reference_frame0)
        assert_tensor_equal(frame_time6, reference_frame_time6)

    @pytest.mark.parametrize("create_from", ("file", "tensor", "bytes"))
    def test_create_decoder(self, create_from):
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

        add_video_stream(decoder)
        frame0, _, _ = get_next_frame(decoder)
        reference_frame0 = NASA_VIDEO.get_frame_data_by_index(0)
        assert_tensor_equal(frame0, reference_frame0)
        reference_frame1 = NASA_VIDEO.get_frame_data_by_index(1)
        frame1, _, _ = get_next_frame(decoder)
        assert_tensor_equal(frame1, reference_frame1)
        seek_to_pts(decoder, 6.0)
        frame_time6, _, _ = get_next_frame(decoder)
        reference_frame_time6 = NASA_VIDEO.get_frame_by_name("time6.000000")
        assert_tensor_equal(frame_time6, reference_frame_time6)

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
        scan_all_streams_to_update_metadata(decoder)
        add_video_stream(decoder)
        metadata = get_json_metadata(decoder)
        metadata_dict = json.loads(metadata)
        assert metadata_dict["width"] == 480
        assert metadata_dict["height"] == 270
        assert metadata_dict["minPtsSecondsFromScan"] == 0
        assert metadata_dict["maxPtsSecondsFromScan"] == 13.013

    def test_audio_get_json_metadata(self):
        decoder = create_from_file(str(NASA_AUDIO.path))
        metadata = get_json_metadata(decoder)
        metadata_dict = json.loads(metadata)
        assert metadata_dict["durationSeconds"] == pytest.approx(13.25, abs=0.01)

    def test_get_ffmpeg_version(self):
        ffmpeg_dict = get_ffmpeg_library_versions()
        assert len(ffmpeg_dict["libavcodec"]) == 3
        assert len(ffmpeg_dict["libavfilter"]) == 3
        assert len(ffmpeg_dict["libavformat"]) == 3
        assert len(ffmpeg_dict["libavutil"]) == 3
        # The earliest libavutil version is 50 as per:
        # https://www.ffmpeg.org/olddownload.html
        assert ffmpeg_dict["libavutil"][0] > 50
        ffmpeg_version = ffmpeg_dict["ffmpeg_version"]
        split_ffmpeg_version = [int(num) for num in ffmpeg_version.split(".")]
        assert len(split_ffmpeg_version) == 3

    def test_frame_pts_equality(self):
        decoder = create_from_file(str(NASA_VIDEO.path))
        scan_all_streams_to_update_metadata(decoder)
        add_video_stream(decoder)

        # Note that for all of these tests, we store the return value of
        # _test_frame_pts_equality() into a boolean variable, and then do the assertion
        # on that variable. This indirection is necessary because if we do the assertion
        # directly on the call to _test_frame_pts_equality(), and it returns False making
        # the assertion fail, we get a PyTorch segfault.

        # If this fails, there's a good chance that we accidentally truncated a 64-bit
        # floating point value to a 32-bit floating value.
        for i in range(390):
            frame, pts, _ = get_frame_at_index(decoder, stream_index=3, frame_index=i)
            pts_is_equal = _test_frame_pts_equality(
                decoder, stream_index=3, frame_index=i, pts_seconds_to_test=pts.item()
            )
            assert pts_is_equal

    @pytest.mark.parametrize("color_conversion_library", ("filtergraph", "swscale"))
    def test_color_conversion_library(self, color_conversion_library):
        decoder = create_from_file(str(NASA_VIDEO.path))
        _add_video_stream(decoder, color_conversion_library=color_conversion_library)
        frame0, *_ = get_next_frame(decoder)
        reference_frame0 = NASA_VIDEO.get_frame_data_by_index(0)
        assert_tensor_equal(frame0, reference_frame0)
        reference_frame1 = NASA_VIDEO.get_frame_data_by_index(1)
        frame1, *_ = get_next_frame(decoder)
        assert_tensor_equal(frame1, reference_frame1)
        seek_to_pts(decoder, 6.0)
        frame_time6, *_ = get_next_frame(decoder)
        reference_frame_time6 = NASA_VIDEO.get_frame_by_name("time6.000000")
        assert_tensor_equal(frame_time6, reference_frame_time6)

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
        scan_all_streams_to_update_metadata(decoder)
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
        assert_tensor_equal(filtergraph_frame0, swscale_frame0)

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
        scan_all_streams_to_update_metadata(decoder)

        frame0_ref = NASA_VIDEO.get_frame_data_by_index(0)
        if dimension_order == "NHWC":
            frame0_ref = frame0_ref.permute(1, 2, 0)
        expected_shape = frame0_ref.shape

        stream_index = 3
        frame0, *_ = get_frame_at_index(
            decoder, stream_index=stream_index, frame_index=0
        )
        assert frame0.shape == expected_shape
        assert_tensor_equal(frame0, frame0_ref)

        frame0, *_ = get_frame_at_pts(decoder, seconds=0.0)
        assert frame0.shape == expected_shape
        assert_tensor_equal(frame0, frame0_ref)

        frames, *_ = get_frames_in_range(
            decoder, stream_index=stream_index, start=0, stop=3
        )
        assert frames.shape[1:] == expected_shape
        assert_tensor_equal(frames[0], frame0_ref)

        frames, *_ = get_frames_by_pts_in_range(
            decoder, stream_index=stream_index, start_seconds=0, stop_seconds=1
        )
        assert frames.shape[1:] == expected_shape
        assert_tensor_equal(frames[0], frame0_ref)

        frames, *_ = get_frames_at_indices(
            decoder, stream_index=stream_index, frame_indices=[0, 1, 3, 4]
        )
        assert frames.shape[1:] == expected_shape
        assert_tensor_equal(frames[0], frame0_ref)

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
        scan_all_streams_to_update_metadata(decoder)
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
        assert_tensor_equal(filtergraph_frame0, auto_frame0)

    @needs_cuda
    def test_cuda_decoder(self):
        decoder = create_from_file(str(NASA_VIDEO.path))
        scan_all_streams_to_update_metadata(decoder)
        add_video_stream(decoder, device="cuda")
        frame0, pts, duration = get_next_frame(decoder)
        assert frame0.device.type == "cuda"
        frame0_cpu = frame0.to("cpu")
        reference_frame0 = NASA_VIDEO.get_frame_data_by_index(0)
        # GPU decode is not bit-accurate. In the following assertion we ensure
        # not more than 0.3% of values have a difference greater than 20.
        diff = (reference_frame0.float() - frame0_cpu.float()).abs()
        assert (diff > 20).float().mean() <= 0.003
        assert pts == torch.tensor([0])
        torch.testing.assert_close(
            duration, torch.tensor(0.0334).double(), atol=0, rtol=1e-3
        )


if __name__ == "__main__":
    pytest.main()
