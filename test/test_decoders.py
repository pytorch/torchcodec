# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import gc
import json
from unittest.mock import patch

import numpy
import pytest
import torch

from torchcodec import _core, FrameBatch
from torchcodec.decoders import (
    AudioDecoder,
    AudioStreamMetadata,
    VideoDecoder,
    VideoStreamMetadata,
)

from .utils import (
    assert_frames_equal,
    AV1_VIDEO,
    cpu_and_accelerators,
    get_ffmpeg_major_version,
    H264_10BITS,
    H265_10BITS,
    H265_VIDEO,
    in_fbcode,
    NASA_AUDIO,
    NASA_AUDIO_MP3,
    NASA_AUDIO_MP3_44100,
    NASA_VIDEO,
    needs_cuda,
    SINE_MONO_S16,
    SINE_MONO_S32,
    SINE_MONO_S32_44100,
    SINE_MONO_S32_8000,
)


class TestDecoder:
    @pytest.mark.parametrize(
        "Decoder, asset",
        (
            (VideoDecoder, NASA_VIDEO),
            (AudioDecoder, NASA_AUDIO),
            (AudioDecoder, NASA_AUDIO_MP3),
        ),
    )
    @pytest.mark.parametrize(
        "source_kind",
        (
            "str",
            "path",
            "file_like_rawio",
            "file_like_bufferedio",
            "file_like_custom",
            "bytes",
            "tensor",
        ),
    )
    def test_create(self, Decoder, asset, source_kind):
        if source_kind == "str":
            source = str(asset.path)
        elif source_kind == "path":
            source = asset.path
        elif source_kind == "file_like_rawio":
            source = open(asset.path, mode="rb", buffering=0)
        elif source_kind == "file_like_bufferedio":
            source = open(asset.path, mode="rb", buffering=4096)
        elif source_kind == "file_like_custom":
            # This class purposefully does not inherit from io.RawIOBase or
            # io.BufferedReader. We are testing the case when users pass an
            # object that has the right methods but is an arbitrary type.
            class CustomReader:
                def __init__(self, file):
                    self._file = file

                def read(self, size: int) -> bytes:
                    return self._file.read(size)

                def seek(self, offset: int, whence: int) -> bytes:
                    return self._file.seek(offset, whence)

            source = CustomReader(open(asset.path, mode="rb", buffering=0))
        elif source_kind == "bytes":
            path = str(asset.path)
            with open(path, "rb") as f:
                source = f.read()
        elif source_kind == "tensor":
            source = asset.to_tensor()
        else:
            raise ValueError("Oops, double check the parametrization of this test!")

        decoder = Decoder(source)
        assert isinstance(decoder.metadata, _core._metadata.StreamMetadata)

    @pytest.mark.parametrize("Decoder", (VideoDecoder, AudioDecoder))
    def test_create_fails(self, Decoder):
        with pytest.raises(TypeError, match="Unknown source type"):
            Decoder(123)

        # stream index that does not exist
        with pytest.raises(ValueError, match="No valid stream found"):
            Decoder(NASA_VIDEO.path, stream_index=40)

        # stream index that does exist, but it's not audio or video
        with pytest.raises(ValueError, match="No valid stream found"):
            Decoder(NASA_VIDEO.path, stream_index=2)

        # user mistakenly forgets to specify binary reading when creating a file
        # like object from open()
        with pytest.raises(TypeError, match="binary reading?"):
            Decoder(open(NASA_VIDEO.path, "r"))


class TestVideoDecoder:
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_metadata(self, seek_mode):
        decoder = VideoDecoder(NASA_VIDEO.path, seek_mode=seek_mode)
        assert isinstance(decoder.metadata, VideoStreamMetadata)
        assert len(decoder) == decoder._num_frames == 390

        assert decoder.stream_index == decoder.metadata.stream_index == 3
        assert decoder.metadata.duration_seconds == pytest.approx(13.013)
        assert decoder.metadata.average_fps == pytest.approx(29.970029)
        assert decoder.metadata.num_frames == 390
        assert decoder.metadata.height == 270
        assert decoder.metadata.width == 480

    def test_create_bytes_ownership(self):
        # Non-regression test for https://github.com/pytorch/torchcodec/issues/720
        #
        # Note that the bytes object we use to instantiate the decoder does not
        # live past the VideoDecoder destructor. That is what we're testing:
        # that the VideoDecoder takes ownership of the bytes. If it does not,
        # then we will hit errors when we try to actually decode from the bytes
        # later on. By the time we actually decode, the reference on the Python
        # side has gone away, and if we don't have ownership on the C++ side, we
        # will hit runtime errors or segfaults.
        #
        # Also note that if this test fails, OTHER tests will likely
        # mysteriously fail. That's because a failure in this tests likely
        # indicates memory corruption, and the memory we corrupt could easily
        # cause problems in other tests. So if this test fails, fix this test
        # first.
        with open(NASA_VIDEO.path, "rb") as f:
            decoder = VideoDecoder(f.read())

        # Let's ensure that the bytes really go away!
        gc.collect()

        assert decoder[0] is not None
        assert decoder[len(decoder) // 2] is not None
        assert decoder[-1] is not None

    def test_create_fails(self):
        with pytest.raises(ValueError, match="Invalid seek mode"):
            VideoDecoder(NASA_VIDEO.path, seek_mode="blah")

    @pytest.mark.parametrize("num_ffmpeg_threads", (1, 4))
    @pytest.mark.parametrize("device", cpu_and_accelerators())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_getitem_int(self, num_ffmpeg_threads, device, seek_mode):
        decoder = VideoDecoder(
            NASA_VIDEO.path,
            num_ffmpeg_threads=num_ffmpeg_threads,
            device=device,
            seek_mode=seek_mode,
        )

        ref_frame0 = NASA_VIDEO.get_frame_data_by_index(0).to(device)
        ref_frame1 = NASA_VIDEO.get_frame_data_by_index(1).to(device)
        ref_frame180 = NASA_VIDEO.get_frame_data_by_index(180).to(device)
        ref_frame_last = NASA_VIDEO.get_frame_data_by_index(289).to(device)

        assert_frames_equal(ref_frame0, decoder[0])
        assert_frames_equal(ref_frame1, decoder[1])
        assert_frames_equal(ref_frame180, decoder[180])
        assert_frames_equal(ref_frame_last, decoder[-1])

    def test_getitem_numpy_int(self):
        decoder = VideoDecoder(NASA_VIDEO.path)

        ref_frame0 = NASA_VIDEO.get_frame_data_by_index(0)
        ref_frame1 = NASA_VIDEO.get_frame_data_by_index(1)
        ref_frame180 = NASA_VIDEO.get_frame_data_by_index(180)
        ref_frame_last = NASA_VIDEO.get_frame_data_by_index(289)

        # test against numpy.int64
        assert_frames_equal(ref_frame0, decoder[numpy.int64(0)])
        assert_frames_equal(ref_frame1, decoder[numpy.int64(1)])
        assert_frames_equal(ref_frame180, decoder[numpy.int64(180)])
        assert_frames_equal(ref_frame_last, decoder[numpy.int64(-1)])

        # test against numpy.int32
        assert_frames_equal(ref_frame0, decoder[numpy.int32(0)])
        assert_frames_equal(ref_frame1, decoder[numpy.int32(1)])
        assert_frames_equal(ref_frame180, decoder[numpy.int32(180)])
        assert_frames_equal(ref_frame_last, decoder[numpy.int32(-1)])

        # test against numpy.uint64
        assert_frames_equal(ref_frame0, decoder[numpy.uint64(0)])
        assert_frames_equal(ref_frame1, decoder[numpy.uint64(1)])
        assert_frames_equal(ref_frame180, decoder[numpy.uint64(180)])

        # test against numpy.uint32
        assert_frames_equal(ref_frame0, decoder[numpy.uint32(0)])
        assert_frames_equal(ref_frame1, decoder[numpy.uint32(1)])
        assert_frames_equal(ref_frame180, decoder[numpy.uint32(180)])

    @pytest.mark.parametrize("device", cpu_and_accelerators())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_getitem_slice(self, device, seek_mode):
        decoder = VideoDecoder(NASA_VIDEO.path, device=device, seek_mode=seek_mode)

        # ensure that the degenerate case of a range of size 1 works

        ref0 = NASA_VIDEO.get_frame_data_by_range(0, 1).to(device)
        slice0 = decoder[0:1]
        assert slice0.shape == torch.Size(
            [
                1,
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        assert_frames_equal(ref0, slice0)

        ref4 = NASA_VIDEO.get_frame_data_by_range(4, 5).to(device)
        slice4 = decoder[4:5]
        assert slice4.shape == torch.Size(
            [
                1,
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        assert_frames_equal(ref4, slice4)

        ref8 = NASA_VIDEO.get_frame_data_by_range(8, 9).to(device)
        slice8 = decoder[8:9]
        assert slice8.shape == torch.Size(
            [
                1,
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        assert_frames_equal(ref8, slice8)

        ref180 = NASA_VIDEO.get_frame_data_by_index(180).to(device)
        slice180 = decoder[180:181]
        assert slice180.shape == torch.Size(
            [
                1,
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        assert_frames_equal(ref180, slice180[0])

        # contiguous ranges
        ref0_9 = NASA_VIDEO.get_frame_data_by_range(0, 9).to(device)
        slice0_9 = decoder[0:9]
        assert slice0_9.shape == torch.Size(
            [
                9,
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        assert_frames_equal(ref0_9, slice0_9)

        ref4_8 = NASA_VIDEO.get_frame_data_by_range(4, 8).to(device)
        slice4_8 = decoder[4:8]
        assert slice4_8.shape == torch.Size(
            [
                4,
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        assert_frames_equal(ref4_8, slice4_8)

        # ranges with a stride
        ref15_35 = NASA_VIDEO.get_frame_data_by_range(15, 36, 5).to(device)
        slice15_35 = decoder[15:36:5]
        assert slice15_35.shape == torch.Size(
            [
                5,
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        assert_frames_equal(ref15_35, slice15_35)

        ref0_9_2 = NASA_VIDEO.get_frame_data_by_range(0, 9, 2).to(device)
        slice0_9_2 = decoder[0:9:2]
        assert slice0_9_2.shape == torch.Size(
            [
                5,
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        assert_frames_equal(ref0_9_2, slice0_9_2)

        # negative numbers in the slice
        ref386_389 = NASA_VIDEO.get_frame_data_by_range(386, 390).to(device)
        slice386_389 = decoder[-4:]
        assert slice386_389.shape == torch.Size(
            [
                4,
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        assert_frames_equal(ref386_389, slice386_389)

        # slices with upper bound greater than len(decoder) are supported
        slice387_389 = decoder[-3:10000].to(device)
        assert slice387_389.shape == torch.Size(
            [
                3,
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        ref387_389 = NASA_VIDEO.get_frame_data_by_range(387, 390).to(device)
        assert_frames_equal(ref387_389, slice387_389)

        # an empty range is valid!
        empty_frame = decoder[5:5]
        assert_frames_equal(empty_frame, NASA_VIDEO.empty_chw_tensor.to(device))

        # slices that are out-of-range are also valid - they return an empty tensor
        also_empty = decoder[10000:]
        assert_frames_equal(also_empty, NASA_VIDEO.empty_chw_tensor.to(device))

        # should be just a copy
        all_frames = decoder[:].to(device)
        assert all_frames.shape == torch.Size(
            [
                len(decoder),
                NASA_VIDEO.num_color_channels,
                NASA_VIDEO.height,
                NASA_VIDEO.width,
            ]
        )
        for sliced, ref in zip(all_frames, decoder):
            if not (device == "cuda" and get_ffmpeg_major_version() == 4):
                # TODO: remove the "if".
                # See https://github.com/pytorch/torchcodec/issues/428
                assert_frames_equal(sliced, ref)

    def test_device_instance(self):
        # Non-regression test for https://github.com/pytorch/torchcodec/issues/602
        decoder = VideoDecoder(NASA_VIDEO.path, device=torch.device("cpu"))
        assert isinstance(decoder.metadata, VideoStreamMetadata)

    @pytest.mark.parametrize("device", cpu_and_accelerators())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_getitem_fails(self, device, seek_mode):
        decoder = VideoDecoder(NASA_VIDEO.path, device=device, seek_mode=seek_mode)

        with pytest.raises(IndexError, match="Invalid frame index"):
            frame = decoder[1000]  # noqa

        with pytest.raises(IndexError, match="Invalid frame index"):
            frame = decoder[-1000]  # noqa

        with pytest.raises(TypeError, match="Unsupported key type"):
            frame = decoder["0"]  # noqa

        with pytest.raises(TypeError, match="Unsupported key type"):
            frame = decoder[2.3]  # noqa

    @pytest.mark.parametrize("device", cpu_and_accelerators())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_iteration(self, device, seek_mode):
        decoder = VideoDecoder(NASA_VIDEO.path, device=device, seek_mode=seek_mode)

        ref_frame0 = NASA_VIDEO.get_frame_data_by_index(0).to(device)
        ref_frame1 = NASA_VIDEO.get_frame_data_by_index(1).to(device)
        ref_frame9 = NASA_VIDEO.get_frame_data_by_index(9).to(device)
        ref_frame35 = NASA_VIDEO.get_frame_data_by_index(35).to(device)
        ref_frame180 = NASA_VIDEO.get_frame_data_by_index(180).to(device)
        ref_frame_last = NASA_VIDEO.get_frame_data_by_index(289).to(device)

        # Access an arbitrary frame to make sure that the later iteration
        # still works as expected. The underlying C++ decoder object is
        # actually stateful, and accessing a frame will move its internal
        # cursor.
        assert_frames_equal(ref_frame35, decoder[35])

        for i, frame in enumerate(decoder):
            if i == 0:
                assert_frames_equal(ref_frame0, frame)
            elif i == 1:
                assert_frames_equal(ref_frame1, frame)
            elif i == 9:
                assert_frames_equal(ref_frame9, frame)
            elif i == 35:
                assert_frames_equal(ref_frame35, frame)
            elif i == 180:
                assert_frames_equal(ref_frame180, frame)
            elif i == 389:
                assert_frames_equal(ref_frame_last, frame)

    @pytest.mark.slow
    def test_iteration_slow(self):
        decoder = VideoDecoder(NASA_VIDEO.path)
        ref_frame_last = NASA_VIDEO.get_frame_data_by_index(389)

        # Force the decoder to seek around a lot while iterating; this will
        # slow down decoding, but we should still only iterate the exact number
        # of total frames.
        iterations = 0
        for frame in decoder:
            assert_frames_equal(ref_frame_last, decoder[-1])
            iterations += 1

        assert iterations == len(decoder) == 390

    @pytest.mark.parametrize("device", cpu_and_accelerators())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frame_at(self, device, seek_mode):
        decoder = VideoDecoder(NASA_VIDEO.path, device=device, seek_mode=seek_mode)

        ref_frame9 = NASA_VIDEO.get_frame_data_by_index(9).to(device)
        frame9 = decoder.get_frame_at(9)

        assert_frames_equal(ref_frame9, frame9.data)
        assert isinstance(frame9.pts_seconds, float)
        expected_frame_info = NASA_VIDEO.get_frame_info(9)
        assert frame9.pts_seconds == pytest.approx(expected_frame_info.pts_seconds)
        assert isinstance(frame9.duration_seconds, float)
        assert frame9.duration_seconds == pytest.approx(
            expected_frame_info.duration_seconds, rel=1e-3
        )

        # test negative frame index
        frame_minus1 = decoder.get_frame_at(-1)
        ref_frame_minus1 = NASA_VIDEO.get_frame_data_by_index(389).to(device)
        assert_frames_equal(ref_frame_minus1, frame_minus1.data)

        # test numpy.int64
        frame9 = decoder.get_frame_at(numpy.int64(9))
        assert_frames_equal(ref_frame9, frame9.data)

        # test numpy.int32
        frame9 = decoder.get_frame_at(numpy.int32(9))
        assert_frames_equal(ref_frame9, frame9.data)

        # test numpy.uint64
        frame9 = decoder.get_frame_at(numpy.uint64(9))
        assert_frames_equal(ref_frame9, frame9.data)

        # test numpy.uint32
        frame9 = decoder.get_frame_at(numpy.uint32(9))
        assert_frames_equal(ref_frame9, frame9.data)

    @pytest.mark.parametrize("device", cpu_and_accelerators())
    def test_get_frame_at_tuple_unpacking(self, device):
        decoder = VideoDecoder(NASA_VIDEO.path, device=device)

        frame = decoder.get_frame_at(50)
        data, pts, duration = decoder.get_frame_at(50)

        assert_frames_equal(frame.data, data)
        assert frame.pts_seconds == pts
        assert frame.duration_seconds == duration

    @pytest.mark.parametrize("device", cpu_and_accelerators())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frame_at_fails(self, device, seek_mode):
        decoder = VideoDecoder(NASA_VIDEO.path, device=device, seek_mode=seek_mode)

        with pytest.raises(
            IndexError,
            match="negative indices must have an absolute value less than the number of frames",
        ):
            frame = decoder.get_frame_at(-10000)  # noqa

        with pytest.raises(IndexError, match="must be less than"):
            frame = decoder.get_frame_at(10000)  # noqa

    @pytest.mark.parametrize("device", cpu_and_accelerators())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_at(self, device, seek_mode):
        decoder = VideoDecoder(NASA_VIDEO.path, device=device, seek_mode=seek_mode)

        # test positive and negative frame index
        frames = decoder.get_frames_at([35, 25, -1, -2])

        assert isinstance(frames, FrameBatch)

        assert_frames_equal(
            frames[0].data, NASA_VIDEO.get_frame_data_by_index(35).to(device)
        )
        assert_frames_equal(
            frames[1].data, NASA_VIDEO.get_frame_data_by_index(25).to(device)
        )
        assert_frames_equal(
            frames[2].data, NASA_VIDEO.get_frame_data_by_index(389).to(device)
        )
        assert_frames_equal(
            frames[3].data, NASA_VIDEO.get_frame_data_by_index(388).to(device)
        )

        assert frames.pts_seconds.device.type == "cpu"
        expected_pts_seconds = torch.tensor(
            [
                NASA_VIDEO.get_frame_info(35).pts_seconds,
                NASA_VIDEO.get_frame_info(25).pts_seconds,
                NASA_VIDEO.get_frame_info(389).pts_seconds,
                NASA_VIDEO.get_frame_info(388).pts_seconds,
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            frames.pts_seconds, expected_pts_seconds, atol=1e-4, rtol=0
        )

        assert frames.duration_seconds.device.type == "cpu"
        expected_duration_seconds = torch.tensor(
            [
                NASA_VIDEO.get_frame_info(35).duration_seconds,
                NASA_VIDEO.get_frame_info(25).duration_seconds,
                NASA_VIDEO.get_frame_info(389).duration_seconds,
                NASA_VIDEO.get_frame_info(388).duration_seconds,
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            frames.duration_seconds, expected_duration_seconds, atol=1e-4, rtol=0
        )

    @pytest.mark.parametrize("device", cpu_and_accelerators())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_at_fails(self, device, seek_mode):
        decoder = VideoDecoder(NASA_VIDEO.path, device=device, seek_mode=seek_mode)

        with pytest.raises(
            IndexError,
            match="negative indices must have an absolute value less than the number of frames",
        ):
            decoder.get_frames_at([-10000])

        with pytest.raises(IndexError, match="Invalid frame index=390"):
            decoder.get_frames_at([390])

        with pytest.raises(RuntimeError, match="Expected a value of type"):
            decoder.get_frames_at([0.3])

    @pytest.mark.parametrize("device", cpu_and_accelerators())
    def test_get_frame_at_av1(self, device):
        if device == "cuda" and get_ffmpeg_major_version() == 4:
            return

        decoder = VideoDecoder(AV1_VIDEO.path, device=device)
        ref_frame10 = AV1_VIDEO.get_frame_data_by_index(10)
        ref_frame_info10 = AV1_VIDEO.get_frame_info(10)
        decoded_frame10 = decoder.get_frame_at(10)
        assert decoded_frame10.duration_seconds == ref_frame_info10.duration_seconds
        assert decoded_frame10.pts_seconds == ref_frame_info10.pts_seconds
        assert_frames_equal(decoded_frame10.data, ref_frame10.to(device=device))

    @pytest.mark.parametrize("device", cpu_and_accelerators())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frame_played_at(self, device, seek_mode):
        decoder = VideoDecoder(NASA_VIDEO.path, device=device, seek_mode=seek_mode)

        ref_frame_played_at_6 = NASA_VIDEO.get_frame_data_by_index(180).to(device)
        assert_frames_equal(
            ref_frame_played_at_6, decoder.get_frame_played_at(6.006).data
        )
        assert_frames_equal(
            ref_frame_played_at_6, decoder.get_frame_played_at(6.02).data
        )
        assert_frames_equal(
            ref_frame_played_at_6, decoder.get_frame_played_at(6.039366).data
        )
        assert isinstance(decoder.get_frame_played_at(6.02).pts_seconds, float)
        assert isinstance(decoder.get_frame_played_at(6.02).duration_seconds, float)

    def test_get_frame_played_at_h265(self):
        # Non-regression test for https://github.com/pytorch/torchcodec/issues/179
        # We don't parametrize with CUDA because the current GPUs on CI do not
        # support x265:
        # https://github.com/pytorch/torchcodec/pull/350#issuecomment-2465011730
        # Note that because our internal fix-up depends on the key frame index, it
        # only works in exact seeking mode.
        decoder = VideoDecoder(H265_VIDEO.path, seek_mode="exact")
        ref_frame6 = H265_VIDEO.get_frame_data_by_index(5)
        assert_frames_equal(ref_frame6, decoder.get_frame_played_at(0.5).data)

    @pytest.mark.parametrize("device", cpu_and_accelerators())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frame_played_at_fails(self, device, seek_mode):
        decoder = VideoDecoder(NASA_VIDEO.path, device=device, seek_mode=seek_mode)

        with pytest.raises(IndexError, match="Invalid pts in seconds"):
            frame = decoder.get_frame_played_at(-1.0)  # noqa

        with pytest.raises(IndexError, match="Invalid pts in seconds"):
            frame = decoder.get_frame_played_at(100.0)  # noqa

    @pytest.mark.parametrize("device", cpu_and_accelerators())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_played_at(self, device, seek_mode):

        decoder = VideoDecoder(NASA_VIDEO.path, device=device, seek_mode=seek_mode)

        # Note: We know the frame at ~0.84s has index 25, the one at 1.16s has
        # index 35. We use those indices as reference to test against.
        seconds = [0.84, 1.17, 0.85]
        reference_indices = [25, 35, 25]
        frames = decoder.get_frames_played_at(seconds)

        assert isinstance(frames, FrameBatch)

        for i in range(len(reference_indices)):
            assert_frames_equal(
                frames.data[i],
                NASA_VIDEO.get_frame_data_by_index(reference_indices[i]).to(device),
                msg=f"index {i}",
            )

        assert frames.pts_seconds.device.type == "cpu"
        expected_pts_seconds = torch.tensor(
            [NASA_VIDEO.get_frame_info(i).pts_seconds for i in reference_indices],
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            frames.pts_seconds, expected_pts_seconds, atol=1e-4, rtol=0
        )

        assert frames.duration_seconds.device.type == "cpu"
        expected_duration_seconds = torch.tensor(
            [NASA_VIDEO.get_frame_info(i).duration_seconds for i in reference_indices],
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            frames.duration_seconds, expected_duration_seconds, atol=1e-4, rtol=0
        )

    @pytest.mark.parametrize("device", cpu_and_accelerators())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_played_at_fails(self, device, seek_mode):
        decoder = VideoDecoder(NASA_VIDEO.path, device=device, seek_mode=seek_mode)

        with pytest.raises(RuntimeError, match="must be greater than or equal to"):
            decoder.get_frames_played_at([-1])

        with pytest.raises(RuntimeError, match="must be less than"):
            decoder.get_frames_played_at([14])

        with pytest.raises(RuntimeError, match="Expected a value of type"):
            decoder.get_frames_played_at(["bad"])

    @pytest.mark.parametrize("device", cpu_and_accelerators())
    @pytest.mark.parametrize("stream_index", [0, 3, None])
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_in_range(self, stream_index, device, seek_mode):
        decoder = VideoDecoder(
            NASA_VIDEO.path,
            stream_index=stream_index,
            device=device,
            seek_mode=seek_mode,
        )

        # test degenerate case where we only actually get 1 frame
        ref_frames9 = NASA_VIDEO.get_frame_data_by_range(
            start=9, stop=10, stream_index=stream_index
        ).to(device)
        frames9 = decoder.get_frames_in_range(start=9, stop=10)

        assert_frames_equal(ref_frames9, frames9.data)

        assert frames9.pts_seconds.device.type == "cpu"
        assert frames9.pts_seconds[0].item() == pytest.approx(
            NASA_VIDEO.get_frame_info(9, stream_index=stream_index).pts_seconds,
            rel=1e-3,
        )
        assert frames9.duration_seconds.device.type == "cpu"
        assert frames9.duration_seconds[0].item() == pytest.approx(
            NASA_VIDEO.get_frame_info(9, stream_index=stream_index).duration_seconds,
            rel=1e-3,
        )

        # test simple ranges
        ref_frames0_9 = NASA_VIDEO.get_frame_data_by_range(
            start=0, stop=10, stream_index=stream_index
        ).to(device)
        frames0_9 = decoder.get_frames_in_range(start=0, stop=10)
        assert frames0_9.data.shape == torch.Size(
            [
                10,
                NASA_VIDEO.get_num_color_channels(stream_index=stream_index),
                NASA_VIDEO.get_height(stream_index=stream_index),
                NASA_VIDEO.get_width(stream_index=stream_index),
            ]
        )
        assert_frames_equal(ref_frames0_9, frames0_9.data)
        torch.testing.assert_close(
            NASA_VIDEO.get_pts_seconds_by_range(0, 10, stream_index=stream_index),
            frames0_9.pts_seconds,
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            NASA_VIDEO.get_duration_seconds_by_range(0, 10, stream_index=stream_index),
            frames0_9.duration_seconds,
            atol=1e-6,
            rtol=1e-6,
        )

        # test steps
        ref_frames0_8_2 = NASA_VIDEO.get_frame_data_by_range(
            start=0, stop=10, step=2, stream_index=stream_index
        ).to(device)
        frames0_8_2 = decoder.get_frames_in_range(start=0, stop=10, step=2)
        assert frames0_8_2.data.shape == torch.Size(
            [
                5,
                NASA_VIDEO.get_num_color_channels(stream_index=stream_index),
                NASA_VIDEO.get_height(stream_index=stream_index),
                NASA_VIDEO.get_width(stream_index=stream_index),
            ]
        )
        assert_frames_equal(ref_frames0_8_2, frames0_8_2.data)
        torch.testing.assert_close(
            NASA_VIDEO.get_pts_seconds_by_range(0, 10, 2, stream_index=stream_index),
            frames0_8_2.pts_seconds,
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            NASA_VIDEO.get_duration_seconds_by_range(
                0, 10, 2, stream_index=stream_index
            ),
            frames0_8_2.duration_seconds,
            atol=1e-6,
            rtol=1e-6,
        )

        # test numpy.int64 for indices
        frames0_8_2 = decoder.get_frames_in_range(
            start=numpy.int64(0), stop=numpy.int64(10), step=numpy.int64(2)
        )
        assert_frames_equal(ref_frames0_8_2, frames0_8_2.data)

        # an empty range is valid!
        empty_frames = decoder.get_frames_in_range(5, 5)
        assert_frames_equal(
            empty_frames.data,
            NASA_VIDEO.get_empty_chw_tensor(stream_index=stream_index).to(device),
        )
        torch.testing.assert_close(
            empty_frames.pts_seconds, NASA_VIDEO.empty_pts_seconds
        )
        torch.testing.assert_close(
            empty_frames.duration_seconds, NASA_VIDEO.empty_duration_seconds
        )

    @pytest.mark.parametrize("device", cpu_and_accelerators())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_in_range_slice_indices_syntax(self, device, seek_mode):
        decoder = VideoDecoder(
            NASA_VIDEO.path,
            stream_index=3,
            device=device,
            seek_mode=seek_mode,
        )

        # high range ends get capped to num_frames
        frames387_389 = decoder.get_frames_in_range(start=387, stop=1000)
        assert frames387_389.data.shape == torch.Size(
            [
                3,
                NASA_VIDEO.get_num_color_channels(stream_index=3),
                NASA_VIDEO.get_height(stream_index=3),
                NASA_VIDEO.get_width(stream_index=3),
            ]
        )
        ref_frame387_389 = NASA_VIDEO.get_frame_data_by_range(
            start=387, stop=390, stream_index=3
        ).to(device)
        assert_frames_equal(frames387_389.data, ref_frame387_389)

        # negative indices are converted
        frames387_389 = decoder.get_frames_in_range(start=-3, stop=1000)
        assert frames387_389.data.shape == torch.Size(
            [
                3,
                NASA_VIDEO.get_num_color_channels(stream_index=3),
                NASA_VIDEO.get_height(stream_index=3),
                NASA_VIDEO.get_width(stream_index=3),
            ]
        )
        assert_frames_equal(frames387_389.data, ref_frame387_389)

        # "None" as stop is treated as end of the video
        frames387_None = decoder.get_frames_in_range(start=-3, stop=None)
        assert frames387_None.data.shape == torch.Size(
            [
                3,
                NASA_VIDEO.get_num_color_channels(stream_index=3),
                NASA_VIDEO.get_height(stream_index=3),
                NASA_VIDEO.get_width(stream_index=3),
            ]
        )
        reference_frame387_389 = NASA_VIDEO.get_frame_data_by_range(
            start=387, stop=390, stream_index=3
        ).to(device)
        assert_frames_equal(frames387_None.data, reference_frame387_389)

    @pytest.mark.parametrize("device", cpu_and_accelerators())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    @patch("torchcodec._core._metadata._get_stream_json_metadata")
    def test_get_frames_with_missing_num_frames_metadata(
        self, mock_get_stream_json_metadata, device, seek_mode
    ):
        # Create a mock stream_dict to test that initializing VideoDecoder without
        # num_frames_from_header and num_frames_from_content calculates num_frames
        # using the average_fps and duration_seconds metadata.
        mock_stream_dict = {
            "averageFpsFromHeader": 29.97003,
            "beginStreamSecondsFromContent": 0.0,
            "beginStreamSecondsFromHeader": 0.0,
            "bitRate": 128783.0,
            "codec": "h264",
            "durationSecondsFromHeader": 13.013,
            "endStreamSecondsFromContent": 13.013,
            "width": 480,
            "height": 270,
            "mediaType": "video",
            "numFramesFromHeader": None,
            "numFramesFromContent": None,
        }
        # Set the return value of the mock to be the mock_stream_dict
        mock_get_stream_json_metadata.return_value = json.dumps(mock_stream_dict)

        decoder = VideoDecoder(
            NASA_VIDEO.path,
            stream_index=3,
            device=device,
            seek_mode=seek_mode,
        )

        assert decoder.metadata.num_frames_from_header is None
        assert decoder.metadata.num_frames_from_content is None
        assert decoder.metadata.duration_seconds is not None
        assert decoder.metadata.average_fps is not None
        assert decoder.metadata.num_frames == int(
            decoder.metadata.duration_seconds * decoder.metadata.average_fps
        )
        assert len(decoder) == 390

        # Test get_frames_in_range Python logic which uses the num_frames metadata mocked earlier.
        # The frame is read at the C++ level.
        ref_frames9 = NASA_VIDEO.get_frame_data_by_range(
            start=9, stop=10, stream_index=3
        ).to(device)
        frames9 = decoder.get_frames_in_range(start=9, stop=10)
        assert_frames_equal(ref_frames9, frames9.data)

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
    @pytest.mark.parametrize("device", cpu_and_accelerators())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_dimension_order(self, dimension_order, frame_getter, device, seek_mode):
        decoder = VideoDecoder(
            NASA_VIDEO.path,
            dimension_order=dimension_order,
            device=device,
            seek_mode=seek_mode,
        )
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
    @pytest.mark.parametrize("device", cpu_and_accelerators())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_by_pts_in_range(self, stream_index, device, seek_mode):
        decoder = VideoDecoder(
            NASA_VIDEO.path,
            stream_index=stream_index,
            device=device,
            seek_mode=seek_mode,
        )

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
        assert_frames_equal(
            frames0_4.data,
            NASA_VIDEO.get_frame_data_by_range(0, 5, stream_index=stream_index).to(
                device
            ),
        )

        # Range where the stop seconds is about halfway between pts values for two frames.
        also_frames0_4 = decoder.get_frames_played_in_range(
            decoder.get_frame_at(0).pts_seconds,
            decoder.get_frame_at(4).pts_seconds + HALF_DURATION,
        )
        assert_frames_equal(also_frames0_4.data, frames0_4.data)

        # Again, the intention here is to provide the exact values we care about. In practice, our
        # pts values are slightly smaller, so we nudge the start upwards.
        frames5_9 = decoder.get_frames_played_in_range(
            decoder.get_frame_at(5).pts_seconds,
            decoder.get_frame_at(10).pts_seconds,
        )
        assert_frames_equal(
            frames5_9.data,
            NASA_VIDEO.get_frame_data_by_range(5, 10, stream_index=stream_index).to(
                device
            ),
        )

        # Range where we provide start_seconds and stop_seconds that are different, but
        # also should land in the same window of time between two frame's pts values. As
        # a result, we should only get back one frame.
        frame6 = decoder.get_frames_played_in_range(
            decoder.get_frame_at(6).pts_seconds,
            decoder.get_frame_at(6).pts_seconds + HALF_DURATION,
        )
        assert_frames_equal(
            frame6.data,
            NASA_VIDEO.get_frame_data_by_range(6, 7, stream_index=stream_index).to(
                device
            ),
        )

        # Very small range that falls in the same frame.
        frame35 = decoder.get_frames_played_in_range(
            decoder.get_frame_at(35).pts_seconds,
            decoder.get_frame_at(35).pts_seconds + 1e-10,
        )
        assert_frames_equal(
            frame35.data,
            NASA_VIDEO.get_frame_data_by_range(35, 36, stream_index=stream_index).to(
                device
            ),
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
        assert_frames_equal(
            frames7_8.data,
            NASA_VIDEO.get_frame_data_by_range(7, 9, stream_index=stream_index).to(
                device
            ),
        )

        # Start and stop seconds are the same value, which should not return a frame.
        empty_frame = decoder.get_frames_played_in_range(
            NASA_VIDEO.get_frame_info(4, stream_index=stream_index).pts_seconds,
            NASA_VIDEO.get_frame_info(4, stream_index=stream_index).pts_seconds,
        )
        assert_frames_equal(
            empty_frame.data,
            NASA_VIDEO.get_empty_chw_tensor(stream_index=stream_index).to(device),
        )
        torch.testing.assert_close(
            empty_frame.pts_seconds, NASA_VIDEO.empty_pts_seconds, atol=0, rtol=0
        )
        torch.testing.assert_close(
            empty_frame.duration_seconds,
            NASA_VIDEO.empty_duration_seconds,
            atol=0,
            rtol=0,
        )

        # Start and stop seconds land within the first frame.
        frame0 = decoder.get_frames_played_in_range(
            NASA_VIDEO.get_frame_info(0, stream_index=stream_index).pts_seconds,
            NASA_VIDEO.get_frame_info(0, stream_index=stream_index).pts_seconds
            + HALF_DURATION,
        )
        assert_frames_equal(
            frame0.data,
            NASA_VIDEO.get_frame_data_by_range(0, 1, stream_index=stream_index).to(
                device
            ),
        )

        # We should be able to get all frames by giving the beginning and ending time
        # for the stream.
        all_frames = decoder.get_frames_played_in_range(
            decoder.metadata.begin_stream_seconds, decoder.metadata.end_stream_seconds
        )
        assert_frames_equal(all_frames.data, decoder[:])

    @pytest.mark.parametrize("device", cpu_and_accelerators())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_by_pts_in_range_fails(self, device, seek_mode):
        decoder = VideoDecoder(NASA_VIDEO.path, device=device, seek_mode=seek_mode)

        with pytest.raises(ValueError, match="Invalid start seconds"):
            frame = decoder.get_frames_played_in_range(100.0, 1.0)  # noqa

        with pytest.raises(ValueError, match="Invalid start seconds"):
            frame = decoder.get_frames_played_in_range(20, 23)  # noqa

        with pytest.raises(ValueError, match="Invalid stop seconds"):
            frame = decoder.get_frames_played_in_range(0, 23)  # noqa

    @pytest.mark.parametrize("device", cpu_and_accelerators())
    def test_get_key_frame_indices(self, device):
        decoder = VideoDecoder(NASA_VIDEO.path, device=device, seek_mode="exact")
        key_frame_indices = decoder._get_key_frame_indices()

        # The key frame indices were generated from the following command:
        #   $ ffprobe -v error -hide_banner -select_streams v:1 -show_frames -of csv test/resources/nasa_13013.mp4 | grep -n ",I," | cut -d ':' -f 1 > key_frames.txt
        # What it's doing:
        #   1. Calling ffprobe on the second video stream, which is absolute stream index 3.
        #   2. Showing all frames for that stream.
        #   3. Using grep to find the "I" frames, which are the key frames. We also get the line
        #      number, which is also the count of the rames.
        #   4. Using cut to extract just the count for the frame.
        # Finally, because the above produces a count, which is index + 1, we subtract
        # one from all values manually to arrive at the values below.
        # TODO: decide if/how we want to incorporate key frame indices into the utils
        # framework.
        nasa_reference_key_frame_indices = torch.tensor([0, 240])

        torch.testing.assert_close(
            key_frame_indices, nasa_reference_key_frame_indices, atol=0, rtol=0
        )

        decoder = VideoDecoder(AV1_VIDEO.path, device=device, seek_mode="exact")
        key_frame_indices = decoder._get_key_frame_indices()

        # $ ffprobe -v error -hide_banner -select_streams v:0 -show_frames -of csv test/resources/av1_video.mkv | grep -n ",I," | cut -d ':' -f 1 > key_frames.txt
        av1_reference_key_frame_indices = torch.tensor([0])

        torch.testing.assert_close(
            key_frame_indices, av1_reference_key_frame_indices, atol=0, rtol=0
        )

        decoder = VideoDecoder(H265_VIDEO.path, device=device, seek_mode="exact")
        key_frame_indices = decoder._get_key_frame_indices()

        # ffprobe -v error -hide_banner -select_streams v:0 -show_frames -of csv test/resources/h265_video.mp4 | grep -n ",I," | cut -d ':' -f 1 > key_frames.txt
        h265_reference_key_frame_indices = torch.tensor([0, 2, 4, 6, 8])

        torch.testing.assert_close(
            key_frame_indices, h265_reference_key_frame_indices, atol=0, rtol=0
        )

    # TODO investigate why this fails internally.
    @pytest.mark.skipif(in_fbcode(), reason="Compile test fails internally.")
    @pytest.mark.parametrize("device", cpu_and_accelerators())
    def test_compile(self, device):
        decoder = VideoDecoder(NASA_VIDEO.path, device=device)

        @contextlib.contextmanager
        def restore_capture_scalar_outputs():
            try:
                original = torch._dynamo.config.capture_scalar_outputs
                yield
            finally:
                torch._dynamo.config.capture_scalar_outputs = original

        # TODO: We get a graph break because we call Tensor.item() to turn the
        # tensors in FrameBatch into scalars. When we work on compilation and exportability,
        # we should investigate.
        with restore_capture_scalar_outputs():
            torch._dynamo.config.capture_scalar_outputs = True

            @torch.compile(fullgraph=True, backend="eager")
            def get_some_frames(decoder):
                frames = []
                frames.append(decoder.get_frame_at(1))
                frames.append(decoder.get_frame_at(3))
                frames.append(decoder.get_frame_at(5))
                return frames

            frames = get_some_frames(decoder)

            ref_frame1 = NASA_VIDEO.get_frame_data_by_index(1).to(device)
            ref_frame3 = NASA_VIDEO.get_frame_data_by_index(3).to(device)
            ref_frame5 = NASA_VIDEO.get_frame_data_by_index(5).to(device)

            assert_frames_equal(ref_frame1, frames[0].data)
            assert_frames_equal(ref_frame3, frames[1].data)
            assert_frames_equal(ref_frame5, frames[2].data)

    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_pts_to_dts_fallback(self, seek_mode):
        # Non-regression test for
        # https://github.com/pytorch/torchcodec/issues/677 and
        # https://github.com/pytorch/torchcodec/issues/676.
        # More accurately, this is a non-regression test for videos which do
        # *not* specify pts values (all pts values are N/A and set to
        # INT64_MIN), but specify *dts* value - which we fallback to.
        #
        # The test video we have is from
        # https://huggingface.co/datasets/raushan-testing-hf/videos-test/blob/main/sample_video_2.avi
        # We can't check it into the repo due to potential licensing issues, so
        # we have to unconditionally skip this test.#
        # TODO: encode a video with no pts values to unskip this test. Couldn't
        # find a way to do that with FFmpeg's CLI, but this should be doable
        # once we have our own video encoder.
        pytest.skip(reason="TODO: Need video with no pts values.")

        path = "/home/nicolashug/Downloads/sample_video_2.avi"
        decoder = VideoDecoder(path, seek_mode=seek_mode)
        metadata = decoder.metadata

        assert metadata.average_fps == pytest.approx(29.916667)
        assert metadata.duration_seconds_from_header == 9.02507
        assert metadata.duration_seconds == 9.02507
        assert metadata.begin_stream_seconds_from_content == (
            None if seek_mode == "approximate" else 0
        )
        assert metadata.end_stream_seconds_from_content == (
            None if seek_mode == "approximate" else 9.02507
        )

        assert decoder[0].shape == (3, 240, 320)
        decoder[10].shape == (3, 240, 320)
        decoder.get_frame_at(2).data.shape == (3, 240, 320)
        decoder.get_frames_at([2, 10]).data.shape == (2, 3, 240, 320)
        decoder.get_frame_played_at(9).data.shape == (3, 240, 320)
        decoder.get_frames_played_at([2, 4]).data.shape == (2, 3, 240, 320)
        with pytest.raises(AssertionError, match="not equal"):
            torch.testing.assert_close(decoder[0], decoder[10])

    @needs_cuda
    def test_10bit_videos_cuda(self):
        # Assert that we raise proper error on different kinds of 10bit videos.

        # TODO we should investigate how to support 10bit videos on GPU.
        # See https://github.com/pytorch/torchcodec/issues/776

        asset = H265_10BITS

        decoder = VideoDecoder(asset.path, device="cuda")
        with pytest.raises(
            RuntimeError,
            match="The AVFrame is p010le, but we expected AV_PIX_FMT_NV12.",
        ):
            decoder.get_frame_at(0)

    @needs_cuda
    def test_10bit_gpu_fallsback_to_cpu(self):
        # Test for 10-bit videos that aren't supported by NVDEC: we decode and
        # do the color conversion on the CPU.
        # Here we just assert that the GPU results are the same as the CPU
        # results.
        # TODO see other TODO below in test_10bit_videos_cpu: we should validate
        # the frames against a reference.

        # We know from previous tests that the H264_10BITS video isn't supported
        # by NVDEC, so NVDEC decodes it on the CPU.
        asset = H264_10BITS

        decoder_gpu = VideoDecoder(asset.path, device="cuda")
        decoder_cpu = VideoDecoder(asset.path)

        frame_indices = [0, 10, 20, 5]
        for frame_index in frame_indices:
            frame_gpu = decoder_gpu.get_frame_at(frame_index).data
            assert frame_gpu.device.type == "cuda"
            frame_cpu = decoder_cpu.get_frame_at(frame_index).data
            assert_frames_equal(frame_gpu.cpu(), frame_cpu)

        # We also check a batch API just to be on the safe side, making sure the
        # pre-allocated tensor is passed down correctly to the CPU
        # implementation.
        frames_gpu = decoder_gpu.get_frames_at(frame_indices).data
        assert frames_gpu.device.type == "cuda"
        frames_cpu = decoder_cpu.get_frames_at(frame_indices).data
        assert_frames_equal(frames_gpu.cpu(), frames_cpu)

    @pytest.mark.parametrize("asset", (H264_10BITS, H265_10BITS))
    def test_10bit_videos_cpu(self, asset):
        # This just validates that we can decode 10-bit videos on CPU.
        # TODO validate against the ref that the decoded frames are correct

        decoder = VideoDecoder(asset.path)
        decoder.get_frame_at(10)


class TestAudioDecoder:
    @pytest.mark.parametrize("asset", (NASA_AUDIO, NASA_AUDIO_MP3, SINE_MONO_S32))
    def test_metadata(self, asset):
        decoder = AudioDecoder(asset.path)
        assert isinstance(decoder.metadata, AudioStreamMetadata)

        assert (
            decoder.stream_index
            == decoder.metadata.stream_index
            == asset.default_stream_index
        )
        assert decoder.metadata.duration_seconds_from_header == pytest.approx(
            asset.duration_seconds
        )
        assert decoder.metadata.sample_rate == asset.sample_rate
        assert decoder.metadata.num_channels == asset.num_channels
        assert decoder.metadata.sample_format == asset.sample_format

    @pytest.mark.parametrize("asset", (NASA_AUDIO, NASA_AUDIO_MP3))
    def test_error(self, asset):
        decoder = AudioDecoder(asset.path)

        with pytest.raises(ValueError, match="Invalid start seconds"):
            decoder.get_samples_played_in_range(start_seconds=3, stop_seconds=2)

        with pytest.raises(RuntimeError, match="No audio frames were decoded"):
            decoder.get_samples_played_in_range(start_seconds=9999)

    @pytest.mark.parametrize("asset", (NASA_AUDIO, NASA_AUDIO_MP3))
    def test_negative_start(self, asset):
        decoder = AudioDecoder(asset.path)
        samples = decoder.get_samples_played_in_range(start_seconds=-1300)
        reference_samples = decoder.get_samples_played_in_range()
        torch.testing.assert_close(samples.data, reference_samples.data)
        assert samples.pts_seconds == reference_samples.pts_seconds

    @pytest.mark.parametrize("asset", (NASA_AUDIO, NASA_AUDIO_MP3))
    @pytest.mark.parametrize("stop_seconds", (None, "duration", 99999999))
    def test_get_all_samples_with_range(self, asset, stop_seconds):
        decoder = AudioDecoder(asset.path)

        if stop_seconds == "duration":
            stop_seconds = asset.duration_seconds

        samples = decoder.get_samples_played_in_range(stop_seconds=stop_seconds)

        reference_frames = asset.get_frame_data_by_range(
            start=0, stop=asset.get_frame_index(pts_seconds=asset.duration_seconds) + 1
        )

        torch.testing.assert_close(samples.data, reference_frames)
        assert samples.sample_rate == asset.sample_rate
        assert samples.pts_seconds == asset.get_frame_info(idx=0).pts_seconds

    @pytest.mark.parametrize("asset", (NASA_AUDIO, NASA_AUDIO_MP3))
    def test_get_all_samples(self, asset):
        decoder = AudioDecoder(asset.path)
        torch.testing.assert_close(
            decoder.get_all_samples().data,
            decoder.get_samples_played_in_range().data,
        )

    @pytest.mark.parametrize("asset", (NASA_AUDIO, NASA_AUDIO_MP3))
    def test_at_frame_boundaries(self, asset):
        decoder = AudioDecoder(asset.path)

        start_frame_index, stop_frame_index = 10, 40
        start_seconds = asset.get_frame_info(start_frame_index).pts_seconds
        stop_seconds = asset.get_frame_info(stop_frame_index).pts_seconds

        samples = decoder.get_samples_played_in_range(
            start_seconds=start_seconds, stop_seconds=stop_seconds
        )

        reference_frames = asset.get_frame_data_by_range(
            start=start_frame_index, stop=stop_frame_index
        )

        assert samples.pts_seconds == start_seconds
        num_samples = samples.data.shape[1]
        assert (
            num_samples
            == reference_frames.shape[1]
            == (stop_seconds - start_seconds) * decoder.metadata.sample_rate
        )
        torch.testing.assert_close(samples.data, reference_frames)
        assert samples.sample_rate == asset.sample_rate

    @pytest.mark.parametrize("asset", (NASA_AUDIO, NASA_AUDIO_MP3))
    def test_not_at_frame_boundaries(self, asset):
        decoder = AudioDecoder(asset.path)

        start_frame_index, stop_frame_index = 10, 40
        start_frame_info = asset.get_frame_info(start_frame_index)
        stop_frame_info = asset.get_frame_info(stop_frame_index)
        start_seconds = start_frame_info.pts_seconds + (
            start_frame_info.duration_seconds / 2
        )
        stop_seconds = stop_frame_info.pts_seconds + (
            stop_frame_info.duration_seconds / 2
        )
        samples = decoder.get_samples_played_in_range(
            start_seconds=start_seconds, stop_seconds=stop_seconds
        )

        reference_frames = asset.get_frame_data_by_range(
            start=start_frame_index, stop=stop_frame_index + 1
        )

        assert samples.pts_seconds == start_seconds
        num_samples = samples.data.shape[1]
        assert num_samples < reference_frames.shape[1]
        assert (
            num_samples == (stop_seconds - start_seconds) * decoder.metadata.sample_rate
        )
        assert samples.sample_rate == asset.sample_rate

    @pytest.mark.parametrize("asset", (NASA_AUDIO, NASA_AUDIO_MP3))
    def test_start_equals_stop(self, asset):
        decoder = AudioDecoder(asset.path)
        samples = decoder.get_samples_played_in_range(start_seconds=3, stop_seconds=3)
        assert samples.data.shape == (asset.num_channels, 0)

    def test_frame_start_is_not_zero(self):
        # For NASA_AUDIO_MP3, the first frame is not at 0, it's at 0.138125.
        # So if we request (start, stop) = (0.05, None), we shouldn't be
        # truncating anything.

        asset = NASA_AUDIO_MP3
        start_seconds = 0.05  # this is less than the first frame's pts
        stop_frame_index = 10
        stop_seconds = asset.get_frame_info(stop_frame_index).pts_seconds

        decoder = AudioDecoder(asset.path)

        samples = decoder.get_samples_played_in_range(
            start_seconds=start_seconds, stop_seconds=stop_seconds
        )

        reference_frames = asset.get_frame_data_by_range(start=0, stop=stop_frame_index)
        torch.testing.assert_close(samples.data, reference_frames)

        # Non-regression test for https://github.com/pytorch/torchcodec/issues/567
        # If we ask for start < stop <= first_frame_pts, we should raise.
        with pytest.raises(RuntimeError, match="No audio frames were decoded"):
            decoder.get_samples_played_in_range(start_seconds=0, stop_seconds=0.05)

        first_frame_pts_seconds = asset.get_frame_info(idx=0).pts_seconds
        with pytest.raises(RuntimeError, match="No audio frames were decoded"):
            decoder.get_samples_played_in_range(
                start_seconds=0, stop_seconds=first_frame_pts_seconds
            )

        # Documenting an edge case: we ask for samples barely beyond the start
        # of the first frame. The C++ decoder returns the first frame, which
        # gets (correctly!) truncated by the AudioDecoder, and we end up with
        # empty data.
        samples = decoder.get_samples_played_in_range(
            start_seconds=0, stop_seconds=first_frame_pts_seconds + 1e-5
        )
        assert samples.data.shape == (2, 0)
        assert samples.pts_seconds == first_frame_pts_seconds
        assert samples.duration_seconds == 0

        # if we ask for a little bit more samples, we get non-empty data
        samples = decoder.get_samples_played_in_range(
            start_seconds=0, stop_seconds=first_frame_pts_seconds + 1e-3
        )
        assert samples.data.shape == (2, 8)
        assert samples.pts_seconds == first_frame_pts_seconds

    def test_single_channel(self):
        asset = SINE_MONO_S32
        decoder = AudioDecoder(asset.path)

        samples = decoder.get_samples_played_in_range(stop_seconds=2)
        assert samples.data.shape[0] == asset.num_channels == 1

    def test_format_conversion(self):
        asset = SINE_MONO_S32
        decoder = AudioDecoder(asset.path)
        assert decoder.metadata.sample_format == asset.sample_format == "s32"

        all_samples = decoder.get_samples_played_in_range()
        assert all_samples.data.dtype == torch.float32

        reference_frames = asset.get_frame_data_by_range(start=0, stop=asset.num_frames)
        torch.testing.assert_close(all_samples.data, reference_frames)

    @pytest.mark.parametrize(
        "start_seconds, stop_seconds",
        (
            (0, None),
            (0, 4),
            (0, 3),
            (2, None),
            (2, 3),
        ),
    )
    def test_sample_rate_conversion(self, start_seconds, stop_seconds):
        # When start_seconds is not exactly 0, we have to increase the tolerance
        # a bit. This is because sample_rate conversion relies on a sliding
        # window of samples: if we start decoding a stream in the middle, the
        # first few samples we're decoding aren't able to take advantage of the
        # preceeding samples for sample-rate conversion. This leads to a
        # slightly different sample-rate conversion that we would otherwise get,
        # had we started the stream from the beginning.
        atol = 1e-6 if start_seconds == 0 else 1e-2
        rtol = 1e-6

        # Upsample
        decoder = AudioDecoder(SINE_MONO_S32_44100.path)
        assert decoder.metadata.sample_rate == 44_100
        frames_44100_native = decoder.get_samples_played_in_range(
            start_seconds=start_seconds, stop_seconds=stop_seconds
        )
        assert frames_44100_native.sample_rate == 44_100

        decoder = AudioDecoder(SINE_MONO_S32.path, sample_rate=44_100)
        frames_upsampled_to_44100 = decoder.get_samples_played_in_range(
            start_seconds=start_seconds, stop_seconds=stop_seconds
        )
        assert decoder.metadata.sample_rate == 16_000
        assert frames_upsampled_to_44100.sample_rate == 44_100

        torch.testing.assert_close(
            frames_upsampled_to_44100.data,
            frames_44100_native.data,
            atol=atol,
            rtol=rtol,
        )

        # Downsample
        decoder = AudioDecoder(SINE_MONO_S32_8000.path)
        assert decoder.metadata.sample_rate == 8000
        frames_8000_native = decoder.get_samples_played_in_range(
            start_seconds=start_seconds, stop_seconds=stop_seconds
        )
        assert frames_8000_native.sample_rate == 8000

        decoder = AudioDecoder(SINE_MONO_S32.path, sample_rate=8000)
        frames_downsampled_to_8000 = decoder.get_samples_played_in_range(
            start_seconds=start_seconds, stop_seconds=stop_seconds
        )
        assert decoder.metadata.sample_rate == 16_000
        assert frames_downsampled_to_8000.sample_rate == 8000

        torch.testing.assert_close(
            frames_downsampled_to_8000.data,
            frames_8000_native.data,
            atol=atol,
            rtol=rtol,
        )

    def test_sample_rate_conversion_stereo(self):
        # Non-regression test for https://github.com/pytorch/torchcodec/pull/584
        asset = NASA_AUDIO_MP3
        assert asset.sample_rate == 8000
        assert asset.num_channels == 2
        decoder = AudioDecoder(asset.path, sample_rate=44_100)
        decoder.get_samples_played_in_range()

    def test_downsample_empty_frame(self):
        # Non-regression test for
        # https://github.com/pytorch/torchcodec/pull/586: when downsampling  by
        # a great factor, if an input frame has a small amount of sample, the
        # resampled frame (as output by swresample) may contain zero sample. We
        # make sure we handle this properly.
        #
        # NASA_AUDIO_MP3_44100's first frame has only 47 samples which triggers
        # the test scenario:
        # ```
        # » ffprobe -v error -hide_banner -select_streams a:0 -show_frames -of json test/resources/nasa_13013.mp4.audio_44100.mp3 | grep nb_samples | head -n 3
        # "nb_samples": 47,
        # "nb_samples": 1152,
        # "nb_samples": 1152,
        # ```
        asset = NASA_AUDIO_MP3_44100
        assert asset.sample_rate == 44_100
        decoder = AudioDecoder(asset.path, sample_rate=8_000)
        frames_44100_to_8000 = decoder.get_samples_played_in_range()

        # Just checking correctness now
        asset = NASA_AUDIO_MP3
        assert asset.sample_rate == 8_000
        decoder = AudioDecoder(asset.path)
        frames_8000 = decoder.get_samples_played_in_range()
        torch.testing.assert_close(
            frames_44100_to_8000.data, frames_8000.data, atol=0.03, rtol=0
        )

    def test_s16_ffmpeg4_bug(self):
        # s16 fails on FFmpeg4 but can be decoded on other versions.
        # Debugging logs show that we're hitting:
        # [SWR @ 0x560a7abdaf80] Input channel count and layout are unset
        # which seems to point to:
        # https://github.com/FFmpeg/FFmpeg/blob/40a6963fbd0c47be358a3760480180b7b532e1e9/libswresample/swresample.c#L293-L305
        # ¯\_(ツ)_/¯

        asset = SINE_MONO_S16
        decoder = AudioDecoder(asset.path)
        assert decoder.metadata.sample_rate == asset.sample_rate
        assert decoder.metadata.sample_format == asset.sample_format

        cm = (
            pytest.raises(RuntimeError, match="The frame has 0 channels, expected 1.")
            if get_ffmpeg_major_version() == 4
            else contextlib.nullcontext()
        )
        with cm:
            decoder.get_samples_played_in_range()

    @pytest.mark.parametrize("asset", (NASA_AUDIO, NASA_AUDIO_MP3))
    @pytest.mark.parametrize("sample_rate", (None, 8000, 16_000, 44_1000))
    def test_samples_duration(self, asset, sample_rate):
        decoder = AudioDecoder(asset.path, sample_rate=sample_rate)
        samples = decoder.get_samples_played_in_range(start_seconds=1, stop_seconds=2)
        assert samples.duration_seconds == 1

    @pytest.mark.parametrize("asset", (SINE_MONO_S32, NASA_AUDIO_MP3))
    # Note that we parametrize over sample_rate as well, so that we can ensure
    # that the extra tensor allocation that happens within
    # maybeFlushSwrBuffers() is correct.
    @pytest.mark.parametrize("sample_rate", (None, 16_000))
    # FFmpeg can handle up to AV_NUM_DATA_POINTERS=8 channels
    @pytest.mark.parametrize("num_channels", (1, 2, 8, None))
    def test_num_channels(self, asset, sample_rate, num_channels):
        decoder = AudioDecoder(
            asset.path, sample_rate=sample_rate, num_channels=num_channels
        )
        samples = decoder.get_all_samples()

        if num_channels is None:
            num_channels = asset.num_channels

        assert samples.data.shape[0] == num_channels

    @pytest.mark.parametrize("asset", (SINE_MONO_S32, NASA_AUDIO_MP3))
    def test_num_channels_errors(self, asset):
        with pytest.raises(
            RuntimeError, match="num_channels must be > 0 and <= AV_NUM_DATA_POINTERS"
        ):
            AudioDecoder(asset.path, num_channels=0)
        with pytest.raises(
            RuntimeError, match="num_channels must be > 0 and <= AV_NUM_DATA_POINTERS"
        ):
            # FFmpeg can handle up to AV_NUM_DATA_POINTERS=8 channels
            AudioDecoder(asset.path, num_channels=9)
