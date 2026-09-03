# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import concurrent.futures
import contextlib
import gc
import io
import itertools
import math
import os
import queue
import subprocess
import threading
from functools import partial
from typing import NamedTuple

import numpy
import pytest
import torch
from PIL import Image, ImageOps
from torchcodec import _core, ffmpeg_major_version, FrameBatch
from torchcodec._core.ops import _blocks_demuxer_add_stream, _blocks_demuxer_scan
from torchcodec._frame import Frame
from torchcodec.decoders import (
    AudioDecoder,
    AudioStreamMetadata,
    decode_avif,
    decode_gif,
    decode_heic,
    decode_image,
    decode_jpeg,
    decode_png,
    decode_webp,
    get_nvdec_cache_capacity,
    ImageReadMode,
    set_cuda_backend,
    set_nvdec_cache_capacity,
    VideoDecoder,
    VideoStreamMetadata,
    WavDecoder,
)
from torchcodec.decoders._blocks import (
    AudioConverter,
    AudioPacketDecoder,
    AudioStream,
    ColorConverter,
    Demuxer,
    get_container_metadata,
    Packet,
    RawAudioSamples,
    RawFrame,
    VideoPacketDecoder,
    VideoStream,
)
from torchcodec.decoders._decoder_utils import _get_cuda_backend
from torchcodec.decoders._image_decoders import _source_to_tensor
from torchcodec.encoders import VideoEncoder
from torchcodec.transforms import CenterCrop, RandomCrop, Resize

from .utils import (
    all_supported_devices,
    ANIMATED_GIF,
    ANIMATED_HEIC,
    assert_frames_equal,
    assert_tensor_close_on_at_least,
    AV1_VIDEO,
    BAD_HUFFMAN_JPEG,
    BT2020_LIMITED_RANGE_10BIT,
    BT601_FULL_RANGE,
    BT601_LIMITED_RANGE,
    BT709_FULL_RANGE,
    CMYK_JPEG,
    CORRUPT_JPEG,
    cuda_devices,
    DISCARD_FIRST_KEYFRAME_VIDEO,
    FRAME_EXCEEDS_SCREEN_GIF,
    get_ffmpeg_minor_version,
    get_python_version,
    GRADIENT_10BIT_AVIF,
    GRADIENT_10BIT_HEIC,
    GRADIENT_12BIT_AVIF,
    GRADIENT_16BIT_PNG,
    GRADIENT_AVIF,
    GRADIENT_GIF,
    GRADIENT_HEIC,
    GRADIENT_INTERLACED_PNG,
    GRADIENT_JPEG,
    GRADIENT_MIRRORED_HEIC,
    GRADIENT_PNG,
    GRADIENT_ROTATED_HEIC,
    GRADIENT_WEBP,
    GRAYSCALE_16BIT_PNG,
    GRAYSCALE_ALPHA_PNG,
    GRAYSCALE_JPEG,
    GRAYSCALE_PNG,
    H264_10BITS,
    H265_VIDEO,
    HEAPBOF_PNG,
    in_fbcode,
    IS_WINDOWS,
    make_video_decoder,
    NASA_AUDIO,
    NASA_AUDIO_MP3,
    NASA_AUDIO_MP3_44100,
    NASA_VIDEO,
    NASA_VIDEO_HDR,
    NASA_VIDEO_ROTATED,
    needs_avif,
    needs_cuda,
    needs_ffmpeg_cli,
    needs_heic,
    needs_jpeg,
    needs_png,
    needs_webp,
    psnr,
    RGBA_AVIF,
    RGBA_HEIC,
    RGBA_PNG,
    RGBA_WEBP,
    SIGSEGV_PNG,
    SINE_16_CHANNEL_S16,
    SINE_MONO_F32,
    SINE_MONO_F64,
    SINE_MONO_S16,
    SINE_MONO_S24,
    SINE_MONO_S32,
    SINE_MONO_S32_44100,
    SINE_MONO_S32_8000,
    SINE_MONO_U8,
    SINE_STEREO_MP2_MPEG_PS,
    TEST_NON_ZERO_START,
    TEST_SRC_2_12BIT_HDR,
    TEST_SRC_2_720P,
    TEST_SRC_2_720P_H265,
    TEST_SRC_2_720P_HDR,
    TEST_SRC_2_720P_MPEG4,
    TEST_SRC_2_720P_VP8,
    TEST_SRC_2_720P_VP9,
    TEST_SRC_2_MPEG4_MP4,
    TESTSRC2_444_10BIT_HEVC,
    TESTSRC2_444_12BIT_HEVC,
    TESTSRC2_444_8BIT_HEVC,
    TESTSRC2_AV1_10BIT,
    TESTSRC2_GBRP_HEVC,
    TESTSRC2_GRAY_HEVC,
    TESTSRC2_ODD_HEIGHT_444,
    TESTSRC2_ODD_HEIGHT_AND_WIDTH_444,
    TESTSRC2_ODD_HEIGHT_AND_WIDTH_444_10BIT,
    TESTSRC2_ODD_HEIGHT_AND_WIDTH_MPEG2,
    TESTSRC2_ODD_HEIGHT_AND_WIDTH_VP9,
    TESTSRC2_ODD_HEIGHT_AND_WIDTH_VP9_10BIT,
    TESTSRC2_ODD_HEIGHT_VP9,
    TESTSRC2_ODD_HEIGHT_VP9_10BIT,
    TESTSRC2_ODD_WIDTH_444,
    TESTSRC2_ODD_WIDTH_MPEG2,
    TESTSRC2_ODD_WIDTH_VP9,
    TESTSRC2_ODD_WIDTH_VP9_10BIT,
    TESTSRC2_YUVA420P_FFV1,
    TRANSPARENT_GIF,
    UNSEEKABLE_SWF,
    WAV_ODD_DATA_TRAILING_CHUNK,
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

                def seek(self, offset: int, whence: int) -> int:
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
        with pytest.raises(ValueError, match="40 is not a valid stream"):
            Decoder(NASA_VIDEO.path, stream_index=40)

        # stream index that does exist, but it's not audio or video
        with pytest.raises(ValueError, match=r"not (a|an) (video|audio) stream"):
            Decoder(NASA_VIDEO.path, stream_index=2)

        # user mistakenly forgets to specify binary reading when creating a file
        # like object from open()
        with pytest.raises(TypeError, match="binary reading?"):
            Decoder(open(NASA_VIDEO.path))


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
    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_getitem_int(self, num_ffmpeg_threads, device, seek_mode):
        decoder, device = make_video_decoder(
            NASA_VIDEO.path,
            num_ffmpeg_threads=num_ffmpeg_threads,
            device=device,
            seek_mode=seek_mode,
        )

        ref_frame0 = NASA_VIDEO.get_frame_data_by_index(0).to(device)
        ref_frame1 = NASA_VIDEO.get_frame_data_by_index(1).to(device)
        ref_frame180 = NASA_VIDEO.get_frame_data_by_index(180).to(device)
        ref_frame_last = NASA_VIDEO.get_frame_data_by_index(389).to(device)

        assert_frames_equal(ref_frame0, decoder[0])
        assert_frames_equal(ref_frame1, decoder[1])
        assert_frames_equal(ref_frame180, decoder[180])
        assert_frames_equal(ref_frame_last, decoder[-1])

    def test_getitem_numpy_int(self):
        decoder = VideoDecoder(NASA_VIDEO.path)

        ref_frame0 = NASA_VIDEO.get_frame_data_by_index(0)
        ref_frame1 = NASA_VIDEO.get_frame_data_by_index(1)
        ref_frame180 = NASA_VIDEO.get_frame_data_by_index(180)
        ref_frame_last = NASA_VIDEO.get_frame_data_by_index(389)

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

    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_getitem_slice(self, device, seek_mode):
        if device == "cuda:ffmpeg" and ffmpeg_major_version == 5:
            pytest.skip("CUDA FFmpeg backend has numerical issues on FFmpeg 5")
        device_param = device  # make_video_decoder shadows `device` below
        decoder, device = make_video_decoder(
            NASA_VIDEO.path, device=device, seek_mode=seek_mode
        )

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
            if not (device_param == "cuda:ffmpeg" and ffmpeg_major_version == 4):
                # TODO: remove the "if".
                # See https://github.com/pytorch/torchcodec/issues/428
                assert_frames_equal(sliced, ref)

    def test_device_instance(self):
        # Non-regression test for https://github.com/pytorch/torchcodec/issues/602
        decoder = VideoDecoder(NASA_VIDEO.path, device=torch.device("cpu"))
        assert isinstance(decoder.metadata, VideoStreamMetadata)

    @pytest.mark.parametrize(
        "device_str",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.needs_cuda),
        ],
    )
    def test_device_none_default_device(self, device_str):
        # VideoDecoder defaults to device=None, which should respect both
        # torch.device() context manager and torch.set_default_device().

        # Test with context manager
        with torch.device(device_str):
            decoder = VideoDecoder(NASA_VIDEO.path)
            assert decoder[0].device.type == device_str

        # Test with set_default_device
        original_device = torch.get_default_device()
        try:
            torch.set_default_device(device_str)
            decoder = VideoDecoder(NASA_VIDEO.path)
            assert decoder[0].device.type == device_str
        finally:
            torch.set_default_device(original_device)

    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_getitem_fails(self, device, seek_mode):
        decoder, _ = make_video_decoder(
            NASA_VIDEO.path, device=device, seek_mode=seek_mode
        )

        with pytest.raises(IndexError, match="Invalid frame index"):
            frame = decoder[1000]  # noqa

        with pytest.raises(IndexError, match="Invalid frame index"):
            frame = decoder[-1000]  # noqa

        with pytest.raises(TypeError, match="Unsupported key type"):
            frame = decoder["0"]  # noqa

        with pytest.raises(TypeError, match="Unsupported key type"):
            frame = decoder[2.3]  # noqa

    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_iteration(self, device, seek_mode):
        if device == "cuda:ffmpeg" and ffmpeg_major_version == 5:
            pytest.skip("CUDA FFmpeg backend has numerical issues on FFmpeg 5")
        decoder, device = make_video_decoder(
            NASA_VIDEO.path, device=device, seek_mode=seek_mode
        )

        ref_frame0 = NASA_VIDEO.get_frame_data_by_index(0).to(device)
        ref_frame1 = NASA_VIDEO.get_frame_data_by_index(1).to(device)
        ref_frame9 = NASA_VIDEO.get_frame_data_by_index(9).to(device)
        ref_frame35 = NASA_VIDEO.get_frame_data_by_index(35).to(device)
        ref_frame180 = NASA_VIDEO.get_frame_data_by_index(180).to(device)
        ref_frame_last = NASA_VIDEO.get_frame_data_by_index(389).to(device)

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

    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frame_at(self, device, seek_mode):
        decoder, device = make_video_decoder(
            NASA_VIDEO.path, device=device, seek_mode=seek_mode
        )

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

    @pytest.mark.parametrize("device", all_supported_devices())
    def test_get_frame_at_tuple_unpacking(self, device):
        decoder, _ = make_video_decoder(NASA_VIDEO.path, device=device)

        frame = decoder.get_frame_at(50)
        data, pts, duration = decoder.get_frame_at(50)

        assert_frames_equal(frame.data, data)
        assert frame.pts_seconds == pts
        assert frame.duration_seconds == duration

    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frame_at_fails(self, device, seek_mode):
        decoder, _ = make_video_decoder(
            NASA_VIDEO.path, device=device, seek_mode=seek_mode
        )

        with pytest.raises(
            IndexError,
            match="negative indices must have an absolute value less than the number of frames",
        ):
            frame = decoder.get_frame_at(-10000)  # noqa

        with pytest.raises(IndexError, match="must be less than"):
            frame = decoder.get_frame_at(10000)  # noqa

    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_at(self, device, seek_mode):
        decoder, device = make_video_decoder(
            NASA_VIDEO.path, device=device, seek_mode=seek_mode
        )

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

    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_at_fails(self, device, seek_mode):
        decoder, _ = make_video_decoder(
            NASA_VIDEO.path, device=device, seek_mode=seek_mode
        )

        with pytest.raises(
            IndexError,
            match="negative indices must have an absolute value less than the number of frames",
        ):
            decoder.get_frames_at([-10000])

        with pytest.raises(IndexError, match="Invalid frame index=390"):
            decoder.get_frames_at([390])

        with pytest.raises(RuntimeError, match="Long but found Float"):
            decoder.get_frames_at([0.3])

    @pytest.mark.parametrize("device", all_supported_devices())
    def test_get_frame_at_av1(self, device):
        if device == "cuda:ffmpeg" and ffmpeg_major_version in (4, 5):
            return

        if "cuda" in device and in_fbcode():
            pytest.skip("decoding on CUDA is not supported internally")

        decoder, device = make_video_decoder(AV1_VIDEO.path, device=device)
        ref_frame10 = AV1_VIDEO.get_frame_data_by_index(10)
        ref_frame_info10 = AV1_VIDEO.get_frame_info(10)
        decoded_frame10 = decoder.get_frame_at(10)
        assert decoded_frame10.duration_seconds == ref_frame_info10.duration_seconds
        assert decoded_frame10.pts_seconds == ref_frame_info10.pts_seconds
        assert_frames_equal(decoded_frame10.data, ref_frame10.to(device=device))

    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frame_played_at(self, device, seek_mode):
        decoder, device = make_video_decoder(
            NASA_VIDEO.path, device=device, seek_mode=seek_mode
        )

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

    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_at_backward_seek_after_eof(self, seek_mode, device):
        # Regression test for https://github.com/meta-pytorch/torchcodec/issues/1339.
        # For HEVC codecs (e.g. libx265), decoding a frame near EOF and recieving EOF,
        # then seeking backwards returns a stale frame instead of the requested earlier frame.
        reference_decoder, _ = make_video_decoder(
            TEST_SRC_2_720P_H265.path, device=device, seek_mode=seek_mode
        )
        decoder, _ = make_video_decoder(
            TEST_SRC_2_720P_H265.path, device=device, seek_mode=seek_mode
        )
        expected_frame0 = reference_decoder.get_frame_at(0)
        expected_frame58 = reference_decoder.get_frame_at(58)

        frame58 = decoder.get_frame_at(58)
        frame0 = decoder.get_frame_at(0)

        assert frame58.pts_seconds == expected_frame58.pts_seconds
        assert frame0.pts_seconds == expected_frame0.pts_seconds
        assert_frames_equal(expected_frame0.data, frame0.data)

    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frame_played_at_fails(self, device, seek_mode):
        decoder, _ = make_video_decoder(
            NASA_VIDEO.path, device=device, seek_mode=seek_mode
        )

        with pytest.raises(IndexError, match="Invalid pts in seconds"):
            frame = decoder.get_frame_played_at(-1.0)  # noqa

        with pytest.raises(IndexError, match="Invalid pts in seconds"):
            frame = decoder.get_frame_played_at(100.0)  # noqa

    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    @pytest.mark.parametrize("input_type", ("list", "tensor"))
    def test_get_frames_played_at(self, device, seek_mode, input_type):
        decoder, device = make_video_decoder(
            NASA_VIDEO.path, device=device, seek_mode=seek_mode
        )

        # Note: We know the frame at ~0.84s has index 25, the one at 1.16s has
        # index 35. We use those indices as reference to test against.
        if input_type == "list":
            seconds = [0.84, 1.17, 0.85]
        else:  # tensor
            seconds = torch.tensor([0.84, 1.17, 0.85])

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

    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_played_at_fails(self, device, seek_mode):
        decoder, _ = make_video_decoder(
            NASA_VIDEO.path, device=device, seek_mode=seek_mode
        )

        with pytest.raises(RuntimeError, match="must be greater than or equal to"):
            decoder.get_frames_played_at([-1])

        with pytest.raises(RuntimeError, match="must be less than"):
            decoder.get_frames_played_at([14])

        with pytest.raises(
            ValueError, match="Couldn't convert timestamps input to a tensor"
        ):
            decoder.get_frames_played_at(["bad"])

    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("stream_index", [0, 3, None])
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_in_range(self, stream_index, device, seek_mode):
        if device == "cuda:ffmpeg" and ffmpeg_major_version == 5:
            pytest.skip("CUDA FFmpeg backend has numerical issues on FFmpeg 5")
        decoder, device = make_video_decoder(
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

    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_in_range_slice_indices_syntax(self, device, seek_mode):
        if device == "cuda:ffmpeg" and ffmpeg_major_version == 5:
            pytest.skip("CUDA FFmpeg backend has numerical issues on FFmpeg 5")
        decoder, device = make_video_decoder(
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
    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_dimension_order(self, dimension_order, frame_getter, device, seek_mode):
        decoder, _ = make_video_decoder(
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
    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_by_pts_in_range(self, stream_index, device, seek_mode):
        if device == "cuda:ffmpeg" and ffmpeg_major_version == 5:
            pytest.skip("CUDA FFmpeg backend has numerical issues on FFmpeg 5")
        decoder, device = make_video_decoder(
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

    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_by_pts_in_range_fails(self, device, seek_mode):
        decoder, _ = make_video_decoder(
            NASA_VIDEO.path, device=device, seek_mode=seek_mode
        )

        with pytest.raises(ValueError, match="Invalid start seconds"):
            frame = decoder.get_frames_played_in_range(100.0, 1.0)  # noqa

        with pytest.raises(ValueError, match="Invalid start seconds"):
            frame = decoder.get_frames_played_in_range(20, 23)  # noqa

        with pytest.raises(ValueError, match="Invalid stop seconds"):
            frame = decoder.get_frames_played_in_range(0, 23)  # noqa

    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_played_in_range_with_fps(self, device, seek_mode):
        if device == "cuda:ffmpeg" and ffmpeg_major_version == 5:
            pytest.skip("CUDA FFmpeg backend has numerical issues on FFmpeg 5")
        decoder, _ = make_video_decoder(
            NASA_VIDEO.path, device=device, seek_mode=seek_mode
        )

        source_fps = decoder.metadata.average_fps
        duration_seconds = 1.0
        start_seconds = decoder.get_frame_at(0).pts_seconds
        frame1_pts = decoder.get_frame_at(1).pts_seconds
        stop_seconds = start_seconds + duration_seconds

        # Test downsampling: request lower fps than source
        fps_low = 5
        frames_low_fps = decoder.get_frames_played_in_range(
            start_seconds, stop_seconds, fps=fps_low
        )
        expected_frames_low = round(duration_seconds * fps_low)
        assert len(frames_low_fps) == expected_frames_low
        # First output frame should be frame 0
        frame0_data = decoder.get_frame_at(0).data
        torch.testing.assert_close(frames_low_fps.data[0], frame0_data, atol=0, rtol=0)
        # Second output frame should NOT be frame 1 (we're downsampling)
        frame1_data = decoder.get_frame_at(1).data
        assert not torch.equal(frames_low_fps.data[1], frame1_data)

        # Test upsampling: request higher fps than source (frames should be duplicated)
        # Request 3x the source fps for a single frame's duration
        fps_high = int(source_fps * 3)
        frames_high_fps = decoder.get_frames_played_in_range(
            start_seconds, frame1_pts, fps=fps_high
        )
        # All frames should be duplicates of frame 0 since we're within frame 0's display time
        frame_duration = frame1_pts - start_seconds
        expected_frames_high = round(frame_duration * fps_high)
        assert len(frames_high_fps) == expected_frames_high

        # All duplicated frames should have the same content as frame 0
        frame0_data = decoder.get_frame_at(0).data
        if not (device == "cuda:ffmpeg" and ffmpeg_major_version == 4):
            for i in range(len(frames_high_fps)):
                torch.testing.assert_close(
                    frames_high_fps.data[i], frame0_data, atol=0, rtol=0
                )

        # Test that fps=None returns the original behavior (same as not passing fps)
        frames_no_fps = decoder.get_frames_played_in_range(start_seconds, stop_seconds)
        frames_none_fps = decoder.get_frames_played_in_range(
            start_seconds, stop_seconds, fps=None
        )
        assert len(frames_no_fps) == len(frames_none_fps)
        if not (device == "cuda:ffmpeg" and ffmpeg_major_version == 4):
            torch.testing.assert_close(
                frames_no_fps.data, frames_none_fps.data, atol=0, rtol=0
            )

    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_frames_played_in_range_with_fps_fails(self, device, seek_mode):
        decoder, _ = make_video_decoder(
            NASA_VIDEO.path, device=device, seek_mode=seek_mode
        )

        start_seconds = decoder.get_frame_at(0).pts_seconds
        stop_seconds = start_seconds + 1.0

        with pytest.raises(RuntimeError, match="fps must be positive"):
            decoder.get_frames_played_in_range(start_seconds, stop_seconds, fps=0)

        with pytest.raises(RuntimeError, match="fps must be positive"):
            decoder.get_frames_played_in_range(start_seconds, stop_seconds, fps=-10)

    @pytest.mark.parametrize("fps", [5.0, 15.0, 24.0, 29.97, 30.1, 60.0])
    @pytest.mark.parametrize("full_video", [False, True])
    def test_get_frames_played_in_range_fps_matches_torchvision(self, fps, full_video):
        """Test that TorchCodec's fps output matches torchvision's resampling logic."""
        decoder = VideoDecoder(NASA_VIDEO.path)

        if full_video:
            start_seconds = decoder.metadata.begin_stream_seconds
            stop_seconds = decoder.metadata.end_stream_seconds
        else:
            start_seconds = 0.0
            stop_seconds = start_seconds + 1.0

        # Get resampled frames using our fps feature
        tc_frames_batch = decoder.get_frames_played_in_range(
            start_seconds=start_seconds,
            stop_seconds=stop_seconds,
            fps=fps,
        )

        # Get all source frames in the range
        all_source_frames = decoder.get_frames_played_in_range(
            start_seconds=start_seconds,
            stop_seconds=stop_seconds,
        )

        # Compute expected indices using torchvision's resampling logic:
        # https://github.com/pytorch/vision/blob/1e53952f57462e4c28103835cf1f9e504dbea84b/torchvision/datasets/video_utils.py#L278
        # For each output frame i, select source frame at index floor(i * step)
        # where step = original_fps / target_fps
        original_fps = decoder.metadata.average_fps
        step = original_fps / fps
        expected_indices = (
            (torch.arange(len(tc_frames_batch), dtype=torch.float32) * step)
            .floor()
            .to(torch.int64)
        )
        expected_frames = all_source_frames.data[expected_indices]

        torch.testing.assert_close(
            tc_frames_batch.data,
            expected_frames,
            rtol=0,
            atol=0,
        )

    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_get_all_frames(self, device, seek_mode):
        """Test that get_all_frames returns all frames and is equivalent to get_frames_played_in_range."""
        decoder, _ = make_video_decoder(
            NASA_VIDEO.path, device=device, seek_mode=seek_mode
        )

        all_frames = decoder.get_all_frames()

        assert len(all_frames) == len(decoder)

        frames_in_range = decoder.get_frames_played_in_range(
            start_seconds=decoder.metadata.begin_stream_seconds,
            stop_seconds=decoder.metadata.end_stream_seconds,
        )
        assert len(all_frames) == len(frames_in_range)
        # Use strict bitwise equality, except for FFmpeg 4 and 5 + CUDA FFmpeg
        # interface which has known issues (see #428)
        if not (device == "cuda:ffmpeg" and ffmpeg_major_version in (4, 5)):
            torch.testing.assert_close(
                all_frames.data, frames_in_range.data, atol=0, rtol=0
            )

        fps = 10.0
        all_frames_with_fps = decoder.get_all_frames(fps=fps)
        frames_in_range_with_fps = decoder.get_frames_played_in_range(
            start_seconds=decoder.metadata.begin_stream_seconds,
            stop_seconds=decoder.metadata.end_stream_seconds,
            fps=fps,
        )
        assert len(all_frames_with_fps) == len(frames_in_range_with_fps)
        # Use strict bitwise equality, except for FFmpeg 4 and 5 + CUDA FFmpeg
        # interface which has known issues (see #428)
        if not (device == "cuda:ffmpeg" and ffmpeg_major_version in (4, 5)):
            torch.testing.assert_close(
                all_frames_with_fps.data, frames_in_range_with_fps.data, atol=0, rtol=0
            )

    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_non_zero_start_pts(self, seek_mode):
        """Test that frame retrieval methods return correct PTS values for videos with non-zero start time.

        This is a non-regression test for https://github.com/meta-pytorch/torchcodec/pull/1209
        """
        decoder = VideoDecoder(TEST_NON_ZERO_START.path, seek_mode=seek_mode)

        # Verify the video has a non-zero start time
        assert decoder.metadata.begin_stream_seconds > 0
        expected_start_time = TEST_NON_ZERO_START.get_frame_info(0).pts_seconds
        assert expected_start_time == pytest.approx(8.333, rel=1e-3)

        frame0 = decoder.get_frame_at(0)
        assert frame0.pts_seconds == pytest.approx(expected_start_time, rel=1e-3)

        frame1 = decoder.get_frame_at(1)
        expected_frame1_pts = TEST_NON_ZERO_START.get_frame_info(1).pts_seconds
        assert frame1.pts_seconds == pytest.approx(expected_frame1_pts, rel=1e-3)

        frames = decoder.get_frames_at([0, 1, 2])
        for i, expected_idx in enumerate([0, 1, 2]):
            expected_pts = TEST_NON_ZERO_START.get_frame_info(expected_idx).pts_seconds
            assert frames.pts_seconds[i].item() == pytest.approx(expected_pts, rel=1e-3)

        frame_at_start = decoder.get_frame_played_at(expected_start_time)
        assert frame_at_start.pts_seconds == pytest.approx(
            expected_start_time, rel=1e-3
        )

        frames_range = decoder.get_frames_in_range(0, 3)
        for i in range(3):
            expected_pts = TEST_NON_ZERO_START.get_frame_info(i).pts_seconds
            assert frames_range.pts_seconds[i].item() == pytest.approx(
                expected_pts, rel=1e-3
            )

        # Use the decoder's own PTS value to avoid floating point precision issues
        # between ffprobe's PTS (in JSON) and the decoder's computed PTS
        frame3 = decoder.get_frame_at(3)
        stop_pts = frame3.pts_seconds
        frames_pts_range = decoder.get_frames_played_in_range(
            expected_start_time, stop_pts
        )
        # Should get frames 0, 1, 2 (stop is exclusive)
        assert len(frames_pts_range) == 3
        for i in range(3):
            expected_pts = TEST_NON_ZERO_START.get_frame_info(i).pts_seconds
            assert frames_pts_range.pts_seconds[i].item() == pytest.approx(
                expected_pts, rel=1e-3
            )

    @pytest.mark.parametrize("device", all_supported_devices())
    def test_get_key_frame_indices(self, device):
        decoder, _ = make_video_decoder(
            NASA_VIDEO.path, device=device, seek_mode="exact"
        )
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

        decoder, _ = make_video_decoder(
            AV1_VIDEO.path, device=device, seek_mode="exact"
        )
        key_frame_indices = decoder._get_key_frame_indices()

        # $ ffprobe -v error -hide_banner -select_streams v:0 -show_frames -of csv test/resources/av1_video.mkv | grep -n ",I," | cut -d ':' -f 1 > key_frames.txt
        av1_reference_key_frame_indices = torch.tensor([0])

        torch.testing.assert_close(
            key_frame_indices, av1_reference_key_frame_indices, atol=0, rtol=0
        )

        decoder, _ = make_video_decoder(
            H265_VIDEO.path, device=device, seek_mode="exact"
        )
        key_frame_indices = decoder._get_key_frame_indices()

        # ffprobe -v error -hide_banner -select_streams v:0 -show_frames -of csv test/resources/h265_video.mp4 | grep -n ",I," | cut -d ':' -f 1 > key_frames.txt
        h265_reference_key_frame_indices = torch.tensor([0, 2, 4, 6, 8])

        torch.testing.assert_close(
            key_frame_indices, h265_reference_key_frame_indices, atol=0, rtol=0
        )

    @pytest.mark.parametrize(
        "device", ("cpu", pytest.param("cuda", marks=pytest.mark.needs_cuda))
    )
    def test_discard_first_keyframe(self, device):
        # Non-regression test for TODO
        decoder, device = make_video_decoder(
            DISCARD_FIRST_KEYFRAME_VIDEO.path, device=device
        )

        # The 5 discarded frames (incl. the first keyframe) must be excluded
        # from the frame count: the decoder only ever emits 25 frames.
        assert decoder.metadata.num_frames == 25
        assert len(decoder) == 25

        assert decoder.get_frame_at(0).pts_seconds == 0

        all_frames = decoder[:]
        assert all_frames.shape[0] == 25

        pts = [decoder.get_frame_at(i).pts_seconds for i in range(len(decoder))]
        expected_pts = [0.04 * i for i in range(25)]
        assert pts == pytest.approx(expected_pts, abs=1e-6)

        # The discarded keyframe is not itself an output frame, so it is not
        # reported as a key frame: output frame 0 is a P-frame that depends on
        # it. Only the two non-discarded keyframes (output frames 5 and 15) are
        # reported. key_frames stays in sync with all_frames.
        assert decoder._get_key_frame_indices().tolist() == [5, 15]

        # Random access must agree with sequential decoding: seeking to a frame
        # must return the same pixels as decoding straight through, including
        # frames 0-4, whose keyframe was discarded (we rely on FFmpeg to seek to
        # that keyframe since it is not in our scanned index).
        for i in (0, 4, 5, 6, 15, 16, 24):
            assert_frames_equal(decoder.get_frame_at(i).data, all_frames.data[i])

        # Note that the num_frames_from_header is 30, which is technically
        # correct, but there are only 25 decodeable frames. This means
        # approximate mode will try to decode past frame 25 and fail. Not sure
        # what we can do about this.
        assert decoder.metadata.num_frames_from_header == 30
        decoder_approx, _ = make_video_decoder(
            DISCARD_FIRST_KEYFRAME_VIDEO.path, device=device, seek_mode="approximate"
        )

        with pytest.raises(
            RuntimeError, match="Requested next frame while there are no more frames"
        ):
            # Tries to decode [0, 30) but only 25 frames are decodeable.
            decoder_approx[:]

    # TODO investigate why this is failing from the nightlies of Dec 09 2025.
    @pytest.mark.skip(reason="TODO investigate")
    # TODO investigate why this fails internally.
    @pytest.mark.skipif(in_fbcode(), reason="Compile test fails internally.")
    @pytest.mark.skipif(
        get_python_version() >= (3, 14),
        reason="torch.compile is not supported on Python 3.14+",
    )
    @pytest.mark.parametrize("device", all_supported_devices())
    def test_compile(self, device):
        decoder, device = make_video_decoder(NASA_VIDEO.path, device=device)

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

    # The test video we have is from
    # https://huggingface.co/datasets/raushan-testing-hf/videos-test/blob/main/sample_video_2.avi
    # We can't check it into the repo due to potential licensing issues, so
    # we have to unconditionally skip this test.
    # TODO: encode a video with no pts values to unskip this test. Couldn't
    # find a way to do that with FFmpeg's CLI, but this should be doable
    # once we have our own video encoder.
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    @pytest.mark.skip(reason="TODO: Need video with no pts values.")
    def test_pts_to_dts_fallback(self, seek_mode):
        # Non-regression test for
        # https://github.com/pytorch/torchcodec/issues/677 and
        # https://github.com/pytorch/torchcodec/issues/676.
        # More accurately, this is a non-regression test for videos which do
        # *not* specify pts values (all pts values are N/A and set to
        # INT64_MIN), but specify *dts* value - which we fallback to.
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
    @pytest.mark.parametrize("asset", (BT709_FULL_RANGE, NASA_VIDEO))
    def test_full_and_studio_range_bt709_video(self, asset):
        # Test ensuring result consistency between CPU and GPU decoder on BT709
        # videos, one with full color range, one with studio range.
        # This is a non-regression test for times when we used to not support
        # full range on GPU.
        #
        # NASA_VIDEO is a BT709 studio range video, as can be confirmed with
        # ffprobe -v quiet -select_streams v:0 -show_entries
        # stream=color_space,color_transfer,color_primaries,color_range -of
        # default=noprint_wrappers=1 test/resources/nasa_13013.mp4
        decoder_gpu = VideoDecoder(asset.path, device="cuda")
        decoder_cpu = VideoDecoder(asset.path, device="cpu")

        for frame_index in (0, 10, 20, 5):
            gpu_frame = decoder_gpu.get_frame_at(frame_index).data.cpu()
            cpu_frame = decoder_cpu.get_frame_at(frame_index).data

            torch.testing.assert_close(gpu_frame, cpu_frame, rtol=0, atol=3)

    @needs_cuda
    def test_bt2020_10bit_video(self):
        # Test ensuring result consistency between CPU and default CUDA (NVDEC)
        # decoder on a BT.2020 10-bit video (limited range). This is a
        # non-regression test for BT.2020 color conversion support.
        #
        # bt2020_10bit.mp4 is a BT.2020 limited range 10-bit HEVC video:
        # color_space=bt2020nc, color_range=tv, pix_fmt=yuv420p10le
        #
        # NVDEC decodes 10-bit natively (converting to 8-bit NV12), then our
        # BT.2020 color twist matrix handles the YUV->RGB conversion.
        #
        # TODO investigate CPU vs default CUDA (NVDEC) mismatch on BT.2020 10-bit.
        # See PR #1267 for details.
        asset = BT2020_LIMITED_RANGE_10BIT

        decoder_gpu = VideoDecoder(asset.path, device="cuda")
        decoder_cpu = VideoDecoder(asset.path, device="cpu")

        for frame_index in (0, 10, 20, 5):
            gpu_frame = decoder_gpu.get_frame_at(frame_index).data.cpu()
            cpu_frame = decoder_cpu.get_frame_at(frame_index).data

            assert_tensor_close_on_at_least(gpu_frame, cpu_frame, percentage=90, atol=3)

    @needs_cuda
    @pytest.mark.parametrize(
        "asset",
        (BT601_FULL_RANGE, BT601_LIMITED_RANGE),
    )
    def test_bt601_colorspace(self, asset):
        # Test ensuring result consistency between CPU and default CUDA (NVDEC)
        # decoder on BT.601 videos with full and limited range.
        decoder_gpu = VideoDecoder(asset.path, device="cuda")
        decoder_cpu = VideoDecoder(asset.path, device="cpu")

        for frame_index in (0, 10, 20, 5):
            gpu_frame = decoder_gpu.get_frame_at(frame_index).data.cpu()
            cpu_frame = decoder_cpu.get_frame_at(frame_index).data

            torch.testing.assert_close(gpu_frame, cpu_frame, rtol=0, atol=3)

    @needs_cuda
    @pytest.mark.parametrize(
        "asset",
        (
            TESTSRC2_ODD_WIDTH_444,
            TESTSRC2_ODD_HEIGHT_444,
            TESTSRC2_ODD_HEIGHT_AND_WIDTH_444,
        ),
    )
    @pytest.mark.parametrize("device", ("cuda", "cuda:ffmpeg"))
    @pytest.mark.parametrize("output_dtype", (torch.uint8, torch.float32))
    def test_odd_sized_videos_444(self, asset, device, output_dtype):
        # These are yuv444p H264 videos. On the beta CUDA backend, 4:4:4
        # chroma isn't supported by NVDEC so these go through the CPU
        # fallback path entirely (decoding + color conversion on CPU).
        if output_dtype == torch.float32 and device == "cuda:ffmpeg":
            pytest.skip("float32 output not relevant for cuda:ffmpeg here")

        decoder_gpu, _ = make_video_decoder(
            asset.path, device=device, output_dtype=output_dtype
        )
        if device == "cuda":
            assert decoder_gpu.cpu_fallback
        decoder_cpu = VideoDecoder(asset.path, device="cpu", output_dtype=output_dtype)

        gpu_frame = decoder_gpu.get_frame_at(0).data.cpu()
        cpu_frame = decoder_cpu.get_frame_at(0).data
        assert gpu_frame.shape == cpu_frame.shape
        assert gpu_frame.dtype == output_dtype
        assert_tensor_close_on_at_least(gpu_frame, cpu_frame, percentage=89, atol=3)

        gpu_frames = decoder_gpu.get_frames_at([0, 1, 2]).data.cpu()
        cpu_frames = decoder_cpu.get_frames_at([0, 1, 2]).data
        assert gpu_frames.shape == cpu_frames.shape
        assert_tensor_close_on_at_least(gpu_frames, cpu_frames, percentage=89, atol=3)

    @needs_cuda
    @pytest.mark.parametrize(
        "asset, percentage",
        (
            (TESTSRC2_444_8BIT_HEVC, 99),
            (TESTSRC2_444_10BIT_HEVC, 99),
            (TESTSRC2_444_12BIT_HEVC, 99),
            (TESTSRC2_AV1_10BIT, 89),
        ),
    )
    @pytest.mark.parametrize("output_dtype", (torch.uint8, torch.float32))
    def test_nvdec_native_decoding(self, asset, percentage, output_dtype):
        # Streams NVDEC can decode but that used to hit the CPU fallback,
        # because we only ever asked it for an NV12 or a P016 surface:
        # - 4:4:4, which needs the YUV444 surfaces.
        # - AV1 10-bit, for which NVDEC offers only P016, so a uint8 request
        #   found no 8-bit surface to decode into (float32 already worked).
        decoder_gpu = VideoDecoder(asset.path, device="cuda", output_dtype=output_dtype)
        decoder_cpu = VideoDecoder(asset.path, device="cpu", output_dtype=output_dtype)
        assert not decoder_gpu.cpu_fallback

        gpu_frame = decoder_gpu.get_frame_at(0).data
        cpu_frame = decoder_cpu.get_frame_at(0).data
        assert gpu_frame.shape == cpu_frame.shape
        assert gpu_frame.dtype == output_dtype
        assert_tensor_close_on_at_least(
            gpu_frame.cpu(), cpu_frame, percentage=percentage, atol=3
        )

        gpu_frames = decoder_gpu.get_frames_at([0, 1, 2]).data
        cpu_frames = decoder_cpu.get_frames_at([0, 1, 2]).data
        assert gpu_frames.shape == cpu_frames.shape
        assert gpu_frames.dtype == output_dtype
        assert_tensor_close_on_at_least(
            gpu_frames.cpu(), cpu_frames, percentage=percentage, atol=3
        )

    @needs_cuda
    @pytest.mark.parametrize(
        "asset",
        (
            TESTSRC2_ODD_WIDTH_VP9,
            TESTSRC2_ODD_HEIGHT_VP9,
            TESTSRC2_ODD_HEIGHT_AND_WIDTH_VP9,
            TESTSRC2_ODD_WIDTH_VP9_10BIT,
            TESTSRC2_ODD_HEIGHT_VP9_10BIT,
            TESTSRC2_ODD_HEIGHT_AND_WIDTH_VP9_10BIT,
        ),
    )
    @pytest.mark.parametrize("output_dtype", (torch.uint8, torch.float32))
    def test_odd_sized_videos_vp9(self, asset, output_dtype):
        # These are VP9 yuv420p / yuv420p10le videos. VP9 supports odd
        # dimensions with 4:2:0 chroma. They are decoded by NVDEC directly
        # (no CPU fallback), exercising convertNV12FrameToRGB (uint8) and
        # convertP016FrameToRGB16 (float32) with odd dimensions.
        decoder_gpu, _ = make_video_decoder(
            asset.path, device="cuda", output_dtype=output_dtype
        )
        assert not decoder_gpu.cpu_fallback
        decoder_cpu = VideoDecoder(asset.path, device="cpu", output_dtype=output_dtype)

        gpu_frame = decoder_gpu.get_frame_at(0).data.cpu()
        cpu_frame = decoder_cpu.get_frame_at(0).data
        assert gpu_frame.shape == cpu_frame.shape
        assert gpu_frame.dtype == output_dtype
        assert_tensor_close_on_at_least(gpu_frame, cpu_frame, percentage=89, atol=3)

        gpu_frames = decoder_gpu.get_frames_at([0, 1, 2]).data.cpu()
        cpu_frames = decoder_cpu.get_frames_at([0, 1, 2]).data
        assert gpu_frames.shape == cpu_frames.shape
        assert_tensor_close_on_at_least(gpu_frames, cpu_frames, percentage=89, atol=3)

    @needs_cuda
    @pytest.mark.parametrize(
        "asset", (TESTSRC2_ODD_WIDTH_MPEG2, TESTSRC2_ODD_HEIGHT_AND_WIDTH_MPEG2)
    )
    @pytest.mark.parametrize("output_dtype", (torch.uint8, torch.float32))
    def test_odd_sized_video_420_cpu_fallback(self, asset, output_dtype):
        # MPEG-2 isn't decoded by NVDEC, so these yuv420p videos go through the
        # CPU fallback: the frame is decoded on the CPU, then uploaded as NV12
        # (or P016) for the GPU color conversion. Their odd dimensions mean the
        # upload has to pad the frame to even ones, which the color conversion
        # then crops away - if it padded by rescaling instead, or forgot to
        # crop, the frames below would be shifted or too large.
        decoder_gpu, _ = make_video_decoder(
            asset.path, device="cuda", output_dtype=output_dtype
        )
        assert decoder_gpu.cpu_fallback
        decoder_cpu = VideoDecoder(asset.path, device="cpu", output_dtype=output_dtype)

        gpu_frames = decoder_gpu.get_frames_at([0, 1, 2]).data.cpu()
        cpu_frames = decoder_cpu.get_frames_at([0, 1, 2]).data
        expected_shape = (3, 3, asset.height, asset.width)
        assert gpu_frames.shape == expected_shape
        assert cpu_frames.shape == expected_shape
        assert gpu_frames.dtype == output_dtype

        if asset is TESTSRC2_ODD_HEIGHT_AND_WIDTH_MPEG2:
            # An odd height stops swscale from using its fast unscaled
            # yuv420p -> rgb converter, which pairs each chroma row with exactly
            # two luma rows. It falls back to the general path and *resizes* the
            # chroma plane's ceil(height / 2) rows onto `height` rows - a ratio
            # just under 2, interpolated. We replicate chroma exactly 2x, like
            # NVDEC does, so the two disagree along every colour edge in the
            # frame. Only ~79% of samples land within 5, hence the loose bound;
            # the pixels themselves are checked exactly by
            # TestBlocks::test_matches_video_decoder, where both sides are ours.
            percentage, atol = 75, 5
        else:
            percentage, atol = 98, 3
        assert_tensor_close_on_at_least(
            gpu_frames,
            cpu_frames,
            percentage=percentage,
            atol=atol if output_dtype == torch.uint8 else atol / 255,
        )

    @needs_cuda
    def test_10bit_gpu_fallsback_to_cpu(self):
        # Test for 10-bit videos that aren't supported by NVDEC: we decode and
        # do the color conversion on the CPU.
        # Here we just assert that the GPU results are the same as the CPU
        # results.
        #
        # This test exercises the FFmpeg CUDA interface specifically: its CPU
        # fallback delegates directly to CpuDeviceInterface, so the output
        # matches a pure CPU decoder bit-for-bit. The NVDEC interface
        # has a different fallback path that round-trips through GPU NV12 (an
        # 8-bit format) and produces different output for 10-bit content.

        # We know from previous tests that the H264_10BITS video isn't supported
        # by NVDEC, so NVDEC decodes it on the CPU.
        asset = H264_10BITS

        with set_cuda_backend("ffmpeg"):
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

    def setup_frame_mappings(tmp_path, file, stream_index):
        json_path = tmp_path / "custom_frame_mappings.json"
        custom_frame_mappings = NASA_VIDEO.generate_custom_frame_mappings(stream_index)
        if file:
            # Write the custom frame mappings to a JSON file
            with open(json_path, "w") as f:
                f.write(custom_frame_mappings)
            return json_path
        else:
            # Return the custom frame mappings as a JSON string
            return custom_frame_mappings

    @needs_ffmpeg_cli
    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize("stream_index", [0, 3])
    @pytest.mark.parametrize(
        "method",
        (
            partial(setup_frame_mappings, file=True),
            partial(setup_frame_mappings, file=False),
        ),
    )
    def test_custom_frame_mappings_json_and_bytes(
        self, tmp_path, device, stream_index, method
    ):
        if device == "cuda:ffmpeg" and ffmpeg_major_version == 5:
            pytest.skip("CUDA FFmpeg backend has numerical issues on FFmpeg 5")
        custom_frame_mappings = method(tmp_path=tmp_path, stream_index=stream_index)
        # Optionally open the custom frame mappings file if it is a file path
        # or use a null context if it is a string.
        with (
            open(custom_frame_mappings)
            if hasattr(custom_frame_mappings, "read")
            else contextlib.nullcontext()
        ) as custom_frame_mappings:
            decoder, device = make_video_decoder(
                NASA_VIDEO.path,
                stream_index=stream_index,
                device=device,
                custom_frame_mappings=custom_frame_mappings,
            )
        frame_0 = decoder.get_frame_at(0)
        frame_5 = decoder.get_frame_at(5)
        assert_frames_equal(
            frame_0.data,
            NASA_VIDEO.get_frame_data_by_index(0, stream_index=stream_index).to(device),
        )
        assert_frames_equal(
            frame_5.data,
            NASA_VIDEO.get_frame_data_by_index(5, stream_index=stream_index).to(device),
        )
        frames0_5 = decoder.get_frames_played_in_range(
            frame_0.pts_seconds, frame_5.pts_seconds
        )
        assert_frames_equal(
            frames0_5.data,
            NASA_VIDEO.get_frame_data_by_range(0, 5, stream_index=stream_index).to(
                device
            ),
        )

    @needs_ffmpeg_cli
    @pytest.mark.parametrize("device", all_supported_devices())
    @pytest.mark.parametrize(
        "custom_frame_mappings,expected_match",
        [
            pytest.param(
                None,
                "seek_mode",
                id="valid_content_approximate",
            ),
            ("{}", "The input is empty or missing the required 'frames' key."),
            (
                '{"valid": "json"}',
                "The input is empty or missing the required 'frames' key.",
            ),
            (
                '{"frames": [{"missing": "keys"}]}',
                "keys are required in the frame metadata.",
            ),
        ],
    )
    def test_custom_frame_mappings_init_fails(
        self, device, custom_frame_mappings, expected_match
    ):
        if custom_frame_mappings is None:
            custom_frame_mappings = NASA_VIDEO.generate_custom_frame_mappings(0)
        with pytest.raises(ValueError, match=expected_match):
            VideoDecoder(
                NASA_VIDEO.path,
                stream_index=0,
                device=device,
                custom_frame_mappings=custom_frame_mappings,
                seek_mode=("approximate" if expected_match == "seek_mode" else "exact"),
            )

    @pytest.mark.parametrize("device", all_supported_devices())
    def test_custom_frame_mappings_init_fails_invalid_json(self, tmp_path, device):
        invalid_json_path = tmp_path / "invalid_json"
        with open(invalid_json_path, "w+") as f:
            f.write("invalid input")

        # Test both file object and string
        with open(invalid_json_path) as file_obj:
            for custom_frame_mappings in [
                file_obj,
                file_obj.read(),
            ]:
                with pytest.raises(ValueError, match="Invalid custom frame mappings"):
                    VideoDecoder(
                        NASA_VIDEO.path,
                        stream_index=0,
                        device=device,
                        custom_frame_mappings=custom_frame_mappings,
                    )

    def test_get_frames_at_tensor_indices(self):
        # Non-regression test for tensor support in get_frames_at() and
        # get_frames_played_at()
        decoder = VideoDecoder(NASA_VIDEO.path)

        decoder.get_frames_at(torch.tensor([0, 10], dtype=torch.int))
        decoder.get_frames_at(torch.tensor([0, 10], dtype=torch.float))

        decoder.get_frames_played_at(torch.tensor([0, 1], dtype=torch.int))
        decoder.get_frames_played_at(torch.tensor([0, 1], dtype=torch.float))

    # Note [NVDEC vs FFmpeg CUDA pixel mismatches]:
    # These tests compare the NVDEC (beta) CUDA backend against the FFmpeg
    # CUDA backend. There are two known sources of pixel mismatches:
    #
    # 1. FFmpeg 4: small pixel differences on a few pixels (< 1%), cause
    #    unknown. We don't investigate further since FFmpeg 4 is not a
    #    priority.
    #
    # 2. MPEG4 asset: NVCUVID's parser reports matrix_coefficients=1
    #    (BT.709) for the MPEG4 asset, even though the bitstream has no
    #    color metadata. This is an NVIDIA-internal heuristic. FFmpeg's
    #    parser leaves colorspace as UNSPECIFIED, which both swscale (CPU)
    #    and our color conversion code treat as BT.601. So the NVDEC
    #    backend uses BT.709 while the FFmpeg CUDA backend (and CPU) use
    #    BT.601 for this asset, leading to different RGB output.

    @needs_cuda
    @pytest.mark.parametrize(
        "asset",
        (
            NASA_VIDEO,
            TEST_SRC_2_720P,
            BT709_FULL_RANGE,
            TEST_SRC_2_720P_H265,
            pytest.param(
                AV1_VIDEO,
                marks=pytest.mark.skipif(
                    in_fbcode(), reason="AV1 CUDA not supported internally"
                ),
            ),
            TEST_SRC_2_720P_VP9,
            TEST_SRC_2_720P_VP8,
            TEST_SRC_2_720P_MPEG4,
        ),
    )
    @pytest.mark.parametrize("contiguous_indices", (True, False))
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_nvdec_cuda_interface_get_frame_at(
        self, asset, contiguous_indices, seek_mode
    ):
        with set_cuda_backend("ffmpeg"):
            ref_decoder = VideoDecoder(asset.path, device="cuda", seek_mode=seek_mode)
        nvdec_decoder = VideoDecoder(asset.path, device="cuda", seek_mode=seek_mode)

        assert ref_decoder.metadata == nvdec_decoder.metadata

        if contiguous_indices:
            indices = range(len(ref_decoder))
        else:
            indices = range(0, len(ref_decoder), 10)

        for frame_index in indices:
            ref_frame = ref_decoder.get_frame_at(frame_index)
            nvdec_frame = nvdec_decoder.get_frame_at(frame_index)
            # See Note [NVDEC vs FFmpeg CUDA pixel mismatches]
            if ffmpeg_major_version > 5 and asset is not TEST_SRC_2_720P_MPEG4:
                torch.testing.assert_close(
                    nvdec_frame.data, ref_frame.data, rtol=0, atol=0
                )

            assert nvdec_frame.pts_seconds == ref_frame.pts_seconds
            assert nvdec_frame.duration_seconds == ref_frame.duration_seconds

    @needs_cuda
    @pytest.mark.parametrize(
        "asset",
        (
            NASA_VIDEO,
            TEST_SRC_2_720P,
            BT709_FULL_RANGE,
            TEST_SRC_2_720P_H265,
            pytest.param(
                AV1_VIDEO,
                marks=pytest.mark.skipif(
                    in_fbcode(), reason="AV1 CUDA not supported internally"
                ),
            ),
            TEST_SRC_2_720P_VP9,
            TEST_SRC_2_720P_VP8,
            TEST_SRC_2_720P_MPEG4,
        ),
    )
    @pytest.mark.parametrize("contiguous_indices", (True, False))
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_nvdec_cuda_interface_get_frames_at(
        self, asset, contiguous_indices, seek_mode
    ):
        with set_cuda_backend("ffmpeg"):
            ref_decoder = VideoDecoder(asset.path, device="cuda", seek_mode=seek_mode)
        nvdec_decoder = VideoDecoder(asset.path, device="cuda", seek_mode=seek_mode)

        assert ref_decoder.metadata == nvdec_decoder.metadata

        if contiguous_indices:
            indices = range(len(ref_decoder))
        else:
            indices = range(0, len(ref_decoder), 10)
        indices = list(indices)

        ref_frames = ref_decoder.get_frames_at(indices)
        nvdec_frames = nvdec_decoder.get_frames_at(indices)
        # See Note [NVDEC vs FFmpeg CUDA pixel mismatches]
        if ffmpeg_major_version > 5 and asset is not TEST_SRC_2_720P_MPEG4:
            torch.testing.assert_close(
                nvdec_frames.data, ref_frames.data, rtol=0, atol=0
            )
        torch.testing.assert_close(nvdec_frames.pts_seconds, ref_frames.pts_seconds)
        torch.testing.assert_close(
            nvdec_frames.duration_seconds, ref_frames.duration_seconds
        )

    @needs_cuda
    @pytest.mark.parametrize(
        "asset",
        (
            NASA_VIDEO,
            TEST_SRC_2_720P,
            BT709_FULL_RANGE,
            TEST_SRC_2_720P_H265,
            pytest.param(
                AV1_VIDEO,
                marks=pytest.mark.skipif(
                    in_fbcode(), reason="AV1 CUDA not supported internally"
                ),
            ),
            TEST_SRC_2_720P_VP9,
            TEST_SRC_2_720P_VP8,
            TEST_SRC_2_720P_MPEG4,
        ),
    )
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_nvdec_cuda_interface_get_frame_played_at(self, asset, seek_mode):
        with set_cuda_backend("ffmpeg"):
            ref_decoder = VideoDecoder(asset.path, device="cuda", seek_mode=seek_mode)
        nvdec_decoder = VideoDecoder(asset.path, device="cuda", seek_mode=seek_mode)

        assert ref_decoder.metadata == nvdec_decoder.metadata

        timestamps = torch.linspace(
            0, ref_decoder.metadata.duration_seconds - 1e-4, steps=10
        )
        for pts in timestamps:
            ref_frame = ref_decoder.get_frame_played_at(pts)
            nvdec_frame = nvdec_decoder.get_frame_played_at(pts)
            # See Note [NVDEC vs FFmpeg CUDA pixel mismatches]
            if ffmpeg_major_version > 5 and asset is not TEST_SRC_2_720P_MPEG4:
                torch.testing.assert_close(
                    nvdec_frame.data, ref_frame.data, rtol=0, atol=0
                )

            assert nvdec_frame.pts_seconds == ref_frame.pts_seconds
            assert nvdec_frame.duration_seconds == ref_frame.duration_seconds

    @needs_cuda
    @pytest.mark.parametrize(
        "asset",
        (
            NASA_VIDEO,
            TEST_SRC_2_720P,
            BT709_FULL_RANGE,
            TEST_SRC_2_720P_H265,
            pytest.param(
                AV1_VIDEO,
                marks=pytest.mark.skipif(
                    in_fbcode(), reason="AV1 CUDA not supported internally"
                ),
            ),
            TEST_SRC_2_720P_VP9,
            TEST_SRC_2_720P_VP8,
            TEST_SRC_2_720P_MPEG4,
        ),
    )
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_nvdec_cuda_interface_get_frames_played_at(self, asset, seek_mode):
        with set_cuda_backend("ffmpeg"):
            ref_decoder = VideoDecoder(asset.path, device="cuda", seek_mode=seek_mode)
        nvdec_decoder = VideoDecoder(asset.path, device="cuda", seek_mode=seek_mode)

        assert ref_decoder.metadata == nvdec_decoder.metadata

        timestamps = torch.linspace(
            0, ref_decoder.metadata.duration_seconds - 1e-4, steps=10
        ).tolist()

        ref_frames = ref_decoder.get_frames_played_at(timestamps)
        nvdec_frames = nvdec_decoder.get_frames_played_at(timestamps)
        # See Note [NVDEC vs FFmpeg CUDA pixel mismatches]
        if ffmpeg_major_version > 5 and asset is not TEST_SRC_2_720P_MPEG4:
            torch.testing.assert_close(
                nvdec_frames.data, ref_frames.data, rtol=0, atol=0
            )
        torch.testing.assert_close(nvdec_frames.pts_seconds, ref_frames.pts_seconds)
        torch.testing.assert_close(
            nvdec_frames.duration_seconds, ref_frames.duration_seconds
        )

    @needs_cuda
    @pytest.mark.parametrize(
        "asset",
        (
            NASA_VIDEO,
            TEST_SRC_2_720P,
            BT709_FULL_RANGE,
            TEST_SRC_2_720P_H265,
            pytest.param(
                AV1_VIDEO,
                marks=pytest.mark.skipif(
                    in_fbcode(), reason="AV1 CUDA not supported internally"
                ),
            ),
            TEST_SRC_2_720P_VP9,
            TEST_SRC_2_720P_VP8,
            TEST_SRC_2_720P_MPEG4,
        ),
    )
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_nvdec_cuda_interface_backwards(self, asset, seek_mode):
        with set_cuda_backend("ffmpeg"):
            ref_decoder = VideoDecoder(asset.path, device="cuda", seek_mode=seek_mode)
        nvdec_decoder = VideoDecoder(asset.path, device="cuda", seek_mode=seek_mode)

        assert ref_decoder.metadata == nvdec_decoder.metadata

        for frame_index in [0, 1, 2, 1, 0, 100, 10, 50, 20, 200, 150, 150, 150, 389, 2]:
            # This is ugly, but OK: the indices values above are relevant for
            # the NASA_VIDEO.  We need to avoid going out of bounds for other
            # videos so we cap the frame_index. This test still serves its
            # purpose: no matter what the range of the video, we're still doing
            # backwards seeks.
            frame_index = min(frame_index, len(ref_decoder) - 1)

            ref_frame = ref_decoder.get_frame_at(frame_index)
            nvdec_frame = nvdec_decoder.get_frame_at(frame_index)
            # See Note [NVDEC vs FFmpeg CUDA pixel mismatches]
            if ffmpeg_major_version > 5 and asset is not TEST_SRC_2_720P_MPEG4:
                torch.testing.assert_close(
                    nvdec_frame.data, ref_frame.data, rtol=0, atol=0
                )

            assert nvdec_frame.pts_seconds == ref_frame.pts_seconds
            assert nvdec_frame.duration_seconds == ref_frame.duration_seconds

    @needs_cuda
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_cuda_mpeg4_mp4_first_frame(self, seek_mode):
        # non-regression test for
        # https://github.com/meta-pytorch/torchcodec/issues/1340.
        decoder = VideoDecoder(
            TEST_SRC_2_MPEG4_MP4.path, device="cuda", seek_mode=seek_mode
        )
        with set_cuda_backend("ffmpeg"):
            ref_decoder = VideoDecoder(
                TEST_SRC_2_MPEG4_MP4.path, device="cuda", seek_mode=seek_mode
            )

        expected_frame0 = ref_decoder.get_frame_at(0)
        frame0 = decoder.get_frame_at(0)

        assert frame0.pts_seconds == expected_frame0.pts_seconds
        assert frame0.duration_seconds == expected_frame0.duration_seconds
        assert frame0.data.shape == expected_frame0.data.shape
        # Strict pixel equality is skipped — see Note [NVDEC vs FFmpeg CUDA
        # pixel mismatches] (BT.601 vs BT.709 color matrix mismatch between the
        # ffmpeg and default cuda backend for this MPEG4 asset).

    @pytest.mark.skip(reason="Assets not checked in; run manually with them present.")
    @needs_cuda
    @pytest.mark.parametrize(
        "path", ("./youtube-1HYJQESw3hs.mp4", "./youtube-c_B4XII1L6A.mp4")
    )
    @pytest.mark.parametrize("seek_mode", ("exact", "approximate"))
    def test_cuda_open_gop_late_idr(self, path, seek_mode):
        # Non-regression test for open-GOP H.264 files whose only IDR is not at
        # the start of the stream: the in-band SPS/PPS only shows up at that
        # late IDR. NVDEC used to have no usable sequence header until then, so
        # it dropped every frame before the IDR, returned shifted frames, and
        # hit a premature EOF on a full/sequential decode. CPU is unaffected and
        # serves as the reference here.
        cpu_decoder = VideoDecoder(path, device="cpu", seek_mode=seek_mode)
        cuda_decoder = VideoDecoder(path, device="cuda", seek_mode=seek_mode)

        assert cpu_decoder.metadata == cuda_decoder.metadata
        num_frames = cpu_decoder.metadata.num_frames

        # Full sequential decode must not raise and must yield every frame (the
        # `dec[:]` / `dec[:54]` failures we're guarding against).
        assert cuda_decoder[:].shape[0] == num_frames

        # Frames must not be shifted: pts must match the CPU reference exactly,
        # and pixels must be close. Exact pixel equality is skipped because CPU
        # and CUDA use different color-conversion matrices; a shifted/wrong frame
        # would show a large diff, so a tolerant mean check cleanly catches it.
        for i in [0, 1, 53, num_frames // 2, num_frames - 1]:
            cpu_frame = cpu_decoder.get_frame_at(i)
            cuda_frame = cuda_decoder.get_frame_at(i)
            assert cuda_frame.pts_seconds == cpu_frame.pts_seconds
            mean_abs_diff = (
                (cuda_frame.data.cpu().float() - cpu_frame.data.float()).abs().mean()
            )
            assert mean_abs_diff < 5

    @needs_cuda
    def test_nvdec_cuda_interface_cpu_fallback(self):
        # Non-regression test for the CPU fallback behavior of the NVDEC CUDA
        # interface.
        # We know that the H265_VIDEO asset isn't supported by NVDEC, its
        # dimensions are too small. We also know that the FFmpeg CUDA interface
        # fallbacks to the CPU path in such cases. We assert that we fall back
        # to the CPU path, too.

        with set_cuda_backend("ffmpeg"):
            ref_dec = VideoDecoder(H265_VIDEO.path, device="cuda")

        # Before accessing any frames, status should be unknown
        assert not ref_dec.cpu_fallback.status_known

        ref_frame = ref_dec.get_frame_at(0)

        assert "FFmpeg CUDA" in str(ref_dec.cpu_fallback)
        assert ref_dec.cpu_fallback.status_known
        assert ref_dec.cpu_fallback

        nvdec_dec = VideoDecoder(H265_VIDEO.path, device="cuda")

        assert "CUDA" in str(nvdec_dec.cpu_fallback)
        assert "FFmpeg CUDA" not in str(nvdec_dec.cpu_fallback)
        # For the NVDEC interface, status is known immediately
        assert nvdec_dec.cpu_fallback.status_known
        assert nvdec_dec.cpu_fallback

        nvdec_frame = nvdec_dec.get_frame_at(0)

        assert psnr(ref_frame.data, nvdec_frame.data) > 25

    @needs_cuda
    def test_nvdec_cpu_fallback_yuv444(self, tmp_path):
        # Non-regression test for https://github.com/meta-pytorch/torchcodec/issues/1414
        num_frames = 5
        frames = torch.randint(0, 256, size=(num_frames, 3, 64, 64), dtype=torch.uint8)
        path = str(tmp_path / "yuv444.mp4")
        VideoEncoder(frames=frames, frame_rate=30).to_file(
            path, pixel_format="yuv444p", crf=0
        )

        cpu_decoder = VideoDecoder(path, device="cpu")
        cuda_decoder = VideoDecoder(path, device="cuda")
        assert cuda_decoder.cpu_fallback

        cpu_frames = cpu_decoder.get_frames_in_range(start=0, stop=num_frames).data
        cuda_frames = cuda_decoder.get_frames_in_range(start=0, stop=num_frames).data

        # The CUDA path uploads these as yuv444p and color-converts them with
        # our kernel, which truncates where swscale rounds.
        torch.testing.assert_close(cpu_frames, cuda_frames.cpu(), rtol=0, atol=1)

    @needs_cuda
    def test_nvdec_cuda_interface_error(self):
        with pytest.raises(RuntimeError, match="torch_parse_device_string"):
            VideoDecoder(NASA_VIDEO.path, device="cuda:0:bad_variant")

    @needs_cuda
    def test_set_cuda_backend(self):
        # Tests for the set_cuda_backend() context manager.

        with pytest.raises(ValueError, match="Invalid CUDA backend"):
            with set_cuda_backend("bad_backend"):
                pass

        # set_cuda_backend() is meant to be used as a context manager. Using it
        # as a global call does nothing because the "context" is exited right
        # away. This is a good thing, we prefer users to use it as a CM only.
        set_cuda_backend("ffmpeg")
        assert _get_cuda_backend() == "default"  # Not changed to "ffmpeg".

        # Case insensitive
        with set_cuda_backend("FFMPEG"):
            assert _get_cuda_backend() == "ffmpeg"

        # "nvdec" is the public-facing name for the NVDEC CUDA backend.
        # Internally it maps to the "default" variant value.
        with set_cuda_backend("nvdec"):
            assert _get_cuda_backend() == "default"

        # Check that the default backend is NVDEC
        assert _get_cuda_backend() == "default"
        dec = VideoDecoder(H265_VIDEO.path, device="cuda")
        assert "CUDA" in str(dec.cpu_fallback)
        assert "FFmpeg CUDA" not in str(dec.cpu_fallback)

        # Check that setting "ffmpeg" effectively uses the FFmpeg CUDA backend.
        # We also show that this affects decoder creation only. When the decoder
        # is created with a given backend, it stays in this backend for the rest
        # of its life. This is normal and intended.
        with set_cuda_backend("ffmpeg"):
            dec = VideoDecoder(H265_VIDEO.path, device="cuda")
        assert _get_cuda_backend() == "default"
        assert "FFmpeg CUDA" in str(dec.cpu_fallback)
        with set_cuda_backend("nvdec"):
            assert "FFmpeg CUDA" in str(dec.cpu_fallback)

        # Hacky way to ensure passing "cuda:1" is supported by both backends. We
        # just check that there's an error when passing cuda:N where N is too
        # high.
        bad_device_number = torch.cuda.device_count() + 1
        for backend in ("ffmpeg", "nvdec"):
            with pytest.raises(RuntimeError, match="torch_call_dispatcher"):
                with set_cuda_backend(backend):
                    VideoDecoder(H265_VIDEO.path, device=f"cuda:{bad_device_number}")

    @contextlib.contextmanager
    def restore_nvdec_cache_capacity(self):
        try:
            original = get_nvdec_cache_capacity()
            yield
        finally:
            set_nvdec_cache_capacity(original)
            assert get_nvdec_cache_capacity() == original

    def test_nvdec_cache_capacity(self):
        with self.restore_nvdec_cache_capacity():
            set_nvdec_cache_capacity(42)
            assert get_nvdec_cache_capacity() == 42

            set_nvdec_cache_capacity(0)
            assert get_nvdec_cache_capacity() == 0

            set_nvdec_cache_capacity(1)
            assert get_nvdec_cache_capacity() == 1

            with pytest.raises(
                RuntimeError, match="NVDEC cache capacity must be non-negative"
            ):
                set_nvdec_cache_capacity(-1)

            # Capacity is unchanged after the failed call above.
            assert get_nvdec_cache_capacity() == 1

    @needs_cuda
    def test_nvdec_cache_capacity_eviction(self):
        def create_decoder():
            dec = VideoDecoder(NASA_VIDEO.path, device="cuda")
            dec[0]
            del dec
            gc.collect()

        # Evict any leftover cached decoders from previous tests
        with self.restore_nvdec_cache_capacity():
            set_nvdec_cache_capacity(0)

        with self.restore_nvdec_cache_capacity():
            assert _core._get_nvdec_cache_size(device_index=0) == 0

            # Create decoder, it should be in the cache
            create_decoder()
            assert _core._get_nvdec_cache_size(device_index=0) == 1

            # Set capacity to 1, decoder should still be there
            set_nvdec_cache_capacity(1)
            assert _core._get_nvdec_cache_size(device_index=0) == 1
            # Set capacity to 0, this should evict it
            set_nvdec_cache_capacity(0)
            assert _core._get_nvdec_cache_size(device_index=0) == 0

            # Create a new decoder, it's not cached since capacity is 0
            create_decoder()
            assert _core._get_nvdec_cache_size(device_index=0) == 0

    def test_cpu_fallback_no_fallback_on_cpu_device(self):
        """Test that CPU device doesn't trigger fallback (it's not a fallback scenario)."""
        decoder = VideoDecoder(NASA_VIDEO.path, device="cpu")

        assert decoder.cpu_fallback.status_known
        _ = decoder[0]

        assert not decoder.cpu_fallback
        assert "No fallback required" in str(decoder.cpu_fallback)

    @pytest.mark.parametrize("dimension_order", ["NCHW", "NHWC"])
    @pytest.mark.parametrize(
        # We are skipping over cuda:ffmpeg because we do not support rotation
        # metadata for the FFmpeg CUDA interface.
        "device",
        ("cpu", pytest.param("cuda", marks=pytest.mark.needs_cuda)),
    )
    def test_rotation_applied_to_frames(self, dimension_order, device):
        """Test that rotation is correctly applied to decoded frames.

        Compares frames from NASA_VIDEO_ROTATED (which has 90-degree rotation
        metadata) with manually rotated frames from NASA_VIDEO.
        Tests all decoding methods to ensure rotation is applied consistently.
        """
        decoder, _ = make_video_decoder(
            NASA_VIDEO.path,
            device=device,
            stream_index=NASA_VIDEO.default_stream_index,
            dimension_order=dimension_order,
        )
        decoder_rotated, _ = make_video_decoder(
            NASA_VIDEO_ROTATED.path,
            device=device,
            stream_index=NASA_VIDEO_ROTATED.default_stream_index,
            dimension_order=dimension_order,
        )

        # Rotation dims for single frame (CHW or HWC) and batch (NCHW or NHWC)
        # Rotation dims are (H, W) dimensions for each format
        frame_rot_dims = (1, 2) if dimension_order == "NCHW" else (0, 1)  # CHW vs HWC
        batch_rot_dims = (2, 3) if dimension_order == "NCHW" else (1, 2)  # NCHW vs NHWC

        # Test __getitem__ / get_frame_at (single frame by index)
        for idx in [0, 5, 10]:
            frame = decoder[idx]
            frame_rotated = decoder_rotated[idx]
            expected = torch.rot90(frame, k=1, dims=frame_rot_dims)
            torch.testing.assert_close(expected, frame_rotated, atol=0, rtol=0)

        # Test get_frames_at (multiple frames by indices)
        indices = [0, 5, 10]
        frames = decoder.get_frames_at(indices)
        frames_rotated = decoder_rotated.get_frames_at(indices)
        expected = torch.rot90(frames.data, k=1, dims=batch_rot_dims)
        torch.testing.assert_close(expected, frames_rotated.data, atol=0, rtol=0)

        # Test get_frames_in_range (frames by index range)
        frames_range = decoder.get_frames_in_range(start=0, stop=6, step=2)
        frames_range_rotated = decoder_rotated.get_frames_in_range(
            start=0, stop=6, step=2
        )
        expected = torch.rot90(frames_range.data, k=1, dims=batch_rot_dims)
        torch.testing.assert_close(expected, frames_range_rotated.data, atol=0, rtol=0)

        # Test get_frame_played_at (single frame by timestamp)
        pts = decoder_rotated.metadata.begin_stream_seconds
        frame_at_pts = decoder.get_frame_played_at(pts)
        frame_at_pts_rotated = decoder_rotated.get_frame_played_at(pts)
        expected = torch.rot90(frame_at_pts.data, k=1, dims=frame_rot_dims)
        torch.testing.assert_close(expected, frame_at_pts_rotated.data, atol=0, rtol=0)

        # Test get_frames_played_at (multiple frames by timestamps)
        pts_list = [
            decoder_rotated.metadata.begin_stream_seconds,
            decoder_rotated.metadata.begin_stream_seconds + 0.15,
        ]
        frames_at_pts = decoder.get_frames_played_at(pts_list)
        frames_at_pts_rotated = decoder_rotated.get_frames_played_at(pts_list)
        expected = torch.rot90(frames_at_pts.data, k=1, dims=batch_rot_dims)
        torch.testing.assert_close(expected, frames_at_pts_rotated.data, atol=0, rtol=0)

        # Test get_frames_played_in_range (frames by timestamp range)
        start_seconds = decoder_rotated.metadata.begin_stream_seconds
        stop_seconds = start_seconds + 0.2
        frames_in_range = decoder.get_frames_played_in_range(
            start_seconds=start_seconds, stop_seconds=stop_seconds
        )
        frames_in_range_rotated = decoder_rotated.get_frames_played_in_range(
            start_seconds=start_seconds, stop_seconds=stop_seconds
        )
        expected = torch.rot90(frames_in_range.data, k=1, dims=batch_rot_dims)
        torch.testing.assert_close(
            expected, frames_in_range_rotated.data, atol=0, rtol=0
        )

        # Test get_all_frames (all frames in video)
        # Note: NASA_VIDEO_ROTATED has fewer frames than NASA_VIDEO, so we compare
        # the first N frames where N is the number of frames in the rotated video
        all_frames = decoder.get_all_frames()
        all_frames_rotated = decoder_rotated.get_all_frames()
        num_frames_rotated = all_frames_rotated.data.shape[0]
        expected = torch.rot90(
            all_frames.data[:num_frames_rotated], k=1, dims=batch_rot_dims
        )
        torch.testing.assert_close(expected, all_frames_rotated.data, atol=0, rtol=0)

    @pytest.mark.parametrize(
        "desired_H, desired_W",
        [
            (100, 150),
            (150, 100),
            (100, 100),
        ],
    )
    @pytest.mark.parametrize("TransformClass", [Resize, CenterCrop, RandomCrop])
    def test_rotation_with_transform(self, TransformClass, desired_H, desired_W):
        """Test that transforms work correctly with rotated videos.

        When a user specifies a transform with (H, W), they expect the final output to be
        (H, W) regardless of the video's rotation metadata. This test verifies
        that the transform is applied correctly such that the final output matches
        the user's requested dimensions.
        """
        decoder = VideoDecoder(
            NASA_VIDEO_ROTATED.path,
            transforms=[TransformClass((desired_H, desired_W))],
        )
        frame = decoder[0]

        assert frame.shape == (3, desired_H, desired_W)

        # Also test batch APIs
        frames = decoder.get_frames_at([0, 1])
        assert frames.data.shape == (2, 3, desired_H, desired_W)

    def test_rotation_with_transform_pipeline(self):
        """Test that a pipeline of multiple transforms works correctly with rotated videos.

        This test verifies that chaining multiple transforms (e.g., Resize -> Resize -> Crop)
        works as expected when the video has rotation metadata. Each transform should
        operate on the output of the previous transform in post-rotation coordinate space.
        """
        decoder = VideoDecoder(
            NASA_VIDEO_ROTATED.path,
            transforms=[Resize((400, 300)), Resize((300, 250)), CenterCrop((100, 100))],
        )
        frame = decoder[0]
        assert frame.shape == (3, 100, 100)

        frames = decoder.get_frames_at([0, 1])
        assert frames.data.shape == (2, 3, 100, 100)

    @needs_cuda
    @pytest.mark.parametrize("device", cuda_devices())
    def test_cpu_fallback_h265_video(self, device):
        """Test that H265 video triggers CPU fallback on CUDA interfaces."""
        # H265_VIDEO is known to trigger CPU fallback on CUDA
        # because its dimensions are too small
        decoder, _ = make_video_decoder(H265_VIDEO.path, device=device)

        if "ffmpeg" in device:
            # For FFmpeg interface, status is unknown until first frame is decoded
            assert not decoder.cpu_fallback.status_known
            decoder.get_frame_at(0)
            assert decoder.cpu_fallback.status_known
            assert decoder.cpu_fallback
            # FFmpeg interface doesn't know the specific reason
            assert "Unknown reason - try the 'nvdec' backend to know more" in str(
                decoder.cpu_fallback
            )
        else:
            # For the NVDEC interface, status is known immediately
            assert decoder.cpu_fallback.status_known
            assert decoder.cpu_fallback
            # The NVDEC interface provides the specific reason for fallback
            assert "Video not supported" in str(decoder.cpu_fallback)

    @needs_cuda
    @pytest.mark.parametrize("device", cuda_devices())
    def test_cpu_fallback_no_fallback_on_supported_video(self, device):
        """Test that supported videos don't trigger fallback on CUDA."""
        decoder, _ = make_video_decoder(NASA_VIDEO.path, device=device)

        decoder[0]

        assert not decoder.cpu_fallback
        assert "No fallback required" in str(decoder.cpu_fallback)

    @needs_cuda
    def test_beta_backend_still_supported_for_bc(self):
        with set_cuda_backend("beta"):
            dec = VideoDecoder(NASA_VIDEO.path, device="cuda")
        dec[0]
        assert dec.cpu_fallback._backend == "CUDA"

    @staticmethod
    def _assert_float32_frame_matches_rgb48_ref(frame_data, asset, frame_index):
        is_cuda = frame_data.device.type == "cuda"
        frame_as_uint16 = (frame_data * 65535).round().to(torch.uint16).cpu()
        ref = asset.get_frame_data_by_index_rgb48(frame_index)
        if is_cuda:
            atol = 3 / 255 * 65535
            assert_tensor_close_on_at_least(
                frame_as_uint16, ref, atol=atol, percentage=90
            )
        else:
            torch.testing.assert_close(frame_as_uint16, ref, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "device",
        ("cpu", pytest.param("cuda", marks=pytest.mark.needs_cuda)),
    )
    @pytest.mark.parametrize(
        "asset",
        (NASA_VIDEO, NASA_VIDEO_HDR, TEST_SRC_2_720P_HDR, TEST_SRC_2_12BIT_HDR),
    )
    @pytest.mark.parametrize("output_dtype", (torch.uint8, "default"))
    def test_output_dtype_uint8(self, asset, device, output_dtype):
        if output_dtype == "default":
            decoder = VideoDecoder(asset.path, device=device)
        else:
            decoder = VideoDecoder(asset.path, output_dtype=torch.uint8, device=device)

        frame_indices = [0, 5, 10]
        for frame_index in frame_indices:
            ffmpeg_ref = asset.get_frame_data_by_index(frame_index)
            frame = decoder[frame_index]
            assert frame.dtype == torch.uint8
            if device == "cuda":
                atol = 3
                cpu_ref = VideoDecoder(asset.path)[frame_index]
                assert_tensor_close_on_at_least(
                    frame.data.cpu(), ffmpeg_ref, atol=atol, percentage=90
                )
                assert_tensor_close_on_at_least(
                    frame.data.cpu(), cpu_ref, atol=atol, percentage=90
                )
            else:
                assert_frames_equal(frame.data, ffmpeg_ref)

    @pytest.mark.parametrize(
        "device",
        ("cpu", pytest.param("cuda", marks=pytest.mark.needs_cuda)),
    )
    @pytest.mark.parametrize(
        "asset",
        (NASA_VIDEO, NASA_VIDEO_HDR, TEST_SRC_2_720P_HDR, TEST_SRC_2_12BIT_HDR),
    )
    def test_output_dtype_float32(self, asset, device):
        decoder = VideoDecoder(asset.path, output_dtype=torch.float32, device=device)
        frame_indices = [0, 5, 10]

        # None of those should go through the CPU fallback. This assert is
        # particularly important for NASA_VIDEO which is an SDR video. NVDEC
        # will typically not support P016 on SDR videos, so in such cases we
        # fallback to NV12 instead of falling back to the CPU. This NV12
        # fallback can only be done for SDR videos, not HDR videos where we'd be
        # losing precision.
        assert not decoder.cpu_fallback

        for frame_index in frame_indices:
            frame = decoder[frame_index]
            assert frame.dtype == torch.float32

            self._assert_float32_frame_matches_rgb48_ref(frame.data, asset, frame_index)

    @pytest.mark.parametrize(
        "device",
        ("cpu", pytest.param("cuda", marks=pytest.mark.needs_cuda)),
    )
    @pytest.mark.parametrize(
        "asset, is_hdr",
        (
            (NASA_VIDEO, False),
            (NASA_VIDEO_HDR, True),
            (TEST_SRC_2_720P_HDR, True),
            (TEST_SRC_2_12BIT_HDR, True),
        ),
    )
    def test_output_dtype_auto(self, asset, is_hdr, device):
        decoder = VideoDecoder(asset.path, output_dtype="auto", device=device)
        frame_indices = [0, 5, 10]
        for frame_index in frame_indices:
            frame = decoder[frame_index]
            if is_hdr:
                assert frame.dtype == torch.float32
            else:
                assert frame.dtype == torch.uint8

    @pytest.mark.parametrize(
        "device",
        ("cpu", pytest.param("cuda", marks=pytest.mark.needs_cuda)),
    )
    @pytest.mark.parametrize(
        "asset",
        (NASA_VIDEO, NASA_VIDEO_HDR, TEST_SRC_2_720P_HDR, TEST_SRC_2_12BIT_HDR),
    )
    def test_output_dtype_float32_batch_apis(self, asset, device):
        decoder = VideoDecoder(asset.path, output_dtype=torch.float32, device=device)
        indices = [0, 5, 10]

        # get_frame_at
        self._assert_float32_frame_matches_rgb48_ref(
            decoder.get_frame_at(0).data, asset, 0
        )

        # get_frames_at
        frames = decoder.get_frames_at(indices)
        for i, idx in enumerate(indices):
            self._assert_float32_frame_matches_rgb48_ref(frames.data[i], asset, idx)

        # get_frames_in_range
        frames_range = decoder.get_frames_in_range(start=5, stop=11)
        self._assert_float32_frame_matches_rgb48_ref(frames_range.data[0], asset, 5)
        self._assert_float32_frame_matches_rgb48_ref(frames_range.data[5], asset, 10)

    @pytest.mark.parametrize(
        "device",
        ("cpu", pytest.param("cuda", marks=pytest.mark.needs_cuda)),
    )
    @pytest.mark.parametrize(
        "asset",
        (NASA_VIDEO, NASA_VIDEO_HDR, TEST_SRC_2_720P_HDR, TEST_SRC_2_12BIT_HDR),
    )
    def test_output_dtype_float32_pts_apis(self, asset, device):
        decoder = VideoDecoder(asset.path, output_dtype=torch.float32, device=device)
        indices = [0, 5, 10]

        pts_seconds_ref = [decoder.get_frame_at(i).pts_seconds for i in indices]

        # get_frame_played_at
        for pts, idx in zip(pts_seconds_ref, indices):
            frame = decoder.get_frame_played_at(pts)
            self._assert_float32_frame_matches_rgb48_ref(frame.data, asset, idx)

        # get_frames_played_in_range (full range)
        frames = decoder.get_frames_played_in_range(
            start_seconds=0,
            stop_seconds=pts_seconds_ref[-1] + 1e-4,
        )
        for idx in indices:
            self._assert_float32_frame_matches_rgb48_ref(frames.data[idx], asset, idx)

        # get_frames_played_in_range (single-frame ranges)
        for pts, idx in zip(pts_seconds_ref, indices):
            frames = decoder.get_frames_played_in_range(
                start_seconds=pts, stop_seconds=pts + 1e-4
            )
            self._assert_float32_frame_matches_rgb48_ref(frames.data[0], asset, idx)

        # get_frames_played_at
        frames = decoder.get_frames_played_at(pts_seconds_ref)
        for i, idx in enumerate(indices):
            self._assert_float32_frame_matches_rgb48_ref(frames.data[i], asset, idx)

    @pytest.mark.parametrize("bad_dtype", (torch.float64, torch.int32, "not_a_dtype"))
    def test_output_dtype_invalid(self, bad_dtype):
        with pytest.raises(ValueError, match="Invalid output_dtype"):
            VideoDecoder(NASA_VIDEO.path, output_dtype=bad_dtype)

    @needs_cuda
    @pytest.mark.parametrize("output_dtype", (torch.float32, "auto"))
    def test_output_dtype_not_uint8_ffmpeg_cuda_backend(self, output_dtype):
        with set_cuda_backend("ffmpeg"):
            with pytest.raises(ValueError, match="not supported with the 'ffmpeg'"):
                VideoDecoder(NASA_VIDEO.path, output_dtype=output_dtype, device="cuda")

    @needs_cuda
    def test_output_dtype_float32_cpu_fallback(self):
        # H264_10BITS triggers CPU fallback on NVDEC. float32 output should
        # still work via the CPU fallback path.
        asset = H264_10BITS

        decoder_cpu = VideoDecoder(asset.path, output_dtype=torch.float32)
        decoder_cuda = VideoDecoder(
            asset.path, output_dtype=torch.float32, device="cuda"
        )

        assert decoder_cuda.cpu_fallback

        for frame_index in [0, 5, 10]:
            cpu_frame = decoder_cpu[frame_index]
            cuda_frame = decoder_cuda[frame_index]
            assert cuda_frame.dtype == torch.float32
            assert cuda_frame.data.device.type == "cuda"
            assert_tensor_close_on_at_least(
                cuda_frame.data.cpu(), cpu_frame.data, atol=3 / 255, percentage=85
            )


class TestAudioDecoder:
    @pytest.mark.parametrize(
        "asset", (NASA_AUDIO, NASA_AUDIO_MP3, SINE_MONO_S32, SINE_16_CHANNEL_S16)
    )
    def test_metadata(self, asset):
        decoder = AudioDecoder(asset.path)
        assert isinstance(decoder.metadata, AudioStreamMetadata)

        assert (
            decoder.stream_index
            == decoder.metadata.stream_index
            == asset.default_stream_index
        )

        expected_duration_seconds_from_header = asset.duration_seconds
        if asset == NASA_AUDIO_MP3 and ffmpeg_major_version >= 8:
            expected_duration_seconds_from_header = 13.056

        assert decoder.metadata.duration_seconds_from_header == pytest.approx(
            expected_duration_seconds_from_header
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

    def test_fresh_decoder_seek(self, tmp_path):
        # Non-regression test: on a fresh decoder, get_all_samples() (i.e.
        # start_seconds=0) must not crash. This used to fail on FLAC with
        # "Could not seek file to pts=-9223372036854775808" when we
        # unconditionally seeked on a fresh decoder.
        from torchcodec.encoders import AudioEncoder

        path = str(tmp_path / "test.flac")
        AudioEncoder(torch.rand(1, 1000), sample_rate=16000).to_file(path)
        AudioDecoder(path).get_all_samples()

    def test_unseekable_format(self):
        decoder = AudioDecoder(UNSEEKABLE_SWF.path)
        samples = decoder.get_all_samples()
        assert samples.data.shape == (1, 89856)

        with pytest.raises(RuntimeError, match="'swf' format does not support seeking"):
            decoder.get_samples_played_in_range(start_seconds=1)

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

    @pytest.mark.parametrize("asset", (NASA_AUDIO, NASA_AUDIO_MP3, SINE_16_CHANNEL_S16))
    def test_get_all_samples(self, asset):
        decoder = AudioDecoder(asset.path)
        torch.testing.assert_close(
            decoder.get_all_samples().data,
            decoder.get_samples_played_in_range().data,
        )

    def test_decode_from_tensor_odd_sized_wav(self):
        # Non-regression test for https://github.com/meta-pytorch/torchcodec/issues/1378
        # WAV files with an odd-sized data chunk and a trailing metadata chunk
        # used to crash when decoded from a bytes tensor, because FFmpeg seeks
        # past EOF and the AVIO read callback threw instead of returning
        # AVERROR_EOF.
        asset = WAV_ODD_DATA_TRAILING_CHUNK
        samples_from_path = AudioDecoder(asset.path).get_all_samples()
        samples_from_tensor = AudioDecoder(asset.to_tensor()).get_all_samples()
        torch.testing.assert_close(
            samples_from_path.data, samples_from_tensor.data, rtol=0, atol=0
        )

    @pytest.mark.parametrize("start_seconds", (1, 2, 2.5, 3))
    def test_seek_mpeg_program_stream(self, start_seconds):
        # Non-regression test for https://github.com/meta-pytorch/torchcodec/issues/1610
        # Seeking in an MPEG program stream lands on a container-level byte
        # offset, so the packets we get back until the parser resyncs onto a
        # frame boundary don't decode. We used to error out instead of skipping
        # them. It takes more than one packet at some of the offsets below.
        decoder = AudioDecoder(SINE_STEREO_MP2_MPEG_PS.path)

        all_samples = decoder.get_all_samples()
        samples = decoder.get_samples_played_in_range(start_seconds=start_seconds)

        assert samples.pts_seconds == start_seconds
        offset = round(
            (start_seconds - all_samples.pts_seconds) * decoder.metadata.sample_rate
        )
        reference = all_samples.data[:, offset:]
        assert samples.data.shape == reference.shape
        torch.testing.assert_close(samples.data, reference, rtol=0, atol=1e-4)

    def test_corrupt_data_raises(self, tmp_path):
        # We tolerate undecodable packets right after a seek (see
        # test_seek_mpeg_program_stream), but corrupt data in the middle of a
        # stream must still be reported rather than silently truncating the
        # output.
        data = bytearray(NASA_AUDIO_MP3.path.read_bytes()[:20_000])
        for i in range(4_000, len(data), 7):
            data[i] ^= 0xFF
        path = tmp_path / "corrupt.mp3"
        path.write_bytes(bytes(data))

        with pytest.raises(RuntimeError, match="Invalid data found"):
            AudioDecoder(path).get_all_samples()

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

    LOSSY_ASSETS = (NASA_AUDIO, NASA_AUDIO_MP3, NASA_AUDIO_MP3_44100)

    @pytest.mark.parametrize(
        "asset",
        (
            SINE_MONO_S32_44100,
            SINE_MONO_S32_8000,
            SINE_MONO_U8,
            SINE_MONO_F32,
            SINE_16_CHANNEL_S16,
            NASA_AUDIO,
            NASA_AUDIO_MP3_44100,
        ),
    )
    @pytest.mark.parametrize(
        "out_sample_rate, increment",
        (
            (8_000, 1 / 3),
            (8_000, 0.7),
            (8_000, 0.3),
            (16_000, 1.0),
            (16_000, 1.2),
            (16_000, 1 / 3),
            (16_000, 0.3),
            (16_001, 1.0),
            (16_001, 1 / 3),
            (22_050, 0.7),
            (48_000, 1.2),
        ),
    )
    def test_resample_chunked_matches_full(self, asset, out_sample_rate, increment):
        # Reading a resampled stream in consecutive chunks must return exactly
        # the same samples as decoding it in one go.
        # See [Audio resampling and frame alignment]

        full = AudioDecoder(asset.path, sample_rate=out_sample_rate).get_all_samples()
        actual_duration = (
            full.pts_seconds + full.data.shape[1] / out_sample_rate - full.pts_seconds
        )

        decoder = AudioDecoder(asset.path, sample_rate=out_sample_rate)
        chunks = []
        for i in range(math.ceil(actual_duration / increment)):
            start = full.pts_seconds + i * increment
            chunks.append(
                decoder.get_samples_played_in_range(start, start + increment).data
            )
        chunks = torch.cat(chunks, dim=1)

        assert chunks.shape == full.data.shape
        if asset in self.LOSSY_ASSETS:
            assert_tensor_close_on_at_least(
                chunks, full.data, atol=0, rtol=0, percentage=80
            )
            assert_tensor_close_on_at_least(
                chunks, full.data, atol=0.1, rtol=0, percentage=95
            )
        else:
            torch.testing.assert_close(chunks, full.data, atol=0, rtol=0)

    @pytest.mark.parametrize("out_sample_rate", (8_000, 16_000))
    @pytest.mark.parametrize("stop_seconds", (1.45, 1.91, 2.1))
    def test_resample_chunked_matches_full_postroll(
        self, out_sample_rate, stop_seconds
    ):
        # Test for resampling post-roll. Basically a subset of
        # test_resample_chunked_matches_full but with a focus on postroll:
        # stop_seconds values are chosen such that they actually fail on main
        # and exercise the post-roll fix.
        asset = SINE_MONO_S32_44100
        full = (
            AudioDecoder(asset.path, sample_rate=out_sample_rate).get_all_samples().data
        )

        decoder = AudioDecoder(asset.path, sample_rate=out_sample_rate)
        samples = decoder.get_samples_played_in_range(1.0, stop_seconds).data

        start = round(1.0 * out_sample_rate)
        torch.testing.assert_close(
            samples, full[:, start : start + samples.shape[1]], atol=0, rtol=0
        )

    @pytest.mark.parametrize("boundary_sample", (6_000, 20_000, 30_000))
    def test_chunk_boundary_half_sample(self, boundary_sample):
        # Reading the stream in two chunks, splitting right in the middle of a
        # sample. That sample must be returned by exactly one of the two chunks,
        # and the two chunks must independently agree on which one.
        # This is ensured by a stable rounding (see offset_of()).
        asset = SINE_MONO_S32
        boundary = (boundary_sample + 0.5) / asset.sample_rate

        full = AudioDecoder(asset.path).get_all_samples()
        end_seconds = full.pts_seconds + full.data.shape[1] / asset.sample_rate

        decoder = AudioDecoder(asset.path)
        chunks = [
            decoder.get_samples_played_in_range(full.pts_seconds, boundary).data,
            decoder.get_samples_played_in_range(boundary, end_seconds).data,
        ]

        torch.testing.assert_close(torch.cat(chunks, dim=1), full.data, atol=0, rtol=0)

    def test_decode_s16_ffmpeg4(self):
        # Non-regression test for https://github.com/pytorch/torchcodec/issues/843
        # Ensures that decoding s16 on FFmpeg4 handles
        # unset input channel count and layout

        asset = SINE_MONO_S16
        decoder = AudioDecoder(asset.path)
        assert decoder.metadata.sample_rate == asset.sample_rate
        assert decoder.metadata.sample_format == asset.sample_format

        test_samples = decoder.get_samples_played_in_range()
        assert test_samples.data.shape[0] == decoder.metadata.num_channels
        assert test_samples.sample_rate == decoder.metadata.sample_rate
        reference_frames = asset.get_frame_data_by_range(
            start=0, stop=1, stream_index=0
        )
        torch.testing.assert_close(
            test_samples.data[0], reference_frames, atol=0, rtol=0
        )

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
    @pytest.mark.parametrize(
        "num_channels",
        (
            1,
            2,
            8,
            16,
            pytest.param(
                24,
                marks=pytest.mark.skipif(
                    ffmpeg_major_version == 4 and get_ffmpeg_minor_version() < 4,
                    reason="24 channel layout requires FFmpeg >= 4.4",
                ),
            ),
            None,
        ),
    )
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
        with pytest.raises(RuntimeError, match="num_channels must be > 0"):
            AudioDecoder(asset.path, num_channels=0)
        for num_channels in (15, 23):
            with pytest.raises(RuntimeError, match="Couldn't initialize SwrContext:"):
                decoder = AudioDecoder(asset.path, num_channels=num_channels)
                # Call get_all_samples to trigger num_channels conversion.
                # FFmpeg fails to find a default layout for certain channel counts,
                # which causes SwrContext to fail to initialize.
                decoder.get_all_samples()


class TestWavDecoder:

    def test_non_wav_file_raises_error(self):
        with pytest.raises(RuntimeError, match="Missing RIFF header"):
            WavDecoder(NASA_AUDIO.path)

    @pytest.mark.parametrize(
        "start_seconds,stop_seconds",
        [
            (0.0, 1.0),
            (0.2, 0.6),
            (1.0, 1.0),
            (0.0, None),
            (-1.0, 1.0),
            (-1.0, None),
            (None, None),
        ],
    )
    @pytest.mark.parametrize(
        "asset",
        (
            SINE_MONO_S32,
            SINE_MONO_S24,
            SINE_MONO_S16,
            SINE_MONO_U8,
            SINE_MONO_F32,
            SINE_MONO_F64,
            SINE_16_CHANNEL_S16,
        ),
    )
    @pytest.mark.parametrize(
        "source_kind", ("path", "str", "bytes", "tensor", "file_like")
    )
    def test_against_audio_decoder(
        self, asset, start_seconds, stop_seconds, source_kind
    ):
        file_handle = None
        if source_kind == "path":
            source = asset.path
        elif source_kind == "str":
            source = str(asset.path)
        elif source_kind == "bytes":
            source = asset.path.read_bytes()
        elif source_kind == "tensor":
            source = asset.to_tensor()
        elif source_kind == "file_like":
            file_handle = open(asset.path, "rb")
            source = file_handle

        wav_dec = WavDecoder(source)
        audio_dec = AudioDecoder(asset.path)

        assert isinstance(wav_dec.metadata, AudioStreamMetadata)
        assert wav_dec.stream_index == audio_dec.metadata.stream_index
        assert wav_dec.metadata == audio_dec.metadata

        if start_seconds is None and stop_seconds is None:
            wav_samples = wav_dec.get_all_samples()
            audio_samples = audio_dec.get_all_samples()
        else:
            wav_samples = wav_dec.get_samples_played_in_range(
                start_seconds, stop_seconds
            )
            audio_samples = audio_dec.get_samples_played_in_range(
                start_seconds, stop_seconds
            )
        torch.testing.assert_close(wav_samples.data, audio_samples.data, rtol=0, atol=0)
        assert wav_samples.pts_seconds == audio_samples.pts_seconds

        if file_handle is not None:
            file_handle.close()

    def test_get_samples_played_in_range_errors(self):
        wav_dec = WavDecoder(SINE_MONO_S32.path)
        with pytest.raises(
            ValueError,
            match="Invalid start seconds: 2.0. It must be less than or equal to stop seconds \\(1.0\\).",
        ):
            wav_dec.get_samples_played_in_range(2.0, 1.0)

        with pytest.raises(
            RuntimeError,
            match="No samples to decode. This is probably because start_seconds is too high\\(10\\)",
        ):
            wav_dec.get_samples_played_in_range(10.0, None)

        with pytest.raises(
            RuntimeError,
            match="No samples to decode. This is probably because start_seconds is too high\\(10\\)",
        ):
            wav_dec.get_samples_played_in_range(10.0, 12.0)

    def test_start_equals_stop_returns_empty(self):
        wav_dec = WavDecoder(SINE_MONO_S32.path)
        samples = wav_dec.get_samples_played_in_range(0.5, 0.5)
        assert samples.data.shape[1] == 0
        assert samples.pts_seconds == pytest.approx(0.5)

    def test_multiple_calls_with_backward_seeks(self):
        wav_dec = WavDecoder(SINE_MONO_S32.path)
        audio_dec = AudioDecoder(SINE_MONO_S32.path)

        ranges = [
            (0.0, 0.3),
            (0.5, 0.8),
            (0.2, 0.4),
            (0.7, None),
            (0.0, 0.1),
            (0.6, 0.9),
            (0.1, 0.5),
        ]
        for start, stop in ranges:
            wav_samples = wav_dec.get_samples_played_in_range(start, stop)
            audio_samples = audio_dec.get_samples_played_in_range(start, stop)
            torch.testing.assert_close(
                wav_samples.data, audio_samples.data, rtol=0, atol=0
            )
            assert wav_samples.pts_seconds == audio_samples.pts_seconds


def _block_devices():
    return ("cpu", pytest.param("cuda", marks=pytest.mark.needs_cuda))


def _is_msb_aligned(pix_fmt):
    # By msb-aligned we mean 10- or 12-bit samples that are stored in the most
    # significant bits of a uint16. They thus span the whole uint16 value range
    # (in steps of 64 or 16) rather than the [0, 1024) or [0, 4_096) ranges of
    # lsb-aligned samples, so they must be shifted down to be compared to
    # those. 8-bit samples fill their uint8, so they are msb-aligned by
    # definition.

    # NVDEC's P010 and P012 are msb-aligned. P016 nominally uses all 16 bits.
    return pix_fmt.startswith("p0")


# TODO_API_BREAKDOWN CC P2: this entire class should probably be folded in the
# test/utils asset class.
class _PlanesCase(NamedTuple):
    """A video, how many significant bits its samples carry, and the pixel
    format its frames come out in on each device."""

    video: object
    bit_depth: int
    cpu_pix_fmt: str
    cuda_pix_fmt: str
    # One per component of the pixel format, so three for YUV and RGB, one for
    # grayscale, and one more when the format has an alpha component.
    cpu_num_planes: int = 3
    cuda_num_planes: int = 3
    # FFmpeg 6 added P012. Before that, NVDEC's 12-bit surface can only be
    # described as p016le, which claims 16 bits instead of 12. Set this for the
    # sources that hit it: same samples either way (they're msb-aligned, so
    # p016le describes them just as validly, with 4 zeroed low bits), but both
    # the format and the depth we report change with the FFmpeg version.
    needs_p016_before_ffmpeg6: bool = False

    def pix_fmt(self, device):
        return self.cuda_pix_fmt if device == "cuda" else self.cpu_pix_fmt

    def num_planes(self, device):
        return self.cuda_num_planes if device == "cuda" else self.cpu_num_planes


# Sources with more than 8 bits per sample.
_HDR_VIDEOS = (
    NASA_VIDEO_HDR,
    TEST_SRC_2_720P_HDR,
    TEST_SRC_2_12BIT_HDR,
    TESTSRC2_ODD_WIDTH_VP9_10BIT,
    TESTSRC2_ODD_HEIGHT_VP9_10BIT,
    TESTSRC2_ODD_HEIGHT_AND_WIDTH_VP9_10BIT,
    TESTSRC2_ODD_HEIGHT_AND_WIDTH_444_10BIT,
)


# Videos spanning the pixel-format axes RawFrame.planes has to handle: 4:2:0 vs
# 4:4:4 chroma, even vs odd dims (chroma rounds up), and 8- vs 10-/12-bit
# (uint8 vs uint16 planes). All are YUV, so planes are (Y, U, V) - the sources
# whose frames aren't three YUV planes are in _NON_YUV_PLANES_VIDEOS below.
_PLANES_VIDEOS = (
    _PlanesCase(NASA_VIDEO, 8, "yuv420p", "nv12"),  # even dims
    _PlanesCase(TESTSRC2_ODD_HEIGHT_AND_WIDTH_VP9, 8, "yuv420p", "nv12"),  # odd
    # 4:4:4 (full-res chroma). NVDEC can't decode H264 4:4:4, so these fall back
    # to the CPU and are uploaded as 4:4:4 rather than have their chroma halved.
    _PlanesCase(TESTSRC2_ODD_HEIGHT_AND_WIDTH_444, 8, "yuv444p", "yuv444p"),
    _PlanesCase(
        TESTSRC2_ODD_HEIGHT_AND_WIDTH_444_10BIT, 10, "yuv444p10le", "yuv444p16le"
    ),
    # HEVC 4:4:4, which NVDEC decodes natively into its YUV444 surfaces. The
    # 16-bit one is the only 4:4:4 surface above 8 bits, so 10- and 12-bit
    # sources both land in yuv444p16le.
    _PlanesCase(TESTSRC2_444_8BIT_HEVC, 8, "yuv444p", "yuv444p"),
    _PlanesCase(TESTSRC2_444_10BIT_HEVC, 10, "yuv444p10le", "yuv444p16le"),
    _PlanesCase(TESTSRC2_444_12BIT_HEVC, 12, "yuv444p12le", "yuv444p16le"),
    _PlanesCase(TESTSRC2_ODD_HEIGHT_AND_WIDTH_VP9_10BIT, 10, "yuv420p10le", "p010le"),
    _PlanesCase(
        TEST_SRC_2_12BIT_HDR,
        12,
        "yuv420p12le",
        "p012le",
        needs_p016_before_ffmpeg6=True,
    ),
)


# Sources whose frames aren't three YUV planes on the CPU. NVDEC decodes none of
# them - monochrome, planar RGB and FFV1 all send it to the CPU fallback - and
# the fallback converts to an NVDEC surface format before uploading, so on CUDA
# they are three YUV planes like everything else. That conversion is what
# RawFrame.pix_fmt promises ("on CUDA it is always an NVDEC surface format"),
# and it is lossy for the two formats that carry something YUV can't: grayscale
# gains neutral chroma, and alpha is dropped outright.
_NON_YUV_PLANES_VIDEOS = (
    _PlanesCase(TESTSRC2_GRAY_HEVC, 8, "gray", "nv12", cpu_num_planes=1),
    # Planar RGB. The planes come out (R, G, B), which is *not* the order the
    # format stores them in: FFmpeg's gbrp is green, blue, red.
    _PlanesCase(TESTSRC2_GBRP_HEVC, 8, "gbrp", "yuv444p"),
    # Alpha, which is full size like luma rather than subsampled like chroma.
    _PlanesCase(TESTSRC2_YUVA420P_FFV1, 8, "yuva420p", "nv12", cpu_num_planes=4),
)


def _planes_ids(case):
    return case.video.path.stem


class _CustomReader:
    # A file-like object that isn't an io.IOBase subclass, i.e. one recognized
    # by duck-typing alone.
    def __init__(self, file):
        self._file = file

    def read(self, size: int) -> bytes:
        return self._file.read(size)

    def seek(self, offset: int, whence: int) -> int:
        return self._file.seek(offset, whence)


_BLOCKS_SOURCES = (
    pytest.param(lambda path: path, id="path"),
    pytest.param(lambda path: str(path), id="str"),
    pytest.param(lambda path: path.read_bytes(), id="bytes"),
    pytest.param(
        lambda path: torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8),
        id="tensor",
    ),
    pytest.param(lambda path: open(path, "rb"), id="file_like"),
    pytest.param(lambda path: io.BytesIO(path.read_bytes()), id="bytes_io"),
    pytest.param(
        lambda path: _CustomReader(open(path, "rb", buffering=0)), id="custom_reader"
    ),
)


class TestBlocks:

    @pytest.mark.parametrize("device", _block_devices())
    def test_block_output_types(self, device):
        # Demuxer yields Packets, VideoPacketDecoder yields RawFrames, and
        # ColorConverter yields Frames with the expected shape/dtype.
        demuxer, decoder, converter = self._make_blocks(NASA_VIDEO.path, device)

        num_packets = 0
        for packet in demuxer:
            assert isinstance(packet, Packet)
            num_packets += 1
            for decoded in decoder.decode(packet):
                assert isinstance(decoded, RawFrame)
                frame = converter.convert(decoded)
                assert isinstance(frame, Frame)
                assert frame.data.ndim == 3  # CHW
                assert frame.data.shape[0] == 3  # channels first
                assert frame.data.dtype == torch.uint8
                assert frame.data.device.type == device
                assert frame.duration_seconds >= 0

        assert num_packets > 0

    # ===== multi-stream Demuxer =====

    @pytest.mark.parametrize(
        "streams, expected_indices, expected_types",
        (
            ("video", [3], [VideoStream]),
            ("audio", [4], [AudioStream]),
            (("video", "audio"), [3, 4], [VideoStream, AudioStream]),
            # Order is the caller's, not the container's.
            (("audio", "video"), [4, 3], [AudioStream, VideoStream]),
            (0, [0], [VideoStream]),
            ((3, 1), [3, 1], [VideoStream, AudioStream]),
            # "all" skips the two subtitle streams.
            ("all", [0, 1, 3, 4], [VideoStream, AudioStream] * 2),
        ),
    )
    def test_stream_selection(self, streams, expected_indices, expected_types):
        demuxer = Demuxer(NASA_VIDEO.path, streams=streams)

        assert [s.index for s in demuxer.streams] == expected_indices
        assert [type(s) for s in demuxer.streams] == expected_types

    def test_stream_selection_defaults_to_best_video(self):
        assert [s.index for s in Demuxer(NASA_VIDEO.path).streams] == [3]

    @pytest.mark.parametrize(
        "streams, match",
        (
            ((), "streams is empty"),
            (("video", "video"), "already being demuxed"),
            # 3 is the best video stream, so this names it twice.
            (("video", 3), "already being demuxed"),
            (2, "which cannot be decoded"),
            (99, "not a valid stream"),
            (("all", "audio"), "can only be used on its own"),
            ("subtitles", "Invalid stream selector"),
            (1.0, "Invalid stream selector"),
        ),
    )
    def test_stream_selection_errors(self, streams, match):
        with pytest.raises((ValueError, RuntimeError), match=match):
            Demuxer(NASA_VIDEO.path, streams=streams)

    def test_audio_stream_has_no_scan(self):
        (audio,) = Demuxer(NASA_VIDEO.path, streams="audio").streams
        assert not hasattr(audio, "scan")

    def test_one_pass_matches_separate_demuxers(self):
        # The whole point of following both streams at once: what comes out has
        # to be exactly what two separate demuxers give, sample for sample.
        demuxer = Demuxer(NASA_VIDEO.path, streams=("video", "audio"))
        video, audio = demuxer.streams
        decoders = {s.index: s.make_decoder() for s in demuxer.streams}

        frames, samples = [], []
        for packet in demuxer:
            target = frames if packet.stream_index == video.index else samples
            target += decoders[packet.stream_index].decode(packet)
        frames += decoders[video.index].drain()
        samples += decoders[audio.index].drain()

        video_only = Demuxer(NASA_VIDEO.path, streams="video")
        expected_frames = list(
            self._decode(video_only.streams[0].make_decoder(), video_only)
        )
        audio_only = Demuxer(NASA_VIDEO.path, streams="audio")
        expected_samples = list(
            self._decode(audio_only.streams[0].make_decoder(), audio_only)
        )

        assert len(frames) == len(expected_frames) > 0
        assert len(samples) == len(expected_samples) > 0
        for got, expected in zip(frames, expected_frames):
            assert got.pts_seconds == expected.pts_seconds
            torch.testing.assert_close(got.planes, expected.planes, atol=0, rtol=0)
        for got, expected in zip(samples, expected_samples):
            assert got.pts_seconds == expected.pts_seconds
            torch.testing.assert_close(got.data, expected.data, atol=0, rtol=0)

    def test_packets_are_tagged_with_their_stream(self):
        demuxer = Demuxer(NASA_VIDEO.path, streams=("video", "audio"))
        video, audio = demuxer.streams

        seen = {video.index: 0, audio.index: 0}
        for packet in demuxer:
            seen[packet.stream_index] += 1

        assert seen[video.index] > 0
        assert seen[audio.index] > 0

    # ===== metadata =====

    def test_stream_metadata_matches_the_decoders(self):
        # The blocks report the header tier and nothing else, but where a field
        # exists on both sides it has to say the same thing - same name, same
        # value.
        demuxer = Demuxer(NASA_VIDEO.path, streams=("video", "audio"))
        video, audio = demuxer.streams

        expected_video = VideoDecoder(NASA_VIDEO.path).metadata
        for field in (
            "stream_index",
            "codec",
            "bit_rate",
            "duration_seconds_from_header",
            "begin_stream_seconds_from_header",
            "width",
            "height",
            "num_frames_from_header",
            "average_fps_from_header",
            "pixel_aspect_ratio",
            "rotation",
            "color_primaries",
            "color_space",
            "color_transfer_characteristic",
            "pixel_format",
        ):
            assert getattr(video.metadata, field) == getattr(expected_video, field)

        expected_audio = AudioDecoder(NASA_VIDEO.path).metadata
        for field in ("sample_rate", "num_channels", "sample_format", "codec"):
            assert getattr(audio.metadata, field) == getattr(expected_audio, field)

    def test_stream_metadata_has_no_content_tier(self):
        # The whole point: nothing here is derived from content, so nothing
        # here silently changes meaning depending on whether a scan happened.
        (video,) = Demuxer(NASA_VIDEO.path).streams

        for field in (
            "num_frames_from_content",
            "begin_stream_seconds_from_content",
            "end_stream_seconds_from_content",
            "num_frames",
            "average_fps",
            "duration_seconds",
            "begin_stream_seconds",
            "end_stream_seconds",
        ):
            assert not hasattr(video.metadata, field), field

        # Scanning doesn't change that; the exact answers live on the index.
        index = video.scan()
        assert not hasattr(video.metadata, "num_frames")
        assert video.metadata.num_frames_from_header == index.num_frames_from_content

    def test_demuxer_metadata_has_no_stream_list(self):
        demuxer = Demuxer(NASA_VIDEO.path)
        # Older FFmpeg reports this container's duration differently.
        expected_duration = 16.57 if ffmpeg_major_version <= 5 else 13.056

        assert demuxer.metadata.duration_seconds_from_header == pytest.approx(
            expected_duration
        )
        assert demuxer.metadata.best_video_stream_index == 3
        assert demuxer.metadata.best_audio_stream_index == 4
        # Streams are described by demuxer.streams[i].metadata, and only there.
        assert not hasattr(demuxer.metadata, "streams")

    def test_get_container_metadata(self):
        metadata = get_container_metadata(NASA_VIDEO.path)
        # Older FFmpeg reports this container's duration differently.
        expected_duration = 16.57 if ffmpeg_major_version <= 5 else 13.056

        assert metadata.duration_seconds_from_header == pytest.approx(expected_duration)
        assert metadata.best_video_stream_index == 3
        # Every stream, including the two subtitle ones a demuxer can't follow.
        assert [type(s).__name__ for s in metadata.streams] == [
            "VideoStreamHeaderMetadata",
            "AudioStreamHeaderMetadata",
            "StreamMetadata",
            "VideoStreamHeaderMetadata",
            "AudioStreamHeaderMetadata",
            "StreamMetadata",
        ]
        assert [s.stream_index for s in metadata.streams] == list(range(6))

    def test_get_container_metadata_reads_no_packets(self):
        class CountingFileLike:
            def __init__(self, path):
                self._file = open(path, "rb")
                self.bytes_read = 0

            def read(self, size):
                data = self._file.read(size)
                self.bytes_read += len(data)
                return data

            def seek(self, offset, whence):
                return self._file.seek(offset, whence)

        probed = CountingFileLike(NASA_VIDEO.path)
        get_container_metadata(probed)

        demuxed = CountingFileLike(NASA_VIDEO.path)
        list(Demuxer(demuxed))

        assert probed.bytes_read < demuxed.bytes_read

    def test_adding_a_stream_after_demuxing_raises(self):
        # Demuxer follows its streams from construction, so this is only
        # reachable underneath it - but the demuxer is what enforces it, and a
        # stream added late would start from wherever the container now is.
        demuxer = Demuxer(NASA_VIDEO.path)
        demuxer.next_packet()

        with pytest.raises(RuntimeError, match="before the first packet"):
            _blocks_demuxer_add_stream(demuxer._handle, 4)

    # The three decode stages, each expressed as a generator that transforms an
    # iterator of inputs into an iterator of outputs. They compose directly (the
    # sequential pipeline is just convert(decode(demux()))), and a thread
    # boundary between any two stages is inserted with prefetch() below.

    @staticmethod
    def _demux(demuxer):
        yield from demuxer

    @staticmethod
    def _decode(decoder, packets):
        for packet in packets:
            yield from decoder.decode(packet)
        yield from decoder.drain()

    @staticmethod
    def _convert(converter, frames):
        for frame in frames:
            yield converter.convert(frame)

    @staticmethod
    def prefetch(upstream, buffer_size=8):
        # Run `upstream` (a generator chaining one or more stages) on a
        # background thread, yielding its items through a bounded queue. This is
        # the only threading primitive: where you insert it decides which stages
        # overlap. The queue applies backpressure (the worker blocks in q.put()
        # when the buffer is full), so it runs ~buffer_size items ahead.
        q: queue.Queue = queue.Queue(maxsize=buffer_size)
        eof = object()
        error = []

        def worker():
            try:
                for item in upstream:
                    q.put(item)
            except Exception as e:  # surface failures instead of hanging
                error.append(e)
            finally:
                q.put(eof)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        def drain():
            while (item := q.get()) is not eof:
                yield item
            thread.join()  # worker enqueued eof and is finishing; make it explicit
            if error:
                raise error[0]

        return drain()

    @staticmethod
    def _make_blocks(path, device):
        demuxer = Demuxer(path)
        decoder = demuxer.streams[0].make_decoder(device)
        converter = ColorConverter(device=device)
        return demuxer, decoder, converter

    def _decoded_frames(self, path, device):
        # demux + decode, as a single generator of RawFrames (pts order).
        demuxer, decoder, _ = self._make_blocks(path, device)
        return self._decode(decoder, self._demux(demuxer))

    def _decode_sequential(self, path, device):
        # demux -> decode -> color-convert, all on the calling thread.
        demuxer, decoder, converter = self._make_blocks(path, device)
        return list(
            self._convert(converter, self._decode(decoder, self._demux(demuxer)))
        )

    def _decode_prefetch_frames(self, path, device):
        # [demux + decode] on one thread || [color-convert] on another.
        demuxer, decoder, converter = self._make_blocks(path, device)

        frames = self.prefetch(self._decode(decoder, self._demux(demuxer)))
        return list(self._convert(converter, frames))

    def _decode_prefetch_packets(self, path, device):
        # [demux] on one thread || [decode + color-convert] on another.
        demuxer, decoder, converter = self._make_blocks(path, device)
        packets = self.prefetch(self._demux(demuxer))
        return list(self._convert(converter, self._decode(decoder, packets)))

    def _decode_prefetch_packets_and_frames(self, path, device):
        # [demux] || [decode] || [color-convert], each on its own thread.
        demuxer, decoder, converter = self._make_blocks(path, device)
        packets = self.prefetch(self._demux(demuxer))
        frames = self.prefetch(self._decode(decoder, packets))
        return list(self._convert(converter, frames))

    def _to_frame_batch(self, frames):
        return FrameBatch(
            data=torch.stack([f.data for f in frames]),
            pts_seconds=torch.tensor(
                [f.pts_seconds for f in frames], dtype=torch.float64
            ),
            duration_seconds=torch.tensor(
                [f.duration_seconds for f in frames], dtype=torch.float64
            ),
        )

    @staticmethod
    def _assert_matches_video_decoder(got, ref, video):
        # We typically want exact equality but cannot achieve it on CUDA for HDR
        # videos that are downscaled to uint8: the VideoDecoder will ask NVDEC
        # to output an 8bit surface while the VideoPacketDecoder will decode on the
        # 16bit surface (by contract), so there are minor differences.
        if got.is_cuda and got.dtype == torch.uint8 and video in _HDR_VIDEOS:
            assert_tensor_close_on_at_least(got, ref, percentage=99, atol=2)
        else:
            torch.testing.assert_close(got, ref, atol=0, rtol=0)

    # Every codec, container and pixel format we care about.
    _ALL_VIDEOS = (
        NASA_VIDEO,
        BT709_FULL_RANGE,
        NASA_VIDEO_HDR,
        TEST_SRC_2_720P_HDR,
        TEST_SRC_2_12BIT_HDR,
        # NVDEC can't decode this one (too small), so on CUDA this covers
        # the CPU-fallback path: the decoder hands out CPU frames and the
        # converter has to notice and upload them itself.
        H265_VIDEO,
        # Video with a non-zero start time: exercises pts propagation.
        TEST_NON_ZERO_START,
        # Odd dimensions: NVDEC decodes to even-aligned surfaces, so the
        # converter has to crop back to the real width/height.
        TESTSRC2_ODD_WIDTH_VP9,
        TESTSRC2_ODD_HEIGHT_VP9,
        TESTSRC2_ODD_HEIGHT_AND_WIDTH_VP9,
        # yuv444p (odd dims too). NVDEC can't decode 4:4:4, so on CUDA these
        # take the CPU-fallback path.
        TESTSRC2_ODD_WIDTH_444,
        TESTSRC2_ODD_HEIGHT_444,
        TESTSRC2_ODD_HEIGHT_AND_WIDTH_444,
        TESTSRC2_ODD_HEIGHT_AND_WIDTH_444_10BIT,
        # Odd dimensions with 4:2:0 chroma, taking the CPU-fallback path
        # (MPEG-2 isn't decoded by NVDEC): the frame is uploaded padded to
        # even dimensions, and the converter must crop it back. Both sides
        # of this comparison use our own kernels, so unlike the
        # CPU-reference tests the odd-height one can be compared exactly.
        TESTSRC2_ODD_WIDTH_MPEG2,
        TESTSRC2_ODD_HEIGHT_AND_WIDTH_MPEG2,
        # HEVC 4:4:4: NVDEC decodes these natively instead.
        TESTSRC2_444_8BIT_HEVC,
        TESTSRC2_444_10BIT_HEVC,
        TESTSRC2_444_12BIT_HEVC,
        # First keyframe is marked AV_PKT_FLAG_DISCARD by an mp4 edit list.
        DISCARD_FIRST_KEYFRAME_VIDEO,
        # 90-degree display matrix: the converter must rotate.
        NASA_VIDEO_ROTATED,
    )

    @pytest.mark.parametrize("video", _ALL_VIDEOS)
    @pytest.mark.parametrize(
        "decode_method",
        (
            _decode_sequential,
            _decode_prefetch_frames,
            _decode_prefetch_packets,
            _decode_prefetch_packets_and_frames,
        ),
        ids=lambda f: f.__name__.removeprefix("_decode_"),
    )
    @pytest.mark.parametrize("device", _block_devices())
    def test_matches_video_decoder(self, video, decode_method, device):
        if (
            video
            in (
                TESTSRC2_ODD_WIDTH_VP9,
                TESTSRC2_ODD_HEIGHT_VP9,
                TESTSRC2_ODD_HEIGHT_AND_WIDTH_VP9,
            )
            and ffmpeg_major_version == 4
        ):
            pytest.skip("FFmpeg 4 returns one more frame")

        got = self._to_frame_batch(decode_method(self, video.path, device))
        ref = VideoDecoder(video.path, device=device).get_all_frames()

        assert got.data.device.type == device
        assert got.data.shape == ref.data.shape
        self._assert_matches_video_decoder(got.data, ref.data, video)
        torch.testing.assert_close(got.pts_seconds, ref.pts_seconds, atol=0, rtol=0)
        torch.testing.assert_close(
            got.duration_seconds, ref.duration_seconds, atol=0, rtol=0
        )

    @pytest.mark.parametrize("device", _block_devices())
    def test_discard_first_keyframe(self, device):
        # The leading GOP of this asset is trimmed by an mp4 edit list, so its
        # packets are flagged AV_PKT_FLAG_DISCARD: they must be decoded (the
        # frame at pts=0 references the discarded keyframe) but never output.
        expected_pts = [pytest.approx(0.04 * i, abs=1e-6) for i in range(25)]

        path = DISCARD_FIRST_KEYFRAME_VIDEO.path
        frames = self._decode_sequential(path, device)
        assert [frame.pts_seconds for frame in frames] == expected_pts

        # Seeking to the start makes FFmpeg land on the discarded keyframe, so
        # those packets are sent down a second time: the tracking must survive
        # the seek.
        frames = list(self._frames_after_seek(self._make_blocks(path, device), 0))
        assert [frame.pts_seconds for frame in frames] == expected_pts

    @pytest.mark.parametrize(
        "video", (NASA_VIDEO, NASA_VIDEO_HDR, TEST_SRC_2_12BIT_HDR)
    )
    @pytest.mark.parametrize("output_dtype", (torch.uint8, torch.float32, "auto"))
    @pytest.mark.parametrize("device", _block_devices())
    def test_output_dtype_matches_video_decoder(self, video, output_dtype, device):
        # The blocks pipeline must agree with VideoDecoder for every dtype, on
        # both devices.
        demuxer, decoder, _ = self._make_blocks(video.path, device)
        converter = ColorConverter(device=device, output_dtype=output_dtype)
        got = self._to_frame_batch(
            list(self._convert(converter, self._decode(decoder, self._demux(demuxer))))
        )
        ref = VideoDecoder(
            video.path, device=device, output_dtype=output_dtype
        ).get_all_frames()

        assert got.data.dtype == ref.data.dtype
        assert got.data.shape == ref.data.shape
        self._assert_matches_video_decoder(got.data, ref.data, video)

    @pytest.mark.parametrize("device", _block_devices())
    def test_output_dtype_auto_is_resolved_per_frame(self, device):
        converter = ColorConverter(device=device, output_dtype="auto")
        dtypes = []
        for video in (NASA_VIDEO, NASA_VIDEO_HDR, NASA_VIDEO):
            frame = next(self._decoded_frames(video.path, device))
            dtypes.append(converter.convert(frame).data.dtype)

        assert dtypes == [torch.uint8, torch.float32, torch.uint8]

    @pytest.mark.parametrize("device", _block_devices())
    def test_invalid_output_dtype(self, device):
        with pytest.raises(ValueError, match="Invalid output_dtype"):
            ColorConverter(device=device, output_dtype=torch.uint16)

    @pytest.mark.parametrize("device", _block_devices())
    def test_color_converter_reused_across_videos(self, device):
        # A single unbound ColorConverter must correctly convert frames from
        # different videos - here interleaved frame-by-frame, so the converter
        # switches input resolution/format/rotation on every call.
        converter = ColorConverter(device=device)
        videos = [NASA_VIDEO, BT709_FULL_RANGE, NASA_VIDEO_ROTATED]
        generators = [self._decoded_frames(v.path, device) for v in videos]
        outputs = [[] for _ in videos]

        done = [False] * len(videos)
        while not all(done):
            for i, gen in enumerate(generators):
                if done[i]:
                    continue
                decoded = next(gen, None)
                if decoded is None:
                    done[i] = True
                    continue
                outputs[i].append(converter.convert(decoded))

        for video, frames in zip(videos, outputs):
            got = self._to_frame_batch(frames)
            ref = VideoDecoder(video.path, device=device).get_all_frames()
            assert got.data.shape == ref.data.shape
            torch.testing.assert_close(got.data, ref.data, atol=0, rtol=0)

    @needs_cuda
    @pytest.mark.parametrize(
        ("frame_device", "converter_device"), (("cpu", "cuda"), ("cuda", "cpu"))
    )
    def test_converter_refuses_other_devices(self, frame_device, converter_device):
        frame = next(self._decoded_frames(NASA_VIDEO.path, frame_device))
        converter = ColorConverter(device=converter_device)
        with pytest.raises(
            RuntimeError, match="only converts frames that are already on its own"
        ):
            converter.convert(frame)

    @needs_cuda
    @pytest.mark.parametrize(
        "converter_device", ("cuda", "cuda:0", torch.device("cuda"))
    )
    def test_converter_accepts_every_spelling_of_the_same_device(
        self, converter_device
    ):
        # The frame names a concrete GPU ("cuda:0") while the converter may have
        # been given a bare "cuda". Those are the same place, and a pipeline
        # that spells them differently must not trip the check.
        frame = next(self._decoded_frames(NASA_VIDEO.path, "cuda"))
        converted = ColorConverter(device=converter_device).convert(frame)
        assert converted.data.device.type == "cuda"

    @pytest.mark.parametrize("device", _block_devices())
    def test_set_cuda_backend_is_a_noop(self, device):
        # The blocks always use the NVDEC CUDA backend. Asking for the "ffmpeg"
        # one changes nothing, rather than silently producing something else.
        with set_cuda_backend("ffmpeg"):
            got = self._to_frame_batch(self._decode_sequential(NASA_VIDEO.path, device))
        ref = self._to_frame_batch(self._decode_sequential(NASA_VIDEO.path, device))
        torch.testing.assert_close(got.data, ref.data, atol=0, rtol=0)

    def _first_frame(self, path, device):
        # The first RawFrame of a video
        demuxer, decoder, converter = self._make_blocks(path, device)
        frame = next(self._decode(decoder, self._demux(demuxer)))
        return frame, converter

    @pytest.mark.parametrize("device_str", _block_devices())
    def test_device_none_default_device(self, device_str):
        # VideoPacketDecoder and ColorConverter default to device=None, which should
        # respect both the torch.device() context manager and
        # torch.set_default_device().

        def assert_first_frame_is_on_default_device():
            # Note the absence of any device parameter.
            demuxer = Demuxer(NASA_VIDEO.path)
            decoder = demuxer.streams[0].make_decoder()
            converter = ColorConverter()
            decoded = next(self._decode(decoder, self._demux(demuxer)))
            assert decoded.planes[0].device.type == device_str
            assert converter.convert(decoded).data.device.type == device_str

        with torch.device(device_str):
            assert_first_frame_is_on_default_device()

        original_device = torch.get_default_device()
        try:
            torch.set_default_device(device_str)
            assert_first_frame_is_on_default_device()
        finally:
            torch.set_default_device(original_device)

    @pytest.mark.parametrize("device", _block_devices())
    def test_device_torch_device_instance(self, device):
        # device can be a torch.device instance, not just a string.
        frame, converter = self._first_frame(NASA_VIDEO.path, torch.device(device))
        assert frame.planes[0].device.type == device
        assert converter.convert(frame).data.device.type == device

    @pytest.mark.parametrize(
        "case", _PLANES_VIDEOS + _NON_YUV_PLANES_VIDEOS, ids=_planes_ids
    )
    @pytest.mark.parametrize("device", _block_devices())
    def test_planes_structure(self, case, device):
        # planes shape/dtype/device and the accompanying metadata.
        frame, converter = self._first_frame(case.video.path, device)
        planes, pix_fmt = frame.planes, frame.pix_fmt

        expected_pix_fmt = case.pix_fmt(device)
        if (
            device == "cuda"
            and case.needs_p016_before_ffmpeg6
            and ffmpeg_major_version < 6
        ):
            expected_pix_fmt = "p016le"

        # bit_depth is the depth of the pixel format. It usually matches the
        # source's, but not always - it depends on nvdec's capabilities.
        expected_bit_depth = (
            16 if expected_pix_fmt in ("p016le", "yuv444p16le") else case.bit_depth
        )

        assert pix_fmt == expected_pix_fmt
        # "gbr" is what a planar RGB frame reports: its samples are already RGB,
        # so there is no YUV matrix to name.
        assert frame.colorspace in ("bt709", "bt2020nc", "smpte170m", "gbr", "unknown")
        assert frame.color_range in ("tv", "pc", "unknown")  # FFmpeg has only these

        # All planes are 2D views living on the frame's own device.
        assert len(planes) == case.num_planes(device)
        for plane in planes:
            assert plane.ndim == 2
            assert plane.device.type == device

        assert frame.bit_depth == expected_bit_depth

        expected_dtype = torch.uint16 if frame.bit_depth > 8 else torch.uint8
        assert all(plane.dtype == expected_dtype for plane in planes)

        if device == "cuda" and expected_dtype == torch.uint16:
            # Whatever depth the format claims, a 16-bit CUDA surface holds the
            # source's samples msb-aligned, with the unused low bits zeroed.
            # That's what lets test_cuda_planes_match_cpu shift them
            # back down by 16 - bit_depth and compare against the CPU planes.
            unused_low_bits = (1 << (16 - case.bit_depth)) - 1
            for plane in planes:
                assert (plane.to(torch.int32) & unused_low_bits).count_nonzero() == 0

        height, width = converter.convert(frame).data.shape[1:]
        # The first plane is always full size: luma, or red for a planar RGB
        # format. So is a trailing alpha one, when the format has it.
        assert planes[0].shape == (height, width) == (frame.height, frame.width)
        if len(planes) == 4:
            assert planes[3].shape == (height, width)

        # Below is just a fancy way to divide by 2 accounting for odd sizes,
        # matching the FFmpeg logic
        subsampled = not (pix_fmt.startswith("gbr") or "444" in pix_fmt)
        log2_h, log2_w = (1, 1) if subsampled else (0, 0)
        expected_chroma_shape = (
            (height + (1 << log2_h) - 1) >> log2_h,
            (width + (1 << log2_w) - 1) >> log2_w,
        )
        # Planes 1 and 2 are the subsampled ones for YUV, and full-size green
        # and blue for planar RGB. Grayscale has neither.
        for plane in planes[1:3]:
            assert plane.shape == expected_chroma_shape

    @pytest.mark.parametrize("device", _block_devices())
    def test_planes_are_not_rotated_but_color_conversion_rotates(self, device):
        # The planes are views on the decoder's own memory, so they're in the
        # source's pre-rotation geometry, and so are the dims the frame reports.
        # Only the converted frame is rotated.
        frame, converter = self._first_frame(NASA_VIDEO_ROTATED.path, device)
        Y = frame.planes[0]

        height = NASA_VIDEO_ROTATED.get_height()  # post-rotation
        width = NASA_VIDEO_ROTATED.get_width()
        assert Y.shape == (width, height) == (frame.height, frame.width)
        assert frame.rotation_degrees == 90
        assert converter.convert(frame).data.shape == (3, height, width)

    @pytest.mark.parametrize("device", _block_devices())
    def test_no_rotation(self, device):
        frame, _ = self._first_frame(NASA_VIDEO.path, device)
        assert frame.rotation_degrees == 0

    @pytest.mark.needs_cuda
    @pytest.mark.parametrize("record_stream", (True, False))
    def test_storage_record_stream(self, record_stream):
        # Using VideoPacketDecoder on one stream and consuming the frames on a
        # different stream requires the user to call record_stream() on the
        # frame storage.
        # Without the record_stream() call the decoder's next frame may be
        # handed the same buffer and overwrites it while the read is still
        # queued.
        # See [Standalone Frame Storage and the need for record_stream]
        video = NASA_VIDEO.path
        decode_stream = torch.cuda.Stream()
        read_stream = torch.cuda.Stream()

        def run(separate_stream):
            demuxer, decoder, _ = self._make_blocks(video, "cuda")
            reads = []
            for packet in itertools.chain(demuxer, [None]):
                with torch.cuda.stream(decode_stream):
                    decoded = (
                        decoder.drain() if packet is None else decoder.decode(packet)
                    )
                with torch.cuda.stream(
                    read_stream if separate_stream else decode_stream
                ):
                    while decoded:
                        frame = decoded.pop(0)
                        torch.cuda._sleep(20_000_000)  # ~10ms, fall behind
                        reads.append(frame.planes[0].clone())
                        if separate_stream and record_stream:
                            frame.storage.record_stream(read_stream)
            torch.cuda.synchronize()
            return reads

        ref = run(separate_stream=False)
        got = run(separate_stream=True)
        wrong = sum(1 for a, b in zip(ref, got) if not torch.equal(a, b))
        if record_stream:
            assert wrong == 0
        else:
            # Guards the test itself: without the call this really does corrupt,
            # so the assertion above is meaningful.
            assert wrong > 0

    @pytest.mark.needs_cuda
    def test_backlogged_converter_on_separate_stream(self):
        # Similar test to test_storage_record_stream(), but with the ColorConverter on a
        # separate stream. In this case, *we* call record_stream() on behalf of
        # the user.
        # See [Standalone Frame Storage and the need for record_stream]
        video = NASA_VIDEO.path
        demuxer, decoder, converter = self._make_blocks(video, "cuda")
        decode_stream = torch.cuda.Stream()
        convert_stream = torch.cuda.Stream()

        frames = []
        for packet in itertools.chain(demuxer, [None]):
            with torch.cuda.stream(decode_stream):
                decoded = decoder.drain() if packet is None else decoder.decode(packet)
            with torch.cuda.stream(convert_stream):
                while decoded:
                    torch.cuda._sleep(20_000_000)  # ~10ms
                    frames.append(converter.convert(decoded.pop(0)))
        torch.cuda.synchronize()

        got = self._to_frame_batch(frames)
        ref = VideoDecoder(video, device="cuda").get_all_frames()
        torch.testing.assert_close(got.data, ref.data, atol=0, rtol=0)

    @pytest.mark.needs_cuda
    @pytest.mark.parametrize("case", _PLANES_VIDEOS, ids=_planes_ids)
    def test_cuda_planes_match_cpu(self, case):
        # NVDEC and FFmpeg's software decoder produce the very same samples, so
        # the raw planes must match bit-for-bit across devices, once the NVDEC
        # ones are shifted back down to the CPU planes' scale.
        cpu = self._first_frame(case.video.path, "cpu")[0]
        cuda = self._first_frame(case.video.path, "cuda")[0]

        # Every 16-bit CUDA surface spans the full uint16 range, whether it says
        # so (p016le, yuv444p16le) or reports the source's depth (p010le,
        # p012le). The CPU planes are at the source's own scale.
        is_uint16 = cuda.planes[0].dtype == torch.uint16
        shift = 16 - case.bit_depth if is_uint16 else 0
        assert len(cpu.planes) == len(cuda.planes)
        for cpu_plane, cuda_plane in zip(cpu.planes, cuda.planes):
            torch.testing.assert_close(
                cpu_plane.to(torch.int32),
                cuda_plane.cpu().to(torch.int32) >> shift,
                atol=0,
                rtol=0,
            )

    @pytest.mark.parametrize("case", _PLANES_VIDEOS, ids=_planes_ids)
    @pytest.mark.parametrize("device", _block_devices())
    def test_color_converter_honors_sample_bit_depth(self, case, device):
        # Paint the planes with the limited-range black and white points,
        # expressed in the frame's own sample scale, and check the converter
        # maps them to 0 and 255. This is what pins its notion of how many bits
        # a sample carries and where they sit: misreading either sends both ends
        # to the same extreme.
        frame, converter = self._first_frame(case.video.path, device)
        assert frame.color_range in ("tv", "unknown")  # unknown is treated as tv

        def sample(value_8bit):
            value = value_8bit << (frame.bit_depth - 8)
            if _is_msb_aligned(frame.pix_fmt):
                value <<= 16 - frame.bit_depth
            return value

        Y, U, V = frame.planes
        U.fill_(sample(128))  # no color
        V.fill_(sample(128))

        for luma, expected in ((235, 255), (16, 0)):
            Y.fill_(sample(luma))
            data = converter.convert(frame).data.to(torch.int16)
            # Loose bound: swscale's bt2020 handling lands a couple of levels
            # short of the endpoints. Misreading the depth is off by a factor
            # of 16 or 64, which lands at the opposite endpoint - never 5
            # levels away.
            assert (data - expected).abs().max() <= 5

    @pytest.mark.parametrize("device", _block_devices())
    def test_planes_mutation_visible_in_color_convert(self, device):
        # The planes are views into the frame's own memory, so editing them and
        # color-converting the frame reflects the edit.
        frame, converter = self._first_frame(NASA_VIDEO.path, device)

        original = converter.convert(frame).data.clone()

        planes = frame.planes
        for value, plane in zip((50, 100, 150), planes):
            plane.fill_(value)  # overwrite Y, U, V with distinct constants
        edited = converter.convert(frame).data

        assert not torch.equal(edited, original)
        max_abs_diff = (edited.to(torch.int32) - original.to(torch.int32)).abs().max()
        assert max_abs_diff.item() > 50

        # since every plane is now a constant value, each RGB channel is a
        # single color across all pixels.
        for channel in edited:
            assert channel.min().item() == channel.max().item()

    @pytest.mark.parametrize("device", _block_devices())
    def test_planes_are_views(self, device):
        # The planes view the frame's own buffer, and they're cached: asking
        # twice hands back the very same tensors.
        frame, _ = self._first_frame(NASA_VIDEO.path, device)
        planes_a = frame.planes
        planes_b = frame.planes
        assert all(a is b for a, b in zip(planes_a, planes_b))

        original = int(planes_a[0][0, 0].item())
        planes_a[0][0, 0] = original ^ 0xFF  # flip first pixel in Y
        assert int(planes_b[0][0, 0].item()) == (original ^ 0xFF)

    @pytest.mark.parametrize("case", _PLANES_VIDEOS, ids=_planes_ids)
    @pytest.mark.parametrize("device", _block_devices())
    def test_planes_outlive_frame(self, case, device):
        # The views keep the frame alive, so they stay valid (readable and
        # writable) after the RawFrame they came from is dropped.
        # grep for this test name to see associated comment in the code.
        frame, _ = self._first_frame(case.video.path, device)
        planes = frame.planes
        saved = [plane.clone() for plane in planes]

        del frame
        gc.collect()

        for plane, snapshot in zip(planes, saved):
            torch.testing.assert_close(
                plane.to(torch.int32), snapshot.to(torch.int32), atol=0, rtol=0
            )
        planes[0][0, 0] = 0  # still writable: the memory is valid

    @pytest.mark.parametrize("video", (NASA_VIDEO, BT709_FULL_RANGE))
    @pytest.mark.parametrize("device", _block_devices())
    def test_neutral_chroma_is_grayscale(self, video, device):
        # Forcing the chroma planes to neutral (128) yields a gray image
        # (R == G == B). Not testing much, but fun - isn't it?
        frame, converter = self._first_frame(video.path, device)
        planes = frame.planes
        assert planes[0].dtype == torch.uint8

        _, U, V = planes
        U.fill_(128)
        V.fill_(128)

        r, g, b = converter.convert(frame).data.to(torch.int16)
        torch.testing.assert_close(r, g, atol=1, rtol=0)
        torch.testing.assert_close(g, b, atol=1, rtol=0)

    @pytest.mark.needs_cuda
    @pytest.mark.parametrize(
        "video, expected_pix_fmt",
        (
            # Too small for NVDEC.
            (H265_VIDEO, "nv12"),
            # H264 4:4:4, which NVDEC can't decode. Uploading it as NV12 would
            # halve its chroma resolution, so it stays 4:4:4.
            (TESTSRC2_ODD_HEIGHT_AND_WIDTH_444, "yuv444p"),
            (TESTSRC2_ODD_HEIGHT_AND_WIDTH_444_10BIT, "yuv444p16le"),
        ),
    )
    def test_cpu_fallback_is_on_cuda(self, video, expected_pix_fmt):
        # A CUDA VideoPacketDecoder hands out CUDA frames even for the streams it has
        # to decode on the CPU, and they're in an NVDEC surface format like any
        # other CUDA frame.
        assert VideoDecoder(video.path, device="cuda").cpu_fallback

        frame, _ = self._first_frame(video.path, "cuda")
        assert frame.pix_fmt == expected_pix_fmt
        assert all(plane.device.type == "cuda" for plane in frame.planes)

    @pytest.mark.parametrize(
        "pix_fmt, codec, container",
        (
            ("pal8", "rawvideo", "nut"),
            # Before FFmpeg 8 the nut muxer has no rawvideo tag for the float
            # formats and silently writes a bogus one, so the file reads back as
            # rgb555le. EXR in mkv stores gbrpf32le properly on all versions.
            ("gbrpf32le", "exr", "mkv"),
        ),
    )
    def test_planes_of_non_viewable_format(self, tmp_path, pix_fmt, codec, container):
        # Palettised and float formats can't be handed out as views. Everything
        # *but* the planes still works, which is what lets a caller check
        # pix_fmt before reaching for them.
        path = tmp_path / f"{pix_fmt}.{container}"
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10:duration=1",
                "-c:v", codec, "-pix_fmt", pix_fmt, str(path),
            ],
            check=True,
        )  # fmt: skip

        frame, _ = self._first_frame(path, "cpu")
        assert frame.pix_fmt == pix_fmt
        assert (frame.width, frame.height) == (64, 48)
        with pytest.raises(RuntimeError, match=f"Cannot expose {pix_fmt} as a view"):
            frame.planes

    # ===== seeking =====

    def _frames_after_seek(self, blocks, seconds):
        # Seek, then yield every frame that comes out from there. A seek
        # invalidates the frames the decoder holds as references, hence the
        # reset(). The first frames yielded typically precede `seconds`: a
        # decoder can only start on a keyframe.
        demuxer, decoder, converter = blocks
        demuxer.seek(seconds)
        decoder.reset()
        for packet in demuxer:
            for raw_frame in decoder.decode(packet):
                yield converter.convert(raw_frame)
        # Seeking into the last GOP can leave the codec holding frames until
        # it's told the stream ended.
        for raw_frame in decoder.drain():
            yield converter.convert(raw_frame)

    def _first_frame_after_seek(self, blocks, seconds):
        return next(self._frames_after_seek(blocks, seconds))

    @pytest.mark.parametrize(
        "video",
        (
            NASA_VIDEO,
            # 5 keyframes for 10 frames, and they're reordered: the densest
            # seeking any of our assets asks for.
            H265_VIDEO,
            TEST_SRC_2_720P_MPEG4,
        ),
    )
    @pytest.mark.parametrize("device", _block_devices())
    def test_seek_to_keyframe_matches_get_frame_played_at(self, video, device):
        # On a keyframe, seeking and taking the next frame is the same thing as
        # VideoDecoder.get_frame_played_at: the seek lands on that keyframe, so
        # it's the first thing the decoder outputs. This holds for every
        # keyframe of the file, not just a lucky one.
        video_decoder = VideoDecoder(video.path, device=device)
        for keyframe_index in video_decoder._get_key_frame_indices():
            seconds = video_decoder.get_frame_at(keyframe_index).pts_seconds

            blocks = self._make_blocks(video.path, device)
            got = self._first_frame_after_seek(blocks, seconds)
            expected = video_decoder.get_frame_played_at(seconds)

            assert got.pts_seconds == expected.pts_seconds == seconds
            assert_frames_equal(got.data, expected.data)

    @pytest.mark.parametrize("device", _block_devices())
    def test_seek_to_non_keyframe_differs_from_get_frame_played_at(self, device):
        # On anything else, the two differ: a decoder can only start on a
        # keyframe, so the seek goes *back* to the one preceding the target and
        # the first frame out is an earlier one. Getting to the frame
        # VideoDecoder returns means decoding forward from there, and what we
        # decode along the way must be the real frames - the ones VideoDecoder
        # gives for those same indices, in that same order.
        num_frames = 10
        video_decoder = VideoDecoder(NASA_VIDEO.path, device=device)
        keyframe_index = video_decoder._get_key_frame_indices()[1]
        non_keyframe_index = keyframe_index + 3
        seconds = video_decoder.get_frame_at(non_keyframe_index).pts_seconds

        blocks = self._make_blocks(NASA_VIDEO.path, device)
        got = list(
            itertools.islice(self._frames_after_seek(blocks, seconds), num_frames)
        )
        expected = video_decoder.get_frames_in_range(
            keyframe_index, keyframe_index + num_frames
        )

        assert video_decoder.get_frame_played_at(seconds).pts_seconds == seconds
        assert got[0].pts_seconds < seconds  # we landed before the target
        assert seconds in [frame.pts_seconds for frame in got]  # and we reach it

        assert len(got) == num_frames
        for got_frame, expected_pts, expected_data in zip(
            got, expected.pts_seconds, expected.data
        ):
            assert got_frame.pts_seconds == expected_pts
            assert_frames_equal(got_frame.data, expected_data)

    @pytest.mark.parametrize("seconds", (0.3, 0.5, 0.7))
    @pytest.mark.parametrize("device", _block_devices())
    def test_seek_to_non_keyframe_can_land_past_target(self, seconds, device):
        # And sometimes decoding forward doesn't get you there at all. This
        # file's keyframes are reordered, and FFmpeg resolves a seek against
        # *decode* timestamps, so it can land on a keyframe that is displayed
        # after the target - leaving the frames in between unreachable, however
        # far forward we decode. VideoDecoder's exact mode gets it right by
        # scanning the file for a presentation-order index, which is what we'd
        # need too.
        blocks = self._make_blocks(H265_VIDEO.path, device)

        got = self._first_frame_after_seek(blocks, seconds)

        assert got.pts_seconds > seconds
        exact = VideoDecoder(H265_VIDEO.path, seek_mode="exact", device=device)
        assert exact.get_frame_played_at(seconds).pts_seconds == seconds

    @pytest.mark.parametrize("video", (NASA_VIDEO, H265_VIDEO, TEST_SRC_2_720P_MPEG4))
    @pytest.mark.parametrize("device", _block_devices())
    def test_seek_matches_video_decoder_approximate_get_frame_played_at(
        self, video, device
    ):
        # Our seek is VideoDecoder's approximate one, frame for frame.
        #
        # It holds against get_frame_played_at() and against nothing else, and
        # that scoping is the point. get_frame_played_at() is the only
        # VideoDecoder API that seeks straight to the timestamp it was given:
        # it turns `seconds` into a pts and hands that to FFmpeg, which is the
        # same two steps Demuxer.seek() takes, so the match is structural
        # rather than a property of these files. Every other API goes through
        # a frame index, which approximate mode derives from the header's
        # average fps and converts back into a pts - a round trip nothing in
        # the blocks performs, and one that doesn't come back where it started
        # unless the file is constant-frame-rate.
        video_decoder = VideoDecoder(video.path, device=device)
        num_frames = video_decoder.metadata.num_frames

        for index in range(0, num_frames, max(1, num_frames // 10)):
            frame = video_decoder.get_frame_at(index)
            # Aim at the middle of a frame, so that no target lands on a frame
            # boundary where the two sides could round differently.
            seconds = frame.pts_seconds + frame.duration_seconds / 2

            # A fresh VideoDecoder per target: it skips the seek when the
            # target is just ahead of the last frame it decoded, which would
            # hide the very behaviour we're comparing against.
            expected = VideoDecoder(
                video.path, seek_mode="approximate", device=device
            ).get_frame_played_at(seconds)

            blocks = self._make_blocks(video.path, device)
            got = next(
                frame
                for frame in self._frames_after_seek(blocks, seconds)
                # VideoDecoder's own criterion: the first frame that hasn't
                # finished playing by then. Not `pts_seconds >= seconds`, which
                # would skip the target frame whenever we land right on it.
                if frame.pts_seconds + frame.duration_seconds > seconds
            )

            assert got.pts_seconds == expected.pts_seconds
            assert_frames_equal(got.data, expected.data)

    @pytest.mark.parametrize("device", _block_devices())
    def test_seek_without_reset_raises(self, device):
        # Skipping the reset used to give you stale frames: a decoder holds a
        # few back, and those belong to wherever we were *before* the seek. They
        # aren't corrupt - the landing keyframe gives the codec a clean slate -
        # they're just from the wrong place, so nothing complains and the caller
        # silently gets the wrong frames. The packets carry which side of the
        # seek they came from, so that is now an error instead.
        video_decoder = VideoDecoder(NASA_VIDEO.path, device=device)
        keyframe_index = video_decoder._get_key_frame_indices()[1]
        seconds = video_decoder.get_frame_at(keyframe_index).pts_seconds

        demuxer = Demuxer(NASA_VIDEO.path)
        decoder = demuxer.streams[0].make_decoder(device)
        num_decoded = 0
        for packet in demuxer:  # decode a bit, so frames pile up in the codec
            num_decoded += len(decoder.decode(packet))
            if num_decoded >= 3:
                break

        demuxer.seek(seconds)  # ... and no reset()
        with pytest.raises(RuntimeError, match="seeked since this decoder"):
            next(frame for packet in demuxer for frame in decoder.decode(packet))

        # With the reset, that same seek starts exactly where it was asked to.
        blocks = self._make_blocks(NASA_VIDEO.path, device)
        assert self._first_frame_after_seek(blocks, seconds).pts_seconds == seconds

    def test_seek_without_reset_raises_for_every_decoder(self):
        # The reason this check exists: with several streams there are several
        # decoders to remember, and forgetting one is invisible otherwise.
        demuxer = Demuxer(NASA_VIDEO.path, streams=("video", "audio"))
        video, audio = demuxer.streams
        decoders = {s.index: s.make_decoder() for s in demuxer.streams}

        # Both decoders have to have seen a packet: one that was never fed
        # anything has nothing stale to hold on to, and needs no reset.
        fed = set()
        for packet in demuxer:
            decoders[packet.stream_index].decode(packet)
            fed.add(packet.stream_index)
            if fed == {video.index, audio.index}:
                break

        demuxer.seek(4.0)
        decoders[video.index].reset()  # ... and forget the audio one

        for packet in demuxer:
            if packet.stream_index == audio.index:
                with pytest.raises(RuntimeError, match="seeked since this decoder"):
                    decoders[audio.index].decode(packet)
                break

    def test_seek_without_converter_reset_raises(self):
        # A seek invalidates the resampler's state too, and the demuxer cannot
        # know the converter exists - so the samples carry the check onward.
        demuxer = Demuxer(NASA_AUDIO_MP3.path, streams="audio")
        decoder = demuxer.streams[0].make_decoder()
        converter = AudioConverter(sample_rate=16_000)

        converted = False
        for packet in demuxer:
            for raw in decoder.decode(packet):
                converter.convert(raw)
                converted = True
            if converted:
                break

        demuxer.seek(2.0)
        decoder.reset()  # ... and forget the converter

        raws = []
        for packet in demuxer:
            raws = decoder.decode(packet)
            if raws:
                break
        with pytest.raises(RuntimeError, match="seeked since this converter"):
            converter.convert(raws[0])

        converter.reset()
        converter.convert(raws[0])  # and now it is fine

    @pytest.mark.parametrize("video", (NASA_VIDEO, H265_VIDEO, TEST_SRC_2_720P_MPEG4))
    @pytest.mark.parametrize("device", _block_devices())
    def test_seek_backwards(self, video, device):
        # Walking the keyframes in reverse, on a single demuxer and decoder:
        # every seek here goes back over ground that was already decoded, which
        # is the case a decoder can't get right on its own - it has to be told
        # to drop what it holds.
        video_decoder = VideoDecoder(video.path, device=device)
        blocks = self._make_blocks(video.path, device)

        for keyframe_index in reversed(list(video_decoder._get_key_frame_indices())):
            seconds = video_decoder.get_frame_at(keyframe_index).pts_seconds

            got = self._first_frame_after_seek(blocks, seconds)
            expected = video_decoder.get_frame_played_at(seconds)

            assert got.pts_seconds == expected.pts_seconds == seconds
            assert_frames_equal(got.data, expected.data)

    @pytest.mark.parametrize("video", (NASA_VIDEO, H265_VIDEO, TEST_SRC_2_720P_MPEG4))
    @pytest.mark.parametrize("device", _block_devices())
    def test_seek_is_repeatable(self, video, device):
        # Seeking to the same place twice, and going elsewhere and back, lands
        # on the same frame every time: neither block keeps state that biases
        # the next seek.
        video_decoder = VideoDecoder(video.path, device=device)
        keyframe_indices = video_decoder._get_key_frame_indices()
        seconds = video_decoder.get_frame_at(keyframe_indices[0]).pts_seconds
        elsewhere = video_decoder.get_frame_at(keyframe_indices[-1]).pts_seconds
        blocks = self._make_blocks(video.path, device)

        first = self._first_frame_after_seek(blocks, seconds)
        again = self._first_frame_after_seek(blocks, seconds)
        self._first_frame_after_seek(blocks, elsewhere)
        after_going_elsewhere = self._first_frame_after_seek(blocks, seconds)

        for frame in (again, after_going_elsewhere):
            assert frame.pts_seconds == first.pts_seconds
            assert_frames_equal(frame.data, first.data)

    @pytest.mark.parametrize("video", _ALL_VIDEOS)
    @pytest.mark.parametrize("device", _block_devices())
    def test_seek_to_mid_file_frame(self, video, device):
        # Breadth: containers keep their own index and codecs their own
        # reordering, so seeking is worth trying on every asset the sequential
        # decode is tried on. One target per file, halfway in, reached the way
        # a caller would reach it: seek, then decode forward dropping the
        # frames between the keyframe we land on and the one we want.
        if video is H265_VIDEO:
            pytest.skip(
                "Its seek lands past the target, see "
                "test_seek_to_non_keyframe_can_land_past_target"
            )
        video_decoder = VideoDecoder(video.path, device=device)
        expected = video_decoder.get_frame_at(video_decoder.metadata.num_frames // 2)

        blocks = self._make_blocks(video.path, device)
        got = next(
            frame
            for frame in self._frames_after_seek(blocks, expected.pts_seconds)
            if frame.pts_seconds >= expected.pts_seconds
        )

        assert got.pts_seconds == expected.pts_seconds
        self._assert_matches_video_decoder(got.data, expected.data, video)

    @pytest.mark.parametrize("device", _block_devices())
    def test_independent_pipelines_seek_in_parallel(self, device):
        # What seeking is for: targets spread over a file, one set of blocks
        # per thread, each seeking to its own part of the stream and decoding
        # forward to the frame it wants. The pipelines share nothing - not even
        # the container, since each opens its own - so each has to come back
        # with exactly what VideoDecoder decodes for that frame.
        frame_indices = (10, 100, 200, 350)
        video_decoder = VideoDecoder(NASA_VIDEO.path, device=device)
        expected_frames = [video_decoder.get_frame_at(i) for i in frame_indices]

        def decode_frame_played_at(seconds):
            blocks = self._make_blocks(NASA_VIDEO.path, device)
            return next(
                frame
                for frame in self._frames_after_seek(blocks, seconds)
                if frame.pts_seconds >= seconds
            )

        with concurrent.futures.ThreadPoolExecutor(len(frame_indices)) as pool:
            got_frames = list(
                pool.map(
                    decode_frame_played_at,
                    [frame.pts_seconds for frame in expected_frames],
                )
            )

        for got, expected in zip(got_frames, expected_frames):
            assert got.pts_seconds == expected.pts_seconds
            assert_frames_equal(got.data, expected.data)

    @pytest.mark.parametrize("device", _block_devices())
    def test_seek_while_demuxing_on_another_thread(self, device):
        # Demuxing on one thread and decoding on another. Each block belongs to
        # the thread that drives it, so the seek happens on the demux thread
        # and the reset on the decode thread - and the reset has to land
        # between the last pre-seek packet and the first post-seek one, which
        # is a matter of *ordering*, not timing. Sequencing it through the
        # queue is all it takes: the demux thread posts a marker where it
        # seeked, the same way prefetch() posts one at end of stream.
        num_frames = 5
        reset_marker = object()
        video_decoder = VideoDecoder(NASA_VIDEO.path, device=device)
        keyframe_index = video_decoder._get_key_frame_indices()[1]
        seconds = video_decoder.get_frame_at(keyframe_index).pts_seconds

        demuxer, decoder, converter = self._make_blocks(NASA_VIDEO.path, device)

        def demux_then_seek():
            for _, packet in zip(range(6), demuxer):
                yield packet
            demuxer.seek(seconds)
            yield reset_marker
            for _, packet in zip(range(15), demuxer):
                yield packet

        frames = []
        seeked = False
        for item in self.prefetch(demux_then_seek()):
            if item is reset_marker:
                decoder.reset()
                seeked = True
                continue
            decoded_frames = decoder.decode(item)
            if seeked:
                frames.extend(converter.convert(frame) for frame in decoded_frames)

        expected = video_decoder.get_frames_in_range(
            keyframe_index, keyframe_index + num_frames
        )
        assert len(frames) >= num_frames
        for frame, expected_pts, expected_data in zip(
            frames, expected.pts_seconds, expected.data
        ):
            assert frame.pts_seconds == expected_pts
            assert_frames_equal(frame.data, expected_data)

    @pytest.mark.parametrize("seconds", (1.0, 2.0, 3.0))
    @pytest.mark.parametrize("device", _block_devices())
    def test_seek_mpeg_program_stream(self, seconds, device):
        # Seeking in an MPEG program stream lands on a container byte offset
        # rather than on a keyframe, so the packets that follow are typically
        # mid-GOP ones whose reference frames were never demuxed. The frames
        # they'd produce are simply missing from the output - what must not
        # happen is the decoder emitting them anyway, decoded against whatever
        # references it happens to hold.
        #
        # VideoDecoder can't act as a reference here: it fails on this asset's
        # video stream in both seek modes. We compare against a sequential
        # decode of the same file through the blocks instead.
        video = SINE_STEREO_MP2_MPEG_PS
        sequential = {
            frame.pts_seconds: frame.data
            for frame in self._decode_sequential(video.path, device)
        }

        demuxer, decoder, converter = self._make_blocks(video.path, device)
        demuxer.seek(seconds)
        decoder.reset()
        got = list(
            self._convert(converter, self._decode(decoder, self._demux(demuxer)))
        )

        assert len(got) > 0
        for frame in got:
            assert_frames_equal(frame.data, sequential[frame.pts_seconds])

    @pytest.mark.parametrize(
        "video",
        (
            NASA_VIDEO,
            # Starts at 8.33s, so "before the start" is a positive timestamp
            # here - including 0, which for most files is the start itself.
            TEST_NON_ZERO_START,
        ),
    )
    @pytest.mark.parametrize("seconds_before_start", (0, 1e-4, 5.0, 1e9))
    @pytest.mark.parametrize("device", _block_devices())
    def test_seek_before_start(self, video, seconds_before_start, device):
        # Asking for a time before the stream starts isn't an error: there's a
        # sensible place to land, and that's the beginning.
        video_decoder = VideoDecoder(video.path, device=device)
        begin = video_decoder.metadata.begin_stream_seconds
        blocks = self._make_blocks(video.path, device)

        got = self._first_frame_after_seek(blocks, begin - seconds_before_start)

        expected = video_decoder.get_frame_at(0)
        assert got.pts_seconds == expected.pts_seconds
        assert_frames_equal(got.data, expected.data)

    @pytest.mark.parametrize("seconds", (-1e9, -1.0, 0.0, 1e-3, 1e9))
    @pytest.mark.parametrize("device", _block_devices())
    def test_seek_on_single_frame_video(self, seconds, device):
        # A file with a single frame: wherever we aim, there's only one place
        # to land.
        video = TEST_SRC_2_MPEG4_MP4
        video_decoder = VideoDecoder(video.path, device=device)
        assert video_decoder.metadata.num_frames == 1
        blocks = self._make_blocks(video.path, device)

        got = self._first_frame_after_seek(blocks, seconds)

        expected = video_decoder.get_frame_at(0)
        assert got.pts_seconds == expected.pts_seconds
        assert_frames_equal(got.data, expected.data)

    @pytest.mark.parametrize("seconds_past_end", (0, 10, 1e9))
    @pytest.mark.parametrize("device", _block_devices())
    def test_seek_past_end(self, seconds_past_end, device):
        # Same on the other side: we land on the last keyframe, because that's
        # the closest the file gets to the target. Everything decodable from
        # there is *before* the target.
        video_decoder = VideoDecoder(NASA_VIDEO.path, device=device)
        end = video_decoder.metadata.end_stream_seconds
        last_keyframe_index = video_decoder._get_key_frame_indices()[-1]
        blocks = self._make_blocks(NASA_VIDEO.path, device)

        got = self._first_frame_after_seek(blocks, end + seconds_past_end)

        expected = video_decoder.get_frame_at(last_keyframe_index)
        assert got.pts_seconds == expected.pts_seconds < end
        assert_frames_equal(got.data, expected.data)

    @needs_ffmpeg_cli
    @pytest.mark.skipif(IS_WINDOWS, reason="os.mkfifo isn't available on Windows")
    def test_seek_on_non_seekable_source_raises(self, tmp_path):
        # A named pipe fed by a live stream: there's nothing to seek in, and
        # FFmpeg reports that as a bare EPERM, which we turn into something
        # that names the format and says what's wrong.
        fifo_path = tmp_path / "live.ts"
        os.mkfifo(fifo_path)
        ffmpeg = subprocess.Popen(
            # fmt: off
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=30",
                "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                "-g", "30", "-f", "mpegts", "-y", str(fifo_path),
            ],
            # fmt: on
        )
        try:
            demuxer = Demuxer(fifo_path)
            decoder = demuxer.streams[0].make_decoder()
            num_decoded = 0
            for packet in demuxer:  # make sure the stream is really flowing
                num_decoded += len(decoder.decode(packet))
                if num_decoded > 0:
                    break

            with pytest.raises(RuntimeError, match="does not support seeking"):
                demuxer.seek(60)
        finally:
            ffmpeg.kill()
            ffmpeg.wait()

    # ----- drain() -----

    @pytest.mark.parametrize("device", _block_devices())
    def test_decode_after_drain_raises(self, device):
        # Draining ends the stream as far as the codec is concerned, and it
        # ignores anything sent afterwards. Rather than silently decoding
        # nothing, say so.
        demuxer = Demuxer(H265_VIDEO.path)
        decoder = demuxer.streams[0].make_decoder(device)
        packet = demuxer.next_packet()
        decoder.decode(packet)
        decoder.drain()

        with pytest.raises(RuntimeError, match="has been drained"):
            decoder.decode(packet)

    # ===== scanning =====

    @pytest.mark.parametrize(
        "video",
        (*_ALL_VIDEOS, TEST_SRC_2_720P_MPEG4, TEST_SRC_2_720P_VP9),
    )
    def test_scan_matches_video_decoder_index(self, video):
        odd_dimension_vp9 = (
            TESTSRC2_ODD_WIDTH_VP9,
            TESTSRC2_ODD_HEIGHT_VP9,
            TESTSRC2_ODD_HEIGHT_AND_WIDTH_VP9,
        )
        if ffmpeg_major_version == 4 and any(video is v for v in odd_dimension_vp9):
            # The scan finds 25 frames where VideoDecoder's index has 24. Only
            # these three assets, and only on FFmpeg 4: the 720p VP9 one in the
            # same parametrization is fine.
            pytest.skip("Scan/index frame-count mismatch on FFmpeg 4.")

        # A scan is the same pass over the file that seek_mode="exact" makes at
        # construction, so the index has to agree with VideoDecoder on
        # everything the pass produces: how many frames there are, when each of
        # them is displayed and for how long, and which ones are keyframes.
        index = Demuxer(video.path).streams[0].scan()
        video_decoder = VideoDecoder(video.path, seek_mode="exact")
        frames = video_decoder.get_all_frames()

        assert len(index) == video_decoder.metadata.num_frames
        assert index.num_frames_from_content == video_decoder.metadata.num_frames
        torch.testing.assert_close(
            index.pts_seconds, frames.pts_seconds, atol=0, rtol=0
        )
        torch.testing.assert_close(
            index.duration_seconds, frames.duration_seconds, atol=0, rtol=0
        )
        torch.testing.assert_close(
            index.key_frame_indices,
            video_decoder._get_key_frame_indices(),
            atol=0,
            rtol=0,
        )
        assert (
            index.begin_stream_seconds_from_content
            == video_decoder.metadata.begin_stream_seconds
        )
        assert (
            index.end_stream_seconds_from_content
            == video_decoder.metadata.end_stream_seconds
        )
        assert index.average_fps_from_content == video_decoder.metadata.average_fps

    @pytest.mark.parametrize(
        "video",
        (
            NASA_VIDEO,
            H265_VIDEO,
            TEST_SRC_2_720P_MPEG4,
            # The only asset that leaves gaps between its frames: one tick,
            # after every third frame.
            TEST_SRC_2_720P_VP9,
        ),
    )
    def test_index_at_matches_video_decoder(self, video):
        # index_at() looks up which frame is on screen at a timestamp;
        # get_frame_played_at() decodes to find that same frame. They must
        # agree.
        index = Demuxer(video.path).streams[0].scan()
        video_decoder = VideoDecoder(video.path, seek_mode="exact")

        for i in range(len(index) - 1):  # -1: each frame needs its successor
            frame_start = float(index.pts_seconds[i])
            frame_end = frame_start + float(index.duration_seconds[i])
            next_frame_start = float(index.pts_seconds[i + 1])

            targets = {
                "inside the frame": frame_start + (frame_end - frame_start) / 4,
                "first instant": frame_start,
                # The boundary itself where the frames touch, a moment when
                # nothing is on screen where they don't.
                "just after the end": (frame_end + next_frame_start) / 2,
            }
            for description, seconds in targets.items():
                expected = video_decoder.get_frame_played_at(seconds).pts_seconds
                got = float(index.pts_seconds[index.index_at(seconds)])
                assert got == expected, f"frame {i}, {description} ({seconds}s)"

    def test_scan_skips_discarded_packets(self):
        # This file's mp4 edit list flags its first packets - including the
        # first keyframe - as AV_PKT_FLAG_DISCARD. They're demuxed but never
        # become frames, so counting packets would put the index out of step
        # with both the decoder and VideoDecoder.
        video = DISCARD_FIRST_KEYFRAME_VIDEO
        index = Demuxer(video.path).streams[0].scan()

        assert len(list(Demuxer(video.path))) == 30  # number of packets
        assert len(index) == 25  # number of frames
        assert VideoDecoder(video.path, seek_mode="exact").metadata.num_frames == 25

    @pytest.mark.parametrize("video", (NASA_VIDEO, H265_VIDEO, TEST_SRC_2_720P_MPEG4))
    def test_key_frame_seconds_for(self, video):
        # It must return the last keyframe that isn't after the target, with no
        # keyframe left in between.
        index = Demuxer(video.path).streams[0].scan()
        key_frame_seconds = index.pts_seconds[index.key_frame_indices].tolist()

        for i in range(len(index)):
            start = float(index.pts_seconds[i])
            for seconds in (start, start + float(index.duration_seconds[i]) / 2):
                got = index.key_frame_seconds_for(seconds)
                assert got <= seconds
                assert [k for k in key_frame_seconds if got < k <= seconds] == []

        for k in key_frame_seconds:
            assert index.key_frame_seconds_for(k) == k

    def test_key_frame_seconds_for_when_the_keyframe_was_trimmed(self):
        # This file's first frames decode from a keyframe that the mp4 edit
        # list flagged for discard, so it isn't in the index at all and there's
        # nothing to point at. Fall back to the start of the stream, which is
        # where FFmpeg goes looking for it anyway.
        index = Demuxer(DISCARD_FIRST_KEYFRAME_VIDEO.path).streams[0].scan()
        pts_seconds = index.pts_seconds

        assert index.key_frame_indices.tolist() == [5, 15]
        for i in range(5):
            # pts_seconds[0] isn't a key frame but it's the first frame, so it's the best we can do.
            assert index.key_frame_seconds_for(pts_seconds[i]) == pts_seconds[0]
        for i in range(5, 15):
            assert index.key_frame_seconds_for(pts_seconds[i]) == pts_seconds[5]

    @pytest.mark.parametrize("video", (NASA_VIDEO, H265_VIDEO, TEST_SRC_2_720P_MPEG4))
    @pytest.mark.parametrize("device", _block_devices())
    def test_scan_seek_matches_video_decoder_exact(self, video, device):
        # What the scan is for. Snapping the target back to the preceding
        # keyframe before seeking is the entire difference between
        # VideoDecoder's two seek modes for a timestamp lookup, so doing it
        # here makes the blocks reproduce the exact one - including on
        # H265_VIDEO, where a plain seek lands past the target
        # (test_seek_to_non_keyframe_can_land_past_target) and this must not.
        index = Demuxer(video.path).streams[0].scan()

        num_targets = 10

        for i in range(0, len(index), max(1, len(index) // num_targets)):
            # Aim at the middle of a frame, so that no target lands on a frame
            # boundary where the two sides could round differently.
            seconds = float(index.pts_seconds[i] + index.duration_seconds[i] / 2)
            target_pts = float(index.pts_seconds[index.index_at(seconds)])
            seek_to = index.key_frame_seconds_for(seconds)

            # A fresh VideoDecoder per target: it skips the seek when the
            # target is just ahead of the last frame it decoded, which would
            # hide the very behaviour we're comparing against.
            expected = VideoDecoder(
                video.path, seek_mode="exact", device=device
            ).get_frame_played_at(seconds)

            blocks = self._make_blocks(video.path, device)
            frames = self._frames_after_seek(blocks, seek_to)
            got = next(frame for frame in frames if frame.pts_seconds >= target_pts)

            assert got.pts_seconds == expected.pts_seconds
            assert_frames_equal(got.data, expected.data)

    @pytest.mark.parametrize("video", (NASA_VIDEO, H265_VIDEO, TEST_SRC_2_720P_MPEG4))
    @pytest.mark.parametrize("device", _block_devices())
    def test_scan_leaves_demuxer_at_the_start(self, video, device):
        # A scan reads the file all the way to the end and then rewinds, so a
        # pipeline built on that same demuxer still decodes the whole stream.
        # The decoder needs no reset(): the scan happened before it was fed
        # anything.
        demuxer = Demuxer(video.path)
        index = demuxer.streams[0].scan()

        decoder = demuxer.streams[0].make_decoder(device)
        converter = ColorConverter(device=device)
        frames = list(
            self._convert(converter, self._decode(decoder, self._demux(demuxer)))
        )

        assert len(frames) == len(index)
        for frame, expected_pts in zip(frames, index.pts_seconds):
            assert frame.pts_seconds == expected_pts

    def test_scan_after_demuxing_raises(self):
        # The scan rewinds the container, which would silently desynchronise
        # every decoder already being fed from it.
        demuxer = Demuxer(NASA_VIDEO.path)
        demuxer.next_packet()

        with pytest.raises(RuntimeError, match="before any packet is demuxed"):
            demuxer.streams[0].scan()

    def test_scan_is_cached(self):
        (video,) = Demuxer(NASA_VIDEO.path).streams
        assert video.scan() is video.scan()

    def test_scan_rewind_matches_a_fresh_demuxer(self):
        # scan() rewinds with a seek to 0 rather than reopening the container,
        # and on a file whose edit list discards the first packets those are not
        # obviously the same thing. What comes out after a scan has to be
        # exactly what a demuxer that never scanned gives.
        video = DISCARD_FIRST_KEYFRAME_VIDEO

        scanned = Demuxer(video.path)
        scanned.streams[0].scan()
        after_scan = [
            frame.pts_seconds
            for frame in self._decode(scanned.streams[0].make_decoder(), scanned)
        ]

        fresh = Demuxer(video.path)
        never_scanned = [
            frame.pts_seconds
            for frame in self._decode(fresh.streams[0].make_decoder(), fresh)
        ]

        assert after_scan == never_scanned
        assert len(after_scan) == 25

    def test_scanning_several_video_streams_reads_the_file_once(self):
        # Sorting and building tensors is per-stream, but the I/O - which is
        # what a scan actually costs - is shared.
        class CountingFileLike:
            def __init__(self, path):
                self._file = open(path, "rb")
                self.bytes_read = 0

            def read(self, size):
                data = self._file.read(size)
                self.bytes_read += len(data)
                return data

            def seek(self, offset, whence):
                return self._file.seek(offset, whence)

        one_stream = CountingFileLike(NASA_VIDEO.path)
        (only,) = Demuxer(one_stream, streams=0).streams
        only.scan()

        two_streams = CountingFileLike(NASA_VIDEO.path)
        left, right = Demuxer(two_streams, streams=(0, 3)).streams
        first = left.scan()
        after_first_scan = two_streams.bytes_read
        second = right.scan()

        assert two_streams.bytes_read == after_first_scan  # no further I/O
        assert two_streams.bytes_read == pytest.approx(one_stream.bytes_read, rel=0.01)
        assert len(first) > 0 and len(second) > 0

    def test_scanning_a_non_video_stream_raises(self):
        # Unreachable through the Python API, where AudioStream simply has no
        # scan() - which is the point of the stream classes. The check below it
        # is still worth keeping honest.
        demuxer = Demuxer(NASA_VIDEO.path, streams="audio")

        with pytest.raises(RuntimeError, match="Only a video stream can be scanned"):
            _blocks_demuxer_scan(demuxer._handle, demuxer.streams[0].index)

    # ===== source kinds =====

    @pytest.mark.parametrize("make_source", _BLOCKS_SOURCES)
    def test_source_kinds(self, make_source):
        # Every source kind demuxes into the very same frames as the path does.
        got = self._decode_sequential(make_source(NASA_VIDEO.path), "cpu")
        expected = VideoDecoder(NASA_VIDEO.path).get_all_frames()

        assert len(got) == len(expected)
        for got_frame, expected_pts, expected_data in zip(
            got, expected.pts_seconds, expected.data
        ):
            assert got_frame.pts_seconds == expected_pts
            assert_frames_equal(got_frame.data, expected_data)

    @pytest.mark.parametrize("make_source", _BLOCKS_SOURCES)
    def test_seek_on_every_source_kind(self, make_source):
        # Seeking anything but a path goes through the AVIO seek callback -
        # back into Python for a file-like, into our own buffer for bytes and
        # tensors - instead of through FFmpeg's own file I/O.
        video_decoder = VideoDecoder(NASA_VIDEO.path)
        seconds = video_decoder.get_frame_at(
            video_decoder._get_key_frame_indices()[1]
        ).pts_seconds

        blocks = self._make_blocks(make_source(NASA_VIDEO.path), "cpu")
        got = self._first_frame_after_seek(blocks, seconds)
        expected = video_decoder.get_frame_played_at(seconds)

        assert got.pts_seconds == expected.pts_seconds == seconds
        assert_frames_equal(got.data, expected.data)

    # ===== stream_index =====

    @pytest.mark.parametrize(
        "selector, stream_index", (("video", None), (0, 0), (3, 3))
    )
    def test_stream_index(self, selector, stream_index):
        # nasa_13013.mp4 has two video streams, 0 and 3, of different sizes,
        # and 3 is the best one, i.e. the one "video" resolves to.
        demuxer = Demuxer(NASA_VIDEO.path, streams=selector)
        decoder = demuxer.streams[0].make_decoder()
        converter = ColorConverter()
        got = [
            converter.convert(raw_frame)
            for raw_frame in itertools.islice(
                self._decode(decoder, self._demux(demuxer)), 10
            )
        ]

        expected = VideoDecoder(NASA_VIDEO.path, stream_index=stream_index)[:10]

        assert len(got) == len(expected) == 10
        for got_frame, expected_data in zip(got, expected):
            assert_frames_equal(got_frame.data, expected_data)

    def test_audio_only_file_raises(self):
        with pytest.raises(RuntimeError, match="No valid video stream found"):
            Demuxer(NASA_AUDIO_MP3.path, streams="video")

    def test_video_only_file_raises(self):
        with pytest.raises(RuntimeError, match="No valid audio stream found"):
            Demuxer(H265_VIDEO.path, streams="audio")

    # ===== audio streams =====

    @pytest.mark.parametrize(
        "asset", (NASA_AUDIO_MP3, NASA_AUDIO, SINE_MONO_S32, SINE_16_CHANNEL_S16)
    )
    def test_audio_yields_packets(self, asset):
        packets = list(Demuxer(asset.path, streams="audio"))
        assert len(packets) > 0
        assert all(isinstance(packet, Packet) for packet in packets)

    def test_audio_picks_the_audio_stream_of_a_video_file(self):
        # nasa_13013.mp4 has video streams (0, 3) and aac streams (1, 4).
        # Following one or the other gives different, non-empty packet streams.
        num_audio_packets = len(list(Demuxer(NASA_VIDEO.path, streams="audio")))
        num_video_packets = len(list(Demuxer(NASA_VIDEO.path, streams="video")))
        assert num_audio_packets > 0
        assert num_video_packets > 0
        assert num_audio_packets != num_video_packets

    @pytest.mark.parametrize("selector", ("audio", 1, 4))
    def test_audio_stream_selector(self, selector):
        assert len(list(Demuxer(NASA_VIDEO.path, streams=selector))) > 0

    @pytest.mark.parametrize("make_source", _BLOCKS_SOURCES)
    def test_audio_source_kinds(self, make_source):
        # Every source kind demuxes the very same packets as the path does.
        expected = len(list(Demuxer(NASA_AUDIO_MP3.path, streams="audio")))
        got = Demuxer(make_source(NASA_AUDIO_MP3.path), streams="audio")
        assert len(list(got)) == expected

    def test_audio_seek(self):
        # Seeking past the start leaves fewer packets to demux.
        num_packets_from_start = len(
            list(Demuxer(NASA_AUDIO_MP3.path, streams="audio"))
        )

        demuxer = Demuxer(NASA_AUDIO_MP3.path, streams="audio")
        demuxer.seek(NASA_AUDIO_MP3.duration_seconds / 2)
        num_packets_after_seek = len(list(demuxer))

        assert 0 < num_packets_after_seek < num_packets_from_start

    # ===== Audio decoding: RawAudioSamples =====

    @staticmethod
    def _decode_audio(asset, stream_index=None, seek_seconds=None):
        demuxer = Demuxer(
            asset.path, streams="audio" if stream_index is None else stream_index
        )
        decoder = demuxer.streams[0].make_decoder()
        if seek_seconds is not None:
            demuxer.seek(seek_seconds)
            decoder.reset()
        chunks = []
        for packet in demuxer:
            chunks += decoder.decode(packet)
        chunks += decoder.drain()
        return chunks

    @pytest.mark.parametrize(
        "asset, sample_format, dtype",
        (
            (SINE_MONO_U8, "u8", torch.uint8),
            (SINE_MONO_S16, "s16", torch.int16),
            (SINE_MONO_S32, "s32", torch.int32),
            # FFmpeg has no 24-bit sample format, so a 24-bit source is s32.
            (SINE_MONO_S24, "s32", torch.int32),
            (SINE_MONO_F32, "flt", torch.float32),
            (SINE_MONO_F64, "dbl", torch.float64),
            (SINE_STEREO_MP2_MPEG_PS, "s16p", torch.int16),
            (SINE_16_CHANNEL_S16, "s16", torch.int16),
            (NASA_AUDIO_MP3, "fltp", torch.float32),
        ),
    )
    def test_audio_raw_samples_dtype_and_shape(self, asset, sample_format, dtype):
        # The decoder hands out the codec's own sample type, always as
        # [num_channels, num_samples]. The packed formats above (no trailing
        # 'p') are the ones exercising the de-interleaving path.
        chunks = self._decode_audio(asset)
        assert len(chunks) > 0

        for chunk in chunks:
            assert isinstance(chunk, RawAudioSamples)
            assert chunk.sample_format == sample_format
            assert chunk.data.dtype == dtype
            assert chunk.data.ndim == 2
            assert chunk.data.is_contiguous()
            assert chunk.num_channels == asset.num_channels
            assert chunk.sample_rate == asset.sample_rate
            assert chunk.duration_seconds >= 0

    @pytest.mark.parametrize(
        "asset",
        (
            SINE_MONO_U8,
            SINE_MONO_S16,
            SINE_MONO_S32,
            SINE_MONO_F32,
            SINE_MONO_F64,
            SINE_STEREO_MP2_MPEG_PS,
            SINE_16_CHANNEL_S16,
            NASA_AUDIO_MP3,
            NASA_AUDIO,
        ),
    )
    @pytest.mark.parametrize("seek_fraction", (None, 1 / 3, 2 / 3))
    def test_audio_raw_samples_match_audio_decoder(self, asset, seek_fraction):
        # We hand out the true source samples: normalizing them the way FFmpeg
        # does reproduces AudioDecoder's output bit for bit. This is also what
        # pins the de-interleaving, most visibly on the 16-channel asset.
        #
        # It holds after a seek too, but only once the caller has done the
        # pre-roll these blocks don't do: a lossy codec decodes its first frames
        # after a seek from a flushed state, so they come out subtly wrong -
        # plausible, but not what whole-file decoding gives - until it
        # re-primes. Dropping those frames is exactly what pre-rolling means,
        # and everything from there on is bit exact again. Without the drop,
        # mp3 and aac diverge over their first ~1000-1600 samples and match
        # perfectly after that.
        if seek_fraction is not None and asset is SINE_STEREO_MP2_MPEG_PS:
            pytest.skip(
                "MPEG-PS resync after a seek is unreliable for both the blocks "
                "and AudioDecoder (which raises seeking this file to 2.8s), so "
                "it can't tell us anything here. See "
                "test_audio_decoder_mpeg_ps_resync_after_seek."
            )

        seek_seconds = (
            None if seek_fraction is None else asset.duration_seconds * seek_fraction
        )
        chunks = self._decode_audio(asset, seek_seconds=seek_seconds)
        if seek_seconds is not None:
            # Same number of frames SingleStreamDecoder pre-rolls by, see
            # Note [Audio pre-roll and post-roll].
            chunks = chunks[4:]
        assert len(chunks) > 0

        raw = torch.cat([chunk.data for chunk in chunks], dim=1)

        if raw.dtype == torch.uint8:
            got = (raw.to(torch.float32) - 128) / 128
        elif raw.dtype in (torch.int16, torch.int32):
            got = raw.to(torch.float32) / -float(torch.iinfo(raw.dtype).min)
        else:
            got = raw.to(torch.float32)

        decoder = AudioDecoder(asset.path)
        if seek_seconds is None:
            expected = decoder.get_all_samples().data
        else:
            # Re-anchor on the first frame we kept, so both sides start on the
            # same sample.
            expected = decoder.get_samples_played_in_range(
                start_seconds=chunks[0].pts_seconds
            ).data
        torch.testing.assert_close(got, expected, atol=0, rtol=0)

    def test_audio_raw_samples_pts(self):
        chunks = self._decode_audio(SINE_MONO_S16)
        pts = [chunk.pts_seconds for chunk in chunks]
        assert pts == sorted(pts)
        assert pts[0] == pytest.approx(0, abs=1e-6)

    def test_decoder_output_type_follows_the_stream(self):
        # The two decoders are one class in C++; the split is a Python-level
        # one, so that each has an exact output type and its own arguments.
        for selector, decoder_class, expected_type in (
            ("audio", AudioPacketDecoder, RawAudioSamples),
            ("video", VideoPacketDecoder, RawFrame),
        ):
            demuxer = Demuxer(NASA_VIDEO.path, streams=selector)
            decoder = demuxer.streams[0].make_decoder()
            assert isinstance(decoder, decoder_class)
            # A codec needs more than one packet before it outputs anything.
            decoded = []
            while not decoded:
                decoded = decoder.decode(demuxer.next_packet())
            assert isinstance(decoded[0], expected_type)

    def test_audio_decoder_takes_no_device(self):
        (audio,) = Demuxer(NASA_AUDIO_MP3.path, streams="audio").streams
        with pytest.raises(TypeError, match="device"):
            audio.make_decoder(device="cuda")

    def test_audio_decoder_mpeg_ps_resync_after_seek(self):
        # Seeking an MPEG program stream lands on a container-level byte
        # offset, so the parser resumes mid-frame and the packets it rebuilds
        # are AVERROR_INVALIDDATA until it resyncs. That's a property of the
        # container, not of the codec: it applies to this file's audio stream
        # exactly as it does to a video one. Without the resync handling this
        # raises "Failed to send packet to decoder" on the very first packet.
        asset = SINE_STEREO_MP2_MPEG_PS
        demuxer = Demuxer(asset.path, streams="audio")
        decoder = demuxer.streams[0].make_decoder()
        demuxer.seek(asset.duration_seconds / 2)
        decoder.reset()

        chunks = []
        for packet in demuxer:
            chunks += decoder.decode(packet)
        chunks += decoder.drain()

        assert len(chunks) > 0
        num_samples = sum(chunk.num_samples for chunk in chunks)
        assert 0 < num_samples < asset.duration_seconds * asset.sample_rate

    # ===== AudioConverter =====

    @staticmethod
    def _convert_audio(asset, drain=True, seek_seconds=None, **converter_kwargs):
        demuxer = Demuxer(asset.path, streams="audio")
        decoder = demuxer.streams[0].make_decoder()
        converter = AudioConverter(**converter_kwargs)

        raw_chunks = []
        if seek_seconds is not None:
            demuxer.seek(seek_seconds)
            decoder.reset()
            converter.reset()
        for packet in demuxer:
            raw_chunks += decoder.decode(packet)
        raw_chunks += decoder.drain()
        if seek_seconds is not None:
            # The caller's pre-roll, see
            # test_audio_raw_samples_match_audio_decoder.
            raw_chunks = raw_chunks[4:]

        chunks = [converter.convert(raw) for raw in raw_chunks]
        if drain:
            chunks.append(converter.drain())
        return chunks

    @staticmethod
    def _cat(chunks):
        return torch.cat([chunk.data for chunk in chunks], dim=1)

    def _converted_and_expected(self, asset, seek_fraction, **converter_kwargs):
        # Runs the blocks pipeline, seeking first when asked, and returns it
        # alongside the AudioDecoder output covering the same region.
        seek_seconds = (
            None if seek_fraction is None else asset.duration_seconds * seek_fraction
        )
        chunks = self._convert_audio(
            asset, seek_seconds=seek_seconds, **converter_kwargs
        )
        decoder = AudioDecoder(asset.path, **converter_kwargs)
        if seek_seconds is None:
            expected = decoder.get_all_samples().data
        else:
            # Re-anchor on the first chunk we produced, so both sides start on
            # the same sample.
            expected = decoder.get_samples_played_in_range(
                start_seconds=chunks[0].pts_seconds
            ).data
        return self._cat(chunks), expected

    @pytest.mark.parametrize(
        "asset",
        (
            SINE_MONO_U8,
            SINE_MONO_S16,
            SINE_MONO_S32,
            SINE_MONO_F32,
            SINE_MONO_F64,
            SINE_STEREO_MP2_MPEG_PS,
            SINE_16_CHANNEL_S16,
            NASA_AUDIO_MP3,
            NASA_AUDIO,
        ),
    )
    @pytest.mark.parametrize("seek_fraction", (None, 1 / 3, 2 / 3))
    def test_audio_converter_matches_audio_decoder(self, asset, seek_fraction):
        # No resampling, no remix: the converter only normalizes the sample
        # type, which is frame-local, so this is bit exact - after a seek too,
        # once the caller has pre-rolled the codec.
        if seek_fraction is not None and asset is SINE_STEREO_MP2_MPEG_PS:
            pytest.skip("MPEG-PS seek resync, see the raw-samples test.")
        got, expected = self._converted_and_expected(asset, seek_fraction)
        torch.testing.assert_close(got, expected, atol=0, rtol=0)

    @pytest.mark.parametrize(
        "asset, num_channels",
        (
            (NASA_AUDIO_MP3, 1),
            (NASA_AUDIO_MP3, 2),
            (SINE_STEREO_MP2_MPEG_PS, 1),
            (SINE_MONO_S16, 2),
            (SINE_16_CHANNEL_S16, 2),
        ),
    )
    @pytest.mark.parametrize("seek_fraction", (None, 1 / 3, 2 / 3))
    def test_audio_converter_num_channels(self, asset, num_channels, seek_fraction):
        # Remixing is a matrix over one frame's samples, so it is bit exact
        # too, seek or no seek.
        if seek_fraction is not None and asset is SINE_STEREO_MP2_MPEG_PS:
            pytest.skip("MPEG-PS seek resync, see the raw-samples test.")
        got, expected = self._converted_and_expected(
            asset, seek_fraction, num_channels=num_channels
        )
        assert got.shape[0] == num_channels
        torch.testing.assert_close(got, expected, atol=0, rtol=0)

    @pytest.mark.parametrize(
        "asset, sample_rate",
        (
            (NASA_AUDIO_MP3, 16_000),
            (NASA_AUDIO_MP3, 8_000),
            (NASA_AUDIO_MP3_44100, 16_000),
            (SINE_MONO_S32, 8_000),
            (SINE_MONO_S32, 44_100),
            (SINE_STEREO_MP2_MPEG_PS, 16_000),
        ),
    )
    @pytest.mark.parametrize("seek_fraction", (None, 1 / 3, 2 / 3))
    def test_audio_converter_sample_rate(self, asset, sample_rate, seek_fraction):
        # Resampling the whole stream from its start is bit exact against
        # AudioDecoder. After a seek it is only *close*, and that is by design:
        # resampling is the one conversion that isn't frame-local, and we don't
        # implement SingleStreamDecoder's alignment grid, so where the seek
        # lands decides whether our output grid coincides with AudioDecoder's.
        #
        # Sweeping the offset on sine_mono_s32.wav at 16000 -> 8000 shows both
        # regimes: most offsets differ only over the resampler's first ~16
        # output samples and are bit exact after that, while duration/3 and
        # duration*2/3 land off-grid and every sample is shifted by a fraction
        # of a sample. Measured across these cases at the two offsets below,
        # that costs at most 6.7e-2 elementwise and 3.3e-2 on average, on
        # samples in [-1, 1]; the bounds are ~3x and ~2x that.
        if seek_fraction is not None and asset is SINE_STEREO_MP2_MPEG_PS:
            pytest.skip("MPEG-PS seek resync, see the raw-samples test.")

        got, expected = self._converted_and_expected(
            asset, seek_fraction, sample_rate=sample_rate
        )
        if seek_fraction is None:
            torch.testing.assert_close(got, expected, atol=0, rtol=0)
            return

        # The resampler can end up one output sample ahead of AudioDecoder's
        # trimming (16000 -> 44100 does, at both offsets).
        assert abs(got.shape[1] - expected.shape[1]) <= 1
        num_samples = min(got.shape[1], expected.shape[1])
        difference = (got[:, :num_samples] - expected[:, :num_samples]).abs()
        assert difference.max() < 0.2
        assert difference.mean() < 0.07

    def test_audio_converter_without_drain_loses_samples_when_resampling(self):
        with_drain = self._cat(self._convert_audio(NASA_AUDIO_MP3, sample_rate=16_000))
        without = self._cat(
            self._convert_audio(NASA_AUDIO_MP3, drain=False, sample_rate=16_000)
        )
        assert without.shape[1] < with_drain.shape[1]
        torch.testing.assert_close(
            with_drain[:, : without.shape[1]], without, atol=0, rtol=0
        )

    def test_audio_converter_drain_is_empty_without_resampling(self):
        chunks = self._convert_audio(NASA_AUDIO_MP3)
        assert chunks[-1].data.shape[1] == 0

    @pytest.mark.parametrize("sample_rate", (None, 16_000))
    def test_audio_converter_pts_is_contiguous(self, sample_rate):
        chunks = self._convert_audio(NASA_AUDIO_MP3, sample_rate=sample_rate)
        for current, following in zip(chunks, chunks[1:]):
            assert current.pts_seconds + current.duration_seconds == pytest.approx(
                following.pts_seconds, abs=1e-9
            )

    def test_audio_converter_reset_allows_another_stream(self):
        converter = AudioConverter(sample_rate=16_000)

        def convert_all(asset):
            demuxer = Demuxer(asset.path, streams="audio")
            decoder = demuxer.streams[0].make_decoder()
            chunks = []
            for packet in demuxer:
                chunks += [converter.convert(raw) for raw in decoder.decode(packet)]
            chunks += [converter.convert(raw) for raw in decoder.drain()]
            chunks.append(converter.drain())
            return self._cat(chunks)

        first = convert_all(NASA_AUDIO_MP3)
        converter.reset()
        second = convert_all(SINE_MONO_S32)

        for got, asset in ((first, NASA_AUDIO_MP3), (second, SINE_MONO_S32)):
            expected = (
                AudioDecoder(asset.path, sample_rate=16_000).get_all_samples().data
            )
            torch.testing.assert_close(got, expected, atol=0, rtol=0)

    def test_audio_converter_changing_stream_without_reset_raises(self):
        # swresample is configured from the first samples it sees and its
        # buffered state belongs to that configuration, so we make the caller
        # reset() rather than silently reconfiguring.
        converter = AudioConverter()
        converter.convert(self._decode_audio(NASA_AUDIO_MP3)[0])
        with pytest.raises(RuntimeError, match="Call reset\\(\\) to convert"):
            converter.convert(self._decode_audio(SINE_MONO_S32)[0])

    def test_audio_converter_convert_after_drain_raises(self):
        converter = AudioConverter()
        converter.convert(self._decode_audio(NASA_AUDIO_MP3)[0])
        converter.drain()
        with pytest.raises(RuntimeError, match="has been drained"):
            converter.convert(self._decode_audio(NASA_AUDIO_MP3)[0])

        converter.reset()
        converter.convert(self._decode_audio(NASA_AUDIO_MP3)[0])  # no raise

    def test_audio_converter_drain_before_convert_raises(self):
        with pytest.raises(RuntimeError, match="hasn't converted any samples"):
            AudioConverter().drain()

    @pytest.mark.parametrize("kwargs", ({"sample_rate": 0}, {"num_channels": -1}))
    def test_audio_converter_invalid_args_raise(self, kwargs):
        with pytest.raises(RuntimeError, match="must be > 0"):
            AudioConverter(**kwargs)

    def test_audio_pipeline_after_seek(self):
        # Seeking gives you the tail of the stream, and the three blocks each
        # need to be told about it. We deliberately do NOT assert that these
        # samples match the corresponding slice of a whole-file decode: with no
        # pre-roll and no alignment grid, they don't, and that's documented.
        seek_seconds = NASA_AUDIO_MP3.duration_seconds / 2
        num_samples_from_start = self._cat(
            self._convert_audio(NASA_AUDIO_MP3, sample_rate=16_000)
        ).shape[1]

        demuxer = Demuxer(NASA_AUDIO_MP3.path, streams="audio")
        decoder = demuxer.streams[0].make_decoder()
        converter = AudioConverter(sample_rate=16_000)
        demuxer.seek(seek_seconds)
        decoder.reset()
        converter.reset()

        chunks = []
        for packet in demuxer:
            chunks += [converter.convert(raw) for raw in decoder.decode(packet)]
        chunks += [converter.convert(raw) for raw in decoder.drain()]
        chunks.append(converter.drain())

        samples = self._cat(chunks)
        assert samples.shape[0] == 2
        assert 0 < samples.shape[1] < num_samples_from_start
        assert chunks[0].pts_seconds == pytest.approx(seek_seconds, abs=0.2)

    @pytest.mark.parametrize("stream_index", (-1, 6, 1000))
    def test_invalid_stream_index_raises(self, stream_index):
        with pytest.raises(RuntimeError, match="is not a valid stream"):
            Demuxer(NASA_VIDEO.path, streams=stream_index)

    def test_bad_source_type_raises(self):
        with pytest.raises(TypeError, match="Unknown source type"):
            Demuxer(123)

        # user mistakenly forgets to specify binary reading when creating a
        # file-like object from open()
        with pytest.raises(TypeError, match="binary reading?"):
            Demuxer(open(NASA_VIDEO.path))


# Small helpers to avoid having to always specify the same skip marks and decode_fn
def _jpeg_param(*values):
    return pytest.param(decode_jpeg, *values, marks=pytest.mark.needs_jpeg, id="jpeg")


def _jpeg_cuda_param(*values):
    return pytest.param(
        partial(decode_jpeg, device="cuda"),
        *values,
        marks=(pytest.mark.needs_jpeg, pytest.mark.needs_cuda),
        id="jpeg_cuda",
    )


def _png_param(*values):
    return pytest.param(decode_png, *values, marks=pytest.mark.needs_png, id="png")


def _webp_param(*values):
    return pytest.param(decode_webp, *values, marks=pytest.mark.needs_webp, id="webp")


def _gif_param(*values):
    return pytest.param(decode_gif, *values, id="gif")


def _avif_param(*values):
    return pytest.param(decode_avif, *values, marks=pytest.mark.needs_avif, id="avif")


def _heic_param(*values):
    return pytest.param(decode_heic, *values, marks=pytest.mark.needs_heic, id="heic")


class TestImageDecoder:

    # ===== shared helpers =====

    def _save_debug(self, decoded, reference, path="debug.png"):
        # Debugging helper: dump decoded and reference frames side-by-side.
        from torchvision.io import write_png
        from torchvision.utils import make_grid

        grid = make_grid([decoded, reference], padding=10)
        write_png(grid, str(path))

    @staticmethod
    def _pil_to_tensor(img):
        t = torch.from_numpy(numpy.array(img))
        return t.permute(2, 0, 1) if t.ndim == 3 else t.unsqueeze(0)

    @staticmethod
    def _scriptable_decode(kind: str, data: torch.Tensor, mode: int) -> torch.Tensor:
        if kind == "jpeg":
            return torch.ops.torchcodec_ns.decode_jpeg(data, mode)
        elif kind == "png":
            return torch.ops.torchcodec_ns.decode_png(data, mode)
        elif kind == "webp":
            return torch.ops.torchcodec_ns.decode_webp(data, mode)
        elif kind == "gif":
            return torch.ops.torchcodec_ns.decode_gif(data, mode)
        else:
            assert kind == "avif"
            return torch.ops.torchcodec_ns.decode_avif(data, mode)

    @staticmethod
    def _make_transparent_png(path, kind):
        # A PNG can encode transparency via a tRNS chunk instead of a full alpha
        # channel: a transparent colorkey for gray/RGB images, or per-palette-
        # entry alpha for palette images. The left half is transparent.
        h, w = 16, 20
        if kind == "rgb":
            arr = numpy.empty((h, w, 3), numpy.uint8)
            arr[:, : w // 2] = (10, 20, 30)
            arr[:, w // 2 :] = (200, 100, 50)
            Image.fromarray(arr, "RGB").save(path, transparency=(10, 20, 30))
        elif kind == "gray":
            arr = numpy.empty((h, w), numpy.uint8)
            arr[:, : w // 2] = 42
            arr[:, w // 2 :] = 200
            Image.fromarray(arr, "L").save(path, transparency=42)
        else:
            assert kind == "palette"
            px = numpy.zeros((h, w), numpy.uint8)
            px[:, w // 2 :] = 1
            im = Image.fromarray(px, "P")
            im.putpalette([10, 20, 30, 200, 100, 50])
            im.info["transparency"] = bytes([0, 255])  # per-index alpha
            im.save(path)

    # ===== cross-codec tests: basics & API =====

    @pytest.mark.filterwarnings(
        "ignore:`torch.jit.script` is deprecated:DeprecationWarning"
    )
    @pytest.mark.parametrize(
        "kind, asset",
        (
            pytest.param(
                "jpeg", GRADIENT_JPEG, marks=pytest.mark.needs_jpeg, id="jpeg"
            ),
            pytest.param("png", GRADIENT_PNG, marks=pytest.mark.needs_png, id="png"),
            pytest.param(
                "webp", GRADIENT_WEBP, marks=pytest.mark.needs_webp, id="webp"
            ),
            pytest.param("gif", GRADIENT_GIF, id="gif"),
            pytest.param(
                "avif", GRADIENT_AVIF, marks=pytest.mark.needs_avif, id="avif"
            ),
        ),
    )
    def test_torchscript(self, kind, asset):
        # This is just to ensure some sort of BC from torchvision. Zero
        # guarantee we'll keep supporting torchscript.
        data = _source_to_tensor(asset.path)
        scripted = torch.jit.script(self._scriptable_decode)
        eager = getattr(torch.ops.torchcodec_ns, f"decode_{kind}")
        rgb = 3  # the raw ops take an int mode; 3 is RGB
        torch.testing.assert_close(
            scripted(kind, data, rgb),
            eager(data, rgb),
            atol=0,
            rtol=0,
        )

    @pytest.mark.parametrize(
        "decode_fn, asset",
        (
            _jpeg_param(GRAYSCALE_JPEG),
            _jpeg_cuda_param(GRAYSCALE_JPEG),
            _png_param(GRAYSCALE_PNG),
            _webp_param(RGBA_WEBP),
            _gif_param(GRADIENT_GIF),
            _avif_param(RGBA_AVIF),
            _heic_param(RGBA_HEIC),
        ),
    )
    def test_default_mode_is_rgb(self, decode_fn, asset):
        # The default output mode is RGB, so the decoded output always has 3
        # channels regardless of the source: a grayscale source is expanded from
        # 1 channel, an RGBA source has its alpha stripped.
        decoded = decode_fn(asset.path)
        assert decoded.shape[0] == 3

    @pytest.mark.parametrize(
        "make_source",
        (
            pytest.param(lambda a: str(a.path), id="str"),
            pytest.param(lambda a: a.path.read_bytes(), id="bytes"),
            pytest.param(
                lambda a: torch.frombuffer(
                    bytearray(a.path.read_bytes()), dtype=torch.uint8
                ),
                id="tensor",
            ),
        ),
    )
    @pytest.mark.parametrize(
        "decode_fn, asset",
        (
            _jpeg_param(GRADIENT_JPEG),
            _jpeg_cuda_param(GRADIENT_JPEG),
            _png_param(GRADIENT_PNG),
            _webp_param(GRADIENT_WEBP),
            _gif_param(GRADIENT_GIF),
            _avif_param(GRADIENT_AVIF),
            _heic_param(GRADIENT_HEIC),
        ),
    )
    def test_source_kinds(self, decode_fn, asset, make_source):
        # A str path, bytes, and a uint8 tensor of the encoded data must all
        # decode to the same result as a pathlib.Path.
        assert_frames_equal(decode_fn(make_source(asset)), decode_fn(asset.path))

    @pytest.mark.parametrize(
        "decode_fn",
        (
            _jpeg_param(),
            _png_param(),
            _webp_param(),
            _gif_param(),
            _avif_param(),
            _heic_param(),
        ),
    )
    def test_bad_source_type_raises(self, decode_fn):
        with pytest.raises(TypeError, match="Unknown source type"):
            decode_fn(123)

    # ===== cross-codec tests: decode_image (format autodetection) =====

    @pytest.mark.parametrize(
        "make_source",
        (
            pytest.param(lambda a: a.path, id="path"),
            pytest.param(lambda a: str(a.path), id="str"),
            pytest.param(lambda a: a.path.read_bytes(), id="bytes"),
            pytest.param(
                lambda a: torch.frombuffer(
                    bytearray(a.path.read_bytes()), dtype=torch.uint8
                ),
                id="tensor",
            ),
        ),
    )
    @pytest.mark.parametrize(
        "mode",
        ("UNCHANGED", "RGB", "GRAY_ALPHA"),
    )
    @pytest.mark.parametrize(
        "decode_fn, asset",
        (
            _jpeg_param(GRADIENT_JPEG),
            _png_param(RGBA_PNG),
            _webp_param(RGBA_WEBP),
            _gif_param(GRADIENT_GIF),
            _avif_param(RGBA_AVIF),
            _heic_param(RGBA_HEIC),
        ),
    )
    def test_decode_image(self, decode_fn, asset, mode, make_source):
        # decode_image detects the format and must produce exactly what the
        # format-specific decoder produces, for every mode and source kind.
        assert_frames_equal(
            decode_image(make_source(asset), mode=mode),
            decode_fn(asset.path, mode=mode),
        )

    @pytest.mark.parametrize("output_dtype", (torch.uint8, torch.uint16, "auto"))
    @pytest.mark.parametrize(
        "decode_fn, asset",
        (
            _jpeg_param(GRADIENT_JPEG),
            _png_param(GRADIENT_16BIT_PNG),
            _webp_param(GRADIENT_WEBP),
            _gif_param(GRADIENT_GIF),
            _avif_param(GRADIENT_10BIT_AVIF),
            _heic_param(GRADIENT_10BIT_HEIC),
        ),
    )
    def test_decode_image_output_dtype(self, decode_fn, asset, output_dtype):
        # decode_image must expose output_dtype and forward it, producing exactly
        # what the format-specific decoder produces. The PNG/AVIF assets are
        # >8-bit so the uint16/"auto" paths are meaningfully exercised.
        from_image = decode_image(asset.path, output_dtype=output_dtype)
        from_decoder = decode_fn(asset.path, output_dtype=output_dtype)
        assert from_image.dtype == from_decoder.dtype
        torch.testing.assert_close(
            from_image.to(torch.int64), from_decoder.to(torch.int64), atol=0, rtol=0
        )

    def test_decode_image_unrecognized_format_raises(self):
        garbage = torch.arange(64, dtype=torch.uint8)
        with pytest.raises(ValueError, match="Unsupported or unrecognized"):
            decode_image(garbage)

    @pytest.mark.parametrize("mode", ("RGBA", ImageReadMode.RGBA))
    def test_decode_image_rgba_alias(self, mode):
        # "RGBA" (string or enum) is an undocumented alias for "RGB_ALPHA".
        reference = decode_image(RGBA_PNG.path, mode="RGB_ALPHA")
        assert_frames_equal(decode_image(RGBA_PNG.path, mode=mode), reference)

    # ===== cross-codec tests: output modes =====

    @pytest.mark.parametrize(
        "decode_fn, fmt, ext, save_kwargs, source_mode",
        (
            _jpeg_param("JPEG", "jpg", {"quality": 95}, "L"),
            _jpeg_param("JPEG", "jpg", {"quality": 95}, "RGB"),
            _jpeg_param("JPEG", "jpg", {"quality": 95}, "CMYK"),
            _png_param("PNG", "png", {}, "L"),
            _png_param("PNG", "png", {}, "LA"),
            _png_param("PNG", "png", {}, "RGB"),
            _png_param("PNG", "png", {}, "RGBA"),
            _png_param("PNG", "png", {}, "P"),
            _webp_param("WEBP", "webp", {"lossless": True}, "RGB"),
            _webp_param("WEBP", "webp", {"lossless": True}, "RGBA"),
            _gif_param("GIF", "gif", {}, "L"),
            _gif_param("GIF", "gif", {}, "RGB"),
            _gif_param("GIF", "gif", {}, "P"),
            # Only an RGB source for AVIF: it's lossy and, unlike webp, has no
            # lossless mode here. An RGBA source would additionally hit the
            # alpha-drop divergence from PIL (see test_avif_against_pil).
            _avif_param("AVIF", "avif", {}, "RGB"),
        ),
    )
    @pytest.mark.parametrize(
        "output_mode, pil_mode, num_expected_channels",
        (
            ("UNCHANGED", None, None),
            ("GRAY", "L", 1),
            ("GRAY_ALPHA", "LA", 2),
            ("RGB", "RGB", 3),
            ("RGB_ALPHA", "RGBA", 4),
        ),
    )
    def test_all_source_to_all_output_modes(
        self,
        tmp_path,
        decode_fn,
        fmt,
        ext,
        save_kwargs,
        source_mode,
        output_mode,
        pil_mode,
        num_expected_channels,
    ):
        # Test that every input color mode is decodable to every output mode.

        h, w = 40, 60
        xs = numpy.linspace(0, 255, w)
        ys = numpy.linspace(0, 255, h)
        r = numpy.broadcast_to(xs, (h, w))
        g = numpy.broadcast_to(ys[:, None], (h, w))
        base = numpy.stack([r, g, (r + g) / 2], axis=-1).astype(numpy.uint8)

        path = tmp_path / f"{source_mode}.{ext}"
        Image.fromarray(base, mode="RGB").convert(source_mode).save(
            path, fmt, **save_kwargs
        )

        decoded = decode_fn(path, mode=output_mode)
        assert decoded.dtype == torch.uint8

        reference = self._pil_to_tensor(Image.open(path).convert(pil_mode))

        if output_mode == "UNCHANGED":
            num_expected_channels = reference.shape[0]
        assert decoded.shape[0] == num_expected_channels
        assert decoded.shape == reference.shape
        assert_tensor_close_on_at_least(decoded, reference, percentage=99, atol=2)

        source_has_alpha = source_mode in ("LA", "RGBA")
        if output_mode in ("GRAY_ALPHA", "RGB_ALPHA") and not source_has_alpha:
            assert (decoded[-1] == 255).all()

    # ===== cross-codec tests: output_dtype =====

    @pytest.mark.parametrize(
        "decode_fn, asset",
        (
            _jpeg_param(GRADIENT_JPEG),
            _jpeg_cuda_param(GRADIENT_JPEG),
            _png_param(GRADIENT_PNG),
            _webp_param(GRADIENT_WEBP),
            _gif_param(GRADIENT_GIF),
            _avif_param(GRADIENT_AVIF),
            _heic_param(GRADIENT_HEIC),
        ),
    )
    @pytest.mark.parametrize(
        "mode",
        (
            "UNCHANGED",
            "GRAY",
            "GRAY_ALPHA",
            "RGB",
            "RGB_ALPHA",
        ),
    )
    def test_output_dtype_8bit_source(self, decode_fn, asset, mode):
        # For an 8-bit source, uint8 (the default) and "auto" both yield uint8,
        # while uint16 widens to the full 16-bit range. This holds for every
        # output mode.
        default = decode_fn(asset.path, mode=mode)
        uint8 = decode_fn(asset.path, mode=mode, output_dtype=torch.uint8)
        auto = decode_fn(asset.path, mode=mode, output_dtype="auto")
        uint16 = decode_fn(asset.path, mode=mode, output_dtype=torch.uint16)

        assert uint8.dtype == torch.uint8
        torch.testing.assert_close(default, uint8, atol=0, rtol=0)  # uint8 default
        assert auto.dtype == torch.uint8  # 8-bit source
        torch.testing.assert_close(auto, uint8, atol=0, rtol=0)
        assert uint16.dtype == torch.uint16
        assert uint16.shape == uint8.shape

        # The widened output is the uint8 output scaled to the full 16-bit range
        # (a factor of 257 = 65535 / 255): exact for the codecs that widen by
        # byte replication, within rounding for AVIF (which converts at 16-bit
        # precision).
        if decode_fn is decode_avif:
            downscaled = (uint16.to(torch.float32) / 257).round()
            assert_tensor_close_on_at_least(
                downscaled, uint8.to(torch.float32), percentage=99, atol=1
            )
        else:
            torch.testing.assert_close(
                uint16.to(torch.int64), uint8.to(torch.int64) * 257, atol=0, rtol=0
            )

    @pytest.mark.parametrize(
        "decode_fn, asset",
        (
            _jpeg_param(GRADIENT_JPEG),
            _png_param(GRADIENT_PNG),
            _webp_param(GRADIENT_WEBP),
            _gif_param(GRADIENT_GIF),
            _avif_param(GRADIENT_AVIF),
            _heic_param(GRADIENT_HEIC),
        ),
    )
    @pytest.mark.parametrize("bad_dtype", (torch.float32, torch.int32, "uint8"))
    def test_output_dtype_invalid(self, decode_fn, asset, bad_dtype):
        # Only torch.uint8, torch.uint16 and the string "auto" are accepted.
        with pytest.raises(ValueError, match="Invalid output_dtype"):
            decode_fn(asset.path, output_dtype=bad_dtype)

    # ===== cross-codec tests: EXIF / orientation =====

    @pytest.mark.parametrize(
        "decode_fn, fmt, ext, save_kwargs",
        (
            _jpeg_param("JPEG", "jpg", {"quality": 95}),
            _png_param("PNG", "png", {}),
            _webp_param("WEBP", "webp", {"lossless": True}),
            # Note that avif doesn't encode exif data, it has its own metadata
            # for it, but it seems that PIL can still encode this fine.
            _avif_param("AVIF", "avif", {"quality": 100}),
        ),
    )
    @pytest.mark.parametrize("orientation", (0, 1, 2, 3, 4, 5, 6, 7, 8))
    def test_exif_orientation(
        self, tmp_path, orientation, decode_fn, fmt, ext, save_kwargs
    ):
        arr = torch.randint(0, 256, (100, 101, 3), dtype=torch.uint8).numpy()
        img = Image.fromarray(arr)
        exif = img.getexif()
        exif[0x0112] = orientation  # 0x0112 is the EXIF orientation tag
        path = tmp_path / f"exif_{orientation}.{ext}"
        img.save(path, fmt, exif=exif.tobytes(), **save_kwargs)

        decoded = decode_fn(path, mode="RGB")
        reference = self._pil_to_tensor(ImageOps.exif_transpose(Image.open(path)))

        assert decoded.shape == reference.shape
        assert_tensor_close_on_at_least(decoded, reference, percentage=99, atol=2)

    @pytest.mark.parametrize(
        "decode_fn, fmt, ext, save_kwargs",
        (
            _jpeg_param("JPEG", "jpg", {"quality": 95}),
            _png_param("PNG", "png", {}),
            _webp_param("WEBP", "webp", {"lossless": True}),
        ),
    )
    @pytest.mark.parametrize("size", (65533, 1, 7, 10, 23, 33))
    def test_invalid_exif(self, tmp_path, size, decode_fn, fmt, ext, save_kwargs):
        # Malformed EXIF must not crash. Inspired by a Pillow test.
        arr = torch.randint(0, 256, (100, 101, 3), dtype=torch.uint8).numpy()
        img = Image.fromarray(arr)
        path = tmp_path / f"invalid_exif_{size}.{ext}"
        img.save(path, fmt, exif=b"1" * size, **save_kwargs)

        decoded = decode_fn(path, mode="RGB")
        assert decoded.shape == (3, 100, 101)

        # For JPEG the output should also match PIL, which ignores the bad EXIF.
        # We can't check this for PNG: PIL's exif_transpose raises on a malformed
        # eXIf chunk instead of ignoring it, so there's no clean reference.
        if decode_fn is decode_jpeg:
            reference = self._pil_to_tensor(ImageOps.exif_transpose(Image.open(path)))
            assert decoded.shape == reference.shape
            assert_tensor_close_on_at_least(decoded, reference, percentage=99, atol=2)

    @needs_heic
    @needs_png
    @pytest.mark.parametrize(
        "asset, orientation",
        (
            (GRADIENT_ROTATED_HEIC, 6),  # 90-degree rotation (HEIF irot)
            (GRADIENT_MIRRORED_HEIC, 2),  # horizontal mirror (HEIF imir)
        ),
    )
    def test_heic_orientation(self, asset, orientation):
        # Grouped with the EXIF orientation test above: HEIC carries orientation
        # through its own HEIF transforms (irot/imir) rather than EXIF, so it
        # needs a separate test but exercises the same orientation behavior.
        #
        # We check against a png reference decoded by PIL because encoding an
        # HEIC would require pillow_heic which we don't want to install on the
        # CI.
        decoded = decode_heic(asset.path, mode="RGB")
        ref_img = Image.open(GRADIENT_PNG.path).convert("RGB")
        ref_img.getexif()[0x0112] = orientation  # 0x0112 is the EXIF orientation tag
        reference = self._pil_to_tensor(ImageOps.exif_transpose(ref_img))
        assert decoded.shape == reference.shape
        assert_tensor_close_on_at_least(decoded, reference, percentage=99, atol=2)

    # ===== cross-codec tests: malformed / corrupt input =====

    @pytest.mark.parametrize(
        "decode_fn, ext, match",
        (
            _jpeg_param("jpg", "Not a JPEG"),
            _png_param("png", "Not a PNG file"),
            _webp_param("webp", "WebPGetFeatures failed"),
            _gif_param("gif", "DGifOpen"),
            _avif_param("avif", "avifDecoderParse failed"),
        ),
    )
    def test_not_an_image_raises(self, tmp_path, decode_fn, ext, match):
        path = tmp_path / f"garbage.{ext}"
        path.write_bytes(b"\x00" * 100)
        with pytest.raises(RuntimeError, match=match):
            decode_fn(path)

    @pytest.mark.parametrize(
        "decode_fn, asset, ext, match",
        (
            _jpeg_param(GRADIENT_JPEG, "jpg", "Image is incomplete or truncated"),
            _png_param(GRADIENT_PNG, "png", "Out of bound read"),
            _webp_param(GRADIENT_WEBP, "webp", "Failed to decode the WebP bitstream"),
            _gif_param(GRADIENT_GIF, "gif", "DGifSlurp"),
            _avif_param(GRADIENT_AVIF, "avif", "avifDecoderParse failed"),
        ),
    )
    @pytest.mark.parametrize("div", (2, 3, 4))
    def test_truncated_raises(self, tmp_path, div, decode_fn, asset, ext, match):
        # A file truncated mid-stream must raise, not crash.
        data = asset.path.read_bytes()
        path = tmp_path / f"truncated.{ext}"
        path.write_bytes(data[: len(data) // div])
        with pytest.raises(RuntimeError, match=match):
            decode_fn(path)

    @pytest.mark.parametrize(
        "decode_fn",
        (
            _jpeg_param(),
            _png_param(),
            _webp_param(),
            _gif_param(),
            _avif_param(),
        ),
    )
    @pytest.mark.parametrize(
        "make_bad, match",
        (
            (lambda t: t[None], "1-dimensional"),
            (lambda t: t.to(torch.float32), "uint8"),
            (lambda t: t[:0], "non-empty"),
        ),
    )
    def test_bad_encoded_data_raises(self, decode_fn, make_bad, match):
        data = torch.randint(0, 256, (100,), dtype=torch.uint8)
        with pytest.raises(RuntimeError, match=match):
            decode_fn(make_bad(data))

    @pytest.mark.parametrize(
        "decode_fn, asset",
        (
            _jpeg_param(GRADIENT_JPEG),
            _jpeg_cuda_param(GRADIENT_JPEG),
            _png_param(GRADIENT_PNG),
            _webp_param(GRADIENT_WEBP),
            _gif_param(GRADIENT_GIF),
            _avif_param(GRADIENT_AVIF),
            _heic_param(GRADIENT_HEIC),
        ),
    )
    def test_non_contiguous_encoded_data(self, decode_fn, asset):
        # A non-contiguous tensor of encoded bytes must decode to the same result
        # as the contiguous data: the decoders make it contiguous internally
        # rather than erroring.
        contiguous = torch.frombuffer(
            bytearray(asset.path.read_bytes()), dtype=torch.uint8
        )
        padded = torch.empty(contiguous.numel() * 2, dtype=torch.uint8)
        padded[::2] = contiguous
        non_contiguous = padded[::2]
        assert not non_contiguous.is_contiguous()
        torch.testing.assert_close(non_contiguous, contiguous, atol=0, rtol=0)
        assert_frames_equal(decode_fn(non_contiguous), decode_fn(contiguous))

    # ===== JPEG =====

    @needs_jpeg
    def test_mode_str_and_enum(self):
        # The canonical mode form is an uppercase string (used everywhere else in
        # the suite), but the argument is case-insensitive, and the ImageReadMode
        # enum is still accepted for backward compatibility with torchvision. All
        # these spellings must produce the same result.
        path = GRADIENT_JPEG.path
        reference = decode_jpeg(path, mode="GRAY_ALPHA")
        for mode in (
            "gray_alpha",
            "Gray_Alpha",
            "GRAY_ALPHA",
            ImageReadMode.GRAY_ALPHA,
        ):
            assert_frames_equal(decode_jpeg(path, mode=mode), reference)

        with pytest.raises(ValueError, match="Invalid mode"):
            decode_jpeg(path, mode="not_a_mode")

    @needs_jpeg
    @pytest.mark.parametrize("asset", (GRADIENT_JPEG, GRAYSCALE_JPEG, CMYK_JPEG))
    @pytest.mark.parametrize(
        "mode, pil_mode",
        (
            ("UNCHANGED", None),
            ("GRAY", "L"),
            ("GRAY_ALPHA", "LA"),
            ("RGB", "RGB"),
            ("RGB_ALPHA", "RGBA"),
        ),
    )
    def test_jpeg_against_pil(self, asset, mode, pil_mode):
        decoded = decode_jpeg(asset.path, mode=mode)

        reference = self._pil_to_tensor(Image.open(asset.path).convert(pil_mode))

        assert decoded.shape == reference.shape
        assert_tensor_close_on_at_least(decoded, reference, percentage=99, atol=2)

        if mode in ("GRAY_ALPHA", "RGB_ALPHA"):
            # The synthesized alpha channel must be fully opaque.
            assert (decoded[-1] == 255).all()

    @needs_jpeg
    def test_jpeg_batch_cpu(self):
        # A list of sources decodes each independently and returns a list (one
        # tensor per source), matching the CUDA batch API. a single source still
        # returns a bare tensor.
        sources = [GRADIENT_JPEG.path, GRAYSCALE_JPEG.path, GRADIENT_JPEG.path]
        batch = decode_jpeg(sources, mode="RGB")
        assert isinstance(batch, list)
        assert len(batch) == len(sources)
        for decoded, src in zip(batch, sources):
            single = decode_jpeg(src, mode="RGB")
            assert isinstance(single, torch.Tensor)
            assert_frames_equal(decoded, single)

        # A one-element list returns a one-element list, not a bare tensor.
        as_list = decode_jpeg([GRADIENT_JPEG.path], mode="RGB")
        assert isinstance(as_list, list) and len(as_list) == 1
        assert_frames_equal(as_list[0], decode_jpeg(GRADIENT_JPEG.path, mode="RGB"))

    @needs_jpeg
    def test_bad_huffman_decodes(self):
        # A JPEG with a bad Huffman table is still decodable; just make sure it
        # doesn't raise.
        decode_jpeg(BAD_HUFFMAN_JPEG.path)

    @needs_jpeg
    def test_corrupt_jpeg_raises(self):
        # Non-regression test ported from torchvision.
        with pytest.raises(RuntimeError, match="Unsupported marker type"):
            decode_jpeg(CORRUPT_JPEG.path)

    # ===== JPEG on CUDA (nvJPEG) =====

    @needs_cuda
    @needs_jpeg
    @pytest.mark.parametrize("orientation", (0, 1, 2, 3, 4, 5, 6, 7, 8))
    def test_cuda_jpeg_exif_orientation(self, orientation):
        base = Image.open(GRADIENT_JPEG.path).convert("RGB")
        exif = base.getexif()
        exif[0x0112] = orientation  # 0x0112 == EXIF Orientation tag
        buf = io.BytesIO()
        base.save(buf, format="JPEG", exif=exif, quality=95)
        data = torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8)

        cpu = decode_jpeg(data, mode="RGB")
        gpu = decode_jpeg(data, mode="RGB", device="cuda")
        assert gpu.shape == cpu.shape
        assert_tensor_close_on_at_least(gpu.cpu(), cpu, percentage=99, atol=3)

    @needs_cuda
    @needs_jpeg
    @pytest.mark.parametrize("asset", (GRADIENT_JPEG, GRAYSCALE_JPEG))
    @pytest.mark.parametrize(
        "mode", ("UNCHANGED", "RGB", "GRAY", "RGB_ALPHA", "GRAY_ALPHA")
    )
    def test_cuda_jpeg_matches_cpu(self, asset, mode):
        # Note that this will only exercise either the HW path or SW path
        # depending on the GPU.
        cpu = decode_jpeg(asset.path, mode=mode)
        gpu = decode_jpeg(asset.path, mode=mode, device="cuda")
        assert gpu.device.type == "cuda"
        assert gpu.dtype == torch.uint8
        assert gpu.shape == cpu.shape
        assert_tensor_close_on_at_least(gpu.cpu(), cpu, percentage=99, atol=3)

    @needs_cuda
    @needs_jpeg
    @pytest.mark.parametrize("mode", ("UNCHANGED", "GRAY", "RGB"))
    def test_cuda_jpeg_batch(self, mode):
        assets = [GRADIENT_JPEG, GRAYSCALE_JPEG, GRADIENT_JPEG]
        sources = [a.path for a in assets]
        batch = decode_jpeg(sources, mode=mode, device="cuda")
        assert isinstance(batch, list)
        assert len(batch) == len(sources)
        for decoded, asset in zip(batch, assets):
            assert decoded.device.type == "cuda"
            single = decode_jpeg(asset.path, mode=mode, device="cuda")
            assert decoded.shape == single.shape
            torch.testing.assert_close(decoded, single, atol=0, rtol=0)

        # UNCHANGED must preserve each source's native channel count.
        if mode == "UNCHANGED":
            assert batch[0].shape[0] == 3
            assert batch[1].shape[0] == 1
            assert batch[2].shape[0] == 3

    @needs_cuda
    @needs_jpeg
    def test_cuda_jpeg_single_vs_list_return_type(self):
        single = decode_jpeg(GRADIENT_JPEG.path, mode="RGB", device="cuda")
        assert isinstance(single, torch.Tensor)
        as_list = decode_jpeg([GRADIENT_JPEG.path], mode="RGB", device="cuda")
        assert isinstance(as_list, list) and len(as_list) == 1
        torch.testing.assert_close(as_list[0], single, atol=0, rtol=0)

    @needs_cuda
    @needs_jpeg
    def test_cuda_jpeg_errors(self):
        # Corrupt input raises.
        with pytest.raises(RuntimeError, match="nvjpegDecode failed:"):
            decode_jpeg(CORRUPT_JPEG.path, device="cuda")

        cuda_data = torch.frombuffer(
            bytearray(GRADIENT_JPEG.path.read_bytes()), dtype=torch.uint8
        ).cuda()
        with pytest.raises(RuntimeError, match="must be on the CPU"):
            decode_jpeg(cuda_data, device="cuda")

    @needs_cuda
    @needs_jpeg
    def test_cuda_jpeg_multithreaded(self):
        # Many threads decoding concurrently on CUDA.
        sources = [GRADIENT_JPEG.path, GRAYSCALE_JPEG.path, GRADIENT_JPEG.path]
        reference = [decode_jpeg(s, mode="RGB") for s in sources]

        num_workers = 10
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = [
                ex.submit(decode_jpeg, sources, mode="RGB", device="cuda")
                for _ in range(num_workers)
            ]
            results = [f.result() for f in futures]

        assert len(results) == num_workers
        for decoded in results:
            assert len(decoded) == len(sources)
            for got, ref in zip(decoded, reference):
                assert got.device.type == "cuda"
                assert got.shape == ref.shape
                assert_tensor_close_on_at_least(got.cpu(), ref, percentage=99, atol=3)

    # ===== PNG =====

    @needs_png
    @pytest.mark.parametrize(
        "asset", (GRADIENT_PNG, GRAYSCALE_PNG, GRAYSCALE_ALPHA_PNG, RGBA_PNG)
    )
    @pytest.mark.parametrize(
        "mode, pil_mode",
        (
            ("UNCHANGED", None),
            ("GRAY", "L"),
            ("GRAY_ALPHA", "LA"),
            ("RGB", "RGB"),
            ("RGB_ALPHA", "RGBA"),
        ),
    )
    def test_png_against_pil(self, asset, mode, pil_mode):
        decoded = decode_png(asset.path, mode=mode)

        reference = self._pil_to_tensor(Image.open(asset.path).convert(pil_mode))

        assert decoded.shape == reference.shape
        assert_tensor_close_on_at_least(decoded, reference, percentage=99, atol=2)

    @needs_png
    @pytest.mark.parametrize("asset", (GRAYSCALE_16BIT_PNG, GRADIENT_16BIT_PNG))
    @pytest.mark.parametrize(
        "mode",
        (
            "UNCHANGED",
            "GRAY",
            "GRAY_ALPHA",
            "RGB",
            "RGB_ALPHA",
        ),
    )
    def test_output_dtype_16bit_png(self, asset, mode):
        # 16-bit PNGs (grayscale and RGB) are genuine >8-bit sources, exercising
        # the decoder's 16-bit path for every output mode.
        default = decode_png(asset.path, mode=mode)
        uint8 = decode_png(asset.path, mode=mode, output_dtype=torch.uint8)
        uint16 = decode_png(asset.path, mode=mode, output_dtype=torch.uint16)
        auto = decode_png(asset.path, mode=mode, output_dtype="auto")

        # >8-bit source: "auto" preserves 16 bits, but the default is still uint8.
        assert uint16.dtype == torch.uint16
        assert auto.dtype == torch.uint16
        torch.testing.assert_close(
            auto.to(torch.int64), uint16.to(torch.int64), atol=0, rtol=0
        )
        assert uint8.dtype == torch.uint8
        torch.testing.assert_close(default, uint8, atol=0, rtol=0)
        assert uint16.shape == uint8.shape

        # Genuine 16-bit content: not merely 8-bit values scaled by 257 (which
        # would make every sample divisible by 257). A full-range color channel
        # also reaches the top of the 16-bit range; GRAY luma of a gradient never
        # does, so we only check that for the color-carrying modes.
        assert (uint16.to(torch.int64) % 257 != 0).any()
        if mode not in ("GRAY", "GRAY_ALPHA"):
            assert uint16.to(torch.int64).max() > 60000

        # uint8 output is the 16-bit output scaled down by 257 (full-range).
        expected8 = (uint16.to(torch.float32) / 257).round()
        assert_tensor_close_on_at_least(
            uint8.to(torch.float32), expected8, percentage=99, atol=1
        )

        # Synthesized alpha stays fully opaque at each dtype's max.
        if mode in ("GRAY_ALPHA", "RGB_ALPHA"):
            assert (uint16[-1].to(torch.int64) == 65535).all()
            assert (uint8[-1].to(torch.int64) == 255).all()

        # PIL can read a 16-bit *grayscale* PNG back exactly (it can't for RGB),
        # so for that source we assert the 16-bit values are reproduced exactly.
        # The grayscale source maps to every output color channel (it's
        # replicated across RGB), so we compare each color channel to it.
        if asset is GRAYSCALE_16BIT_PNG:
            src = torch.from_numpy(
                numpy.array(Image.open(asset.path)).astype(numpy.int64)
            )
            has_alpha = mode in ("GRAY_ALPHA", "RGB_ALPHA")
            num_color = uint16.shape[0] - (1 if has_alpha else 0)
            for c in range(num_color):
                torch.testing.assert_close(
                    uint16[c].to(torch.int64), src, atol=0, rtol=0
                )

    @needs_png
    @pytest.mark.parametrize("kind", ("rgb", "gray", "palette"))
    @pytest.mark.parametrize(
        "output_mode, pil_mode",
        (
            ("GRAY_ALPHA", "LA"),
            ("RGB_ALPHA", "RGBA"),
        ),
    )
    def test_png_trns_transparency(self, tmp_path, kind, output_mode, pil_mode):
        # tRNS transparency must be honored (not decoded as fully opaque) when
        # decoding to an alpha mode.
        path = tmp_path / f"{kind}.png"
        self._make_transparent_png(path, kind)

        decoded = decode_png(path, mode=output_mode)
        reference = self._pil_to_tensor(Image.open(path).convert(pil_mode))
        assert decoded.shape == reference.shape

        # The alpha channel (the transparency itself) must match exactly: the
        # left half is transparent (0), the right half opaque (255).
        alpha, ref_alpha = decoded[-1], reference[-1]
        assert_tensor_close_on_at_least(alpha, ref_alpha, percentage=99, atol=2)
        assert (alpha == 0).any()
        assert (alpha == 255).any()

        # Color must match where visible. The color under fully-transparent
        # pixels is irrelevant and PIL fills it differently than a straight
        # gray/RGB conversion, so we don't compare it.
        visible = (alpha > 0).unsqueeze(0).expand(decoded.shape[0] - 1, -1, -1)
        assert_tensor_close_on_at_least(
            decoded[:-1][visible],
            reference[:-1][visible],
            percentage=99,
            atol=2,
        )

    @needs_png
    @pytest.mark.parametrize("shape", ((27, 27), (60, 60), (105, 105)))
    def test_1bit_png(self, tmp_path, shape):
        # 1-bit (black & white) PNGs are an edge case for the bit-depth handling:
        # libpng packs 8 pixels per byte and we must expand them to full uint8.
        # Ported from torchvision.
        gen = torch.Generator().manual_seed(0)
        pixels = (torch.rand(shape, generator=gen) > 0.5).numpy()
        path = tmp_path / "1bit.png"
        Image.fromarray(pixels).save(path)

        decoded = decode_png(path, mode="GRAY")
        reference = self._pil_to_tensor(Image.open(path).convert("L"))
        assert decoded.shape == reference.shape
        assert_frames_equal(decoded, reference)

    @needs_png
    @pytest.mark.parametrize("mode, pil_mode", (("UNCHANGED", None), ("RGB", "RGB")))
    def test_interlaced_png(self, mode, pil_mode):
        # Ported from torchvision
        asset = GRADIENT_INTERLACED_PNG
        assert Image.open(asset.path).info.get("interlace") == 1

        decoded = decode_png(asset.path, mode=mode)
        reference = self._pil_to_tensor(Image.open(asset.path).convert(pil_mode))
        assert decoded.shape == reference.shape
        assert_frames_equal(decoded, reference)

    @needs_png
    def test_corrupt_png_raises(self, tmp_path):
        # Corrupting the IHDR chunk type makes libpng raise an error (its stored
        # CRC no longer matches). This exercizes the error callback and the
        # setjmp/longjmp handling.
        data = bytearray(GRADIENT_PNG.path.read_bytes())
        data[12:16] = b"XXXX"  # the "IHDR" chunk type, at a fixed offset
        path = tmp_path / "corrupt.png"
        path.write_bytes(bytes(data))
        with pytest.raises(RuntimeError, match="CRC error"):
            decode_png(path)

    @needs_png
    @pytest.mark.parametrize("asset", (SIGSEGV_PNG, HEAPBOF_PNG))
    def test_corrupt_png_out_of_bound_read_raises(self, asset):
        # Non-regression test ported from torchvision.
        with pytest.raises(RuntimeError, match="Out of bound read"):
            decode_png(asset.path)

    # ===== WEBP =====

    @needs_webp
    @pytest.mark.parametrize("asset", (GRADIENT_WEBP, RGBA_WEBP))
    @pytest.mark.parametrize(
        "mode, pil_mode",
        (
            ("UNCHANGED", None),
            ("GRAY", "L"),
            ("GRAY_ALPHA", "LA"),
            ("RGB", "RGB"),
            ("RGB_ALPHA", "RGBA"),
        ),
    )
    def test_webp_against_pil(self, asset, mode, pil_mode):
        decoded = decode_webp(asset.path, mode=mode)

        reference = self._pil_to_tensor(Image.open(asset.path).convert(pil_mode))

        assert decoded.shape == reference.shape
        assert_tensor_close_on_at_least(decoded, reference, percentage=99, atol=2)

    @needs_webp
    @pytest.mark.parametrize(
        "mode, pil_mode",
        (
            ("UNCHANGED", None),
            ("GRAY", "L"),
            ("GRAY_ALPHA", "LA"),
            ("RGB", "RGB"),
            ("RGB_ALPHA", "RGBA"),
        ),
    )
    def test_animated_webp(self, tmp_path, mode, pil_mode):
        from PIL import ImageSequence

        path = tmp_path / "animated.webp"
        frames = [
            Image.fromarray(
                torch.randint(0, 256, (16, 16, 3), dtype=torch.uint8).numpy()
            )
            for _ in range(3)
        ]
        frames[0].save(
            path,
            "WEBP",
            save_all=True,
            append_images=frames[1:],
            duration=100,
            lossless=True,
        )

        decoded = decode_webp(path, mode=mode)
        pil = Image.open(path)

        assert decoded.ndim == 4
        assert decoded.shape[0] == pil.n_frames

        for i, frame in enumerate(ImageSequence.Iterator(pil)):
            reference = self._pil_to_tensor(frame.convert(pil_mode))
            assert decoded[i].shape == reference.shape
            assert_tensor_close_on_at_least(
                decoded[i], reference, percentage=99, atol=2
            )

        if mode in ("GRAY_ALPHA", "RGB_ALPHA"):
            # The source is opaque, so the alpha channel must be fully opaque.
            assert (decoded[:, -1] == 255).all()

    @needs_webp
    def test_animated_webp_transparency(self, tmp_path):
        # An animated WebP with real transparency: an opaque red square that
        # moves over a transparent background, one frame per position. We assert
        # against directly-constructed expectations rather than PIL: for these
        # small transparent WebPs, PIL's animation reader flattens the
        # background to opaque, whereas our libwebpdemux-based decode faithfully
        # preserves the transparent background. The frames are saved lossless,
        # so the opaque pixels round-trip exactly.
        path = tmp_path / "animated_transparent.webp"
        frames = []
        for i in range(3):
            arr = numpy.zeros((16, 24, 4), dtype=numpy.uint8)
            arr[4:12, i * 6 : i * 6 + 6] = (255, 0, 0, 255)
            frames.append(Image.fromarray(arr, "RGBA"))
        frames[0].save(
            path,
            "WEBP",
            save_all=True,
            append_images=frames[1:],
            duration=100,
            lossless=True,
        )

        decoded = decode_webp(path, mode="RGB_ALPHA")
        assert decoded.shape == (3, 4, 16, 24)

        # channels-last (C, H, W) -> (H, W, C) per frame for easy indexing.
        frames_hwc = decoded.permute(0, 2, 3, 1)
        for i in range(3):
            frame = frames_hwc[i]
            square = frame[4:12, i * 6 : i * 6 + 6]
            assert (square == torch.tensor([255, 0, 0, 255], dtype=torch.uint8)).all()

            # Everything outside the square is fully transparent.
            bg_mask = torch.ones((16, 24), dtype=torch.bool)
            bg_mask[4:12, i * 6 : i * 6 + 6] = False
            assert (frame[bg_mask] == 0).all()

    # ===== GIF =====

    @pytest.mark.parametrize(
        "mode, pil_mode",
        (
            ("UNCHANGED", None),
            ("GRAY", "L"),
            ("GRAY_ALPHA", "LA"),
            ("RGB", "RGB"),
            ("RGB_ALPHA", "RGBA"),
        ),
    )
    def test_gif_against_pil(self, mode, pil_mode):
        decoded = decode_gif(GRADIENT_GIF.path, mode=mode)

        reference = self._pil_to_tensor(Image.open(GRADIENT_GIF.path).convert(pil_mode))

        assert decoded.shape == reference.shape
        assert_tensor_close_on_at_least(decoded, reference, percentage=99, atol=2)

        if mode in ("GRAY_ALPHA", "RGB_ALPHA"):
            # GIF carries no real alpha, so the synthesized channel is opaque.
            assert (decoded[-1] == 255).all()

    def test_animated_gif(self):
        # An animated GIF decodes to a batched (N, C, H, W) tensor, one frame per
        # image, matching PIL's per-frame RGB decode.
        from PIL import ImageSequence

        decoded = decode_gif(ANIMATED_GIF.path)
        pil = Image.open(ANIMATED_GIF.path)

        assert decoded.ndim == 4
        assert decoded.shape[0] == pil.n_frames

        for i, frame in enumerate(ImageSequence.Iterator(pil)):
            reference = self._pil_to_tensor(frame.convert("RGB"))
            assert decoded[i].shape == reference.shape
            assert_tensor_close_on_at_least(
                decoded[i], reference, percentage=99, atol=2
            )

    @pytest.mark.parametrize("disposal", (1, 2, 3))
    def test_gif_disposal_methods(self, tmp_path, disposal):
        # Each frame paints a colored square in a different quadrant over a common
        # white base; PIL writes them as partial frames with the given disposal
        # method. We check our per-frame compositing matches PIL's, which
        # exercises: keying the base canvas off the *previous* frame's disposal,
        # restoring only that frame's rectangle to background (method 2), and
        # restoring the prior canvas (method 3). See DecodeGif.cpp.
        from PIL import ImageSequence

        # Palette: 0=green, 1=red, 2=blue, 3=white.
        palette = [0, 255, 0, 255, 0, 0, 0, 0, 255, 255, 255, 255]

        def make(square_index, y, x):
            arr = numpy.full((8, 8), 3, dtype=numpy.uint8)  # white base
            arr[y : y + 3, x : x + 3] = square_index
            img = Image.fromarray(arr, "P")
            img.putpalette(palette)
            return img

        frames = [make(0, 0, 0), make(1, 0, 5), make(2, 5, 0), make(2, 5, 5)]
        path = tmp_path / "disposal.gif"
        # The first frame is always "leave in place" so the tested disposal method
        # applies to frames that have a well-defined prior canvas ("restore to
        # previous" for the very first frame is ill-defined and decoders differ).
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            disposal=[1, disposal, disposal, 0],
            loop=0,
        )

        decoded = decode_gif(path)
        pil = Image.open(path)
        assert decoded.shape[0] == pil.n_frames
        for i, frame in enumerate(ImageSequence.Iterator(pil)):
            reference = self._pil_to_tensor(frame.convert("RGB"))
            assert_tensor_close_on_at_least(
                decoded[i], reference, percentage=99, atol=2
            )

    def test_gif_transparency(self):
        # A GIF with a transparent index over a non-zero background color (the
        # "welcome2" case). The alpha-preserving modes return a real alpha
        # channel matching Pillow; RGB composites the transparency over the
        # background color instead.
        asset = TRANSPARENT_GIF

        # UNCHANGED on a transparent GIF yields RGBA (like PNG's UNCHANGED, which
        # keeps the source's native channels); RGB_ALPHA and GRAY_ALPHA likewise
        # carry a real alpha channel.
        cases = (
            ("UNCHANGED", "RGBA"),
            ("RGB_ALPHA", "RGBA"),
            ("GRAY_ALPHA", "LA"),
        )
        for mode, pil_mode in cases:
            decoded = decode_gif(asset.path, mode=mode)
            reference = self._pil_to_tensor(Image.open(asset.path).convert(pil_mode))
            assert decoded.shape == reference.shape
            # The alpha channel must match Pillow exactly. The color channels are
            # only meaningful where opaque: the value under a fully-transparent
            # pixel is unspecified and differs between decoders.
            alpha = reference[-1]
            torch.testing.assert_close(decoded[-1], alpha, atol=0, rtol=0)
            opaque = alpha > 0
            assert_tensor_close_on_at_least(
                decoded[:-1, opaque], reference[:-1, opaque], percentage=99, atol=2
            )

        # RGB has no alpha: transparency is composited over the GIF background
        # color. Opaque pixels match Pillow; transparent ones intentionally
        # differ (we show the background color, Pillow shows the transparent
        # index's own color), so we only compare where opaque.
        rgb = decode_gif(asset.path, mode="RGB")
        assert rgb.shape[0] == 3
        pil_rgb = self._pil_to_tensor(Image.open(asset.path).convert("RGB"))
        opaque = decode_gif(asset.path, mode="UNCHANGED")[-1] > 0
        assert_tensor_close_on_at_least(
            rgb[:, opaque], pil_rgb[:, opaque], percentage=99, atol=2
        )

    def test_gif_first_frame_larger_than_canvas(self):
        # Non-regression test: when the first frame is larger than the logical
        # screen, the output is sized to the frame and the out-of-screen border
        # must be initialized (transparent here) rather than left as
        # uninitialized memory. The border is transparent, so its alpha must
        # match Pillow. A regression would leave it as garbage.
        asset = FRAME_EXCEEDS_SCREEN_GIF
        decoded = decode_gif(asset.path, mode="RGB_ALPHA")
        reference = self._pil_to_tensor(Image.open(asset.path).convert("RGBA"))
        assert decoded.shape == reference.shape == (4, asset.height, asset.width)
        torch.testing.assert_close(decoded[-1], reference[-1], atol=0, rtol=0)

    # ===== AVIF =====

    @needs_avif
    @pytest.mark.parametrize("asset", (GRADIENT_AVIF, RGBA_AVIF))
    @pytest.mark.parametrize(
        "mode, pil_mode",
        (
            ("UNCHANGED", None),
            ("GRAY", "L"),
            ("GRAY_ALPHA", "LA"),
            ("RGB", "RGB"),
            ("RGB_ALPHA", "RGBA"),
        ),
    )
    def test_avif_against_pil(self, asset, mode, pil_mode):
        if asset.num_channels == 4 and mode in (
            "RGB",
            "GRAY",
        ):
            # For an AVIF that carries a real alpha channel, decoding to a mode
            # that drops the alpha (RGB, and GRAY which is derived from RGB)
            # diverges from PIL: libavif plainly ignores the alpha channel, so
            # transparent pixels keep their raw (often dark) color, while PIL
            # blends. Both are defensible, so we don't compare in that case.
            pytest.skip("AVIF RGB/GRAY on an alpha image diverges from PIL by design")

        decoded = decode_avif(asset.path, mode=mode)

        reference = self._pil_to_tensor(Image.open(asset.path).convert(pil_mode))

        assert decoded.shape == reference.shape
        assert_tensor_close_on_at_least(decoded, reference, percentage=99, atol=2)

    @needs_avif
    @pytest.mark.parametrize("asset", (GRADIENT_10BIT_AVIF, GRADIENT_12BIT_AVIF))
    @pytest.mark.parametrize(
        "mode",
        (
            "UNCHANGED",
            "GRAY",
            "GRAY_ALPHA",
            "RGB",
            "RGB_ALPHA",
        ),
    )
    def test_output_dtype_high_bit_depth_avif(self, asset, mode):
        # A 10/12-bit AVIF is a genuine >8-bit source.
        default = decode_avif(asset.path, mode=mode)
        uint8 = decode_avif(asset.path, mode=mode, output_dtype=torch.uint8)
        uint16 = decode_avif(asset.path, mode=mode, output_dtype=torch.uint16)
        auto = decode_avif(asset.path, mode=mode, output_dtype="auto")

        # >8-bit source: "auto" preserves the precision, but the default is uint8.
        assert uint8.dtype == torch.uint8
        torch.testing.assert_close(default, uint8, atol=0, rtol=0)
        assert uint16.dtype == torch.uint16
        assert auto.dtype == torch.uint16
        torch.testing.assert_close(
            auto.to(torch.int64), uint16.to(torch.int64), atol=0, rtol=0
        )
        assert uint16.shape == uint8.shape

        # Genuine >8-bit precision: the 16-bit samples are not merely 8-bit
        # values scaled by 257 (which would make every sample divisible by 257).
        # A full-range color channel also reaches the top of the 16-bit range;
        # GRAY luma of a gradient never does, so we skip that check for gray.
        assert (uint16.to(torch.int64) % 257 != 0).any()
        if mode not in ("GRAY", "GRAY_ALPHA"):
            assert uint16.to(torch.int64).max() > 60000

        # The uint8 output matches PIL's (8-bit) AVIF decode. Note libavif
        # decodes to 8- and 16-bit independently (they aren't related by a clean
        # 257x factor), so we validate the 8-bit path against PIL rather than
        # against the 16-bit output. GRAY/GRAY_ALPHA additionally exercise the
        # Python uint16 gray-conversion helpers (the source has no alpha, so
        # there's no alpha-drop divergence from PIL).
        pil_mode = {
            "UNCHANGED": "RGB",  # source has no alpha
            "GRAY": "L",
            "GRAY_ALPHA": "LA",
            "RGB": "RGB",
            "RGB_ALPHA": "RGBA",
        }[mode]
        reference = self._pil_to_tensor(Image.open(asset.path).convert(pil_mode))
        assert uint8.shape == reference.shape
        assert_tensor_close_on_at_least(uint8, reference, percentage=99, atol=2)

    @needs_avif
    def test_avif_num_threads(self):
        reference = decode_avif(GRADIENT_AVIF.path)
        for num_threads in (1, 2, 4):
            decoded = decode_avif(GRADIENT_AVIF.path, num_threads=num_threads)
            torch.testing.assert_close(decoded, reference, atol=0, rtol=0)

        for bad in (0, -1):
            with pytest.raises(RuntimeError, match="num_threads must be >= 1"):
                decode_avif(GRADIENT_AVIF.path, num_threads=bad)

    @needs_avif
    @pytest.mark.parametrize(
        "mode, pil_mode",
        (
            ("UNCHANGED", None),
            ("GRAY", "L"),
            ("GRAY_ALPHA", "LA"),
            ("RGB", "RGB"),
            ("RGB_ALPHA", "RGBA"),
        ),
    )
    def test_animated_avif(self, tmp_path, mode, pil_mode):
        # An animated AVIF decodes to a batched (N, C, H, W) tensor, one frame
        # per image. We use distinct solid-color frames: AVIF is lossy, but
        # solid colors survive the YUV round-trip cleanly, so the frames match
        # PIL's per-frame decode closely and the frame ordering is verifiable.
        from PIL import ImageSequence

        path = tmp_path / "animated.avif"
        colors = [(200, 30, 30), (30, 200, 30), (30, 30, 200)]
        frames = [
            Image.fromarray(numpy.full((16, 16, 3), c, dtype=numpy.uint8))
            for c in colors
        ]
        frames[0].save(
            path,
            "AVIF",
            save_all=True,
            append_images=frames[1:],
            duration=100,
            quality=100,
        )

        decoded = decode_avif(path, mode=mode)
        pil = Image.open(path)

        assert decoded.ndim == 4
        assert decoded.shape[0] == pil.n_frames == len(colors)

        for i, frame in enumerate(ImageSequence.Iterator(pil)):
            reference = self._pil_to_tensor(frame.convert(pil_mode))
            assert decoded[i].shape == reference.shape
            assert_tensor_close_on_at_least(
                decoded[i], reference, percentage=99, atol=3
            )

        if mode in ("GRAY_ALPHA", "RGB_ALPHA"):
            # The source is opaque, so the alpha channel must be fully opaque.
            assert (decoded[:, -1] == 255).all()

    # ===== HEIC =====

    @needs_heic
    @needs_png
    @pytest.mark.parametrize(
        "heic_asset, png_asset",
        (
            (GRADIENT_HEIC, GRADIENT_PNG),
            (RGBA_HEIC, RGBA_PNG),
        ),
    )
    @pytest.mark.parametrize(
        "mode", ("UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA")
    )
    def test_heic_against_png(self, heic_asset, png_asset, mode):
        # The HEIC assets are the SAME gradients as the corresponding PNGs, saved
        # losslessly (4:4:4), so decode_heic must match decode_png up to the tiny
        # rounding of the RGB<->YUV round-trip. We compare against PNG rather than
        # PIL because Pillow needs the extra pillow-heif plugin to read HEIC.
        decoded = decode_heic(heic_asset.path, mode=mode)
        reference = decode_png(png_asset.path, mode=mode)
        assert decoded.shape == reference.shape
        assert decoded.dtype == reference.dtype == torch.uint8
        assert_tensor_close_on_at_least(decoded, reference, percentage=99, atol=2)

    @needs_heic
    @pytest.mark.parametrize(
        "mode", ("UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA")
    )
    def test_output_dtype_high_bit_depth_heic(self, mode):
        # A 10-bit HEIC is a genuine >8-bit source, so "auto" preserves 16 bits
        # while the default stays uint8.
        asset = GRADIENT_10BIT_HEIC
        default = decode_heic(asset.path, mode=mode)
        uint8 = decode_heic(asset.path, mode=mode, output_dtype=torch.uint8)
        uint16 = decode_heic(asset.path, mode=mode, output_dtype=torch.uint16)
        auto = decode_heic(asset.path, mode=mode, output_dtype="auto")

        assert uint8.dtype == torch.uint8
        torch.testing.assert_close(default, uint8, atol=0, rtol=0)
        assert uint16.dtype == torch.uint16
        assert auto.dtype == torch.uint16  # >8-bit source
        torch.testing.assert_close(
            auto.to(torch.int64), uint16.to(torch.int64), atol=0, rtol=0
        )
        assert uint16.shape == uint8.shape

        # Genuine >8-bit precision: not merely 8-bit values scaled by 257. A
        # full-range color channel also reaches the top of the 16-bit range;
        # GRAY luma of a gradient never does, so we skip that check for gray.
        assert (uint16.to(torch.int64) % 257 != 0).any()
        if mode not in ("GRAY", "GRAY_ALPHA"):
            assert uint16.to(torch.int64).max() > 60000

        # The uint8 output is the 16-bit output scaled down by 257 (full range).
        expected8 = (uint16.to(torch.float32) / 257).round()
        assert_tensor_close_on_at_least(
            uint8.to(torch.float32), expected8, percentage=99, atol=1
        )

        # The alpha channel of a source without alpha is a constant "opaque"
        # value. libheif fills it at the source's native bit depth (e.g. 1023 for
        # 10-bit), which our >8-bit remap bit-replicates to a true 65535 (the top
        # of the uint16 range), the same as for a fully-saturated color channel.
        if mode in ("GRAY_ALPHA", "RGB_ALPHA"):
            assert (uint16[-1] == 65535).all()  # constant, fully opaque

    @needs_heic
    @pytest.mark.parametrize(
        "mode, pil_mode",
        (
            ("UNCHANGED", "RGB"),  # opaque source -> RGB
            ("GRAY", "L"),
            ("GRAY_ALPHA", "LA"),
            ("RGB", "RGB"),
            ("RGB_ALPHA", "RGBA"),
        ),
    )
    def test_animated_heic(self, mode, pil_mode):
        # A multi-image HEIC decodes to a batched (N, C, H, W) tensor, one frame
        # per top-level image. The asset's frames are distinct solid colors
        # (which survive the YUV round-trip cleanly), so we assert their values
        # directly rather than against PIL, which would need pillow-heif.
        colors = [(200, 30, 30), (30, 200, 30), (30, 30, 200)]

        decoded = decode_heic(ANIMATED_HEIC.path, mode=mode)

        assert decoded.ndim == 4
        assert decoded.shape[0] == len(colors)
        assert decoded.shape[2:] == (ANIMATED_HEIC.height, ANIMATED_HEIC.width)
        assert decoded.dtype == torch.uint8

        for i, color in enumerate(colors):
            reference = self._pil_to_tensor(
                Image.fromarray(
                    numpy.full(
                        (ANIMATED_HEIC.height, ANIMATED_HEIC.width, 3),
                        color,
                        dtype=numpy.uint8,
                    )
                ).convert(pil_mode)
            )
            assert decoded[i].shape == reference.shape
            assert_tensor_close_on_at_least(
                decoded[i], reference, percentage=99, atol=3
            )

        if mode in ("GRAY_ALPHA", "RGB_ALPHA"):
            # The source is opaque, so the alpha channel must be fully opaque.
            assert (decoded[:, -1] == 255).all()

    def test_heic_missing_libheif_raises_import_error(self, monkeypatch):
        # If loading libtorchcodec_heic fails (typically because the optional,
        # non-bundled libheif isn't installed), decode_heic must raise a clear,
        # actionable ImportError. We simulate the failure so this runs
        # regardless of whether libheif is actually present.
        from torchcodec import _internally_replaced_utils as iru

        iru.load_heic_library.cache_clear()

        def boom(path):
            raise OSError("libheif.so.1: cannot open shared object file")

        monkeypatch.setattr(torch.ops, "load_library", boom)
        try:
            with pytest.raises(
                ImportError, match="Failed to load the HEIC decoding library"
            ):
                decode_heic(GRADIENT_HEIC.path)
        finally:
            # Don't leak the (failure-poisoned) cache to other tests.
            iru.load_heic_library.cache_clear()
