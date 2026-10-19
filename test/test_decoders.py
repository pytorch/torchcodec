# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import concurrent.futures
import contextlib
import gc
import queue
import threading
from functools import partial

import numpy
import pytest
import torch
from PIL import Image, ImageOps
from torchcodec import _core, ffmpeg_major_version, FrameBatch
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
    ColorConverter,
    DecodedFrame,
    Demuxer,
    Packet,
    PacketDecoder,
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
    IS_ROCM,
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
    TEST_NON_ZERO_START,
    TEST_SRC_2_12BIT_HDR,
    TEST_SRC_2_720P,
    TEST_SRC_2_720P_H265,
    TEST_SRC_2_720P_HDR,
    TEST_SRC_2_720P_MPEG4,
    TEST_SRC_2_720P_VP8,
    TEST_SRC_2_720P_VP9,
    TEST_SRC_2_MPEG4_MP4,
    TESTSRC2_ODD_HEIGHT_444,
    TESTSRC2_ODD_HEIGHT_AND_WIDTH_444,
    TESTSRC2_ODD_HEIGHT_AND_WIDTH_VP9,
    TESTSRC2_ODD_HEIGHT_AND_WIDTH_VP9_10BIT,
    TESTSRC2_ODD_HEIGHT_VP9,
    TESTSRC2_ODD_HEIGHT_VP9_10BIT,
    TESTSRC2_ODD_WIDTH_444,
    TESTSRC2_ODD_WIDTH_VP9,
    TESTSRC2_ODD_WIDTH_VP9_10BIT,
    TRANSPARENT_GIF,
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

        torch.testing.assert_close(cpu_frames, cuda_frames.cpu(), rtol=0, atol=0)

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


class TestBlocks:

    def test_block_output_types(self):
        # Demuxer yields Packets, PacketDecoder yields DecodedFrames, and
        # ColorConverter yields Frames with the expected shape/dtype.
        demuxer = Demuxer(NASA_VIDEO.path)
        decoder = PacketDecoder(demuxer)
        converter = ColorConverter()

        num_packets = 0
        for packet in demuxer:
            assert isinstance(packet, Packet)
            num_packets += 1
            for decoded in decoder.decode(packet):
                assert isinstance(decoded, DecodedFrame)
                frame = converter.convert(decoded)
                assert isinstance(frame, Frame)
                assert frame.data.ndim == 3  # CHW
                assert frame.data.shape[0] == 3  # channels first
                assert frame.data.dtype == torch.uint8
                assert frame.duration_seconds >= 0

        assert num_packets > 0

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
        yield from decoder.flush()

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

    def _decoded_frames(self, path):
        # demux + decode, as a single generator of DecodedFrames (pts order).
        demuxer = Demuxer(path)
        decoder = PacketDecoder(demuxer)
        return self._decode(decoder, self._demux(demuxer))

    def _decode_sequential(self, path):
        # demux -> decode -> color-convert, all on the calling thread.
        demuxer = Demuxer(path)
        decoder = PacketDecoder(demuxer)
        converter = ColorConverter()
        return list(
            self._convert(converter, self._decode(decoder, self._demux(demuxer)))
        )

    def _decode_prefetch_frames(self, path):
        # [demux + decode] on one thread || [color-convert] on another.
        demuxer = Demuxer(path)
        decoder = PacketDecoder(demuxer)
        converter = ColorConverter()
        frames = self.prefetch(self._decode(decoder, self._demux(demuxer)))
        return list(self._convert(converter, frames))

    def _decode_prefetch_packets(self, path):
        # [demux] on one thread || [decode + color-convert] on another.
        demuxer = Demuxer(path)
        decoder = PacketDecoder(demuxer)
        converter = ColorConverter()
        packets = self.prefetch(self._demux(demuxer))
        return list(self._convert(converter, self._decode(decoder, packets)))

    def _decode_prefetch_packets_and_frames(self, path):
        # [demux] || [decode] || [color-convert], each on its own thread.
        demuxer = Demuxer(path)
        decoder = PacketDecoder(demuxer)
        converter = ColorConverter()
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

    @pytest.mark.parametrize("video", (NASA_VIDEO, BT709_FULL_RANGE))
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
    def test_matches_video_decoder(self, video, decode_method):
        got = self._to_frame_batch(decode_method(self, video.path))
        ref = VideoDecoder(video.path).get_all_frames()

        assert got.data.shape == ref.data.shape
        torch.testing.assert_close(got.data, ref.data, atol=0, rtol=0)
        torch.testing.assert_close(got.pts_seconds, ref.pts_seconds, atol=0, rtol=0)
        torch.testing.assert_close(
            got.duration_seconds, ref.duration_seconds, atol=0, rtol=0
        )

    def test_color_converter_reused_across_videos(self):
        # A single unbound ColorConverter must correctly convert frames from two
        # different videos - here interleaved frame-by-frame, so the converter
        # switches input resolution/format on every call.
        converter = ColorConverter()
        videos = [NASA_VIDEO, BT709_FULL_RANGE]
        generators = [self._decoded_frames(v.path) for v in videos]
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
            ref = VideoDecoder(video.path).get_all_frames()
            assert got.data.shape == ref.data.shape
            torch.testing.assert_close(got.data, ref.data, atol=0, rtol=0)


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
        import io

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
        # Corrupt input raises. The message differs by GPU backend: nvJPEG on
        # NVIDIA, rocJPEG on AMD/ROCm. rocJPEG's VCN hardware decoder is more
        # lenient and does not raise on this particular corrupt JPEG.
        if not IS_ROCM:
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
