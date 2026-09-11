import io
import os
import platform
import re
import subprocess
import sys
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchcodec import ffmpeg_major_version
from torchcodec.decoders import AudioDecoder, decode_jpeg, decode_png, VideoDecoder

from torchcodec.encoders import (
    AudioEncoder,
    Encoder,
    JpegEncoder,
    PngEncoder,
    VideoEncoder,
)

from .utils import (
    assert_tensor_close_on_at_least,
    call_ffprobe,
    get_ffmpeg_minor_version,
    GRADIENT_JPEG,
    GRADIENT_PNG,
    GRAYSCALE_JPEG,
    GRAYSCALE_PNG,
    in_fbcode,
    IN_GITHUB_CI,
    IS_WINDOWS,
    NASA_AUDIO_MP3,
    NASA_AUDIO_MP3_44100,
    NASA_VIDEO,
    needs_cuda,
    needs_ffmpeg_cli,
    needs_jpeg,
    needs_png,
    psnr,
    SINE_MONO_S32,
    TEST_SRC_2_720P,
)

IS_WINDOWS_WITH_FFMPEG_LE_70 = IS_WINDOWS and (
    ffmpeg_major_version < 7
    or (ffmpeg_major_version == 7 and get_ffmpeg_minor_version() == 0)
)


@pytest.fixture
def with_ffmpeg_debug_logs():
    # Fixture that sets the ffmpeg logs to DEBUG mode
    previous_log_level = os.environ.get("TORCHCODEC_FFMPEG_LOG_LEVEL", "QUIET")
    os.environ["TORCHCODEC_FFMPEG_LOG_LEVEL"] = "DEBUG"
    yield
    os.environ["TORCHCODEC_FFMPEG_LOG_LEVEL"] = previous_log_level


def validate_frames_properties(*, actual: Path, expected: Path):
    # actual and expected are files containing encoded audio data.  We call
    # `ffprobe` on both, and assert that the frame properties match (pts,
    # duration, etc.)

    # non-exhaustive list of the props we want to test for:
    required_props = (
        "pts",
        "pts_time",
        "sample_fmt",
        "nb_samples",
        "channels",
        "duration",
        "duration_time",
    )
    show_entries = "frame=" + ",".join(required_props)

    frames_actual, frames_expected = (
        call_ffprobe(
            [
                "-select_streams",
                "a:0",
                "-show_frames",
                "-show_entries",
                show_entries,
                f"{f}",
            ]
        )["frames"]
        for f in (actual, expected)
    )

    # frames_actual and frames_expected are both a list of dicts, each dict
    # corresponds to a frame and each key-value pair corresponds to a frame
    # property like pts, nb_samples, etc., similar to the AVFrame fields.
    assert isinstance(frames_actual, list)
    assert all(isinstance(d, dict) for d in frames_actual)

    assert len(frames_actual) > 3  # arbitrary sanity check
    assert len(frames_actual) == len(frames_expected)

    for frame_index, (d_actual, d_expected) in enumerate(
        zip(frames_actual, frames_expected)
    ):
        if ffmpeg_major_version >= 6:
            assert all(required_prop in d_expected for required_prop in required_props)

        for prop in d_expected:
            if prop == "pkt_pos":
                # pkt_pos is the position of the packet *in bytes* in its
                # stream. We don't always match FFmpeg exactly on this,
                # typically on compressed formats like mp3. It's probably
                # because we are not writing the exact same headers, or
                # something like this. In any case, this doesn't seem to be
                # critical.
                continue
            assert (
                d_actual[prop] == d_expected[prop]
            ), f"\nComparing: {actual}\nagainst reference: {expected},\nthe {prop} property is different at frame {frame_index}:"


class TestAudioEncoder:

    def test_bad_input(self):
        with pytest.raises(ValueError, match="Expected samples to be a Tensor"):
            AudioEncoder(samples=123, sample_rate=32_000)
        with pytest.raises(ValueError, match="Expected 1D or 2D samples"):
            AudioEncoder(samples=torch.rand(3, 4, 5), sample_rate=32_000)
        with pytest.raises(ValueError, match="Expected float32 samples"):
            AudioEncoder(
                samples=torch.rand(10, 10, dtype=torch.float64), sample_rate=32_000
            )
        with pytest.raises(ValueError, match="sample_rate = 0 must be > 0"):
            AudioEncoder(samples=torch.rand(10, 10), sample_rate=0)

    def test_equivalence_with_encoder(self, tmp_path):
        # AudioEncoder is a thin wrapper around Encoder. We just check that
        # both produce the same output.
        asset = NASA_AUDIO_MP3
        source_samples = AudioDecoder(str(asset.path)).get_all_samples().data
        sample_rate = asset.sample_rate
        num_channels = source_samples.shape[0]

        # Encode with AudioEncoder
        audio_encoder_output = AudioEncoder(
            source_samples, sample_rate=sample_rate
        ).to_tensor(format="flac")

        # Encode with Encoder
        buf = io.BytesIO()
        enc = Encoder()
        audio = enc.add_audio(sample_rate=sample_rate, num_channels=num_channels)
        with enc.open_file_like(buf, format="flac"):
            audio.add_samples(source_samples)
        encoder_output = torch.frombuffer(buf.getbuffer(), dtype=torch.uint8)

        torch.testing.assert_close(audio_encoder_output, encoder_output, rtol=0, atol=0)

    def test_to_tensor_no_warning(self):
        # Non-regression test for https://github.com/meta-pytorch/torchcodec/issues/1509
        samples = torch.rand(2, 32_000, dtype=torch.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            AudioEncoder(samples, sample_rate=32_000).to_tensor(format="flac")


class TestVideoEncoder:

    def test_bad_input(self, tmp_path):
        with pytest.raises(
            ValueError, match="Expected uint8 frames, got frames.dtype = torch.float32"
        ):
            VideoEncoder(frames=torch.rand(5, 3, 64, 64), frame_rate=30)

        with pytest.raises(
            ValueError, match=r"Expected 4D frames, got frames.shape = torch.Size"
        ):
            VideoEncoder(frames=torch.zeros(10), frame_rate=30)

    def test_equivalence_with_encoder(self, tmp_path):
        # VideoEncoder is a thin wrapper around Encoder. We just check that
        # both produce the same output.
        source_frames = (
            VideoDecoder(str(TEST_SRC_2_720P.path))
            .get_frames_in_range(start=0, stop=10)
            .data
        )
        frame_rate = 30.0

        # Encode with VideoEncoder
        video_encoder_output = VideoEncoder(
            source_frames, frame_rate=frame_rate
        ).to_tensor(format="mp4")

        # Encode with Encoder
        buf = io.BytesIO()
        enc = Encoder()
        video = enc.add_video(
            height=source_frames.shape[2],
            width=source_frames.shape[3],
            frame_rate=frame_rate,
            device=str(source_frames.device),
        )
        with enc.open_file_like(buf, format="mp4"):
            video.add_frames(source_frames)
        encoder_output = torch.frombuffer(buf.getbuffer(), dtype=torch.uint8)

        torch.testing.assert_close(video_encoder_output, encoder_output, rtol=0, atol=0)

    def test_to_tensor_no_warning(self):
        # Non-regression test for https://github.com/meta-pytorch/torchcodec/issues/1509
        frames = torch.randint(0, 256, (10, 3, 64, 64), dtype=torch.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            VideoEncoder(frames, frame_rate=30).to_tensor(format="mp4")


class TestEncoder:
    cpu_and_oss_cuda = (
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.needs_cuda,
                pytest.mark.skipif(in_fbcode(), reason="NVENC not available in fbcode"),
            ],
        ),
    )

    @staticmethod
    def _create_encoder(method, tmp_path, format):
        encoder = Encoder()
        if method == "to_file":
            encoder_output = tmp_path / f"test.{format}"
            open_kwargs = dict(dest=encoder_output)
        elif method == "to_file_like":
            encoder_output = io.BytesIO()
            open_kwargs = dict(dest=encoder_output, format=format)
        else:
            raise ValueError(f"Unknown method: {method}")
        return encoder, encoder_output, open_kwargs

    @staticmethod
    def _open_encoder(enc, open_kwargs):
        if "format" in open_kwargs:
            return enc.open_file_like(open_kwargs["dest"], format=open_kwargs["format"])
        else:
            return enc.open_file(open_kwargs["dest"])

    @staticmethod
    def _get_decoder_source(encoder_output):
        if isinstance(encoder_output, io.BytesIO):
            return encoder_output.getvalue()
        return str(encoder_output)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_double_close(self, tmp_path, method):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        enc.close()
        enc.close()  # double close is a no-op

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_context_manager(self, tmp_path, method):
        enc, encoder_output, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        frames = torch.randint(0, 256, (5, 3, 64, 64), dtype=torch.uint8)
        video = enc.add_video(height=64, width=64, frame_rate=30.0)
        with self._open_encoder(enc, open_kwargs):
            video.add_frames(frames)

        # The output is valid and decodable, proving close() was called by __exit__.
        decoded_frames = (
            VideoDecoder(self._get_decoder_source(encoder_output))
            .get_frames_in_range(start=0, stop=5)
            .data
        )
        assert decoded_frames.shape == frames.shape

    @pytest.mark.parametrize("format", ["mp4", "mov", "mkv"])
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    @pytest.mark.parametrize("device", cpu_and_oss_cuda)
    def test_add_video_and_encode_frames(self, tmp_path, format, method, device):
        source_decoder = VideoDecoder(str(TEST_SRC_2_720P.path))
        source_frames = source_decoder.get_frames_in_range(start=0, stop=10).data.to(
            device
        )
        frame_rate = source_decoder.metadata.average_fps
        percentage, atol = (96, 2) if device == "cuda" else (99, 2)

        enc, encoder_output, open_kwargs = self._create_encoder(
            method, tmp_path, format
        )
        add_video_kwargs = {
            "height": source_frames.shape[2],
            "width": source_frames.shape[3],
            "frame_rate": frame_rate,
            "device": device,
        }
        if device == "cpu":
            add_video_kwargs["pixel_format"] = "yuv444p"
            add_video_kwargs["crf"] = 0
        else:
            add_video_kwargs["extra_options"] = {"qp": "1"}
        video = enc.add_video(**add_video_kwargs)
        self._open_encoder(enc, open_kwargs)
        video.add_frames(source_frames[:5])
        video.add_frames(source_frames[5:])
        enc.close()

        decoded_frames = (
            VideoDecoder(self._get_decoder_source(encoder_output))
            .get_frames_in_range(start=0, stop=10)
            .data
        )
        assert_tensor_close_on_at_least(
            decoded_frames, source_frames.cpu(), percentage=percentage, atol=atol
        )

    def test_open_invalid_path(self):
        enc = Encoder()
        enc.add_video(height=64, width=64, frame_rate=30.0)
        with pytest.raises(RuntimeError, match="make sure it's a valid path"):
            enc.open_file("/nonexistent/dir/test.mp4")

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_open_invalid_format(self, tmp_path, method):
        enc = Encoder()
        enc.add_video(height=64, width=64, frame_rate=30.0)
        if method == "to_file":
            with pytest.raises(RuntimeError, match="check the desired extension"):
                enc.open_file(tmp_path / "test.bad_extension")
        elif method == "to_file_like":
            with pytest.raises(
                RuntimeError,
                match=r"Check the desired format\? Got format=bad_extension",
            ):
                enc.open_file_like(io.BytesIO(), format="bad_extension")

    @pytest.mark.parametrize("format", ["mp4", "mov"])
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    @pytest.mark.parametrize("device", cpu_and_oss_cuda)
    def test_fragmented_mp4(self, format, tmp_path, method, device):
        source_decoder = VideoDecoder(str(TEST_SRC_2_720P.path))
        source_frames = source_decoder.get_frames_in_range(start=0, stop=10).data.to(
            device
        )
        frame_rate = source_decoder.metadata.average_fps
        percentage, atol = (96, 2) if device == "cuda" else (99, 2)

        enc, encoder_output, open_kwargs = self._create_encoder(
            method, tmp_path, format
        )
        # In addition to the fragmentation flag, "flush_packets" and "threads"
        # are necessary to decode frames before close().
        # See frag flags: https://ffmpeg.org/ffmpeg-formats.html#Fragmentation
        # TODO MultiStreamEncoder: Get a better understanding of which options
        # are necessary for reading fragmented mp4s
        extra_options = {
            "movflags": "+frag_every_frame+empty_moov",
            "flush_packets": "1",
            "threads": "1",
        }
        if device == "cuda":
            extra_options.update({"qp": "1", "delay": "0"})
            pixel_format, crf = None, None
        else:
            extra_options["tune"] = "zerolatency"
            pixel_format, crf = "yuv444p", 0
        video = enc.add_video(
            height=source_frames.shape[2],
            width=source_frames.shape[3],
            frame_rate=frame_rate,
            device=device,
            pixel_format=pixel_format,
            crf=crf,
            extra_options=extra_options,
        )
        self._open_encoder(enc, open_kwargs)
        # Here, we decode the available fragmented mp4 frames before calling close()
        for batch in [source_frames[:5], source_frames[5:]]:
            video.add_frames(batch)
            mid_decoder = VideoDecoder(self._get_decoder_source(encoder_output))
            num_available = len(mid_decoder)
            assert num_available > 0
            assert_tensor_close_on_at_least(
                mid_decoder.get_frames_in_range(start=0, stop=num_available).data,
                source_frames[:num_available].cpu(),
                percentage=percentage,
                atol=atol,
            )

        enc.close()
        # After close, all frames must be decodable
        assert_tensor_close_on_at_least(
            VideoDecoder(self._get_decoder_source(encoder_output))
            .get_frames_in_range(start=0, stop=10)
            .data,
            source_frames.cpu(),
            percentage=percentage,
            atol=atol,
        )

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    @pytest.mark.parametrize("device", cpu_and_oss_cuda)
    def test_write_frames_mismatched_dimensions_errors(self, tmp_path, method, device):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        video = enc.add_video(height=256, width=256, frame_rate=30.0, device=device)
        self._open_encoder(enc, open_kwargs)
        # write with wrong size errors
        frames_128 = torch.randint(
            0, 256, (2, 3, 128, 128), dtype=torch.uint8, device=device
        )
        with pytest.raises(RuntimeError, match="same dimensions"):
            video.add_frames(frames_128)
        # write with different size than first also errors
        frames_256 = torch.randint(
            0, 256, (2, 3, 256, 256), dtype=torch.uint8, device=device
        )
        frames_512 = torch.randint(
            0, 256, (2, 3, 512, 512), dtype=torch.uint8, device=device
        )
        video.add_frames(frames_256)
        with pytest.raises(RuntimeError, match="same dimensions"):
            video.add_frames(frames_512)

    @needs_cuda
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_write_frames_different_devices_errors(self, tmp_path, method):
        cpu_frames = torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.uint8)
        cuda_frames = cpu_frames.to("cuda")

        # CPU stream, write CUDA frames
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        video = enc.add_video(height=256, width=256, frame_rate=30.0)
        self._open_encoder(enc, open_kwargs)
        video.add_frames(cpu_frames)
        with pytest.raises(RuntimeError, match="same device"):
            video.add_frames(cuda_frames)
        enc.close()

        # CUDA stream, write CPU frames
        cuda_dir = tmp_path / "cuda"
        cuda_dir.mkdir()
        enc, _, open_kwargs = self._create_encoder(method, cuda_dir, "mp4")
        video = enc.add_video(height=256, width=256, frame_rate=30.0, device="cuda")
        self._open_encoder(enc, open_kwargs)
        with pytest.raises(RuntimeError, match="same device"):
            video.add_frames(cpu_frames)

    @needs_cuda
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_device_None_respects_default_device(self, tmp_path, method):
        source_decoder = VideoDecoder(str(TEST_SRC_2_720P.path))
        source_frames = source_decoder.get_frames_in_range(start=0, stop=5).data.to(
            "cuda"
        )
        frame_rate = source_decoder.metadata.average_fps

        enc, encoder_output, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        with torch.device("cuda:0"):
            video = enc.add_video(
                height=source_frames.shape[2],
                width=source_frames.shape[3],
                frame_rate=frame_rate,
                device=None,
            )
        self._open_encoder(enc, open_kwargs)
        video.add_frames(source_frames)
        enc.close()

        decoded_frames = (
            VideoDecoder(self._get_decoder_source(encoder_output))
            .get_frames_in_range(start=0, stop=5)
            .data
        )
        assert decoded_frames.shape == source_frames.shape

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_device_torch_device_object(self, tmp_path, method):
        frames = torch.randint(0, 256, (5, 3, 64, 64), dtype=torch.uint8)
        enc, encoder_output, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        video = enc.add_video(
            height=64, width=64, frame_rate=30.0, device=torch.device("cpu")
        )
        self._open_encoder(enc, open_kwargs)
        video.add_frames(frames)
        enc.close()

        decoded_frames = (
            VideoDecoder(self._get_decoder_source(encoder_output))
            .get_frames_in_range(start=0, stop=5)
            .data
        )
        assert decoded_frames.shape == frames.shape

    @needs_cuda
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_device_cuda_0_string(self, tmp_path, method):
        source_decoder = VideoDecoder(str(TEST_SRC_2_720P.path))
        source_frames = source_decoder.get_frames_in_range(start=0, stop=5).data.to(
            "cuda:0"
        )
        frame_rate = source_decoder.metadata.average_fps

        enc, encoder_output, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        video = enc.add_video(
            height=source_frames.shape[2],
            width=source_frames.shape[3],
            frame_rate=frame_rate,
            device="cuda:0",
        )
        self._open_encoder(enc, open_kwargs)
        video.add_frames(source_frames)
        enc.close()

        decoded_frames = (
            VideoDecoder(self._get_decoder_source(encoder_output))
            .get_frames_in_range(start=0, stop=5)
            .data
        )
        assert decoded_frames.shape == source_frames.shape

    @needs_cuda
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_write_samples_on_cuda_errors(self, tmp_path, method):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "wav")
        audio = enc.add_audio(sample_rate=44100, num_channels=1)
        self._open_encoder(enc, open_kwargs)
        cuda_samples = torch.randn(1, 1000, device="cuda")
        with pytest.raises(RuntimeError, match="samples must be on CPU, got cuda"):
            audio.add_samples(cuda_samples)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    @pytest.mark.parametrize(
        "device", ("cpu", pytest.param("cuda", marks=pytest.mark.needs_cuda))
    )
    def test_write_frames_without_open_errors(self, tmp_path, method, device):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        video = enc.add_video(height=64, width=64, frame_rate=30.0, device=device)
        frames = torch.randint(0, 256, (5, 3, 64, 64), dtype=torch.uint8, device=device)
        with pytest.raises(
            RuntimeError, match="Call open\\(\\) before addFrames\\(\\)"
        ):
            video.add_frames(frames)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_write_samples_without_open_errors(self, tmp_path, method):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        audio = enc.add_audio(sample_rate=44100, num_channels=1)
        samples = torch.randn(1, 1000)
        with pytest.raises(
            RuntimeError, match="Call open\\(\\) before addSamples\\(\\)"
        ):
            audio.add_samples(samples)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_write_samples_mismatched_channels_errors(self, tmp_path, method):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "wav")
        audio = enc.add_audio(sample_rate=44100, num_channels=1)
        self._open_encoder(enc, open_kwargs)
        samples = torch.randn(2, 1000)  # 2 channels but stream expects 1
        with pytest.raises(RuntimeError, match="Expected 1 channels, got 2"):
            audio.add_samples(samples)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_open_without_stream_errors(self, tmp_path, method):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        with pytest.raises(
            RuntimeError,
            match="Call addVideoStream\\(\\) or addAudioStream\\(\\) before open\\(\\)",
        ):
            self._open_encoder(enc, open_kwargs)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_open_twice_errors(self, tmp_path, method):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        enc.add_video(height=64, width=64, frame_rate=30.0)
        self._open_encoder(enc, open_kwargs)
        with pytest.raises(RuntimeError, match="open\\(\\) was already called"):
            self._open_encoder(enc, open_kwargs)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_open_after_close_errors(self, tmp_path, method):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        enc.add_video(height=64, width=64, frame_rate=30.0)
        self._open_encoder(enc, open_kwargs)
        enc.close()
        with pytest.raises(RuntimeError, match="Cannot open after close\\(\\)"):
            self._open_encoder(enc, open_kwargs)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_write_frames_after_close_errors(self, tmp_path, method):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        video = enc.add_video(height=64, width=64, frame_rate=30.0)
        self._open_encoder(enc, open_kwargs)
        enc.close()
        frames = torch.randint(0, 256, (5, 3, 64, 64), dtype=torch.uint8)
        with pytest.raises(RuntimeError, match="Cannot add frames after close\\(\\)"):
            video.add_frames(frames)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_write_samples_after_close_errors(self, tmp_path, method):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "wav")
        audio = enc.add_audio(sample_rate=44100, num_channels=1)
        self._open_encoder(enc, open_kwargs)
        enc.close()
        samples = torch.randn(1, 1000)
        with pytest.raises(RuntimeError, match="Cannot add samples after close\\(\\)"):
            audio.add_samples(samples)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_add_audio_invalid_bit_rate_errors(self, tmp_path, method):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        enc.add_audio(sample_rate=44100, num_channels=2, bit_rate=-1)
        with pytest.raises(RuntimeError, match="bit_rate=-1 must be >= 0"):
            self._open_encoder(enc, open_kwargs)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_add_audio_and_encode_samples(self, tmp_path, method):
        source_audio = AudioDecoder(str(SINE_MONO_S32.path)).get_all_samples()
        samples = source_audio.data
        sample_rate = source_audio.sample_rate
        num_channels = samples.shape[0]

        enc, encoder_output, open_kwargs = self._create_encoder(method, tmp_path, "wav")
        audio = enc.add_audio(sample_rate=sample_rate, num_channels=num_channels)
        self._open_encoder(enc, open_kwargs)
        chunk_lengths = [1, 50, 1000, 0, 25]
        offset = 0
        for length in chunk_lengths:
            audio.add_samples(samples[:, offset : offset + length])
            offset += length
        audio.add_samples(samples[:, offset:])
        enc.close()

        decoded = AudioDecoder(
            self._get_decoder_source(encoder_output)
        ).get_all_samples()
        assert decoded.data.shape[0] == num_channels
        assert decoded.sample_rate == sample_rate
        torch.testing.assert_close(decoded.data, samples, atol=1e-4, rtol=0)

    @pytest.mark.parametrize(
        "format",
        (
            "mp4",
            pytest.param(
                "mkv",
                marks=pytest.mark.skipif(
                    ffmpeg_major_version < 6,
                    reason="Default audio codec for MKV has low accuracy on older FFmpeg versions.",
                ),
            ),
            "mov",
        ),
    )
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_add_audio_and_video_and_encode(self, tmp_path, format, method):
        source_video_decoder = VideoDecoder(str(NASA_VIDEO.path))
        source_frames = source_video_decoder.get_frames_in_range(
            start=0, stop=len(source_video_decoder)
        ).data

        source_audio = AudioDecoder(str(NASA_AUDIO_MP3_44100.path)).get_all_samples()
        source_samples = source_audio.data
        sample_rate = source_audio.sample_rate

        enc, encoder_output, open_kwargs = self._create_encoder(
            method, tmp_path, format
        )
        video = enc.add_video(
            height=source_frames.shape[2],
            width=source_frames.shape[3],
            frame_rate=source_video_decoder.metadata.average_fps,
            pixel_format="yuv444p",
            crf=0,
        )
        audio = enc.add_audio(
            sample_rate=sample_rate,
            num_channels=source_samples.shape[0],
        )
        self._open_encoder(enc, open_kwargs)
        half_frames = source_frames.shape[0] // 2
        half_samples = source_samples.shape[1] // 2
        video.add_frames(source_frames[:half_frames])
        audio.add_samples(source_samples[:, :half_samples])
        video.add_frames(source_frames[half_frames:])
        audio.add_samples(source_samples[:, half_samples:])
        enc.close()

        source = self._get_decoder_source(encoder_output)

        decoded_video_decoder = VideoDecoder(source)
        decoded_frames = decoded_video_decoder.get_frames_in_range(
            start=0, stop=len(decoded_video_decoder)
        ).data
        assert_tensor_close_on_at_least(
            decoded_frames, source_frames, percentage=99, atol=2
        )

        audio_decoder = AudioDecoder(source)
        decoded_audio = audio_decoder.get_all_samples()
        assert decoded_audio.sample_rate == sample_rate
        assert decoded_audio.data.shape[0] == source_samples.shape[0]
        # Codecs for lossy audio formats (not WAV or FLAC) can add padding which causes
        # sample count to differ, so we only compare the smaller sample count.
        # TODO MultiStreamEncoder: The previous AudioEncoder didn't need
        # padding after introducing a FIFO. Investigate why this is needed.
        num_samples_to_compare = min(
            decoded_audio.data.shape[1], source_samples.shape[1]
        )
        assert_tensor_close_on_at_least(
            decoded_audio.data[:, :num_samples_to_compare],
            source_samples[:, :num_samples_to_compare],
            percentage=96 if format == "mkv" else 99,
            atol=0.1 if format == "mkv" else 0.01,
        )

    @needs_cuda
    @pytest.mark.skipif(in_fbcode(), reason="NVENC not available in fbcode")
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_cuda_video_with_cpu_video_and_cpu_audio(self, tmp_path, method):
        source_video_decoder = VideoDecoder(str(NASA_VIDEO.path))
        source_frames = source_video_decoder.get_frames_in_range(
            start=0, stop=len(source_video_decoder)
        ).data

        source_audio = AudioDecoder(str(NASA_AUDIO_MP3_44100.path)).get_all_samples()
        source_samples = source_audio.data
        sample_rate = source_audio.sample_rate

        enc, encoder_output, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        cuda_video = enc.add_video(
            height=source_frames.shape[2],
            width=source_frames.shape[3],
            frame_rate=source_video_decoder.metadata.average_fps,
            device="cuda",
            extra_options={"qp": "1"},
        )
        cpu_video = enc.add_video(
            height=source_frames.shape[2],
            width=source_frames.shape[3],
            frame_rate=source_video_decoder.metadata.average_fps,
            pixel_format="yuv444p",
            crf=0,
        )
        audio = enc.add_audio(
            sample_rate=sample_rate,
            num_channels=source_samples.shape[0],
        )
        self._open_encoder(enc, open_kwargs)
        half_frames = source_frames.shape[0] // 2
        half_samples = source_samples.shape[1] // 2
        cuda_video.add_frames(source_frames[:half_frames].to("cuda"))
        cpu_video.add_frames(source_frames[:half_frames])
        audio.add_samples(source_samples[:, :half_samples])
        cuda_video.add_frames(source_frames[half_frames:].to("cuda"))
        cpu_video.add_frames(source_frames[half_frames:])
        audio.add_samples(source_samples[:, half_samples:])
        enc.close()

        source = self._get_decoder_source(encoder_output)

        decoded_video_decoder = VideoDecoder(source, stream_index=0)
        decoded_cuda_frames = decoded_video_decoder.get_frames_in_range(
            start=0, stop=len(decoded_video_decoder)
        ).data
        assert_tensor_close_on_at_least(
            decoded_cuda_frames, source_frames, percentage=90, atol=2
        )

        decoded_video_decoder = VideoDecoder(source, stream_index=1)
        decoded_cpu_frames = decoded_video_decoder.get_frames_in_range(
            start=0, stop=len(decoded_video_decoder)
        ).data
        assert_tensor_close_on_at_least(
            decoded_cpu_frames, source_frames, percentage=99, atol=2
        )

        audio_decoder = AudioDecoder(source)
        decoded_audio = audio_decoder.get_all_samples()
        assert decoded_audio.sample_rate == sample_rate
        assert decoded_audio.data.shape[0] == source_samples.shape[0]
        num_samples_to_compare = min(
            decoded_audio.data.shape[1], source_samples.shape[1]
        )
        assert_tensor_close_on_at_least(
            decoded_audio.data[:, :num_samples_to_compare],
            source_samples[:, :num_samples_to_compare],
            percentage=99,
            atol=0.01,
        )

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_multiple_video_streams_and_audio(self, tmp_path, method):
        source_frames_big = torch.randint(0, 256, (5, 3, 256, 256), dtype=torch.uint8)
        source_frames_small = torch.randint(0, 256, (8, 3, 128, 128), dtype=torch.uint8)

        source_audio = AudioDecoder(str(NASA_AUDIO_MP3_44100.path)).get_all_samples()
        source_samples_stereo = source_audio.data
        sample_rate = source_audio.sample_rate
        source_samples_mono = source_samples_stereo[:1]

        enc, encoder_output, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        video_big = enc.add_video(
            height=256, width=256, frame_rate=10.0, pixel_format="yuv444p", crf=0
        )
        video_small = enc.add_video(
            height=128, width=128, frame_rate=25.0, pixel_format="yuv444p", crf=0
        )
        audio_stereo = enc.add_audio(
            sample_rate=sample_rate,
            num_channels=2,
        )
        audio_mono = enc.add_audio(
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._open_encoder(enc, open_kwargs)
        video_big.add_frames(source_frames_big[:3])
        video_small.add_frames(source_frames_small[:4])
        audio_stereo.add_samples(
            source_samples_stereo[:, : source_samples_stereo.shape[1] // 2]
        )
        audio_mono.add_samples(
            source_samples_mono[:, : source_samples_mono.shape[1] // 2]
        )
        video_big.add_frames(source_frames_big[3:])
        video_small.add_frames(source_frames_small[4:])
        audio_stereo.add_samples(
            source_samples_stereo[:, source_samples_stereo.shape[1] // 2 :]
        )
        audio_mono.add_samples(
            source_samples_mono[:, source_samples_mono.shape[1] // 2 :]
        )
        enc.close()

        source = self._get_decoder_source(encoder_output)

        decoded_big = VideoDecoder(source, stream_index=0)
        assert len(decoded_big) == 5
        decoded_big_frames = decoded_big.get_frames_in_range(
            start=0, stop=len(decoded_big)
        ).data
        assert decoded_big_frames.shape == (5, 3, 256, 256)
        assert_tensor_close_on_at_least(
            decoded_big_frames, source_frames_big, percentage=99, atol=2
        )

        decoded_small = VideoDecoder(source, stream_index=1)
        assert len(decoded_small) == 8
        decoded_small_frames = decoded_small.get_frames_in_range(
            start=0, stop=len(decoded_small)
        ).data
        assert decoded_small_frames.shape == (8, 3, 128, 128)
        assert_tensor_close_on_at_least(
            decoded_small_frames, source_frames_small, percentage=99, atol=2
        )

        # stream_index is absolute: 0, 1 are video; 2, 3 are audio
        decoded_stereo = AudioDecoder(source, stream_index=2).get_all_samples()
        assert decoded_stereo.sample_rate == sample_rate
        assert decoded_stereo.data.shape[0] == 2
        num_samples_to_compare = min(
            decoded_stereo.data.shape[1], source_samples_stereo.shape[1]
        )
        assert_tensor_close_on_at_least(
            decoded_stereo.data[:, :num_samples_to_compare],
            source_samples_stereo[:, :num_samples_to_compare],
            percentage=98,
            atol=0.01,
        )

        decoded_mono = AudioDecoder(source, stream_index=3).get_all_samples()
        assert decoded_mono.sample_rate == sample_rate
        assert decoded_mono.data.shape[0] == 1
        num_samples_to_compare = min(
            decoded_mono.data.shape[1], source_samples_mono.shape[1]
        )
        assert_tensor_close_on_at_least(
            decoded_mono.data[:, :num_samples_to_compare],
            source_samples_mono[:, :num_samples_to_compare],
            percentage=98,
            atol=0.01,
        )

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_add_audio_out_num_channels(self, tmp_path, method):
        # We just check that the out_num_channels parameter is respected.
        # Correctness is checked in other tests (like test_audio_against_cli())
        sample_rate = 44_100
        source_stereo = torch.rand(2, 1_000)
        source_mono = torch.rand(1, 1_000)

        enc, encoder_output, open_kwargs = self._create_encoder(method, tmp_path, "mkv")
        # Stream 0: stereo input, mono output
        audio_stereo_to_mono = enc.add_audio(
            sample_rate=sample_rate,
            num_channels=2,
            out_num_channels=1,
        )
        # Stream 1: mono input, stereo output
        audio_mono_to_stereo = enc.add_audio(
            sample_rate=sample_rate,
            num_channels=1,
            out_num_channels=2,
        )
        # Stream 2: stereo input, no out_num_channels (should stay stereo)
        audio_passthrough = enc.add_audio(
            sample_rate=sample_rate,
            num_channels=2,
        )
        self._open_encoder(enc, open_kwargs)
        audio_stereo_to_mono.add_samples(source_stereo)
        audio_mono_to_stereo.add_samples(source_mono)
        audio_passthrough.add_samples(source_stereo)
        enc.close()

        source = self._get_decoder_source(encoder_output)

        decoded_0 = AudioDecoder(source, stream_index=0).get_all_samples()
        assert decoded_0.data.shape[0] == 1

        decoded_1 = AudioDecoder(source, stream_index=1).get_all_samples()
        assert decoded_1.data.shape[0] == 2

        decoded_2 = AudioDecoder(source, stream_index=2).get_all_samples()
        assert decoded_2.data.shape[0] == 2

    @pytest.mark.parametrize(
        "format",
        [
            "mov",
            "mp4",
            "avi",
            "mkv",
            "flv",
        ],
    )
    @pytest.mark.parametrize(
        "encode_params",
        [
            {"pixel_format": "yuv444p", "crf": 0, "preset": None},
            {"pixel_format": "yuv420p", "crf": 30, "preset": None},
            {"pixel_format": "yuv420p", "crf": None, "preset": "ultrafast"},
            {"pixel_format": "yuv420p", "crf": None, "preset": None},
        ],
    )
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    @pytest.mark.parametrize("frame_rate", [30, 29.97])
    @needs_ffmpeg_cli
    @pytest.mark.skipif(
        IS_WINDOWS and ffmpeg_major_version == 8,
        reason="against_cli tests fail on Windows with FFmpeg 8",
    )
    def test_video_against_ffmpeg_cli(
        self, tmp_path, format, encode_params, method, frame_rate
    ):
        pixel_format = encode_params["pixel_format"]
        crf = encode_params["crf"]
        preset = encode_params["preset"]

        if format in ("avi", "flv") and pixel_format == "yuv444p":
            pytest.skip(f"Default codec for {format} does not support {pixel_format}")

        source_frames = (
            VideoDecoder(str(TEST_SRC_2_720P.path))
            .get_frames_in_range(start=0, stop=30)
            .data
        )

        # Encode with FFmpeg CLI
        temp_raw_path = str(tmp_path / "temp_input.raw")
        with open(temp_raw_path, "wb") as f:
            f.write(source_frames.permute(0, 2, 3, 1).cpu().numpy().tobytes())

        ffmpeg_encoded_path = str(tmp_path / f"ffmpeg_output.{format}")
        # Some codecs (ex. MPEG4) do not support CRF or preset.
        # Flags not supported by the selected codec will be ignored.
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",  # Input format
            "-s",
            f"{source_frames.shape[3]}x{source_frames.shape[2]}",
            "-r",
            str(frame_rate),
            "-i",
            temp_raw_path,
        ]
        if pixel_format is not None:  # Output format
            ffmpeg_cmd.extend(["-pix_fmt", pixel_format])
        if preset is not None:
            ffmpeg_cmd.extend(["-preset", preset])
        if crf is not None:
            ffmpeg_cmd.extend(["-crf", str(crf)])
        # Output path must be last
        ffmpeg_cmd.append(ffmpeg_encoded_path)
        subprocess.run(ffmpeg_cmd, check=True)
        ffmpeg_frames = (
            VideoDecoder(ffmpeg_encoded_path).get_frames_in_range(start=0, stop=30).data
        )

        # Encode with Encoder
        enc, encoder_output, open_kwargs = self._create_encoder(
            method, tmp_path, format
        )
        video = enc.add_video(
            height=source_frames.shape[2],
            width=source_frames.shape[3],
            frame_rate=frame_rate,
            pixel_format=pixel_format,
            crf=crf,
            preset=preset,
        )
        self._open_encoder(enc, open_kwargs)
        video.add_frames(source_frames)
        enc.close()

        encoder_frames = (
            VideoDecoder(self._get_decoder_source(encoder_output))
            .get_frames_in_range(start=0, stop=30)
            .data
        )

        # MPEG codec used for avi format does not accept CRF
        percentage = 94 if format == "avi" else 99

        assert ffmpeg_frames.shape[0] == encoder_frames.shape[0]
        # Check that PSNR between both encoded versions is high
        for ff_frame, enc_frame in zip(ffmpeg_frames, encoder_frames):
            assert psnr(ff_frame, enc_frame) > 30
            assert_tensor_close_on_at_least(
                ff_frame, enc_frame, percentage=percentage, atol=2
            )

        # Only compare video metadata on ffmpeg versions >= 6, as older versions
        # are often missing metadata
        if ffmpeg_major_version >= 6 and method == "to_file":
            fields = [
                "duration",
                "duration_ts",
                "r_frame_rate",
                "time_base",
                "nb_frames",
            ]
            ffmpeg_metadata = self._get_video_metadata(
                ffmpeg_encoded_path, fields=fields
            )
            encoder_metadata = self._get_video_metadata(
                str(encoder_output), fields=fields
            )
            assert ffmpeg_metadata == encoder_metadata

            # Check that frame timestamps and duration are the same
            fields = ("pts", "pts_time")
            if format != "flv":
                fields += ("duration", "duration_time")
            ffmpeg_frames_info = self._get_frames_info(
                ffmpeg_encoded_path, fields=fields
            )
            encoder_frames_info = self._get_frames_info(
                str(encoder_output), fields=fields
            )
            assert ffmpeg_frames_info == encoder_frames_info

    @needs_ffmpeg_cli
    @pytest.mark.parametrize("asset", (NASA_AUDIO_MP3, SINE_MONO_S32))
    @pytest.mark.parametrize("bit_rate", (None, 0, 44_100, 999_999_999))
    @pytest.mark.parametrize("num_channels", (None, 1, 2))
    @pytest.mark.parametrize("sample_rate", (8_000, 32_000))
    @pytest.mark.parametrize(
        "format",
        [
            # TODO: https://github.com/pytorch/torchcodec/issues/837
            pytest.param(
                "mp3",
                marks=pytest.mark.skipif(
                    IS_WINDOWS and ffmpeg_major_version <= 5,
                    reason="Encoding mp3 on Windows is weirdly buggy",
                ),
            ),
            pytest.param(
                "wav",
                marks=pytest.mark.skipif(
                    ffmpeg_major_version == 4,
                    reason="Swresample with FFmpeg 4 doesn't work on wav files",
                ),
            ),
            "flac",
        ],
    )
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    @pytest.mark.skipif(
        IS_WINDOWS and ffmpeg_major_version >= 8,
        reason="against_cli tests fail on Windows with FFmpeg 8",
    )
    def test_audio_against_cli(
        self,
        asset,
        bit_rate,
        num_channels,
        sample_rate,
        format,
        method,
        tmp_path,
        capfd,
        with_ffmpeg_debug_logs,
    ):
        # Encodes samples with our encoder and with the FFmpeg CLI, and checks
        # that both decoded outputs are equal
        source_audio = AudioDecoder(str(asset.path)).get_all_samples()
        source_samples = source_audio.data
        in_sample_rate = source_audio.sample_rate
        in_num_channels = source_samples.shape[0]

        out_num_channels = num_channels if num_channels is not None else in_num_channels

        # Encode with FFmpeg CLI
        encoded_by_ffmpeg = tmp_path / f"ffmpeg_output.{format}"
        subprocess.run(
            ["ffmpeg", "-i", str(asset.path)]
            + (["-b:a", f"{bit_rate}"] if bit_rate is not None else [])
            + (["-ac", f"{out_num_channels}"] if num_channels is not None else [])
            + ["-ar", f"{sample_rate}"]
            + [str(encoded_by_ffmpeg)],
            capture_output=True,
            check=True,
        )

        # Encode with Encoder
        enc, encoder_output, open_kwargs = self._create_encoder(
            method, tmp_path, format
        )
        audio = enc.add_audio(
            sample_rate=in_sample_rate,
            num_channels=in_num_channels,
            bit_rate=bit_rate,
            out_num_channels=num_channels,
            out_sample_rate=sample_rate,
        )
        self._open_encoder(enc, open_kwargs)
        audio.add_samples(source_samples)
        enc.close()

        captured = capfd.readouterr()
        if format == "wav":
            assert "Timestamps are unset in a packet" not in captured.err
        if format == "mp3":
            assert "Queue input is backward in time" not in captured.err
        if format in ("flac", "wav"):
            assert "Encoder did not produce proper pts" not in captured.err
        if format in ("flac", "mp3"):
            assert "Application provided invalid" not in captured.err

        if IS_WINDOWS_WITH_FFMPEG_LE_70 and format == "mp3":
            # We're getting a "Could not open input file" on Windows mp3
            # files when decoding.
            # TODO: https://github.com/pytorch/torchcodec/issues/837
            return

        samples_by_us = AudioDecoder(
            self._get_decoder_source(encoder_output)
        ).get_all_samples()
        samples_by_ffmpeg = AudioDecoder(str(encoded_by_ffmpeg)).get_all_samples()

        assert_close = torch.testing.assert_close
        if sample_rate != in_sample_rate:
            if platform.machine().lower() == "aarch64":
                rtol, atol = 0, 1e-2
            else:
                rtol, atol = 0, 1e-3
            if sys.platform == "darwin":
                assert_close = partial(assert_tensor_close_on_at_least, percentage=99)
        elif format == "wav":
            rtol, atol = 0, 1e-4
        elif format == "mp3" and asset is SINE_MONO_S32 and num_channels == 2:
            # Not sure why, this one needs slightly higher tol. With default
            # tolerances, the check fails on ~1% of the samples, so that's
            # probably fine. It might be that the FFmpeg CLI doesn't rely on
            # libswresample for converting channels?
            rtol, atol = 0, 1e-3
        else:
            rtol, atol = None, None

        assert_close(
            samples_by_us.data,
            samples_by_ffmpeg.data,
            rtol=rtol,
            atol=atol,
        )
        assert samples_by_us.sample_rate == samples_by_ffmpeg.sample_rate

        # On FFmpeg >= 9 the CLI encodes FLAC with a block size equal to the
        # number of samples of the first frame it feeds to the encoder (e.g. 47
        # for NASA_AUDIO_MP3, whose first decoded frame is short because of the
        # mp3 encoder delay), instead of the block size that the FLAC encoder
        # picks by default from the sample rate, which is what we use. The
        # decoded samples still match, but the packets can't.
        cli_flac_block_size_differs = format == "flac" and ffmpeg_major_version >= 9
        if method == "to_file" and not cli_flac_block_size_differs:
            validate_frames_properties(
                actual=encoder_output, expected=encoded_by_ffmpeg
            )

    @pytest.mark.parametrize(
        "format",
        [
            "mov",
            "mp4",
            "mkv",
        ],
    )
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_video_round_trip(self, tmp_path, format, method):
        # Test that decode(encode(decode(frames))) == decode(frames)
        source_decoder = VideoDecoder(str(TEST_SRC_2_720P.path))
        source_frames = source_decoder.get_frames_in_range(start=0, stop=30).data
        frame_rate = source_decoder.metadata.average_fps

        enc, encoder_output, open_kwargs = self._create_encoder(
            method, tmp_path, format
        )
        video = enc.add_video(
            height=source_frames.shape[2],
            width=source_frames.shape[3],
            frame_rate=frame_rate,
            pixel_format="yuv444p",
            crf=0,
        )
        self._open_encoder(enc, open_kwargs)
        video.add_frames(source_frames)
        enc.close()

        round_trip_frames = (
            VideoDecoder(self._get_decoder_source(encoder_output))
            .get_frames_in_range(start=0, stop=30)
            .data
        )

        assert source_frames.shape == round_trip_frames.shape
        assert source_frames.dtype == round_trip_frames.dtype

        for s_frame, rt_frame in zip(source_frames, round_trip_frames):
            assert psnr(s_frame, rt_frame) > 30
            torch.testing.assert_close(s_frame, rt_frame, atol=2, rtol=0)

    @pytest.mark.parametrize(
        "format",
        [
            pytest.param(
                "wav",
                marks=pytest.mark.skipif(
                    ffmpeg_major_version == 4,
                    reason="Swresample with FFmpeg 4 doesn't work on wav files",
                ),
            ),
            "flac",
        ],
    )
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_audio_round_trip(self, method, format, tmp_path):
        # Check that decode(encode(samples)) == samples on lossless formats
        asset = NASA_AUDIO_MP3
        source_samples = AudioDecoder(str(asset.path)).get_all_samples().data
        sample_rate = asset.sample_rate
        num_channels = source_samples.shape[0]

        enc, encoder_output, open_kwargs = self._create_encoder(
            method, tmp_path, format
        )
        audio = enc.add_audio(sample_rate=sample_rate, num_channels=num_channels)
        self._open_encoder(enc, open_kwargs)
        audio.add_samples(source_samples)
        enc.close()

        decoded = AudioDecoder(
            self._get_decoder_source(encoder_output)
        ).get_all_samples()

        rtol, atol = (0, 1e-4) if format == "wav" else (None, None)
        torch.testing.assert_close(decoded.data, source_samples, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "format",
        [
            "mov",
            "mp4",
            "avi",
            "mkv",
            "flv",
        ],
    )
    def test_video_to_file_vs_to_file_like(self, tmp_path, format):
        # Test that to_file and to_file_like produce the same results
        source_decoder = VideoDecoder(str(TEST_SRC_2_720P.path))
        source_frames = source_decoder.get_frames_in_range(start=0, stop=10).data
        frame_rate = source_decoder.metadata.average_fps

        pixel_format = "yuv420p" if format in ("avi", "flv") else "yuv444p"
        crf = 0

        # Encode via to_file
        enc_file = Encoder()
        file_path = tmp_path / f"output_file.{format}"
        video_file = enc_file.add_video(
            height=source_frames.shape[2],
            width=source_frames.shape[3],
            frame_rate=frame_rate,
            pixel_format=pixel_format,
            crf=crf,
        )
        enc_file.open_file(file_path)
        video_file.add_frames(source_frames)
        enc_file.close()

        # Encode via to_file_like
        enc_fl = Encoder()
        file_like = io.BytesIO()
        video_fl = enc_fl.add_video(
            height=source_frames.shape[2],
            width=source_frames.shape[3],
            frame_rate=frame_rate,
            pixel_format=pixel_format,
            crf=crf,
        )
        enc_fl.open_file_like(file_like, format=format)
        video_fl.add_frames(source_frames)
        enc_fl.close()

        decoded_from_file = (
            VideoDecoder(str(file_path)).get_frames_in_range(start=0, stop=10).data
        )
        decoded_from_file_like = (
            VideoDecoder(file_like.getvalue())
            .get_frames_in_range(start=0, stop=10)
            .data
        )
        torch.testing.assert_close(
            decoded_from_file, decoded_from_file_like, atol=0, rtol=0
        )

    @pytest.mark.parametrize(
        "format",
        [
            pytest.param(
                "mp3",
                marks=pytest.mark.skipif(
                    IS_WINDOWS and ffmpeg_major_version <= 5,
                    reason="Encoding mp3 on Windows is weirdly buggy",
                ),
            ),
            pytest.param(
                "wav",
                marks=pytest.mark.skipif(
                    ffmpeg_major_version == 4,
                    reason="Swresample with FFmpeg 4 doesn't work on wav files",
                ),
            ),
            "flac",
        ],
    )
    def test_audio_to_file_vs_to_file_like(self, tmp_path, format):
        # Test that to_file and to_file_like produce the same results
        asset = NASA_AUDIO_MP3
        source_samples = AudioDecoder(str(asset.path)).get_all_samples().data
        sample_rate = asset.sample_rate
        num_channels = source_samples.shape[0]

        # Encode via to_file
        enc_file = Encoder()
        file_path = tmp_path / f"output_file.{format}"
        audio_file = enc_file.add_audio(
            sample_rate=sample_rate, num_channels=num_channels
        )
        enc_file.open_file(file_path)
        audio_file.add_samples(source_samples)
        enc_file.close()

        # Encode via to_file_like
        enc_fl = Encoder()
        file_like = io.BytesIO()
        audio_fl = enc_fl.add_audio(sample_rate=sample_rate, num_channels=num_channels)
        enc_fl.open_file_like(file_like, format=format)
        audio_fl.add_samples(source_samples)
        enc_fl.close()

        if IS_WINDOWS_WITH_FFMPEG_LE_70 and format == "mp3":
            # We're getting a "Could not open input file" on Windows mp3
            # files when decoding.
            # TODO: https://github.com/pytorch/torchcodec/issues/837
            return

        decoded_from_file = AudioDecoder(str(file_path)).get_all_samples().data
        decoded_from_file_like = (
            AudioDecoder(file_like.getvalue()).get_all_samples().data
        )
        torch.testing.assert_close(
            decoded_from_file, decoded_from_file_like, atol=0, rtol=0
        )

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    @pytest.mark.parametrize(
        "device",
        (
            "cpu",
            pytest.param(
                "cuda",
                marks=[
                    pytest.mark.needs_cuda,
                    pytest.mark.skipif(
                        in_fbcode(), reason="NVENC not available in fbcode"
                    ),
                    pytest.mark.skipif(
                        ffmpeg_major_version in (4, 5),
                        reason="CUDA + FFmpeg 4 and 5 test is flaky",
                    ),
                ],
            ),
        ),
    )
    def test_video_contiguity(self, method, tmp_path, device):
        # Ensure that 2 sets of video frames with the same pixel values are encoded
        # in the same way, regardless of their memory layout. Here we encode 2 equal
        # frame tensors, one is contiguous while the other is non-contiguous.
        num_frames, channels, height, width = 5, 3, 256, 256
        contiguous_frames = (
            torch.randint(
                0, 256, size=(num_frames, channels, height, width), dtype=torch.uint8
            )
            .contiguous()
            .to(device)
        )
        assert contiguous_frames.is_contiguous()

        # Permute NCHW to NHWC, then update the memory layout, then permute back
        non_contiguous_frames = (
            contiguous_frames.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
        )
        assert not non_contiguous_frames.is_contiguous()
        assert non_contiguous_frames.is_contiguous(memory_format=torch.channels_last)

        torch.testing.assert_close(
            contiguous_frames, non_contiguous_frames, rtol=0, atol=0
        )

        def encode(frames):
            enc, encoder_output, open_kwargs = self._create_encoder(
                method, tmp_path, "mp4"
            )
            add_video_kwargs = dict(
                height=height,
                width=width,
                frame_rate=30,
                device=device,
                crf=0,
            )
            if device == "cpu":
                add_video_kwargs["pixel_format"] = "yuv444p"
            video = enc.add_video(**add_video_kwargs)
            self._open_encoder(enc, open_kwargs)
            video.add_frames(frames)
            enc.close()
            source = self._get_decoder_source(encoder_output)
            if isinstance(source, str):
                with open(source, "rb") as f:
                    return torch.frombuffer(f.read(), dtype=torch.uint8)
            return torch.frombuffer(source, dtype=torch.uint8)

        encoded_from_contiguous = encode(contiguous_frames)
        encoded_from_non_contiguous = encode(non_contiguous_frames)

        torch.testing.assert_close(
            encoded_from_contiguous, encoded_from_non_contiguous, rtol=0, atol=0
        )

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_audio_contiguity(self, method, tmp_path):
        # Ensure that 2 waveforms with the same values are encoded in the same
        # way, regardless of their memory layout. Here we encode 2 equal
        # waveforms, one is row-aligned while the other is column-aligned.
        num_samples = 10_000
        contiguous_samples = torch.rand(2, num_samples).contiguous()
        assert contiguous_samples.stride() == (num_samples, 1)

        non_contiguous_samples = contiguous_samples.T.contiguous().T
        assert non_contiguous_samples.stride() == (1, 2)

        torch.testing.assert_close(
            contiguous_samples, non_contiguous_samples, rtol=0, atol=0
        )

        def encode(samples):
            enc, encoder_output, open_kwargs = self._create_encoder(
                method, tmp_path, "flac"
            )
            audio = enc.add_audio(
                sample_rate=16_000,
                num_channels=2,
            )
            self._open_encoder(enc, open_kwargs)
            audio.add_samples(samples)
            enc.close()
            source = self._get_decoder_source(encoder_output)
            if isinstance(source, str):
                with open(source, "rb") as f:
                    return torch.frombuffer(f.read(), dtype=torch.uint8)
            return torch.frombuffer(source, dtype=torch.uint8)

        encoded_from_contiguous = encode(contiguous_samples)
        encoded_from_non_contiguous = encode(non_contiguous_samples)

        torch.testing.assert_close(
            encoded_from_contiguous, encoded_from_non_contiguous, rtol=0, atol=0
        )

    @staticmethod
    def _get_video_metadata(file_path, fields):
        """Helper function to get video metadata from a file using ffprobe."""
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                f"stream={','.join(fields)}",
                "-of",
                "default=noprint_wrappers=1",
                str(file_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True,
        )
        metadata = {}
        for line in result.stdout.strip().split("\n"):
            if "=" in line:
                key, value = line.split("=", 1)
                metadata[key] = value
        assert all(field in metadata for field in fields)
        return metadata

    @staticmethod
    def _get_frames_info(file_path, fields):
        """Helper function to get frame info (pts, dts, etc.) using ffprobe."""
        parsed = call_ffprobe(
            [
                "-select_streams",
                "v:0",
                "-show_entries",
                f"frame={','.join(fields)}",
                str(file_path),
            ]
        )
        frames = parsed["frames"]
        assert all(field in frame for field in fields for frame in frames)
        return frames

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_add_audio_out_sample_rate(self, tmp_path, method):
        in_sample_rate = 44_100
        source_samples = torch.rand(1, 10_000)

        enc, encoder_output, open_kwargs = self._create_encoder(method, tmp_path, "mkv")
        # Stream 0: 44100 -> 48000
        audio_upsample = enc.add_audio(
            sample_rate=in_sample_rate,
            num_channels=1,
            out_sample_rate=48_000,
        )
        # Stream 1: 44100 -> 32000
        audio_downsample = enc.add_audio(
            sample_rate=in_sample_rate,
            num_channels=1,
            out_sample_rate=32_000,
        )
        # Stream 2: 44100 -> no conversion (stays 44100)
        audio_passthrough = enc.add_audio(
            sample_rate=in_sample_rate,
            num_channels=1,
        )
        self._open_encoder(enc, open_kwargs)
        audio_upsample.add_samples(source_samples)
        audio_downsample.add_samples(source_samples)
        audio_passthrough.add_samples(source_samples)
        enc.close()

        source = self._get_decoder_source(encoder_output)

        decoded_0 = AudioDecoder(source, stream_index=0).get_all_samples()
        assert decoded_0.sample_rate == 48_000

        decoded_1 = AudioDecoder(source, stream_index=1).get_all_samples()
        assert decoded_1.sample_rate == 32_000

        decoded_2 = AudioDecoder(source, stream_index=2).get_all_samples()
        assert decoded_2.sample_rate == 44_100

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_pixel_format_errors(self, method, tmp_path):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        enc.add_video(
            height=64,
            width=64,
            frame_rate=30.0,
            pixel_format="invalid_pix_fmt",
        )
        with pytest.raises(
            RuntimeError,
            match=r"Unknown pixel format: invalid_pix_fmt[\s\S]*Supported pixel formats.*yuv420p",
        ):
            self._open_encoder(enc, open_kwargs)

        enc2, _, open_kwargs2 = self._create_encoder(method, tmp_path, "mp4")
        enc2.add_video(
            height=64,
            width=64,
            frame_rate=30.0,
            pixel_format="rgb24",
        )
        with pytest.raises(
            RuntimeError,
            match=r"Specified pixel format rgb24 is not supported[\s\S]*Supported pixel formats.*yuv420p",
        ):
            self._open_encoder(enc2, open_kwargs2)

    @needs_cuda
    @pytest.mark.skipif(in_fbcode(), reason="NVENC not available in fbcode")
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_pixel_format_gpu_override_errors(self, method, tmp_path):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        enc.add_video(
            height=64,
            width=64,
            frame_rate=30.0,
            device="cuda",
            pixel_format="yuv444p",
        )
        with pytest.raises(
            RuntimeError,
            match="Video encoding on GPU currently only supports the nv12 pixel format",
        ):
            self._open_encoder(enc, open_kwargs)

    @pytest.mark.parametrize(
        "extra_options,error",
        [
            ({"qp": -10}, "qp=-10 is out of valid range"),
            ({"qp": ""}, "Option qp expects a numeric value but got"),
            (
                {"direct-pred": "a"},
                "Option direct-pred expects a numeric value but got 'a'",
            ),
            ({"tune": "not_a_real_tune"}, "avcodec_open2 failed: Invalid argument"),
        ],
    )
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_extra_options_errors(self, method, tmp_path, extra_options, error):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        enc.add_video(
            height=64,
            width=64,
            frame_rate=30.0,
            extra_options=extra_options,
        )
        with pytest.raises(RuntimeError, match=error):
            self._open_encoder(enc, open_kwargs)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_write_frames_wrong_dtype_errors(self, method, tmp_path):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        video = enc.add_video(height=64, width=64, frame_rate=30.0)
        self._open_encoder(enc, open_kwargs)
        float_frames = torch.rand(2, 3, 64, 64, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="must have uint8 dtype"):
            video.add_frames(float_frames)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_write_frames_wrong_ndim_errors(self, method, tmp_path):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        video = enc.add_video(height=64, width=64, frame_rate=30.0)
        self._open_encoder(enc, open_kwargs)
        frames_1d = torch.randint(0, 256, (100,), dtype=torch.uint8)
        with pytest.raises(RuntimeError):
            video.add_frames(frames_1d)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_write_frames_wrong_num_channels_errors(self, method, tmp_path):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        video = enc.add_video(height=64, width=64, frame_rate=30.0)
        self._open_encoder(enc, open_kwargs)
        frames_2ch = torch.randint(0, 256, (2, 2, 64, 64), dtype=torch.uint8)
        with pytest.raises(RuntimeError):
            video.add_frames(frames_2ch)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_add_audio_invalid_sample_rate_errors(self, method, tmp_path):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "wav")
        with pytest.raises(RuntimeError, match="sample_rate must be > 0"):
            enc.add_audio(sample_rate=0, num_channels=1)

        enc2, _, open_kwargs2 = self._create_encoder(method, tmp_path, "wav")
        with pytest.raises(RuntimeError, match="sample_rate must be > 0"):
            enc2.add_audio(sample_rate=-1, num_channels=1)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_add_audio_invalid_num_channels_errors(self, method, tmp_path):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "wav")
        with pytest.raises(RuntimeError, match="num_channels must be > 0"):
            enc.add_audio(sample_rate=44100, num_channels=0)

    def test_to_file_like_custom_file_object(self, tmp_path):
        """Test to_file_like with a custom file-like object that implements write and seek."""

        class CustomFileObject:
            def __init__(self):
                self._file = io.BytesIO()

            def write(self, data):
                return self._file.write(data)

            def seek(self, offset, whence=0):
                return self._file.seek(offset, whence)

            def get_encoded_data(self):
                return self._file.getvalue()

        source_decoder = VideoDecoder(str(TEST_SRC_2_720P.path))
        source_frames = source_decoder.get_frames_in_range(start=0, stop=10).data
        frame_rate = source_decoder.metadata.average_fps

        enc = Encoder()
        file_like = CustomFileObject()
        video = enc.add_video(
            height=source_frames.shape[2],
            width=source_frames.shape[3],
            frame_rate=frame_rate,
            pixel_format="yuv444p",
            crf=0,
        )
        enc.open_file_like(file_like, format="mp4")
        video.add_frames(source_frames)
        enc.close()

        decoded_frames = (
            VideoDecoder(file_like.get_encoded_data())
            .get_frames_in_range(start=0, stop=10)
            .data
        )
        assert_tensor_close_on_at_least(
            decoded_frames, source_frames, percentage=99, atol=2
        )

    def test_to_file_like_custom_file_object_audio(self, tmp_path):
        class CustomFileObject:
            def __init__(self):
                self._file = io.BytesIO()

            def write(self, data):
                return self._file.write(data)

            def seek(self, offset, whence=0):
                return self._file.seek(offset, whence)

            def get_encoded_data(self):
                return self._file.getvalue()

        asset = NASA_AUDIO_MP3
        source_samples = AudioDecoder(str(asset.path)).get_all_samples().data
        sample_rate = asset.sample_rate
        num_channels = source_samples.shape[0]

        enc = Encoder()
        file_like = CustomFileObject()
        audio = enc.add_audio(sample_rate=sample_rate, num_channels=num_channels)
        enc.open_file_like(file_like, format="flac")
        audio.add_samples(source_samples)
        enc.close()

        decoded = AudioDecoder(file_like.get_encoded_data()).get_all_samples()
        torch.testing.assert_close(decoded.data, source_samples, rtol=0, atol=1e-4)

    def test_to_file_like_real_file_video(self, tmp_path):
        """Test to_file_like with a real file opened in binary write mode."""
        source_decoder = VideoDecoder(str(TEST_SRC_2_720P.path))
        source_frames = source_decoder.get_frames_in_range(start=0, stop=10).data
        frame_rate = source_decoder.metadata.average_fps

        file_path = tmp_path / "test_real_file.mp4"
        enc = Encoder()
        video = enc.add_video(
            height=source_frames.shape[2],
            width=source_frames.shape[3],
            frame_rate=frame_rate,
            pixel_format="yuv444p",
            crf=0,
        )
        with open(file_path, "wb") as f:
            enc.open_file_like(f, format="mp4")
            video.add_frames(source_frames)
            enc.close()

        decoded_frames = (
            VideoDecoder(str(file_path)).get_frames_in_range(start=0, stop=10).data
        )
        assert_tensor_close_on_at_least(
            decoded_frames, source_frames, percentage=99, atol=2
        )

    def test_to_file_like_real_file_audio(self, tmp_path):
        """Test to_file_like with a real file opened in binary write mode."""
        asset = NASA_AUDIO_MP3
        source_samples = AudioDecoder(str(asset.path)).get_all_samples().data
        sample_rate = asset.sample_rate
        num_channels = source_samples.shape[0]

        file_path = tmp_path / "test_real_file.flac"
        enc = Encoder()
        audio = enc.add_audio(sample_rate=sample_rate, num_channels=num_channels)
        with open(file_path, "wb") as f:
            enc.open_file_like(f, format="flac")
            audio.add_samples(source_samples)
            enc.close()

        decoded = AudioDecoder(str(file_path)).get_all_samples()
        torch.testing.assert_close(decoded.data, source_samples, rtol=0, atol=1e-4)

    def test_to_file_like_bad_methods_video(self):
        class NoWriteMethod:
            def seek(self, offset, whence=0):
                return 0

        enc = Encoder()
        enc.add_video(height=64, width=64, frame_rate=30.0)
        with pytest.raises(
            RuntimeError, match="File like object must implement a write method"
        ):
            enc.open_file_like(NoWriteMethod(), format="mp4")

        class NoSeekMethod:
            def write(self, data):
                return len(data)

        enc2 = Encoder()
        enc2.add_video(height=64, width=64, frame_rate=30.0)
        with pytest.raises(
            RuntimeError, match="File like object must implement a seek method"
        ):
            enc2.open_file_like(NoSeekMethod(), format="mp4")

    def test_to_file_like_bad_methods_audio(self):
        class NoWriteMethod:
            def seek(self, offset, whence=0):
                return 0

        enc = Encoder()
        enc.add_audio(sample_rate=44100, num_channels=1)
        with pytest.raises(
            RuntimeError, match="File like object must implement a write method"
        ):
            enc.open_file_like(NoWriteMethod(), format="wav")

        class NoSeekMethod:
            def write(self, data):
                return len(data)

        enc2 = Encoder()
        enc2.add_audio(sample_rate=44100, num_channels=1)
        with pytest.raises(
            RuntimeError, match="File like object must implement a seek method"
        ):
            enc2.open_file_like(NoSeekMethod(), format="wav")

    @needs_ffmpeg_cli
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    @pytest.mark.parametrize(
        "format,codec_spec",
        [
            ("mp4", "h264"),
            ("mp4", "hevc"),
            ("mkv", "av1"),
            ("avi", "mpeg4"),
            pytest.param(
                "webm",
                "vp9",
                marks=pytest.mark.skipif(
                    IS_WINDOWS, reason="vp9 codec not available on Windows"
                ),
            ),
        ],
    )
    def test_codec_parameter_utilized(self, tmp_path, method, format, codec_spec):
        # Test the codec parameter is utilized by using ffprobe to check the
        # encoded file's codec spec
        enc, encoder_output, open_kwargs = self._create_encoder(
            method, tmp_path, format
        )
        video = enc.add_video(height=64, width=64, frame_rate=30.0, codec=codec_spec)
        self._open_encoder(enc, open_kwargs)
        video.add_frames(torch.zeros((10, 3, 64, 64), dtype=torch.uint8))
        enc.close()

        if method == "to_file_like":
            return
        actual = self._get_video_metadata(encoder_output, fields=["codec_name"])[
            "codec_name"
        ]
        assert actual == codec_spec

    @needs_ffmpeg_cli
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    @pytest.mark.parametrize(
        "codec_spec,codec_impl",
        [
            ("h264", "libx264"),
            ("av1", "libaom-av1"),
            pytest.param(
                "vp9",
                "libvpx-vp9",
                marks=pytest.mark.skipif(
                    IS_WINDOWS, reason="vp9 codec not available on Windows"
                ),
            ),
        ],
    )
    def test_codec_spec_vs_impl_equivalence(
        self, tmp_path, method, codec_spec, codec_impl
    ):
        # Test that using codec spec gives the same result as using default
        # codec implementation.
        # We cannot directly check codec impl used, so we assert frame equality.
        frames = torch.randint(0, 256, (10, 3, 64, 64), dtype=torch.uint8)

        def encode_with_codec(codec, suffix):
            sub = tmp_path / suffix
            sub.mkdir(exist_ok=True)
            enc, encoder_output, open_kwargs = self._create_encoder(method, sub, "mp4")
            video = enc.add_video(height=64, width=64, frame_rate=30.0, codec=codec)
            self._open_encoder(enc, open_kwargs)
            video.add_frames(frames)
            enc.close()
            return encoder_output

        spec_output = encode_with_codec(codec_spec, "spec")
        impl_output = encode_with_codec(codec_impl, "impl")

        spec_decoded = (
            VideoDecoder(self._get_decoder_source(spec_output))
            .get_frames_in_range(start=0, stop=10)
            .data
        )
        impl_decoded = (
            VideoDecoder(self._get_decoder_source(impl_output))
            .get_frames_in_range(start=0, stop=10)
            .data
        )
        torch.testing.assert_close(spec_decoded, impl_decoded, rtol=0, atol=0)

    @needs_ffmpeg_cli
    @pytest.mark.parametrize(
        "profile,colorspace,color_range",
        [
            ("baseline", "bt709", "tv"),
            ("main", "bt470bg", "pc"),
            ("high", "fcc", "pc"),
        ],
    )
    def test_extra_options_utilized(self, tmp_path, profile, colorspace, color_range):
        # Test setting profile, colorspace, and color_range via extra_options
        # is utilized
        enc = Encoder()
        dest = str(tmp_path / "output.mp4")
        video = enc.add_video(
            height=64,
            width=64,
            frame_rate=30.0,
            extra_options={
                "profile": profile,
                "colorspace": colorspace,
                "color_range": color_range,
            },
        )
        enc.open_file(dest)
        video.add_frames(torch.zeros((5, 3, 64, 64), dtype=torch.uint8))
        enc.close()

        metadata = self._get_video_metadata(
            dest, fields=["profile", "color_space", "color_range"]
        )
        # Validate profile (case-insensitive, baseline is reported as
        # "Constrained Baseline")
        expected_profile = "constrained baseline" if profile == "baseline" else profile
        assert metadata["profile"].lower() == expected_profile
        assert metadata["color_space"] == colorspace
        assert metadata["color_range"] == color_range

    @pytest.mark.parametrize(
        "extra_options",
        (
            {},
            {"color_range": "pc"},
            {"colorspace": "bt709"},
            {"colorspace": "bt709", "color_range": "pc"},
        ),
    )
    def test_color_tags_describe_the_samples(self, tmp_path, extra_options):
        # The colorspace and range a stream advertises are what every decoder
        # reads it back with, so the samples have to be encoded with those.
        # swscale writes limited range BT.601 unless it is told otherwise.
        frames = torch.zeros((5, 3, 64, 64), dtype=torch.uint8)
        # Mid-tones: the extremes clip to the same place either way.
        frames[:, 0], frames[:, 1], frames[:, 2] = 0x40, 0x80, 0x60

        dest = str(tmp_path / "output.mp4")
        enc = Encoder()
        video = enc.add_video(
            height=64,
            width=64,
            frame_rate=30.0,
            # Lossless and unsubsampled, so the only thing left between these
            # frames and the ones we decode back is the color conversion.
            crf=0,
            pixel_format="yuv444p",
            extra_options=extra_options,
        )
        enc.open_file(dest)
        video.add_frames(frames)
        enc.close()

        decoded = VideoDecoder(dest).get_frames_in_range(start=0, stop=5).data
        torch.testing.assert_close(decoded, frames, rtol=0, atol=3)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    @pytest.mark.parametrize("crf", [23, 23.5, -0.9])
    def test_crf_valid_values(self, method, crf, tmp_path):
        enc, encoder_output, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        video = enc.add_video(height=64, width=64, frame_rate=30.0, crf=crf)
        self._open_encoder(enc, open_kwargs)
        video.add_frames(torch.zeros((5, 3, 64, 64), dtype=torch.uint8))
        enc.close()

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    @pytest.mark.parametrize("bit_rate", [None, 0, 44100, 999999999])
    def test_audio_bit_rate_positive_values(self, method, bit_rate, tmp_path):
        enc, encoder_output, open_kwargs = self._create_encoder(method, tmp_path, "wav")
        audio = enc.add_audio(sample_rate=44100, num_channels=1, bit_rate=bit_rate)
        self._open_encoder(enc, open_kwargs)
        audio.add_samples(torch.randn(1, 1000))
        enc.close()

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    @pytest.mark.parametrize("format", ["wav", "mp3", "flac"])
    def test_multiple_audio_formats(self, method, format, tmp_path):
        if IS_WINDOWS and format == "mp3":
            pytest.skip("mp3 encoding not supported on Windows")
        enc, encoder_output, open_kwargs = self._create_encoder(
            method, tmp_path, format
        )
        audio = enc.add_audio(sample_rate=44100, num_channels=1)
        self._open_encoder(enc, open_kwargs)
        audio.add_samples(torch.randn(1, 10_000))
        enc.close()

        decoded = AudioDecoder(
            self._get_decoder_source(encoder_output)
        ).get_all_samples()
        assert decoded.sample_rate == 44100
        assert decoded.data.shape[0] == 1

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_non_integer_frame_rate(self, method, tmp_path):
        enc, encoder_output, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        video = enc.add_video(height=64, width=64, frame_rate=29.97)
        self._open_encoder(enc, open_kwargs)
        video.add_frames(torch.zeros((10, 3, 64, 64), dtype=torch.uint8))
        enc.close()

        decoded_decoder = VideoDecoder(self._get_decoder_source(encoder_output))
        assert abs(decoded_decoder.metadata.average_fps - 29.97) < 0.01

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_preset_parameter(self, method, tmp_path):
        enc, encoder_output, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        video = enc.add_video(height=64, width=64, frame_rate=30.0, preset="ultrafast")
        self._open_encoder(enc, open_kwargs)
        video.add_frames(torch.zeros((5, 3, 64, 64), dtype=torch.uint8))
        enc.close()

        decoded = (
            VideoDecoder(self._get_decoder_source(encoder_output))
            .get_frames_in_range(start=0, stop=5)
            .data
        )
        assert decoded.shape == (5, 3, 64, 64)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_invalid_codec_name_errors(self, method, tmp_path):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        enc.add_video(
            height=64,
            width=64,
            frame_rate=30.0,
            codec="invalid_codec_name",
        )
        with pytest.raises(
            RuntimeError,
            match=r"Video codec invalid_codec_name not found.",
        ):
            self._open_encoder(enc, open_kwargs)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_crf_out_of_range_errors(self, method, tmp_path):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        enc.add_video(height=64, width=64, frame_rate=30.0, crf=-10)
        with pytest.raises(RuntimeError, match=r"crf=-10 is out of valid range"):
            self._open_encoder(enc, open_kwargs)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_invalid_preset_errors(self, method, tmp_path):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp4")
        enc.add_video(height=64, width=64, frame_rate=30.0, preset="fake_preset")
        with pytest.raises(
            RuntimeError,
            match=r"avcodec_open2 failed: Invalid argument",
        ):
            self._open_encoder(enc, open_kwargs)

    @pytest.mark.skipif(
        ffmpeg_major_version == 4,
        reason="On FFmpeg 4  hitting a truncated packet results in AVERROR_INVALIDDATA, which torchcodec does not handle.",
    )
    @pytest.mark.parametrize("format", ["mp4", "mov"])
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_fragmented_mp4_truncation(self, format, method, tmp_path):
        # Test that Encoder can write fragmented files using movflags.
        # Fragmented files store metadata interleaved with data rather than
        # all at the end, making them decodable even if writing is interrupted.
        source_decoder = VideoDecoder(str(TEST_SRC_2_720P.path))
        source_frames = source_decoder.get_frames_in_range(start=0, stop=30).data
        frame_rate = source_decoder.metadata.average_fps

        enc, encoder_output, open_kwargs = self._create_encoder(
            method, tmp_path, format
        )
        video = enc.add_video(
            height=source_frames.shape[2],
            width=source_frames.shape[3],
            frame_rate=frame_rate,
            extra_options={"movflags": "+frag_keyframe+empty_moov"},
        )
        self._open_encoder(enc, open_kwargs)
        video.add_frames(source_frames)
        enc.close()

        source = self._get_decoder_source(encoder_output)

        reference_decoder = VideoDecoder(source)
        reference_frames = [reference_decoder.get_frame_at(i) for i in range(10)]

        # Truncate the file to simulate interrupted write
        if isinstance(encoder_output, io.BytesIO):
            full_content = encoder_output.getvalue()
            truncated_size = int(len(full_content) * 0.5)
            truncated_source = full_content[:truncated_size]
        else:
            with open(encoder_output, "rb") as f:
                full_content = f.read()
            truncated_size = int(len(full_content) * 0.5)
            with open(encoder_output, "wb") as f:
                f.write(full_content[:truncated_size])
            truncated_source = str(encoder_output)

        # Decode the truncated file and verify first 10 frames match reference
        truncated_decoder = VideoDecoder(truncated_source)
        assert len(truncated_decoder) >= 10
        for i in range(10):
            truncated_frame = truncated_decoder.get_frame_at(i)
            torch.testing.assert_close(
                truncated_frame.data, reference_frames[i].data, atol=0, rtol=0
            )

    @needs_ffmpeg_cli
    @needs_cuda
    @pytest.mark.skipif(in_fbcode(), reason="NVENC not available in fbcode")
    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    @pytest.mark.parametrize(
        ("format", "codec"),
        [
            ("mov", None),  # will default to h264_nvenc
            ("mov", "h264_nvenc"),
            ("avi", "h264_nvenc"),
            ("mp4", "hevc_nvenc"),  # use non-default codec
            pytest.param(
                "mkv",
                "av1_nvenc",
                marks=[
                    pytest.mark.skipif(
                        IN_GITHUB_CI, reason="av1_nvenc is not supported on CI"
                    ),
                    pytest.mark.skipif(
                        ffmpeg_major_version == 4,
                        reason="av1_nvenc is not supported on FFmpeg 4",
                    ),
                ],
            ),
        ],
    )
    # We test the color space and color range parameters in this test, because
    # we are required to define matrices specific to these specs for color conversion, see note:
    # [RGB -> YUV Color Conversion, limited color range]
    # BT.601, BT.709, BT.2020
    @pytest.mark.parametrize("color_space", ("bt470bg", "bt709", "bt2020nc", None))
    # Full/PC range, Limited/TV range
    @pytest.mark.parametrize("color_range", ("pc", "tv", None))
    @pytest.mark.skipif(
        IS_WINDOWS and ffmpeg_major_version == 8,
        reason="against_cli tests fail on Windows with FFmpeg 8",
    )
    def test_nvenc_against_ffmpeg_cli(
        self, tmp_path, method, format, codec, color_space, color_range
    ):
        # TODO-VideoEncoder: (P2) Investigate why FFmpeg 4 and 6 fail with
        # non-default color space and range.
        # See https://github.com/meta-pytorch/torchcodec/issues/1140
        if ffmpeg_major_version in (4, 5, 6) and not (
            color_space == "bt470bg" and color_range == "tv"
        ):
            pytest.skip(
                "Non-default color space and range have lower accuracy on FFmpeg 4 and 6"
            )

        # Encode with FFmpeg CLI using nvenc codecs
        device = "cuda"
        qp = 1  # Use near lossless encoding to reduce noise and support av1_nvenc
        source_frames = (
            VideoDecoder(str(TEST_SRC_2_720P.path))
            .get_frames_in_range(start=0, stop=30)
            .data.to(device)
        )
        frame_rate = 30

        temp_raw_path = str(tmp_path / "temp_input.raw")
        with open(temp_raw_path, "wb") as f:
            f.write(source_frames.permute(0, 2, 3, 1).cpu().numpy().tobytes())

        ffmpeg_encoded_path = str(tmp_path / f"ffmpeg_nvenc_output.{format}")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",  # Input format
            "-s",
            f"{source_frames.shape[3]}x{source_frames.shape[2]}",
            "-r",
            str(frame_rate),
            "-i",
            temp_raw_path,
            # CLI requires explicit codec for nvenc
            # Encoder will default to h264_nvenc since the frames are
            # on GPU.
            "-c:v",
            codec if codec is not None else "h264_nvenc",
            "-pix_fmt",
            "nv12",  # Output format is always NV12
            "-qp",
            str(qp),
        ]
        if color_space:
            ffmpeg_cmd.extend(["-colorspace", color_space])
        if color_range:
            ffmpeg_cmd.extend(["-color_range", color_range])
        ffmpeg_cmd.append(ffmpeg_encoded_path)
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)

        enc, encoder_output, open_kwargs = self._create_encoder(
            method, tmp_path, format
        )
        extra_options = {"qp": qp}
        if color_space:
            extra_options["colorspace"] = color_space
        if color_range:
            extra_options["color_range"] = color_range
        video = enc.add_video(
            height=source_frames.shape[2],
            width=source_frames.shape[3],
            frame_rate=frame_rate,
            device=device,
            codec=codec,
            extra_options=extra_options,
        )
        self._open_encoder(enc, open_kwargs)
        video.add_frames(source_frames)
        enc.close()

        ffmpeg_frames = (
            VideoDecoder(ffmpeg_encoded_path).get_frames_in_range(start=0, stop=30).data
        )
        encoder_frames = (
            VideoDecoder(self._get_decoder_source(encoder_output))
            .get_frames_in_range(start=0, stop=30)
            .data
        )

        assert ffmpeg_frames.shape[0] == encoder_frames.shape[0]
        for ff_frame, enc_frame in zip(ffmpeg_frames, encoder_frames):
            assert psnr(ff_frame, enc_frame) > 25
            assert_tensor_close_on_at_least(ff_frame, enc_frame, percentage=96, atol=2)

        if method == "to_file":
            metadata_fields = ["pix_fmt", "color_range", "color_space"]
            ffmpeg_metadata = self._get_video_metadata(
                ffmpeg_encoded_path, metadata_fields
            )
            encoder_metadata = self._get_video_metadata(
                str(encoder_output), metadata_fields
            )
            # pix_fmt nv12 is stored as yuv420p in metadata, unless full
            # range (pc) is used. In that case, h264 and hevc NVENC codecs
            # will use yuvj420p automatically.
            if color_range == "pc" and codec != "av1_nvenc":
                expected_pix_fmt = "yuvj420p"
            else:
                # av1_nvenc does not utilize the yuvj420p pixel format
                expected_pix_fmt = "yuv420p"
            assert (
                encoder_metadata["pix_fmt"]
                == ffmpeg_metadata["pix_fmt"]
                == expected_pix_fmt
            )
            assert encoder_metadata["color_range"] == ffmpeg_metadata["color_range"]
            assert encoder_metadata["color_space"] == ffmpeg_metadata["color_space"]
            # Default values vary by codec, so we only assert when
            # color_range and color_space are not None.
            if color_range is not None:
                # FFmpeg and torchcodec encode color_range as 'unknown' for
                # mov and avi when color_range='tv' and color_space=None on
                # FFmpeg >= 7. Since this failure is rare, I suspect its a bug
                # related to these older container formats on newer FFmpeg
                # versions.
                if not (
                    ffmpeg_major_version >= 7
                    and color_range == "tv"
                    and color_space is None
                    and format in ("mov", "avi")
                ):
                    assert color_range == encoder_metadata["color_range"]
            if color_space is not None:
                assert color_space == encoder_metadata["color_space"]

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_write_samples_wrong_dtype_errors(self, method, tmp_path):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "wav")
        audio = enc.add_audio(sample_rate=44100, num_channels=1)
        self._open_encoder(enc, open_kwargs)
        float64_samples = torch.randn(1, 1000, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="must have float32 dtype"):
            audio.add_samples(float64_samples)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_write_samples_wrong_ndim_errors(self, method, tmp_path):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "wav")
        audio = enc.add_audio(sample_rate=44100, num_channels=1)
        self._open_encoder(enc, open_kwargs)
        samples_3d = torch.randn(1, 1, 1000)
        with pytest.raises(RuntimeError, match="must have 2 dimensions"):
            audio.add_samples(samples_3d)
        samples_1d = torch.randn(1000)
        with pytest.raises(RuntimeError, match="must have 2 dimensions"):
            audio.add_samples(samples_1d)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_add_audio_unsupported_num_channels_errors(self, method, tmp_path):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp3")
        enc.add_audio(sample_rate=44100, num_channels=1, out_num_channels=3)
        avcodec_open2_failed_msg = "avcodec_open2 failed: Invalid argument"
        match = (
            avcodec_open2_failed_msg
            if IS_WINDOWS_WITH_FFMPEG_LE_70
            else re.escape("Desired number of channels (3) is not supported")
        )
        with pytest.raises(RuntimeError, match=match):
            self._open_encoder(enc, open_kwargs)

        sub = tmp_path / "ten_ch"
        sub.mkdir()
        enc2, _, open_kwargs2 = self._create_encoder(method, sub, "wav")
        with pytest.raises(RuntimeError, match="Trying to encode 10 channels"):
            enc2.add_audio(sample_rate=44100, num_channels=10)

    @pytest.mark.parametrize("method", ("to_file", "to_file_like"))
    def test_add_audio_invalid_out_sample_rate_errors(self, method, tmp_path):
        enc, _, open_kwargs = self._create_encoder(method, tmp_path, "mp3")
        enc.add_audio(
            sample_rate=44100,
            num_channels=1,
            out_sample_rate=10,
        )
        avcodec_open2_failed_msg = "avcodec_open2 failed: Invalid argument"
        with pytest.raises(
            RuntimeError,
            match=(
                avcodec_open2_failed_msg
                if IS_WINDOWS_WITH_FFMPEG_LE_70
                else "invalid sample rate=10"
            ),
        ):
            self._open_encoder(enc, open_kwargs)


_cpu_and_cuda = ("cpu", pytest.param("cuda", marks=pytest.mark.needs_cuda))


_image_encoders = (
    pytest.param(PngEncoder, "cpu", marks=pytest.mark.needs_png, id="png"),
    pytest.param(JpegEncoder, "cpu", marks=pytest.mark.needs_jpeg, id="jpeg"),
    pytest.param(
        JpegEncoder,
        "cuda",
        marks=(pytest.mark.needs_jpeg, pytest.mark.needs_cuda),
        id="jpeg_cuda",
    ),
)


class TestImageEncoders:
    def _decode(self, asset, mode):
        # Decode an asset into a CHW uint8 tensor to use as encoder input.
        return decode_png(asset.path, mode=mode)

    def _to_bytes(self, encoder, **kwargs):
        # Encode into an in-memory file-like and return the raw bytes as a 1D
        # uint8 tensor, so the codec-specific tests below can inspect the output.
        buf = io.BytesIO()
        encoder.to_file_like(buf, **kwargs)
        return torch.frombuffer(buf.getvalue(), dtype=torch.uint8)

    # ===== destination handling, all codecs =====

    @pytest.mark.parametrize("Encoder, device", _image_encoders)
    def test_dest_variants_match(self, Encoder, device, tmp_path):
        # All three destinations must produce identical bytes: a file path, a
        # file-like object, and a tensor
        source = self._decode(GRADIENT_PNG, "RGB").to(device)

        path = tmp_path / "out"
        Encoder(source).to_file(path)
        from_path = torch.frombuffer(path.read_bytes(), dtype=torch.uint8)

        from_file_like = self._to_bytes(Encoder(source))
        from_tensor = Encoder(source).to_tensor().cpu()

        torch.testing.assert_close(from_path, from_file_like, rtol=0, atol=0)
        torch.testing.assert_close(from_path, from_tensor, rtol=0, atol=0)

    @pytest.mark.parametrize("Encoder, device", _image_encoders)
    def test_dest_str_and_pathlib(self, Encoder, device, tmp_path):
        # to_file accepts both str and pathlib.Path.
        source = self._decode(GRADIENT_PNG, "RGB").to(device)
        as_str = tmp_path / "str_out"
        as_path = tmp_path / "path_out"
        Encoder(source).to_file(str(as_str))
        Encoder(source).to_file(as_path)
        torch.testing.assert_close(
            torch.frombuffer(as_str.read_bytes(), dtype=torch.uint8),
            torch.frombuffer(as_path.read_bytes(), dtype=torch.uint8),
            rtol=0,
            atol=0,
        )

    @pytest.mark.parametrize("Encoder, device", _image_encoders)
    def test_dest_open_file_object(self, Encoder, device, tmp_path):
        # A real open file (in binary write mode) is a valid to_file_like dest.
        source = self._decode(GRADIENT_PNG, "RGB").to(device)
        path = tmp_path / "out"
        with open(path, "wb") as f:
            Encoder(source).to_file_like(f)
        torch.testing.assert_close(
            torch.frombuffer(path.read_bytes(), dtype=torch.uint8),
            self._to_bytes(Encoder(source)),
            rtol=0,
            atol=0,
        )

    # ===== PNG =====

    @needs_png
    @pytest.mark.parametrize(
        "asset, mode", ((GRADIENT_PNG, "RGB"), (GRAYSCALE_PNG, "GRAY"))
    )
    @pytest.mark.parametrize("compression_level", (0, 6, 9))
    def test_round_trip_png(self, asset, mode, compression_level):
        source = self._decode(asset, mode)
        encoded = self._to_bytes(
            PngEncoder(source), compression_level=compression_level
        )

        assert encoded.dtype == torch.uint8
        assert encoded.ndim == 1
        # PNG file signature.
        assert encoded[:8].tolist() == [137, 80, 78, 71, 13, 10, 26, 10]

        # PNG is lossless, so the round-trip must be exact at any compression level.
        torch.testing.assert_close(
            decode_png(encoded, mode=mode), source, rtol=0, atol=0
        )

    @needs_png
    def test_against_pil_png(self):
        source = self._decode(GRADIENT_PNG, "RGB")
        encoded = self._to_bytes(PngEncoder(source))

        pil_img = Image.open(io.BytesIO(encoded.numpy().tobytes()))
        assert pil_img.format == "PNG"
        pil_tensor = torch.from_numpy(np.asarray(pil_img).copy()).permute(2, 0, 1)
        torch.testing.assert_close(pil_tensor, source, rtol=0, atol=0)

    @needs_png
    def test_compression_level_affects_size_png(self):
        source = self._decode(GRADIENT_PNG, "RGB")
        least = self._to_bytes(PngEncoder(source), compression_level=0).numel()
        most = self._to_bytes(PngEncoder(source), compression_level=9).numel()
        assert least > most

    @needs_png
    def test_default_compression_level_png(self):
        source = self._decode(GRADIENT_PNG, "RGB")
        torch.testing.assert_close(
            self._to_bytes(PngEncoder(source)),
            self._to_bytes(PngEncoder(source), compression_level=6),
            rtol=0,
            atol=0,
        )

    @needs_png
    @pytest.mark.parametrize("compression_level", (-1, 10))
    def test_bad_compression_level_png(self, compression_level):
        source = self._decode(GRADIENT_PNG, "RGB")
        with pytest.raises(RuntimeError, match="between 0 and 9"):
            PngEncoder(source).to_file_like(
                io.BytesIO(), compression_level=compression_level
            )

    @needs_png
    def test_to_tensor_png(self):
        # to_tensor returns the encoded bytes as a 1-D uint8 CPU tensor.
        source = self._decode(GRADIENT_PNG, "RGB")

        encoded = PngEncoder(source).to_tensor()
        assert isinstance(encoded, torch.Tensor)
        assert encoded.dtype == torch.uint8
        assert encoded.ndim == 1
        assert encoded.device.type == "cpu"

        assert encoded[:8].tolist() == [137, 80, 78, 71, 13, 10, 26, 10]
        # PNG is lossless, so the round trip is exact.
        torch.testing.assert_close(
            decode_png(encoded, mode="RGB"), source, rtol=0, atol=0
        )

    # ===== JPEG =====

    # (quality, min_psnr): minimum acceptable round-trip PSNR (dB) per JPEG
    # quality on a smooth gradient. Higher quality preserves more, so the floor
    # rises with quality. The floors sit safely below what both the CPU (libjpeg)
    # and CUDA (nvJPEG) encoders achieve.
    _JPEG_QUALITY_AND_MIN_PSNR = ((25, 35), (75, 40), (95, 42))

    @needs_jpeg
    @pytest.mark.parametrize("device", _cpu_and_cuda)
    @pytest.mark.parametrize("quality, min_psnr", _JPEG_QUALITY_AND_MIN_PSNR)
    def test_round_trip_jpeg(self, device, quality, min_psnr):
        # Encode a CHW uint8 tensor (on CPU via libjpeg, on CUDA via nvJPEG),
        # then decode it back and check the round trip is faithful
        img = decode_jpeg(GRADIENT_JPEG.path, mode="RGB").to(device)

        encoded = self._to_bytes(JpegEncoder(img), quality=quality)
        assert encoded.dtype == torch.uint8
        assert encoded.ndim == 1

        pil_decoded = Image.open(io.BytesIO(encoded.numpy().tobytes()))
        assert pil_decoded.format == "JPEG"

        decoded = decode_jpeg(encoded, mode="RGB")
        assert decoded.shape == img.shape
        assert psnr(decoded, img.cpu()) > min_psnr

        buf = io.BytesIO()
        Image.fromarray(img.cpu().permute(1, 2, 0).numpy()).save(
            buf, format="JPEG", quality=quality
        )
        pil = decode_jpeg(
            torch.frombuffer(buf.getvalue(), dtype=torch.uint8), mode="RGB"
        )
        if device == "cpu":
            # Our CPU encoder and PIL both wrap libjpeg with the same defaults,
            # so they produce near-identical output. We compare decoded pixels
            # (byte-exactness is too fragile across libjpeg builds).
            assert_tensor_close_on_at_least(decoded, pil, percentage=99, atol=2)
        else:
            # nvJPEG is a different implementation from libjpeg (and encodes
            # 4:4:4 chroma vs PIL's default 4:2:0), so we only require perceptual
            # closeness rather than near-identical pixels.
            assert psnr(decoded, pil) > min_psnr

    @needs_jpeg
    @pytest.mark.parametrize("quality, min_psnr", _JPEG_QUALITY_AND_MIN_PSNR)
    def test_round_trip_jpeg_grayscale(self, quality, min_psnr):
        # Grayscale round trip, CPU only: nvJPEG encoding is RGB-only (see
        # test_grayscale_jpeg_cuda_errors).
        img = decode_jpeg(GRAYSCALE_JPEG.path, mode="UNCHANGED")
        assert img.shape[0] == 1

        encoded = self._to_bytes(JpegEncoder(img), quality=quality)
        decoded = decode_jpeg(encoded, mode="UNCHANGED")
        assert decoded.shape == img.shape
        assert psnr(decoded, img) > min_psnr

    @needs_jpeg
    @pytest.mark.parametrize("device", _cpu_and_cuda)
    def test_quality_affects_size_jpeg(self, device):
        img = decode_jpeg(GRADIENT_JPEG.path, mode="RGB").to(device)
        low = self._to_bytes(JpegEncoder(img), quality=10).numel()
        high = self._to_bytes(JpegEncoder(img), quality=95).numel()
        assert high > low

    @needs_jpeg
    @pytest.mark.parametrize("device", _cpu_and_cuda)
    def test_to_tensor_jpeg(self, device):
        # to_tensor returns the encoded bytes as a 1-D uint8 tensor on the same
        # device as the input (CUDA in -> CUDA out, zero-copy).
        img = decode_jpeg(GRADIENT_JPEG.path, mode="RGB").to(device)

        encoded = JpegEncoder(img).to_tensor(quality=90)
        assert isinstance(encoded, torch.Tensor)
        assert encoded.dtype == torch.uint8
        assert encoded.ndim == 1
        assert encoded.device.type == device

        pil_decoded = Image.open(io.BytesIO(encoded.cpu().numpy().tobytes()))
        assert pil_decoded.format == "JPEG"
        assert decode_jpeg(encoded.cpu(), mode="RGB").shape == img.shape

    @needs_jpeg
    @pytest.mark.parametrize("quality", (-1, 101))
    def test_bad_quality_jpeg(self, quality):
        with pytest.raises(
            ValueError,
            match="Image quality should be a positive number between 1 and 100",
        ):
            JpegEncoder(torch.zeros(3, 8, 8, dtype=torch.uint8)).to_file_like(
                io.BytesIO(), quality=quality
            )

    @needs_cuda
    @needs_jpeg
    def test_grayscale_jpeg_cuda_errors(self):
        # nvJPEG encoding only supports 3-channel RGB; grayscale must use the CPU.
        img = torch.zeros(1, 8, 8, dtype=torch.uint8, device="cuda")
        with pytest.raises(RuntimeError, match="number of channels should be 3"):
            JpegEncoder(img).to_file_like(io.BytesIO())

    # ===== shared validation - all codecs =====

    @pytest.mark.parametrize("Encoder, device", _image_encoders)
    def test_bad_dtype(self, Encoder, device):
        img = torch.zeros(3, 8, 8, dtype=torch.float32, device=device)
        with pytest.raises(RuntimeError, match="uint8"):
            Encoder(img).to_file_like(io.BytesIO())

    @pytest.mark.parametrize("Encoder, device", _image_encoders)
    @pytest.mark.parametrize("shape", ((720, 1280), (3, 3, 8, 8)))
    def test_bad_ndim(self, Encoder, device, shape):
        img = torch.zeros(shape, dtype=torch.uint8, device=device)
        with pytest.raises(RuntimeError, match="3-dimensional"):
            Encoder(img).to_file_like(io.BytesIO())

    @pytest.mark.parametrize("Encoder, device", _image_encoders)
    @pytest.mark.parametrize("num_channels", (2, 4))
    def test_bad_num_channels(self, Encoder, device, num_channels):
        # CPU encoders accept 1 or 3 channels; the CUDA (nvJPEG) path is RGB-only,
        # so its message differs. Both reject 2 and 4 channels.
        img = torch.zeros(num_channels, 8, 8, dtype=torch.uint8, device=device)
        with pytest.raises(RuntimeError, match="number of channels should be"):
            Encoder(img).to_file_like(io.BytesIO())

    @pytest.mark.parametrize("Encoder, device", _image_encoders)
    def test_bad_file_like(self, Encoder, device):
        img = torch.zeros(3, 8, 8, dtype=torch.uint8, device=device)

        class NoWriteMethod:
            def seek(self, offset, whence=0):
                return 0

        with pytest.raises(
            RuntimeError, match="File like object must implement a write method"
        ):
            Encoder(img).to_file_like(NoWriteMethod())

        class NoSeekMethod:
            def write(self, data):
                return len(data)

        with pytest.raises(
            RuntimeError, match="File like object must implement a seek method"
        ):
            Encoder(img).to_file_like(NoSeekMethod())

        with pytest.raises(
            RuntimeError, match="File like object must implement a write method"
        ):
            Encoder(img).to_file_like(3)
