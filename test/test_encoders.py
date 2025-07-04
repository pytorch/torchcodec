import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import torch
from torchcodec.decoders import AudioDecoder

from torchcodec.encoders import AudioEncoder

from .utils import (
    get_ffmpeg_major_version,
    in_fbcode,
    NASA_AUDIO_MP3,
    SINE_MONO_S32,
    TestContainerFile,
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

    frames_actual, frames_expected = (
        json.loads(
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-hide_banner",
                    "-select_streams",
                    "a:0",
                    "-show_frames",
                    "-of",
                    "json",
                    f"{f}",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
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

    for frame_index, (d_actual, d_expected) in enumerate(
        zip(frames_actual, frames_expected)
    ):
        if get_ffmpeg_major_version() >= 6:
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

    def decode(self, source) -> torch.Tensor:
        if isinstance(source, TestContainerFile):
            source = str(source.path)
        return AudioDecoder(source).get_all_samples()

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

        encoder = AudioEncoder(samples=torch.rand(2, 100), sample_rate=32_000)

        bad_path = "/bad/path.mp3"
        with pytest.raises(
            RuntimeError,
            match=f"avio_open failed. The destination file is {bad_path}, make sure it's a valid path",
        ):
            encoder.to_file(dest=bad_path)

        bad_extension = "output.bad_extension"
        with pytest.raises(RuntimeError, match="check the desired extension"):
            encoder.to_file(dest=bad_extension)

        bad_format = "bad_format"
        with pytest.raises(
            RuntimeError,
            match=re.escape(f"Check the desired format? Got format={bad_format}"),
        ):
            encoder.to_tensor(format=bad_format)

    @pytest.mark.parametrize("method", ("to_file", "to_tensor"))
    def test_bad_input_parametrized(self, method, tmp_path):
        valid_params = (
            dict(dest=str(tmp_path / "output.mp3"))
            if method == "to_file"
            else dict(format="mp3")
        )

        decoder = AudioEncoder(self.decode(NASA_AUDIO_MP3).data, sample_rate=10)
        with pytest.raises(RuntimeError, match="invalid sample rate=10"):
            getattr(decoder, method)(**valid_params)

        decoder = AudioEncoder(
            self.decode(NASA_AUDIO_MP3).data, sample_rate=NASA_AUDIO_MP3.sample_rate
        )
        with pytest.raises(RuntimeError, match="bit_rate=-1 must be >= 0"):
            getattr(decoder, method)(**valid_params, bit_rate=-1)

        bad_num_channels = 10
        decoder = AudioEncoder(torch.rand(bad_num_channels, 20), sample_rate=16_000)
        with pytest.raises(
            RuntimeError, match=f"Trying to encode {bad_num_channels} channels"
        ):
            getattr(decoder, method)(**valid_params)

        decoder = AudioEncoder(
            self.decode(NASA_AUDIO_MP3).data, sample_rate=NASA_AUDIO_MP3.sample_rate
        )
        for num_channels in (0, 3):
            with pytest.raises(
                RuntimeError,
                match=re.escape(
                    f"Desired number of channels ({num_channels}) is not supported"
                ),
            ):
                getattr(decoder, method)(**valid_params, num_channels=num_channels)

    @pytest.mark.parametrize("method", ("to_file", "to_tensor"))
    @pytest.mark.parametrize("format", ("wav", "flac"))
    def test_round_trip(self, method, format, tmp_path):
        # Check that decode(encode(samples)) == samples on lossless formats

        if get_ffmpeg_major_version() == 4 and format == "wav":
            pytest.skip("Swresample with FFmpeg 4 doesn't work on wav files")

        asset = NASA_AUDIO_MP3
        source_samples = self.decode(asset).data

        encoder = AudioEncoder(source_samples, sample_rate=asset.sample_rate)

        if method == "to_file":
            encoded_path = str(tmp_path / f"output.{format}")
            encoded_source = encoded_path
            encoder.to_file(dest=encoded_path)
        else:
            encoded_source = encoder.to_tensor(format=format)
            assert encoded_source.dtype == torch.uint8
            assert encoded_source.ndim == 1

        rtol, atol = (0, 1e-4) if format == "wav" else (None, None)
        torch.testing.assert_close(
            self.decode(encoded_source).data, source_samples, rtol=rtol, atol=atol
        )

    @pytest.mark.skipif(in_fbcode(), reason="TODO: enable ffmpeg CLI")
    @pytest.mark.parametrize("asset", (NASA_AUDIO_MP3, SINE_MONO_S32))
    @pytest.mark.parametrize("bit_rate", (None, 0, 44_100, 999_999_999))
    @pytest.mark.parametrize("num_channels", (None, 1, 2))
    @pytest.mark.parametrize("sample_rate", (8_000, 32_000))
    @pytest.mark.parametrize("format", ("mp3", "wav", "flac"))
    @pytest.mark.parametrize("method", ("to_file", "to_tensor"))
    def test_against_cli(
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

        if get_ffmpeg_major_version() == 4 and format == "wav":
            pytest.skip("Swresample with FFmpeg 4 doesn't work on wav files")

        encoded_by_ffmpeg = tmp_path / f"ffmpeg_output.{format}"
        subprocess.run(
            ["ffmpeg", "-i", str(asset.path)]
            + (["-b:a", f"{bit_rate}"] if bit_rate is not None else [])
            + (["-ac", f"{num_channels}"] if num_channels is not None else [])
            + ["-ar", f"{sample_rate}"]
            + [
                str(encoded_by_ffmpeg),
            ],
            capture_output=True,
            check=True,
        )

        encoder = AudioEncoder(self.decode(asset).data, sample_rate=asset.sample_rate)
        params = dict(
            bit_rate=bit_rate, num_channels=num_channels, sample_rate=sample_rate
        )
        if method == "to_file":
            encoded_by_us = tmp_path / f"output.{format}"
            encoder.to_file(dest=str(encoded_by_us), **params)
        else:
            encoded_by_us = encoder.to_tensor(format=format, **params)

        captured = capfd.readouterr()
        if format == "wav":
            assert "Timestamps are unset in a packet" not in captured.err
        if format == "mp3":
            assert "Queue input is backward in time" not in captured.err
        if format in ("flac", "wav"):
            assert "Encoder did not produce proper pts" not in captured.err
        if format in ("flac", "mp3"):
            assert "Application provided invalid" not in captured.err

        if sample_rate != asset.sample_rate:
            rtol, atol = 0, 1e-3
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

        samples_by_us = self.decode(encoded_by_us)
        samples_by_ffmpeg = self.decode(encoded_by_ffmpeg)
        torch.testing.assert_close(
            samples_by_us.data,
            samples_by_ffmpeg.data,
            rtol=rtol,
            atol=atol,
        )
        assert samples_by_us.pts_seconds == samples_by_ffmpeg.pts_seconds
        assert samples_by_us.duration_seconds == samples_by_ffmpeg.duration_seconds
        assert samples_by_us.sample_rate == samples_by_ffmpeg.sample_rate

        if method == "to_file":
            validate_frames_properties(actual=encoded_by_us, expected=encoded_by_ffmpeg)
        else:
            assert method == "to_tensor", "wrong test parametrization!"

    @pytest.mark.parametrize("asset", (NASA_AUDIO_MP3, SINE_MONO_S32))
    @pytest.mark.parametrize("bit_rate", (None, 0, 44_100, 999_999_999))
    @pytest.mark.parametrize("num_channels", (None, 1, 2))
    @pytest.mark.parametrize("format", ("mp3", "wav", "flac"))
    def test_to_tensor_against_to_file(
        self, asset, bit_rate, num_channels, format, tmp_path
    ):
        if get_ffmpeg_major_version() == 4 and format == "wav":
            pytest.skip("Swresample with FFmpeg 4 doesn't work on wav files")

        encoder = AudioEncoder(self.decode(asset).data, sample_rate=asset.sample_rate)

        params = dict(bit_rate=bit_rate, num_channels=num_channels)
        encoded_file = tmp_path / f"output.{format}"
        encoder.to_file(dest=str(encoded_file), **params)
        encoded_tensor = encoder.to_tensor(
            format=format, bit_rate=bit_rate, num_channels=num_channels
        )

        torch.testing.assert_close(
            self.decode(encoded_file).data, self.decode(encoded_tensor).data
        )

    def test_encode_to_tensor_long_output(self):
        # Check that we support re-allocating the output tensor when the encoded
        # data is large.
        samples = torch.rand(1, int(1e7))
        encoded_tensor = AudioEncoder(samples, sample_rate=16_000).to_tensor(
            format="flac", bit_rate=44_000
        )

        # Note: this should be in sync with its C++ counterpart for the test to
        # be meaningful.
        INITIAL_TENSOR_SIZE = 10_000_000
        assert encoded_tensor.numel() > INITIAL_TENSOR_SIZE

        torch.testing.assert_close(self.decode(encoded_tensor).data, samples)

    def test_contiguity(self):
        # Ensure that 2 waveforms with the same values are encoded in the same
        # way, regardless of their memory layout. Here we encode 2 equal
        # waveforms, one is row-aligned while the other is column-aligned.
        # TODO: Ideally we'd be testing all encoding methods here

        num_samples = 10_000  # per channel
        contiguous_samples = torch.rand(2, num_samples).contiguous()
        assert contiguous_samples.stride() == (num_samples, 1)

        params = dict(format="flac", bit_rate=44_000)
        encoded_from_contiguous = AudioEncoder(
            contiguous_samples, sample_rate=16_000
        ).to_tensor(**params)

        non_contiguous_samples = contiguous_samples.T.contiguous().T
        assert non_contiguous_samples.stride() == (1, 2)

        torch.testing.assert_close(
            contiguous_samples, non_contiguous_samples, rtol=0, atol=0
        )

        encoded_from_non_contiguous = AudioEncoder(
            non_contiguous_samples, sample_rate=16_000
        ).to_tensor(**params)

        torch.testing.assert_close(
            encoded_from_contiguous, encoded_from_non_contiguous, rtol=0, atol=0
        )

    @pytest.mark.parametrize("num_channels_input", (1, 2))
    @pytest.mark.parametrize("num_channels_output", (1, 2, None))
    @pytest.mark.parametrize("method", ("to_file", "to_tensor"))
    def test_num_channels(
        self, num_channels_input, num_channels_output, method, tmp_path
    ):
        # We just check that the num_channels parameter is respected.
        # Correctness is checked in other tests (like test_against_cli())

        sample_rate = 16_000
        source_samples = torch.rand(num_channels_input, 1_000)
        format = "mp3"

        encoder = AudioEncoder(source_samples, sample_rate=sample_rate)
        params = dict(num_channels=num_channels_output)

        if method == "to_file":
            encoded_path = str(tmp_path / f"output.{format}")
            encoded_source = encoded_path
            encoder.to_file(dest=encoded_path, **params)
        else:
            encoded_source = encoder.to_tensor(format=format, **params)

        if num_channels_output is None:
            num_channels_output = num_channels_input
        assert self.decode(encoded_source).data.shape[0] == num_channels_output

    def test_1d_samples(self):
        # smoke test making sure 1D samples are supported
        samples_1d, sample_rate = torch.rand(1000), 16_000
        samples_2d = samples_1d[None, :]

        torch.testing.assert_close(
            AudioEncoder(samples_1d, sample_rate=sample_rate).to_tensor("wav"),
            AudioEncoder(samples_2d, sample_rate=sample_rate).to_tensor("wav"),
        )
