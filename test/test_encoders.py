import re

import pytest
import torch

from torchcodec.encoders import AudioEncoder


class TestAudioEncoder:

    def test_bad_input(self):
        with pytest.raises(ValueError, match="Expected samples to be a Tensor"):
            AudioEncoder(samples=123, sample_rate=32_000)
        with pytest.raises(ValueError, match="Expected 2D samples"):
            AudioEncoder(samples=torch.rand(10), sample_rate=32_000)
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
