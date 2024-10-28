# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob
import importlib.resources
import os
import shutil
from pathlib import Path

from benchmark_decoders_library import (
    generate_videos,
    plot_data,
    run_benchmarks,
    TorchAudioDecoder,
    TorchcodecNonCompiledWithOptions,
    TVNewAPIDecoderWithBackend,
)


def in_fbcode() -> bool:
    return "FB_PAR_RUNTIME_FILES" in os.environ


def get_test_resource_path(filename: str) -> str:
    if in_fbcode():
        resource = importlib.resources.files(__package__).joinpath(filename)
        with importlib.resources.as_file(resource) as path:
            return os.fspath(path)

    return str(Path(__file__).parent / f"../../test/resources/{filename}")


def main() -> None:
    """Benchmarks the performance of a few video decoders on synthetic videos"""

    resolutions = ["640x480"]
    encodings = ["libx264"]
    fpses = [30]
    gop_sizes = [600]
    durations = [10]
    pix_fmts = ["yuv420p"]
    ffmpeg_path = "/usr/local/bin/ffmpeg"
    videos_path = "/tmp/videos"
    shutil.rmtree(videos_path)
    os.makedirs(videos_path)
    generate_videos(
        resolutions,
        encodings,
        fpses,
        gop_sizes,
        durations,
        pix_fmts,
        ffmpeg_path,
        videos_path,
    )
    video_paths = glob.glob(f"{videos_path}/*.mp4")

    decoder_dict = {}
    decoder_dict["TorchCodec"] = TorchcodecNonCompiledWithOptions()
    decoder_dict["TorchCodec[num_threads=1]"] = TorchcodecNonCompiledWithOptions(
        num_threads=1
    )
    decoder_dict["TorchVision[backend=VideoReader]"] = TVNewAPIDecoderWithBackend(
        "video_reader"
    )
    decoder_dict["TorchAudio"] = TorchAudioDecoder()

    output_png = Path(__file__) / "benchmark_readme_chart.png"
    # These are the number of uniform seeks we do in the seek+decode benchmark.
    num_uniform_samples = 10
    df_data = run_benchmarks(
        decoder_dict,
        video_paths,
        num_uniform_samples,
        10,
        False,
    )
    plot_data(df_data, output_png)


if __name__ == "__main__":
    main()
