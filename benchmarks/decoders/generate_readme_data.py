# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import platform
import shutil
from pathlib import Path

from benchmark_decoders_library import (
    generate_videos,
    retrieve_videos,
    run_benchmarks,
    TorchAudioDecoder,
    TorchCodecPublic,
    TorchVision,
)

NASA_URL = "https://download.pytorch.org/torchaudio/tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4"


def main() -> None:
    """Benchmarks the performance of a few video decoders on synthetic videos"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_run",
        help="Test run only; use small values for experiments to ensure everything works. Does not overwrite the data file.",
        action="store_true",
    )
    args = parser.parse_args()

    # The logic is clearer internally if we invert the boolean. However, we want to
    # maintain the external default that a test run is off by default.
    data_generation_run = not args.test_run

    if data_generation_run:
        resolutions = ["1280x720"]
        encodings = ["libx264"]
        patterns = ["mandelbrot"]
        fpses = [60]
        gop_sizes = [600]
        durations = [120]
        pix_fmts = ["yuv420p"]
        ffmpeg_path = "ffmpeg"
        min_runtime_seconds = 30

        # These are the number of uniform seeks we do in the seek+decode benchmark.
        num_samples = 10
    else:
        resolutions = ["640x480"]
        encodings = ["libx264"]
        patterns = ["mandelbrot"]
        fpses = [30]
        gop_sizes = [20]
        durations = [10]  # if this goes too low, we hit EOF errors in some decoders
        pix_fmts = ["yuv420p"]
        ffmpeg_path = "ffmpeg"
        min_runtime_seconds = 1

        num_samples = 4

    videos_dir_path = "/tmp/torchcodec_benchmarking_videos"
    shutil.rmtree(videos_dir_path, ignore_errors=True)
    os.makedirs(videos_dir_path)

    generate_videos(
        resolutions,
        encodings,
        patterns,
        fpses,
        gop_sizes,
        durations,
        pix_fmts,
        ffmpeg_path,
        videos_dir_path,
    )

    urls_and_dest_paths = [
        (NASA_URL, f"{videos_dir_path}/nasa_960x540_206s_30fps_yuv420p.mp4")
    ]
    retrieve_videos(urls_and_dest_paths)

    decoder_dict = {}
    decoder_dict["torchcodec"] = TorchCodecPublic()
    decoder_dict["torchvision[video_reader]"] = TorchVision("video_reader")
    decoder_dict["torchaudio"] = TorchAudioDecoder()

    video_files_paths = list(Path(videos_dir_path).glob("*.mp4"))
    df_data = run_benchmarks(
        decoder_dict,
        video_files_paths,
        num_samples,
        num_sequential_frames_from_start=[100],
        min_runtime_seconds=min_runtime_seconds,
        benchmark_video_creation=False,
    )
    df_data.append(
        {
            "cpu_count": os.cpu_count(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": str(platform.python_version()),
        }
    )

    if data_generation_run:
        data_json = Path(__file__).parent / "benchmark_readme_data.json"
        with open(data_json, "w") as write_file:
            json.dump(df_data, write_file, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
