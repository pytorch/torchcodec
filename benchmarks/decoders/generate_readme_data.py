# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os
import platform
import shutil
from pathlib import Path

from benchmark_decoders_library import (
    generate_videos,
    run_benchmarks,
    TorchAudioDecoder,
    TorchcodecNonCompiledWithOptions,
    TVNewAPIDecoderWithBackend,
)


def main() -> None:
    """Benchmarks the performance of a few video decoders on synthetic videos"""

    resolutions = ["640x480"]
    encodings = ["libx264"]
    fpses = [30]
    gop_sizes = [600]
    durations = [10]
    pix_fmts = ["yuv420p"]
    ffmpeg_path = "ffmpeg"
    videos_dir_path = "/tmp/torchcodec_benchmarking_videos"
    shutil.rmtree(videos_dir_path, ignore_errors=True)
    os.makedirs(videos_dir_path)
    generate_videos(
        resolutions,
        encodings,
        fpses,
        gop_sizes,
        durations,
        pix_fmts,
        ffmpeg_path,
        videos_dir_path,
    )
    video_files_paths = glob.glob(f"{videos_dir_path}/*.mp4")

    decoder_dict = {}
    decoder_dict["TorchCodec"] = TorchcodecNonCompiledWithOptions()
    decoder_dict["TorchCodec[num_threads=1]"] = TorchcodecNonCompiledWithOptions(
        num_threads=1
    )
    decoder_dict["TorchVision[backend=VideoReader]"] = TVNewAPIDecoderWithBackend(
        "video_reader"
    )
    decoder_dict["TorchAudio"] = TorchAudioDecoder()

    # These are the number of uniform seeks we do in the seek+decode benchmark.
    num_uniform_samples = 10
    df_data = run_benchmarks(
        decoder_dict,
        video_files_paths,
        num_uniform_samples,
        10,
        False,
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

    data_json = Path(__file__).parent / "benchmark_readme_data.json"
    with open(data_json, "w") as write_file:
        json.dump(df_data, write_file, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
