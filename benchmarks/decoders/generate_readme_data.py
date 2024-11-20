# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import platform
import shutil
from pathlib import Path

import torch

from benchmark_decoders_library import (
    BatchParameters,
    DataLoaderInspiredWorkloadParameters,
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

    videos_dir_path = "/tmp/torchcodec_benchmarking_videos"
    if not os.path.exists(videos_dir_path):
        shutil.rmtree(videos_dir_path, ignore_errors=True)
        os.makedirs(videos_dir_path)

        resolutions = ["1920x1080"]
        encodings = ["libx264"]
        patterns = ["mandelbrot"]
        fpses = [60]
        gop_sizes = [600]
        durations = [120]
        pix_fmts = ["yuv420p"]
        ffmpeg_path = "ffmpeg"
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
    decoder_dict["torchcodec[cuda]"] = TorchCodecPublic(device="cuda")
    decoder_dict["torchvision[video_reader]"] = TorchVision("video_reader")
    decoder_dict["torchaudio"] = TorchAudioDecoder()

    # These are the number of uniform seeks we do in the seek+decode benchmark.
    num_samples = 10
    video_files_paths = list(Path(videos_dir_path).glob("*.mp4"))
    assert len(video_files_paths) == 2, "Expected exactly 2 videos"
    results = run_benchmarks(
        decoder_dict,
        video_files_paths,
        num_samples,
        num_sequential_frames_from_start=[100],
        min_runtime_seconds=30,
        benchmark_video_creation=False,
        dataloader_parameters=DataLoaderInspiredWorkloadParameters(
            batch_parameters=BatchParameters(batch_size=50, num_threads=10),
            resize_height=256,
            resize_width=256,
            resize_device="cuda",
        ),
    )
    data_for_writing = {
        "experiments": results,
        "system_metadata": {
            "cpu_count": os.cpu_count(),
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": str(platform.python_version()),
            "cuda": (
                str(torch.cuda.get_device_properties(0))
                if torch.cuda.is_available()
                else "not available"
            ),
        },
    }

    data_json = Path(__file__).parent / "benchmark_readme_data.json"
    with open(data_json, "w") as write_file:
        json.dump(data_for_writing, write_file, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
