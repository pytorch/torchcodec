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

from benchmark_decoders_library import (
    DecordAccurateBatch,
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
    shutil.rmtree(videos_dir_path, ignore_errors=True)
    os.makedirs(videos_dir_path)

    resolutions = ["1280x720"]
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
    decoder_dict["TorchCodec"] = TorchCodecPublic()
    decoder_dict["TorchCodec[cuda]"] = TorchCodecPublic(device="cuda")
    decoder_dict["TorchVision[video_reader]"] = TorchVision("video_reader")
    decoder_dict["TorchAudio"] = TorchAudioDecoder()
    decoder_dict["Decord"] = DecordAccurateBatch()

    # These are the number of uniform seeks we do in the seek+decode benchmark.
    num_samples = 10
    video_files_paths = list(Path(videos_dir_path).glob("*.mp4"))
    df_data = run_benchmarks(
        decoder_dict,
        video_files_paths,
        num_samples,
        num_sequential_frames_from_start=[100],
        min_runtime_seconds=30,
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

    data_json = Path(__file__).parent / "benchmark_readme_data.json"
    with open(data_json, "w") as write_file:
        json.dump(df_data, write_file, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
