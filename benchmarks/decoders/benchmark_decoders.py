# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib.resources
import os
from pathlib import Path

from benchmark_decoders_library import (
    DecordNonBatchDecoderAccurateSeek,
    plot_data,
    run_benchmarks,
    TorchAudioDecoder,
    TorchcodecCompiled,
    TorchCodecNonCompiledBatch,
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
    """Benchmarks the performance of a few video decoders"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bm_video_creation",
        help="Benchmark large video creation memory",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--verbose",
        help="Show verbose output",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--bm_video_speed_min_run_seconds",
        help="Benchmark minimum run time, in seconds, to wait per datapoint",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--bm_video_paths",
        help="Comma-separated paths to videos that you want to benchmark.",
        type=str,
        default=get_test_resource_path("nasa_13013.mp4"),
    )
    parser.add_argument(
        "--decoders",
        help=(
            "Comma-separated list of decoders to benchmark. "
            "Choices are torchcodec, torchaudio, torchvision, decord, tcoptions:num_threads=1+color_conversion_library=filtergraph, torchcodec_compiled"
            "For torchcodec, you can specify options with tcoptions:<plus-separated-options>. "
        ),
        type=str,
        default="decord,tcoptions:,torchvision,torchaudio,torchcodec_compiled,tcoptions:num_threads=1",
    )
    parser.add_argument(
        "--bm_video_dir",
        help="Directory where video files reside. We will run benchmarks on all .mp4 files in this directory.",
        type=str,
        default="",
    )
    parser.add_argument(
        "--plot_path",
        help="Path where the generated plot is stored, if non-empty",
        type=str,
        default="",
    )

    args = parser.parse_args()
    decoders = set(args.decoders.split(","))

    # These are the PTS values we want to extract from the small video.
    num_uniform_samples = 10

    decoder_dict = {}
    for decoder in decoders:
        if decoder == "decord":
            decoder_dict["DecordNonBatchDecoderAccurateSeek"] = (
                DecordNonBatchDecoderAccurateSeek()
            )
        elif decoder == "torchcodec":
            decoder_dict["TorchCodecNonCompiled"] = TorchcodecNonCompiledWithOptions()
        elif decoder == "torchcodec_compiled":
            decoder_dict["TorchcodecCompiled"] = TorchcodecCompiled()
        elif decoder == "torchvision":
            decoder_dict["TVNewAPIDecoderWithBackendVideoReader"] = (
                # We don't compare TorchVision's "pyav" backend because it doesn't support
                # accurate seeks.
                TVNewAPIDecoderWithBackend("video_reader")
            )
        elif decoder == "torchaudio":
            decoder_dict["TorchAudioDecoder"] = TorchAudioDecoder()
        elif decoder.startswith("tcbatchoptions:"):
            options = decoder[len("tcbatchoptions:") :]
            kwargs_dict = {}
            for item in options.split("+"):
                if item.strip() == "":
                    continue
                k, v = item.split("=")
                kwargs_dict[k] = v
            decoder_dict["TorchCodecNonCompiledBatch:" + options] = (
                TorchCodecNonCompiledBatch(**kwargs_dict)
            )
        elif decoder.startswith("tcoptions:"):
            options = decoder[len("tcoptions:") :]
            kwargs_dict = {}
            for item in options.split("+"):
                if item.strip() == "":
                    continue
                k, v = item.split("=")
                kwargs_dict[k] = v
            decoder_dict["TorchcodecNonCompiled:" + options] = (
                TorchcodecNonCompiledWithOptions(**kwargs_dict)
            )
    video_paths = args.bm_video_paths.split(",")
    if args.bm_video_dir:
        video_paths = []
        for entry in os.scandir(args.bm_video_dir):
            if entry.is_file() and entry.name.endswith(".mp4"):
                video_paths.append(entry.path)

    df_data = run_benchmarks(
        decoder_dict,
        video_paths,
        num_uniform_samples,
        args.bm_video_speed_min_run_seconds,
        args.bm_video_creation,
    )
    plot_data(df_data, args.plot_path)


if __name__ == "__main__":
    main()
