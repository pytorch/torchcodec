# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib.resources
import json
import os
import platform
from pathlib import Path

import torch

from benchmark_decoders_library import (
    decoder_registry,
    plot_data,
    run_benchmarks,
    verify_outputs,
)


def in_fbcode() -> bool:
    return "FB_PAR_RUNTIME_FILES" in os.environ


def get_test_resource_path(filename: str) -> str:
    if in_fbcode():
        resource = importlib.resources.files(__package__).joinpath(filename)
        with importlib.resources.as_file(resource) as path:
            return os.fspath(path)

    return str(Path(__file__).parent / f"../../test/resources/{filename}")


def parse_options_code(options_code: str) -> dict[str, str]:
    options = {}
    for item in options_code.split("+"):
        if item.strip() == "":
            continue
        k, v = item.split("=")
        options[k] = v
    return options


def main() -> None:
    """Benchmarks the performance of a few video decoders"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bm-video-creation",
        help="Benchmark large video creation memory",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--verbose",
        help="Show verbose output",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--min-run-seconds",
        help="Benchmark minimum run time, in seconds, to wait per datapoint",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--video-paths",
        help="Comma-separated paths to videos that you want to benchmark.",
        type=str,
        default=get_test_resource_path("nasa_13013.mp4"),
    )
    parser.add_argument(
        "--decoders",
        help=(
            "Comma-separated list of decoders to benchmark. "
            "Choices are: " + ", ".join(decoder_registry.keys()) + ". "
            "To specify options, append a ':' and then value pairs seperated by a '+'. "
            "For example, torchcodec_core:num_threads=1+color_conversion_library=filtergraph."
        ),
        type=str,
        default=(
            "decord,decord_batch,"
            "torchvision,"
            "torchaudio,"
            "torchcodec_core,torchcodec_core:num_threads=1,torchcodec_core_batch,torchcodec_core_nonbatch,"
            "torchcodec_public"
        ),
    )
    parser.add_argument(
        "--video-dir",
        help="Directory where video files reside. We will run benchmarks on all .mp4 files in this directory.",
        type=str,
        default="",
    )
    parser.add_argument(
        "--plot-path",
        help="Path where the generated plot is stored, if non-empty",
        type=str,
        default="benchmarks.png",
    )
    parser.add_argument(
        "--verify-outputs",
        help="Verify that the outputs of the decoders are the same",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--output-json",
        help="Output the results to a JSON file",
        type=str,
        default="",
    )

    args = parser.parse_args()
    specified_decoders = set(args.decoders.split(","))

    # These are the PTS values we want to extract from the small video.
    num_uniform_samples = 10

    decoders_to_run = {}
    for decoder in specified_decoders:
        if ":" in decoder:
            decoder, _, options_code = decoder.partition(":")
            assert decoder in decoder_registry, f"Unknown decoder: {decoder}"
            display = decoder_registry[decoder].display_name + ":" + options_code
            options = parse_options_code(options_code)
        else:
            assert decoder in decoder_registry, f"Unknown decoder: {decoder}"
            display = decoder_registry[decoder].display_name
            options = decoder_registry[decoder].default_options

        kind = decoder_registry[decoder].kind
        decoders_to_run[display] = kind(**options)

    video_paths = args.video_paths.split(",")
    if args.video_dir:
        video_paths = []
        for entry in os.scandir(args.video_dir):
            if entry.is_file() and entry.name.endswith(".mp4"):
                video_paths.append(entry.path)

    if args.verify_outputs:
        verify_outputs(decoders_to_run, video_paths, num_uniform_samples)
    else:
        results = run_benchmarks(
            decoders_to_run,
            video_paths,
            num_uniform_samples,
            num_sequential_frames_from_start=[1, 10, 100],
            min_runtime_seconds=args.min_run_seconds,
            benchmark_video_creation=args.bm_video_creation,
        )
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=2)

        data = {
            "experiments": results,
            "system_metadata": {
                "cpu_count": os.cpu_count(),
                "system": platform.system(),
                "machine": platform.machine(),
                "python_version": str(platform.python_version()),
                "cuda": (
                    torch.cuda.get_device_properties(0).name
                    if torch.cuda.is_available()
                    else "not available"
                ),
            },
        }
        plot_data(data, args.plot_path)


if __name__ == "__main__":
    main()
