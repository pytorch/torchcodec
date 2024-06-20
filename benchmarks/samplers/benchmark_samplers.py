# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import abc
import argparse
import importlib
import os

import decord
import numpy as np
import torch

import torch.utils.benchmark as benchmark
from torchcodec.samplers._video_clip_sampler import (
    IndexBasedSamplerArgs,
    TimeBasedSamplerArgs,
    VideoArgs,
    VideoClipSampler,
)
from torchmultimodal.fb.utils.video_utils import (
    ClipSamplerType,
    VideoClipSampler as tmm_vcs,
)
from torchvision.datasets.video_clip_sampler import (  # @manual=//pytorch/vision:internal_datasets
    TVVideoClipDecoder,
    UniformClipSamplingStrategy,
    VideoClipSampler as ta_vcs,
)


class AbstractSampler:
    def __init__(self):
        pass

    @abc.abstractmethod
    def sample_frames_uniformly(self, video_file, clips_per_video):
        pass


class TorchCodecTimeBasedSampler(AbstractSampler):
    def __init__(self):
        pass

    def sample_frames_uniformly(self, video_file, clips_per_video):
        arr = np.fromfile(video_file, dtype=np.uint8)
        video_tensor = torch.from_numpy(arr)
        video_input = VideoArgs()
        sampler_input = TimeBasedSamplerArgs(
            sampler_type="uniform", clips_per_video=clips_per_video, frames_per_clip=1
        )
        sampler = VideoClipSampler(video_input, sampler_input)
        return sampler(video_tensor)


class TorchCodecIndexBasedSampler(AbstractSampler):
    def __init__(self):
        pass

    def sample_frames_uniformly(self, video_file, clips_per_video):
        arr = np.fromfile(video_file, dtype=np.uint8)
        video_tensor = torch.from_numpy(arr)
        video_input = VideoArgs()
        sampler_input = IndexBasedSamplerArgs(
            sampler_type="uniform", clips_per_video=clips_per_video, frames_per_clip=1
        )
        sampler = VideoClipSampler(video_input, sampler_input)
        return sampler(video_tensor)


class TorchCodecIndexBasedSamplerWithStackedOutput(AbstractSampler):
    """
    On large batch, torch stack has impact on performance, but it's not obvious locally.
    """

    def __init__(self):
        pass

    def sample_frames_uniformly(self, video_file, clips_per_video):
        arr = np.fromfile(video_file, dtype=np.uint8)
        video_tensor = torch.from_numpy(arr)
        video_input = VideoArgs()
        sampler_input = IndexBasedSamplerArgs(
            sampler_type="uniform", clips_per_video=clips_per_video, frames_per_clip=1
        )
        sampler = VideoClipSampler(video_input, sampler_input)
        clips = sampler(video_tensor)
        return torch.stack([clip[0] for clip in clips])


class DecordSampler(AbstractSampler):
    def __init__(self):
        pass

    def sample_frames_uniformly(self, video_file, clips_per_video):
        decord.bridge.set_bridge("torch")
        av_reader = decord.VideoReader(video_file)
        num_frames = len(av_reader)
        frame_indices = np.linspace(0, num_frames - 1, clips_per_video, dtype=int)
        frames = av_reader.get_batch(frame_indices)
        return frames


class TorchMMSamplerWithTorchVisionBackend(AbstractSampler):
    """
    Here we use TorchMultimodal sampler as it's updated version on top of torchvision decoder.
    """

    def __init__(self):
        pass

    def sample_frames_uniformly(self, video_file, clips_per_video):
        arr = np.fromfile(video_file, dtype=np.uint8)
        video_tensor = torch.from_numpy(arr)
        sampler = tmm_vcs(
            clip_sampler_type=ClipSamplerType("UNIFORM"),
            clips_per_video=clips_per_video,
            frames_per_clip=1,
            frame_dilation=1,
        )
        return sampler(video_tensor)


class TorchVisionNewSamplerWithTorchVisionBackend(AbstractSampler):
    def __init__(self):
        pass

    def sample_frames_uniformly(self, video_file, clips_per_video):
        clip_sampling_strategy = UniformClipSamplingStrategy(
            clips_per_video=clips_per_video
        )
        decoder = TVVideoClipDecoder(clip_length_in_frames=1, read_audio_stream=False)
        sampler = ta_vcs(clip_sampling_strategy, decoder)
        return sampler(str(video_file))


def main():
    """Benchmarks the performance of different samplers"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bm_small_video_speed",
        help="Benchmark small video decoding speed",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--bm_large_video_speed",
        help="Benchmark large video decoding speed",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--bm_video_speed_min_run_seconds",
        help="Benchmark minimum run time, in seconds, to wait per datapoint",
        type=float,
        default=5.0,
    )
    args = parser.parse_args()

    small_video_path = importlib.resources.path(__package__, "nasa_13013.mp4")
    small_video_path = os.fspath(str(small_video_path))

    large_video_path = importlib.resources.path(__package__, "853.mp4")
    large_video_path = os.fspath(str(large_video_path))

    clips_per_video = 8

    sampler_dict = {}
    sampler_dict["TorchCodecTimeBasedSampler"] = TorchCodecTimeBasedSampler()
    sampler_dict["TorchCodecIndexBasedSampler"] = TorchCodecIndexBasedSampler()
    sampler_dict["TorchCodecIndexBasedSamplerWithStackedOutput"] = (
        TorchCodecIndexBasedSamplerWithStackedOutput()
    )
    sampler_dict["DecordSampler"] = DecordSampler()
    sampler_dict["TorchMMSamplerWithTorchVisionBackend"] = (
        TorchMMSamplerWithTorchVisionBackend()
    )
    sampler_dict["TorchVisionNewSamplerWithTorchVisionBackend"] = (
        TorchVisionNewSamplerWithTorchVisionBackend()
    )

    results = []

    for sampler_name, sampler in sampler_dict.items():
        if args.bm_small_video_speed:
            sampler_result = benchmark.Timer(
                stmt="sampler.sample_frames_uniformly(video_file, clips_per_video)",
                globals={
                    "video_file": small_video_path,
                    "clips_per_video": clips_per_video,
                    "sampler": sampler,
                },
                label="uniform sampling latency for 700KB video",
                sub_label=sampler_name,
                description=f"uniform sampling {clips_per_video} frames",
            )
            results.append(
                sampler_result.blocked_autorange(
                    min_run_time=args.bm_video_speed_min_run_seconds
                )
            )

        if args.bm_large_video_speed:
            if sampler_name == "TorchMMSamplerWithTorchVisionBackend":
                continue
            sampler_result = benchmark.Timer(
                stmt="sampler.sample_frames_uniformly(video_file, clips_per_video)",
                globals={
                    "video_file": large_video_path,
                    "clips_per_video": clips_per_video,
                    "sampler": sampler,
                },
                label="uniform sampling latency for 50MB video",
                sub_label=sampler_name,
                description=f"uniform sampling {clips_per_video} frames",
            )
            results.append(
                sampler_result.blocked_autorange(
                    min_run_time=args.bm_video_speed_min_run_seconds
                )
            )

    compare = benchmark.Compare(results)
    compare.print()
