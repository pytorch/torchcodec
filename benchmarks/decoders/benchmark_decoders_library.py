import abc
import json
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.utils.benchmark as benchmark
from torchcodec.decoders import VideoDecoder, VideoStreamMetadata

from torchcodec.decoders._core import (
    _add_video_stream,
    create_from_file,
    get_frames_at_indices,
    get_frames_by_pts,
    get_json_metadata,
    get_next_frame,
    scan_all_streams_to_update_metadata,
    seek_to_pts,
)

torch._dynamo.config.cache_size_limit = 100
torch._dynamo.config.capture_dynamic_output_shape_ops = True


class AbstractDecoder:
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_frames_from_video(self, video_file, pts_list):
        pass

    @abc.abstractmethod
    def get_consecutive_frames_from_video(self, video_file, numFramesToDecode):
        pass

    @abc.abstractmethod
    def decode_and_transform(self, video_file, pts_list, height, width, device):
        pass


class DecordAccurate(AbstractDecoder):
    def __init__(self):
        import decord  # noqa: F401

        self.decord = decord
        self.decord.bridge.set_bridge("torch")

    def get_frames_from_video(self, video_file, pts_list):
        decord_vr = self.decord.VideoReader(video_file, ctx=self.decord.cpu())
        frames = []
        fps = decord_vr.get_avg_fps()
        for pts in pts_list:
            decord_vr.seek_accurate(int(pts * fps))
            frame = decord_vr.next()
            frames.append(frame)
        return frames

    def get_consecutive_frames_from_video(self, video_file, numFramesToDecode):
        decord_vr = self.decord.VideoReader(video_file, ctx=self.decord.cpu())
        frames = []
        for _ in range(numFramesToDecode):
            frame = decord_vr.next()
            frames.append(frame)
        return frames


class DecordAccurateBatch(AbstractDecoder):
    def __init__(self):
        import decord  # noqa: F401

        self.decord = decord
        self.decord.bridge.set_bridge("torch")

    def get_frames_from_video(self, video_file, pts_list):
        decord_vr = self.decord.VideoReader(video_file, ctx=self.decord.cpu())
        average_fps = decord_vr.get_avg_fps()
        indices_list = [int(pts * average_fps) for pts in pts_list]
        return decord_vr.get_batch(indices_list)

    def get_consecutive_frames_from_video(self, video_file, numFramesToDecode):
        decord_vr = self.decord.VideoReader(video_file, ctx=self.decord.cpu())
        indices_list = list(range(numFramesToDecode))
        return decord_vr.get_batch(indices_list)


class TorchVision(AbstractDecoder):
    def __init__(self, backend):
        self._backend = backend
        self._print_each_iteration_time = False
        import torchvision  # noqa: F401
        from torchvision.transforms import v2 as transforms_v2

        self.torchvision = torchvision
        self.transforms_v2 = transforms_v2

    def get_frames_from_video(self, video_file, pts_list):
        self.torchvision.set_video_backend(self._backend)
        reader = self.torchvision.io.VideoReader(video_file, "video")
        frames = []
        for pts in pts_list:
            reader.seek(pts)
            frame = next(reader)
            frames.append(frame["data"].permute(1, 2, 0))
        return frames

    def get_consecutive_frames_from_video(self, video_file, numFramesToDecode):
        self.torchvision.set_video_backend(self._backend)
        reader = self.torchvision.io.VideoReader(video_file, "video")
        frames = []
        for _ in range(numFramesToDecode):
            frame = next(reader)
            frames.append(frame["data"].permute(1, 2, 0))
        return frames

    def decode_and_transform(self, video_file, pts_list, height, width, device):
        self.torchvision.set_video_backend(self._backend)
        reader = self.torchvision.io.VideoReader(video_file, "video")
        frames = []
        for pts in pts_list:
            reader.seek(pts)
            frame = next(reader)
            frames.append(frame["data"].permute(1, 2, 0))
        frames = [frame.to(device) for frame in frames]
        frames = self.transforms_v2.functional.resize(frames, (height, width))
        return frames


class TorchCodecCore(AbstractDecoder):
    def __init__(self, num_threads=None, color_conversion_library=None, device="cpu"):
        self._num_threads = int(num_threads) if num_threads else None
        self._color_conversion_library = color_conversion_library
        self._device = device

    def get_frames_from_video(self, video_file, pts_list):
        decoder = create_from_file(video_file)
        scan_all_streams_to_update_metadata(decoder)
        _add_video_stream(
            decoder,
            num_threads=self._num_threads,
            color_conversion_library=self._color_conversion_library,
            device=self._device,
        )
        metadata = json.loads(get_json_metadata(decoder))
        best_video_stream = metadata["bestVideoStreamIndex"]
        frames, *_ = get_frames_by_pts(
            decoder, stream_index=best_video_stream, timestamps=pts_list
        )
        return frames

    def get_consecutive_frames_from_video(self, video_file, numFramesToDecode):
        decoder = create_from_file(video_file)
        _add_video_stream(
            decoder,
            num_threads=self._num_threads,
            color_conversion_library=self._color_conversion_library,
            device=self._device,
        )

        frames = []
        for _ in range(numFramesToDecode):
            frame = get_next_frame(decoder)
            frames.append(frame)

        return frames


class TorchCodecCoreNonBatch(AbstractDecoder):
    def __init__(self, num_threads=None, color_conversion_library=None, device="cpu"):
        self._num_threads = int(num_threads) if num_threads else None
        self._color_conversion_library = color_conversion_library
        self._device = device

    def get_frames_from_video(self, video_file, pts_list):
        decoder = create_from_file(video_file)
        _add_video_stream(
            decoder,
            num_threads=self._num_threads,
            color_conversion_library=self._color_conversion_library,
            device=self._device,
        )

        frames = []
        for pts in pts_list:
            seek_to_pts(decoder, pts)
            frame = get_next_frame(decoder)
            frames.append(frame)

        return frames

    def get_consecutive_frames_from_video(self, video_file, numFramesToDecode):
        decoder = create_from_file(video_file)
        _add_video_stream(
            decoder,
            num_threads=self._num_threads,
            color_conversion_library=self._color_conversion_library,
            device=self._device,
        )

        frames = []
        for _ in range(numFramesToDecode):
            frame = get_next_frame(decoder)
            frames.append(frame)

        return frames


class TorchCodecCoreBatch(AbstractDecoder):
    def __init__(self, num_threads=None, color_conversion_library=None, device="cpu"):
        self._print_each_iteration_time = False
        self._num_threads = int(num_threads) if num_threads else None
        self._color_conversion_library = color_conversion_library
        self._device = device

    def get_frames_from_video(self, video_file, pts_list):
        decoder = create_from_file(video_file)
        scan_all_streams_to_update_metadata(decoder)
        _add_video_stream(
            decoder,
            num_threads=self._num_threads,
            color_conversion_library=self._color_conversion_library,
            device=self._device,
        )
        metadata = json.loads(get_json_metadata(decoder))
        best_video_stream = metadata["bestVideoStreamIndex"]
        frames, *_ = get_frames_by_pts(
            decoder, stream_index=best_video_stream, timestamps=pts_list
        )
        return frames

    def get_consecutive_frames_from_video(self, video_file, numFramesToDecode):
        decoder = create_from_file(video_file)
        scan_all_streams_to_update_metadata(decoder)
        _add_video_stream(
            decoder,
            num_threads=self._num_threads,
            color_conversion_library=self._color_conversion_library,
            device=self._device,
        )
        metadata = json.loads(get_json_metadata(decoder))
        best_video_stream = metadata["bestVideoStreamIndex"]
        indices_list = list(range(numFramesToDecode))
        frames, *_ = get_frames_at_indices(
            decoder, stream_index=best_video_stream, frame_indices=indices_list
        )
        return frames


class TorchCodecPublic(AbstractDecoder):
    def __init__(self, num_ffmpeg_threads=None, device="cpu"):
        self._num_ffmpeg_threads = (
            int(num_ffmpeg_threads) if num_ffmpeg_threads else None
        )
        self._device = device

        import torchvision  # noqa: F401
        from torchvision.transforms import v2 as transforms_v2

        self.torchvision = torchvision
        self.transforms_v2 = transforms_v2

    def get_frames_from_video(self, video_file, pts_list):
        decoder = VideoDecoder(
            video_file, num_ffmpeg_threads=self._num_ffmpeg_threads, device=self._device
        )
        return decoder.get_frames_played_at(pts_list)

    def get_consecutive_frames_from_video(self, video_file, numFramesToDecode):
        decoder = VideoDecoder(
            video_file, num_ffmpeg_threads=self._num_ffmpeg_threads, device=self._device
        )
        frames = []
        count = 0
        for frame in decoder:
            frames.append(frame)
            count += 1
            if count == numFramesToDecode:
                break
        return frames

    def decode_and_transform(self, video_file, pts_list, height, width, device):
        decoder = VideoDecoder(
            video_file, num_ffmpeg_threads=self._num_ffmpeg_threads, device=self._device
        )
        frames = decoder.get_frames_played_at(pts_list)
        frames = self.transforms_v2.functional.resize(frames.data, (height, width))
        return frames


@torch.compile(fullgraph=True, backend="eager")
def compiled_seek_and_next(decoder, pts):
    seek_to_pts(decoder, pts)
    return get_next_frame(decoder)


@torch.compile(fullgraph=True, backend="eager")
def compiled_next(decoder):
    return get_next_frame(decoder)


class TorchCodecCoreCompiled(AbstractDecoder):
    def __init__(self):
        pass

    def get_frames_from_video(self, video_file, pts_list):
        decoder = create_from_file(video_file)
        _add_video_stream(decoder)
        frames = []
        for pts in pts_list:
            frame = compiled_seek_and_next(decoder, pts)
            frames.append(frame)
        return frames

    def get_consecutive_frames_from_video(self, video_file, numFramesToDecode):
        decoder = create_from_file(video_file)
        _add_video_stream(decoder)
        frames = []
        for _ in range(numFramesToDecode):
            frame = compiled_next(decoder)
            frames.append(frame)
        return frames


class TorchAudioDecoder(AbstractDecoder):
    def __init__(self):
        import torchaudio  # noqa: F401

        self.torchaudio = torchaudio

        from torchvision.transforms import v2 as transforms_v2

        self.transforms_v2 = transforms_v2

    def get_frames_from_video(self, video_file, pts_list):
        stream_reader = self.torchaudio.io.StreamReader(src=video_file)
        stream_reader.add_basic_video_stream(frames_per_chunk=1)
        frames = []
        for pts in pts_list:
            stream_reader.seek(pts)
            stream_reader.fill_buffer()
            clip = stream_reader.pop_chunks()
            frames.append(clip[0][0])
        return frames

    def get_consecutive_frames_from_video(self, video_file, numFramesToDecode):
        stream_reader = self.torchaudio.io.StreamReader(src=video_file)
        stream_reader.add_basic_video_stream(frames_per_chunk=1)
        frames = []
        frame_cnt = 0
        for vframe in stream_reader.stream():
            if frame_cnt >= numFramesToDecode:
                break
            frames.append(vframe[0][0])
            frame_cnt += 1

        return frames

    def decode_and_transform(self, video_file, pts_list, height, width, device):
        stream_reader = self.torchaudio.io.StreamReader(src=video_file)
        stream_reader.add_basic_video_stream(frames_per_chunk=1)
        frames = []
        for pts in pts_list:
            stream_reader.seek(pts)
            stream_reader.fill_buffer()
            clip = stream_reader.pop_chunks()
            frames.append(clip[0][0])
        frames = [frame.to(device) for frame in frames]
        frames = self.transforms_v2.functional.resize(frames, (height, width))
        return frames


def create_torchcodec_decoder_from_file(video_file):
    video_decoder = create_from_file(video_file)
    _add_video_stream(video_decoder)
    get_next_frame(video_decoder)
    return video_decoder


def generate_video(command):
    print(command)
    print(" ".join(command))
    subprocess.check_call(command)
    return True


def generate_videos(
    resolutions,
    encodings,
    patterns,
    fpses,
    gop_sizes,
    durations,
    pix_fmts,
    ffmpeg_cli,
    output_dir,
):
    executor = ThreadPoolExecutor(max_workers=20)
    video_count = 0

    futures = []
    for resolution, duration, fps, gop_size, encoding, pattern, pix_fmt in product(
        resolutions, durations, fpses, gop_sizes, encodings, patterns, pix_fmts
    ):
        outfile = f"{output_dir}/{pattern}_{resolution}_{duration}s_{fps}fps_{gop_size}gop_{encoding}_{pix_fmt}.mp4"
        command = [
            ffmpeg_cli,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"{pattern}=s={resolution}",
            "-t",
            str(duration),
            "-c:v",
            encoding,
            "-r",
            str(fps),
            "-g",
            str(gop_size),
            "-pix_fmt",
            pix_fmt,
            outfile,
        ]
        futures.append(executor.submit(generate_video, command))
        video_count += 1

    wait(futures)
    for f in futures:
        assert f.result()
    executor.shutdown(wait=True)
    print(f"Generated {video_count} videos")


def retrieve_videos(urls_and_dest_paths):
    for url, path in urls_and_dest_paths:
        urllib.request.urlretrieve(url, path)


def plot_data(df_data, plot_path):
    plt.rcParams["font.size"] = 18

    # Creating the DataFrame
    df = pd.DataFrame(df_data)

    # Sorting by video, type, and frame_count
    df_sorted = df.sort_values(by=["video", "type", "frame_count"])

    # Group by video first
    grouped_by_video = df_sorted.groupby("video")

    # Define colors (consistent across decoders)
    colors = plt.get_cmap("tab10")

    # Find the unique combinations of (type, frame_count) per video
    video_type_combinations = {
        video: video_group.groupby(["type", "frame_count"]).ngroups
        for video, video_group in grouped_by_video
    }

    # Get the unique videos and the maximum number of (type, frame_count) combinations per video
    unique_videos = list(video_type_combinations.keys())
    max_combinations = max(video_type_combinations.values())

    # Create subplots: each row is a video, and each column is for a unique (type, frame_count)
    fig, axes = plt.subplots(
        nrows=len(unique_videos),
        ncols=max_combinations,
        figsize=(max_combinations * 6, len(unique_videos) * 4),
        sharex=False,
        sharey=True,
    )

    # Handle cases where there's only one row or column
    if len(unique_videos) == 1:
        axes = np.array([axes])  # Make sure axes is a list of lists
    if max_combinations == 1:
        axes = np.expand_dims(axes, axis=1)  # Ensure a 2D array for axes

    # Loop through each video and its sub-groups
    for row, (video, video_group) in enumerate(grouped_by_video):
        sub_group = video_group.groupby(["type", "frame_count"])

        # Loop through each (type, frame_count) group for this video
        for col, ((vtype, vcount), group) in enumerate(sub_group):
            ax = axes[row, col]  # Select the appropriate axis

            # Set the title for the subplot
            base_video = Path(video).name.removesuffix(".mp4")
            ax.set_title(f"{base_video}\n{vcount} x {vtype}", fontsize=11)

            # Plot bars with error bars
            ax.barh(
                group["decoder"],
                group["fps_median"],
                xerr=[
                    group["fps_median"] - group["fps_p75"],
                    group["fps_p25"] - group["fps_median"],
                ],
                color=[colors(i) for i in range(len(group))],
                align="center",
                capsize=5,
                label=group["decoder"],
            )

            # Set the labels
            ax.set_xlabel("FPS")

    # Remove any empty subplots for videos with fewer combinations
    for row in range(len(unique_videos)):
        for col in range(video_type_combinations[unique_videos[row]], max_combinations):
            fig.delaxes(axes[row, col])

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Show plot
    plt.savefig(
        plot_path,
    )


def get_metadata(video_file_path: str) -> VideoStreamMetadata:
    return VideoDecoder(video_file_path).metadata


@dataclass
class BatchParameters:
    num_threads: int
    batch_size: int


@dataclass
class DataLoaderInspiredWorkloadParameters:
    batch_parameters: BatchParameters
    resize_height: int
    resize_width: int
    resize_device: str


def run_batch_using_threads(
    function,
    *args,
    batch_parameters: BatchParameters = BatchParameters(num_threads=8, batch_size=40),
):
    executor = ThreadPoolExecutor(max_workers=batch_parameters.num_threads)
    futures = []
    for _ in range(batch_parameters.batch_size):
        futures.append(executor.submit(function, *args))
    for f in futures:
        # TODO: Add a stronger check here based on arguments to the function.
        assert len(f.result()) > 0
    executor.shutdown(wait=True)


def convert_result_to_df_item(
    result, decoder_name, video_file_path, num_samples, decode_pattern
):
    df_item = {}
    df_item["decoder"] = decoder_name
    df_item["video"] = str(video_file_path)
    df_item["description"] = result.description
    df_item["frame_count"] = num_samples
    df_item["median"] = result.median
    df_item["iqr"] = result.iqr
    df_item["type"] = decode_pattern
    df_item["fps_median"] = num_samples / result.median
    df_item["fps_p75"] = num_samples / result._p75
    df_item["fps_p25"] = num_samples / result._p25
    return df_item


def run_benchmarks(
    decoder_dict: dict[str, AbstractDecoder],
    video_files_paths: list[Path],
    num_samples: int,
    num_sequential_frames_from_start: list[int],
    min_runtime_seconds: float,
    benchmark_video_creation: bool,
    dataloader_parameters: DataLoaderInspiredWorkloadParameters = None,
    batch_parameters: BatchParameters = None,
) -> list[dict[str, str | float | int]]:
    # Ensure that we have the same seed across benchmark runs.
    torch.manual_seed(0)

    print(f"video_files_paths={video_files_paths}")

    results = []
    df_data = []
    verbose = False
    # TODO: change this back before landing.
    min_runtime_seconds = 0.1
    for video_file_path in video_files_paths:
        metadata = get_metadata(video_file_path)
        metadata_label = f"{metadata.codec} {metadata.width}x{metadata.height}, {metadata.duration_seconds}s {metadata.average_fps}fps"

        duration = metadata.duration_seconds
        uniform_pts_list = [i * duration / num_samples for i in range(num_samples)]

        # Note that we are using the same random pts values for all decoders for the same
        # video. However, because we use the duration as part of this calculation, we
        # are using different random pts values across videos.
        random_pts_list = (torch.rand(num_samples) * duration).tolist()

        for decoder_name, decoder in decoder_dict.items():
            print(f"video={video_file_path}, decoder={decoder_name}")

            if dataloader_parameters:
                bp = dataloader_parameters.batch_parameters
                dataloader_result = benchmark.Timer(
                    stmt="run_batch_using_threads(decoder.decode_and_transform, video_file, pts_list, height, width, device, batch_parameters=batch_parameters)",
                    globals={
                        "video_file": str(video_file_path),
                        "pts_list": uniform_pts_list,
                        "decoder": decoder,
                        "run_batch_using_threads": run_batch_using_threads,
                        "batch_parameters": dataloader_parameters.batch_parameters,
                        "height": dataloader_parameters.resize_height,
                        "width": dataloader_parameters.resize_width,
                        "device": dataloader_parameters.resize_device,
                    },
                    label=f"video={video_file_path} {metadata_label}",
                    sub_label=decoder_name,
                    description=f"dataloader[threads={bp.num_threads} batch_size={bp.batch_size}] {num_samples} decode_and_transform()",
                )
                results.append(
                    dataloader_result.blocked_autorange(
                        min_run_time=min_runtime_seconds
                    )
                )
                df_data.append(
                    convert_result_to_df_item(
                        results[-1],
                        decoder_name,
                        video_file_path,
                        num_samples * dataloader_parameters.batch_parameters.batch_size,
                        f"dataloader[threads={bp.num_threads} batch_size={bp.batch_size}] {num_samples} x decode_and_transform()",
                    )
                )

            for kind, pts_list in [
                ("uniform", uniform_pts_list),
                ("random", random_pts_list),
            ]:
                if verbose:
                    print(
                        f"video={video_file_path}, decoder={decoder_name}, pts_list={pts_list}"
                    )
                seeked_result = benchmark.Timer(
                    stmt="decoder.get_frames_from_video(video_file, pts_list)",
                    globals={
                        "video_file": str(video_file_path),
                        "pts_list": pts_list,
                        "decoder": decoder,
                    },
                    label=f"video={video_file_path} {metadata_label}",
                    sub_label=decoder_name,
                    description=f"{kind} {num_samples} seek()+next()",
                )
                results.append(
                    seeked_result.blocked_autorange(min_run_time=min_runtime_seconds)
                )
                df_data.append(
                    convert_result_to_df_item(
                        results[-1],
                        decoder_name,
                        video_file_path,
                        num_samples,
                        f"{kind} seek()+next()",
                    )
                )

                if batch_parameters:
                    seeked_result = benchmark.Timer(
                        stmt="run_batch_using_threads(decoder.get_frames_from_video, video_file, pts_list, batch_parameters=batch_parameters)",
                        globals={
                            "video_file": str(video_file_path),
                            "pts_list": pts_list,
                            "decoder": decoder,
                            "run_batch_using_threads": run_batch_using_threads,
                            "batch_parameters": batch_parameters,
                        },
                        label=f"video={video_file_path} {metadata_label}",
                        sub_label=decoder_name,
                        description=f"batch {kind} {num_samples} seek()+next()",
                    )
                    results.append(
                        seeked_result.blocked_autorange(
                            min_run_time=min_runtime_seconds
                        )
                    )
                    df_data.append(
                        convert_result_to_df_item(
                            results[-1],
                            decoder_name,
                            video_file_path,
                            num_samples * batch_parameters.batch_size,
                            f"batch {kind} seek()+next()",
                        )
                    )

            for num_consecutive_nexts in num_sequential_frames_from_start:
                consecutive_frames_result = benchmark.Timer(
                    stmt="decoder.get_consecutive_frames_from_video(video_file, consecutive_frames_to_extract)",
                    globals={
                        "video_file": str(video_file_path),
                        "consecutive_frames_to_extract": num_consecutive_nexts,
                        "decoder": decoder,
                    },
                    label=f"video={video_file_path} {metadata_label}",
                    sub_label=decoder_name,
                    description=f"{num_consecutive_nexts} next()",
                )
                results.append(
                    consecutive_frames_result.blocked_autorange(
                        min_run_time=min_runtime_seconds
                    )
                )
                df_data.append(
                    convert_result_to_df_item(
                        results[-1],
                        decoder_name,
                        video_file_path,
                        num_consecutive_nexts,
                        f"{num_consecutive_nexts} next()",
                    )
                )

                if batch_parameters:
                    consecutive_frames_result = benchmark.Timer(
                        stmt="run_batch_using_threads(decoder.get_consecutive_frames_from_video, video_file, consecutive_frames_to_extract, batch_parameters=batch_parameters)",
                        globals={
                            "video_file": str(video_file_path),
                            "consecutive_frames_to_extract": num_consecutive_nexts,
                            "decoder": decoder,
                            "run_batch_using_threads": run_batch_using_threads,
                            "batch_parameters": batch_parameters,
                        },
                        label=f"video={video_file_path} {metadata_label}",
                        sub_label=decoder_name,
                        description=f"batch {num_consecutive_nexts} next()",
                    )
                    results.append(
                        consecutive_frames_result.blocked_autorange(
                            min_run_time=min_runtime_seconds
                        )
                    )
                    df_data.append(
                        convert_result_to_df_item(
                            results[-1],
                            decoder_name,
                            video_file_path,
                            num_consecutive_nexts * batch_parameters.batch_size,
                            f"batch {num_consecutive_nexts} next()",
                        )
                    )

        first_video_file_path = video_files_paths[0]
        if benchmark_video_creation:
            metadata = get_metadata(video_file_path)
            metadata_label = f"{metadata.codec} {metadata.width}x{metadata.height}, {metadata.duration_seconds}s {metadata.average_fps}fps"
            creation_result = benchmark.Timer(
                stmt="create_torchcodec_decoder_from_file(video_file)",
                globals={
                    "video_file": str(first_video_file_path),
                    "create_torchcodec_decoder_from_file": create_torchcodec_decoder_from_file,
                },
                label=f"video={first_video_file_path} {metadata_label}",
                sub_label="TorchCodecCore",
                description="create()+next()",
            )
            results.append(
                creation_result.blocked_autorange(
                    min_run_time=2.0,
                )
            )
    compare = benchmark.Compare(results)
    compare.print()
    return df_data
