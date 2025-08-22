import abc
import subprocess
import typing
import urllib.request
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.utils.benchmark as benchmark

from torchcodec._core import (
    _add_video_stream,
    create_from_file,
    get_frames_at_indices,
    get_frames_by_pts,
    get_next_frame,
    seek_to_pts,
)
from torchcodec._frame import FrameBatch
from torchcodec.decoders import VideoDecoder, VideoStreamMetadata

torch._dynamo.config.cache_size_limit = 100
torch._dynamo.config.capture_dynamic_output_shape_ops = True


class AbstractDecoder:
    def __init__(self):
        pass

    @abc.abstractmethod
    def decode_frames(self, video_file, pts_list):
        pass

    def decode_frames_description(self, num_frames: int, kind: str) -> str:
        return f"decode {num_frames} {kind} frames"

    @abc.abstractmethod
    def decode_first_n_frames(self, video_file, n):
        pass

    def decode_first_n_frames_description(self, n) -> str:
        return f"first {n} frames"

    def init_decode_and_resize(self):
        pass

    @abc.abstractmethod
    def decode_and_resize(self, video_file, pts_list, height, width, device):
        pass

    def decode_and_resize_description(
        self, num_frames: int, height: int, width: int
    ) -> str:
        return f"decode {num_frames} -> {height}x{width}"


class DecordAccurate(AbstractDecoder):
    def __init__(self):
        import decord  # noqa: F401

        self.decord = decord
        self.decord.bridge.set_bridge("torch")

    def decode_frames(self, video_file, pts_list):
        decord_vr = self.decord.VideoReader(video_file, ctx=self.decord.cpu())
        frames = []
        fps = decord_vr.get_avg_fps()
        for pts in pts_list:
            decord_vr.seek_accurate(int(pts * fps))
            frame = decord_vr.next()
            frames.append(frame)
        return frames

    def decode_first_n_frames(self, video_file, n):
        decord_vr = self.decord.VideoReader(video_file, ctx=self.decord.cpu())
        frames = []
        for _ in range(n):
            frame = decord_vr.next()
            frames.append(frame)
        return frames


class DecordAccurateBatch(AbstractDecoder):
    def __init__(self):
        import decord  # noqa: F401

        self.decord = decord
        self.decord.bridge.set_bridge("torch")

    def decode_frames(self, video_file, pts_list):
        decord_vr = self.decord.VideoReader(video_file, ctx=self.decord.cpu())
        average_fps = decord_vr.get_avg_fps()
        indices_list = [int(pts * average_fps) for pts in pts_list]
        return decord_vr.get_batch(indices_list)

    def decode_first_n_frames(self, video_file, n):
        decord_vr = self.decord.VideoReader(video_file, ctx=self.decord.cpu())
        indices_list = list(range(n))
        return decord_vr.get_batch(indices_list)


class TorchVision(AbstractDecoder):
    def __init__(self, backend):
        self._backend = backend
        self._print_each_iteration_time = False
        import torchvision  # noqa: F401

        self.torchvision = torchvision

    def init_decode_and_resize(self):
        from torchvision.transforms import v2 as transforms_v2

        self.transforms_v2 = transforms_v2

    def decode_frames(self, video_file, pts_list):
        self.torchvision.set_video_backend(self._backend)
        reader = self.torchvision.io.VideoReader(video_file, "video", num_threads=0)
        frames = []
        for pts in pts_list:
            reader.seek(pts)
            frame = next(reader)
            frames.append(frame["data"].permute(1, 2, 0))
        return frames

    def decode_first_n_frames(self, video_file, n):
        self.torchvision.set_video_backend(self._backend)
        reader = self.torchvision.io.VideoReader(video_file, "video", num_threads=0)
        frames = []
        for _ in range(n):
            frame = next(reader)
            frames.append(frame["data"].permute(1, 2, 0))
        return frames

    def decode_and_resize(self, video_file, pts_list, height, width, device):
        self.torchvision.set_video_backend(self._backend)
        reader = self.torchvision.io.VideoReader(video_file, "video", num_threads=1)
        frames = []
        for pts in pts_list:
            reader.seek(pts)
            frame = next(reader)
            frames.append(frame["data"].permute(1, 2, 0))
        frames = [
            self.transforms_v2.functional.resize(frame.to(device), (height, width))
            for frame in frames
        ]
        return frames


class OpenCVDecoder(AbstractDecoder):
    def __init__(self, backend):
        import cv2

        self.cv2 = cv2

        self._available_backends = {"FFMPEG": cv2.CAP_FFMPEG}
        self._backend = self._available_backends.get(backend)

        self._print_each_iteration_time = False

    def decode_frames(self, video_file, pts_list):
        cap = self.cv2.VideoCapture(video_file, self._backend)
        if not cap.isOpened():
            raise ValueError("Could not open video stream")

        fps = cap.get(self.cv2.CAP_PROP_FPS)
        approx_frame_indices = [int(pts * fps) for pts in pts_list]

        current_frame = 0
        frames = []
        while True:
            ok = cap.grab()
            if not ok:
                raise ValueError("Could not grab video frame")
            if current_frame in approx_frame_indices:  # only decompress needed
                ret, frame = cap.retrieve()
                if ret:
                    frame = self.convert_frame_to_rgb_tensor(frame)
                    frames.append(frame)

            if len(frames) == len(approx_frame_indices):
                break
            current_frame += 1
        cap.release()
        assert len(frames) == len(approx_frame_indices)
        return frames

    def decode_first_n_frames(self, video_file, n):
        cap = self.cv2.VideoCapture(video_file, self._backend)
        if not cap.isOpened():
            raise ValueError("Could not open video stream")

        frames = []
        for i in range(n):
            ok = cap.grab()
            if not ok:
                raise ValueError("Could not grab video frame")
            ret, frame = cap.retrieve()
            if ret:
                frame = self.convert_frame_to_rgb_tensor(frame)
                frames.append(frame)
        cap.release()
        assert len(frames) == n
        return frames

    def decode_and_resize(self, *args, **kwargs):
        raise ValueError(
            "OpenCV doesn't apply antialias while pytorch does by default, this is potentially an unfair comparison"
        )

    def convert_frame_to_rgb_tensor(self, frame):
        # OpenCV uses BGR, change to RGB
        frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        # Update to C, H, W
        frame = np.transpose(frame, (2, 0, 1))
        # Convert to tensor
        frame = torch.from_numpy(frame)
        return frame


class TorchCodecCore(AbstractDecoder):
    def __init__(
        self,
        num_threads: str | None = None,
        color_conversion_library=None,
        device="cpu",
    ):
        self._num_threads = int(num_threads) if num_threads else None
        self._color_conversion_library = color_conversion_library
        self._device = device

    def decode_frames(self, video_file, pts_list):
        decoder = create_from_file(video_file, seek_mode="exact")
        _add_video_stream(
            decoder,
            num_threads=self._num_threads,
            color_conversion_library=self._color_conversion_library,
            device=self._device,
        )
        frames, *_ = get_frames_by_pts(decoder, timestamps=pts_list)
        return frames

    def decode_first_n_frames(self, video_file, n):
        decoder = create_from_file(video_file, seek_mode="approximate")
        _add_video_stream(
            decoder,
            num_threads=self._num_threads,
            color_conversion_library=self._color_conversion_library,
            device=self._device,
        )

        frames = []
        for _ in range(n):
            frame = get_next_frame(decoder)
            frames.append(frame)

        return frames


class TorchCodecCoreNonBatch(AbstractDecoder):
    def __init__(
        self,
        num_threads: str | None = None,
        color_conversion_library=None,
        device="cpu",
    ):
        self._num_threads = num_threads
        self._color_conversion_library = color_conversion_library
        self._device = device

    def init_decode_and_resize(self):
        from torchvision.transforms import v2 as transforms_v2

        self.transforms_v2 = transforms_v2

    def decode_frames(self, video_file, pts_list):
        decoder = create_from_file(video_file, seek_mode="approximate")
        num_threads = int(self._num_threads) if self._num_threads else 0
        _add_video_stream(
            decoder,
            num_threads=num_threads,
            color_conversion_library=self._color_conversion_library,
            device=self._device,
        )

        frames = []
        for pts in pts_list:
            seek_to_pts(decoder, pts)
            frame = get_next_frame(decoder)
            frames.append(frame)

        return frames

    def decode_first_n_frames(self, video_file, n):
        num_threads = int(self._num_threads) if self._num_threads else 0
        decoder = create_from_file(video_file, seek_mode="approximate")
        _add_video_stream(
            decoder,
            num_threads=num_threads,
            color_conversion_library=self._color_conversion_library,
            device=self._device,
        )

        frames = []
        for _ in range(n):
            frame = get_next_frame(decoder)
            frames.append(frame)

        return frames

    def decode_and_resize(self, video_file, pts_list, height, width, device):
        num_threads = int(self._num_threads) if self._num_threads else 1
        decoder = create_from_file(video_file, seek_mode="approximate")
        _add_video_stream(
            decoder,
            num_threads=num_threads,
            color_conversion_library=self._color_conversion_library,
            device=self._device,
        )

        frames = []
        for pts in pts_list:
            seek_to_pts(decoder, pts)
            frame, *_ = get_next_frame(decoder)
            frames.append(frame)

        frames = [
            self.transforms_v2.functional.resize(frame.to(device), (height, width))
            for frame in frames
        ]

        return frames


class TorchCodecCoreBatch(AbstractDecoder):
    def __init__(
        self,
        num_threads: str | None = None,
        color_conversion_library=None,
        device="cpu",
    ):
        self._print_each_iteration_time = False
        self._num_threads = int(num_threads) if num_threads else None
        self._color_conversion_library = color_conversion_library
        self._device = device

    def decode_frames(self, video_file, pts_list):
        decoder = create_from_file(video_file, seek_mode="exact")
        _add_video_stream(
            decoder,
            num_threads=self._num_threads,
            color_conversion_library=self._color_conversion_library,
            device=self._device,
        )
        frames, *_ = get_frames_by_pts(decoder, timestamps=pts_list)
        return frames

    def decode_first_n_frames(self, video_file, n):
        decoder = create_from_file(video_file, seek_mode="exact")
        _add_video_stream(
            decoder,
            num_threads=self._num_threads,
            color_conversion_library=self._color_conversion_library,
            device=self._device,
        )
        indices_list = list(range(n))
        frames, *_ = get_frames_at_indices(decoder, frame_indices=indices_list)
        return frames


class TorchCodecPublic(AbstractDecoder):
    def __init__(
        self,
        num_ffmpeg_threads: str | None = None,
        device="cpu",
        seek_mode="exact",
        stream_index: str | None = None,
    ):
        self._num_ffmpeg_threads = num_ffmpeg_threads
        self._device = device
        self._seek_mode = seek_mode
        self._stream_index = int(stream_index) if stream_index else None

    def init_decode_and_resize(self):
        from torchvision.transforms import v2 as transforms_v2

        self.transforms_v2 = transforms_v2

    def decode_frames(self, video_file, pts_list):
        num_ffmpeg_threads = (
            int(self._num_ffmpeg_threads) if self._num_ffmpeg_threads else 0
        )
        decoder = VideoDecoder(
            video_file,
            num_ffmpeg_threads=num_ffmpeg_threads,
            device=self._device,
            seek_mode=self._seek_mode,
            stream_index=self._stream_index,
        )
        return decoder.get_frames_played_at(pts_list)

    def decode_first_n_frames(self, video_file, n):
        num_ffmpeg_threads = (
            int(self._num_ffmpeg_threads) if self._num_ffmpeg_threads else 0
        )
        decoder = VideoDecoder(
            video_file,
            num_ffmpeg_threads=num_ffmpeg_threads,
            device=self._device,
            seek_mode=self._seek_mode,
            stream_index=self._stream_index,
        )
        frames = []
        count = 0
        for frame in decoder:
            frames.append(frame)
            count += 1
            if count == n:
                break
        return frames

    def decode_and_resize(self, video_file, pts_list, height, width, device):
        num_ffmpeg_threads = (
            int(self._num_ffmpeg_threads) if self._num_ffmpeg_threads else 1
        )
        decoder = VideoDecoder(
            video_file,
            num_ffmpeg_threads=num_ffmpeg_threads,
            device=self._device,
            seek_mode=self._seek_mode,
            stream_index=self._stream_index,
        )
        frames = decoder.get_frames_played_at(pts_list)
        frames = self.transforms_v2.functional.resize(frames.data, (height, width))
        return frames


class TorchCodecPublicNonBatch(AbstractDecoder):
    def __init__(
        self,
        num_ffmpeg_threads: str | None = None,
        device="cpu",
        seek_mode="approximate",
    ):
        self._num_ffmpeg_threads = num_ffmpeg_threads
        self._device = device
        self._seek_mode = seek_mode

    def init_decode_and_resize(self):
        from torchvision.transforms import v2 as transforms_v2

        self.transforms_v2 = transforms_v2

    def decode_frames(self, video_file, pts_list):
        num_ffmpeg_threads = (
            int(self._num_ffmpeg_threads) if self._num_ffmpeg_threads else 0
        )
        decoder = VideoDecoder(
            video_file,
            num_ffmpeg_threads=num_ffmpeg_threads,
            device=self._device,
            seek_mode=self._seek_mode,
        )

        frames = []
        for pts in pts_list:
            frame = decoder.get_frame_played_at(pts)
            frames.append(frame)
        return frames

    def decode_first_n_frames(self, video_file, n):
        num_ffmpeg_threads = (
            int(self._num_ffmpeg_threads) if self._num_ffmpeg_threads else 0
        )
        decoder = VideoDecoder(
            video_file,
            num_ffmpeg_threads=num_ffmpeg_threads,
            device=self._device,
            seek_mode=self._seek_mode,
        )
        frames = []
        count = 0
        for frame in decoder:
            frames.append(frame)
            count += 1
            if count == n:
                break
        return frames

    def decode_and_resize(self, video_file, pts_list, height, width, device):
        num_ffmpeg_threads = (
            int(self._num_ffmpeg_threads) if self._num_ffmpeg_threads else 1
        )
        decoder = VideoDecoder(
            video_file,
            num_ffmpeg_threads=num_ffmpeg_threads,
            device=self._device,
            seek_mode=self._seek_mode,
        )

        frames = []
        for pts in pts_list:
            frame = decoder.get_frame_played_at(pts)
            frames.append(frame)

        frames = [
            self.transforms_v2.functional.resize(frame.to(device), (height, width))
            for frame in frames
        ]
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

    def decode_frames(self, video_file, pts_list):
        decoder = create_from_file(video_file)
        _add_video_stream(decoder)
        frames = []
        for pts in pts_list:
            frame = compiled_seek_and_next(decoder, pts)
            frames.append(frame)
        return frames

    def decode_first_n_frames(self, video_file, n):
        decoder = create_from_file(video_file)
        _add_video_stream(decoder)
        frames = []
        for _ in range(n):
            frame = compiled_next(decoder)
            frames.append(frame)
        return frames


class TorchAudioDecoder(AbstractDecoder):
    def __init__(self, stream_index: str | None = None):
        import torchaudio  # noqa: F401

        self.torchaudio = torchaudio

        self._stream_index = int(stream_index) if stream_index else None

    def init_decode_and_resize(self):
        from torchvision.transforms import v2 as transforms_v2

        self.transforms_v2 = transforms_v2

    def decode_frames(self, video_file, pts_list):
        stream_reader = self.torchaudio.io.StreamReader(src=video_file)
        stream_reader.add_basic_video_stream(
            frames_per_chunk=1,
            decoder_option={"threads": "0"},
            stream_index=self._stream_index,
        )
        frames = []
        for pts in pts_list:
            stream_reader.seek(pts)
            stream_reader.fill_buffer()
            clip = stream_reader.pop_chunks()
            frames.append(clip[0][0])
        return frames

    def decode_first_n_frames(self, video_file, n):
        stream_reader = self.torchaudio.io.StreamReader(src=video_file)
        stream_reader.add_basic_video_stream(
            frames_per_chunk=1,
            decoder_option={"threads": "0"},
            stream_index=self._stream_index,
        )
        frames = []
        frame_cnt = 0
        for vframe in stream_reader.stream():
            if frame_cnt >= n:
                break
            frames.append(vframe[0][0])
            frame_cnt += 1

        return frames

    def decode_and_resize(self, video_file, pts_list, height, width, device):
        stream_reader = self.torchaudio.io.StreamReader(src=video_file)
        stream_reader.add_basic_video_stream(
            frames_per_chunk=1,
            decoder_option={"threads": "1"},
            stream_index=self._stream_index,
        )
        frames = []
        for pts in pts_list:
            stream_reader.seek(pts)
            stream_reader.fill_buffer()
            clip = stream_reader.pop_chunks()
            frames.append(clip[0][0])
        frames = [
            self.transforms_v2.functional.resize(frame.to(device), (height, width))
            for frame in frames
        ]
        return frames


def create_torchcodec_core_decode_first_frame(video_file):
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


def plot_data(json_data, plot_path):
    plt.rcParams["font.size"] = 18

    # Creating the DataFrame
    df = pd.DataFrame(json_data["experiments"])

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
            ax.set_title(f"{base_video}\n{vtype}", fontsize=11)

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

    # Stamp the metadata for the experimental system on the chart.
    plt.gcf().text(
        0.005,
        0.013,
        "\n".join([f"{k}: {v}" for k, v in json_data["system_metadata"].items()]),
        fontsize=11,
        bbox=dict(facecolor="white"),
    )

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


@dataclass
class DecoderKind:
    display_name: str
    kind: typing.Type[AbstractDecoder]
    default_options: dict[str, str] = field(default_factory=dict)


decoder_registry = {
    "decord": DecoderKind("DecordAccurate", DecordAccurate),
    "decord_batch": DecoderKind("DecordAccurateBatch", DecordAccurateBatch),
    "torchcodec_core": DecoderKind("TorchCodecCore", TorchCodecCore),
    "torchcodec_core_batch": DecoderKind("TorchCodecCoreBatch", TorchCodecCoreBatch),
    "torchcodec_core_nonbatch": DecoderKind(
        "TorchCodecCoreNonBatch", TorchCodecCoreNonBatch
    ),
    "torchcodec_core_compiled": DecoderKind(
        "TorchCodecCoreCompiled", TorchCodecCoreCompiled
    ),
    "torchcodec_public": DecoderKind("TorchCodecPublic", TorchCodecPublic),
    "torchcodec_public_nonbatch": DecoderKind(
        "TorchCodecPublicNonBatch", TorchCodecPublicNonBatch
    ),
    "torchvision": DecoderKind(
        # We don't compare against TorchVision's "pyav" backend because it doesn't support
        # accurate seeks.
        "TorchVision[backend=video_reader]",
        TorchVision,
        {"backend": "video_reader"},
    ),
    "torchaudio": DecoderKind("TorchAudio", TorchAudioDecoder),
    "opencv": DecoderKind(
        "OpenCV[backend=FFMPEG]", OpenCVDecoder, {"backend": "FFMPEG"}
    ),
}


def run_benchmarks(
    decoder_dict: dict[str, AbstractDecoder],
    video_files_paths: list[Path],
    num_samples: int,
    num_sequential_frames_from_start: list[int],
    min_runtime_seconds: float,
    benchmark_video_creation: bool,
    dataloader_parameters: DataLoaderInspiredWorkloadParameters = None,
) -> list[dict[str, str | float | int]]:
    # Ensure that we have the same seed across benchmark runs.
    torch.manual_seed(0)

    print(f"video_files_paths={video_files_paths}")

    results = []
    df_data = []
    verbose = False
    for video_file_path in video_files_paths:
        metadata = get_metadata(video_file_path)
        metadata_label = f"{metadata.codec} {metadata.width}x{metadata.height}, {metadata.duration_seconds}s {metadata.average_fps}fps"

        duration = metadata.duration_seconds
        uniform_pts_list = [i * duration / num_samples for i in range(num_samples)]

        # Note that we are using the same random pts values for all decoders for the same
        # video. However, because we use the duration as part of this calculation, we
        # are using different random pts values across videos.
        random_pts_list = (torch.rand(num_samples) * duration).tolist()

        # The decoder items are sorted to perform and display the benchmarks in a consistent order.
        for decoder_name, decoder in sorted(decoder_dict.items(), key=lambda x: x[0]):
            print(f"video={video_file_path}, decoder={decoder_name}")

            if dataloader_parameters:
                bp = dataloader_parameters.batch_parameters
                decoder.init_decode_and_resize()
                description = (
                    f"concurrency {bp.num_threads}"
                    f"batch {bp.batch_size}"
                    + decoder.decode_and_resize_description(
                        num_samples,
                        dataloader_parameters.resize_height,
                        dataloader_parameters.resize_width,
                    )
                )
                dataloader_result = benchmark.Timer(
                    stmt="run_batch_using_threads(decoder.decode_and_resize, video_file, pts_list, height, width, device, batch_parameters=batch_parameters)",
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
                    description=description,
                )
                print(description)
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
                        description,
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
                    stmt="decoder.decode_frames(video_file, pts_list)",
                    globals={
                        "video_file": str(video_file_path),
                        "pts_list": pts_list,
                        "decoder": decoder,
                    },
                    label=f"video={video_file_path} {metadata_label}",
                    sub_label=decoder_name,
                    description=decoder.decode_frames_description(num_samples, kind),
                )
                print(
                    f"{decoder_name} {decoder.decode_frames_description(num_samples, kind)}"
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
                        decoder.decode_frames_description(num_samples, kind),
                    )
                )

            for num_frames in num_sequential_frames_from_start:
                consecutive_frames_result = benchmark.Timer(
                    stmt="decoder.decode_first_n_frames(video_file, n)",
                    globals={
                        "video_file": str(video_file_path),
                        "n": num_frames,
                        "decoder": decoder,
                    },
                    label=f"video={video_file_path} {metadata_label}",
                    sub_label=decoder_name,
                    description=decoder.decode_first_n_frames_description(num_frames),
                )
                print(
                    f"{decoder_name} {decoder.decode_first_n_frames_description(num_frames)}"
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
                        num_frames,
                        decoder.decode_first_n_frames_description(num_frames),
                    )
                )

        if benchmark_video_creation:
            metadata = get_metadata(video_file_path)
            metadata_label = f"{metadata.codec} {metadata.width}x{metadata.height}, {metadata.duration_seconds}s {metadata.average_fps}fps"
            creation_result = benchmark.Timer(
                stmt="create_torchcodec_core_decode_first_frame(video_file)",
                globals={
                    "video_file": str(video_file_path),
                    "create_torchcodec_core_decode_first_frame": create_torchcodec_core_decode_first_frame,
                },
                label=f"video={video_file_path} {metadata_label}",
                sub_label="TorchCodecCore",
                description="create decode first",
            )
            results.append(
                creation_result.blocked_autorange(
                    min_run_time=2.0,
                )
            )
    compare = benchmark.Compare(results)
    compare.print()
    return df_data


def verify_outputs(decoders_to_run, video_paths, num_samples):
    # Reuse TorchCodecPublic decoder stream_index option, if provided.
    options = decoder_registry["torchcodec_public"].default_options
    if torchcodec_decoder := next(
        (
            decoder
            for name, decoder in decoders_to_run.items()
            if "TorchCodecPublic" in name
        ),
        None,
    ):
        options["stream_index"] = (
            str(torchcodec_decoder._stream_index)
            if torchcodec_decoder._stream_index is not None
            else ""
        )
    # Create default TorchCodecPublic decoder to use as a baseline
    torchcodec_public_decoder = TorchCodecPublic(**options)

    # Get frames using each decoder
    for video_file_path in video_paths:
        metadata = get_metadata(video_file_path)
        metadata_label = f"{metadata.codec} {metadata.width}x{metadata.height}, {metadata.duration_seconds}s {metadata.average_fps}fps"
        print(f"{metadata_label=}")

        # Generate uniformly spaced PTS
        duration = metadata.duration_seconds
        pts_list = [i * duration / num_samples for i in range(num_samples)]

        # Get the frames from TorchCodecPublic as the baseline
        torchcodec_public_results = decode_and_adjust_frames(
            torchcodec_public_decoder,
            video_file_path,
            num_samples=num_samples,
            pts_list=pts_list,
        )

        for decoder_name, decoder in decoders_to_run.items():
            print(f"video={video_file_path}, decoder={decoder_name}")

            frames = decode_and_adjust_frames(
                decoder,
                video_file_path,
                num_samples=num_samples,
                pts_list=pts_list,
            )
            for f1, f2 in zip(torchcodec_public_results, frames):
                torch.testing.assert_close(f1, f2)
            print(f"Results of baseline TorchCodecPublic and {decoder_name} match!")


def decode_and_adjust_frames(
    decoder, video_file_path, *, num_samples: int, pts_list: list[float]
):
    frames = []
    # Decode non-sequential frames using decode_frames function
    non_seq_frames = decoder.decode_frames(video_file_path, pts_list)
    # TorchCodec's batch APIs return a FrameBatch, so we need to extract the frames
    if isinstance(non_seq_frames, FrameBatch):
        non_seq_frames = non_seq_frames.data
    frames.extend(non_seq_frames)

    # Decode sequential frames using decode_first_n_frames function
    seq_frames = decoder.decode_first_n_frames(video_file_path, num_samples)
    if isinstance(seq_frames, FrameBatch):
        seq_frames = seq_frames.data
    frames.extend(seq_frames)

    # Check and convert frames to C,H,W for consistency with other decoders.
    if frames[0].shape[-1] == 3:
        frames = [frame.permute(-1, *range(frame.dim() - 1)) for frame in frames]
    return frames
