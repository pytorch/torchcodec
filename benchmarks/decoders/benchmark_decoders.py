# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import argparse
import importlib
import json
import os
import timeit

import torch
import torch.utils.benchmark as benchmark
from torchcodec.decoders import SimpleVideoDecoder

from torchcodec.decoders._core import (
    add_video_stream,
    create_from_file,
    get_frames_at_indices,
    get_json_metadata,
    get_next_frame,
    scan_all_streams_to_update_metadata,
    seek_to_pts,
)

torch._dynamo.config.cache_size_limit = 100
torch._dynamo.config.capture_dynamic_output_shape_ops = True


def in_fbcode() -> bool:
    return "FB_PAR_RUNTIME_FILES" in os.environ


class AbstractDecoder:
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_frames_from_video(self, video_file, pts_list):
        pass


class DecordNonBatchDecoderAccurateSeek(AbstractDecoder):
    def __init__(self):
        import decord  # noqa: F401

        self.decord = decord

        self._print_each_iteration_time = False

    def get_frames_from_video(self, video_file, pts_list):
        self.decord.bridge.set_bridge("torch")
        decord_vr = self.decord.VideoReader(video_file, ctx=self.decord.cpu())
        frames = []
        times = []
        fps = decord_vr.get_avg_fps()
        for pts in pts_list:
            start = timeit.default_timer()
            decord_vr.seek_accurate(int(pts * fps))
            frame = decord_vr.next()
            end = timeit.default_timer()
            times.append(round(end - start, 3))
            frames.append(frame)
        if self._print_each_iteration_time:
            print("decord times=", times, sum(times))
        return frames

    def get_consecutive_frames_from_video(self, video_file, numFramesToDecode):
        self.decord.bridge.set_bridge("torch")
        decord_vr = self.decord.VideoReader(video_file, ctx=self.decord.cpu())
        frames = []
        times = []
        for _ in range(numFramesToDecode):
            start = timeit.default_timer()
            frame = decord_vr.next()
            end = timeit.default_timer()
            times.append(round(end - start, 3))
            frames.append(frame)
        if self._print_each_iteration_time:
            print("decord times=", times, sum(times))
        return frames


class TVNewAPIDecoderWithBackend(AbstractDecoder):
    def __init__(self, backend):
        self._backend = backend
        self._print_each_iteration_time = False
        import torchvision  # noqa: F401

        self.torchvision = torchvision

    def get_frames_from_video(self, video_file, pts_list):
        start = timeit.default_timer()
        self.torchvision.set_video_backend(self._backend)
        reader = self.torchvision.io.VideoReader(video_file, "video")
        create_done = timeit.default_timer()
        frames = []
        for pts in pts_list:
            reader.seek(pts)
            frame = next(reader)
            frames.append(frame["data"].permute(1, 2, 0))
        frames_done = timeit.default_timer()
        if self._print_each_iteration_time:
            del reader
        del_done = timeit.default_timer()
        create_duration = 1000 * round(create_done - start, 3)
        frames_duration = 1000 * round(frames_done - create_done, 3)
        del_duration = 1000 * round(del_done - frames_done, 3)
        total_duration = 1000 * round(del_done - start, 3)
        if self._print_each_iteration_time:
            print(
                f"TV: {create_duration=} {frames_duration=} {del_duration=} {total_duration=}"
            )
        return frames

    def get_consecutive_frames_from_video(self, video_file, numFramesToDecode):
        start = timeit.default_timer()
        self.torchvision.set_video_backend(self._backend)
        reader = self.torchvision.io.VideoReader(video_file, "video")
        create_done = timeit.default_timer()
        frames = []
        for _ in range(numFramesToDecode):
            frame = next(reader)
            frames.append(frame["data"].permute(1, 2, 0))
        frames_done = timeit.default_timer()
        if self._print_each_iteration_time:
            del reader
        del_done = timeit.default_timer()
        create_duration = 1000 * round(create_done - start, 3)
        frames_duration = 1000 * round(frames_done - create_done, 3)
        del_duration = 1000 * round(del_done - frames_done, 3)
        total_duration = 1000 * round(del_done - start, 3)
        if self._print_each_iteration_time:
            print(
                f"TV: consecutive: {create_duration=} {frames_duration=} {del_duration=} {total_duration=} {frames[0].shape=}"
            )
        return frames


class TorchcodecNonCompiledWithOptions(AbstractDecoder):
    def __init__(self, num_threads=None, color_conversion_library=None):
        self._print_each_iteration_time = False
        self._num_threads = int(num_threads) if num_threads else None
        self._color_conversion_library = color_conversion_library

    def get_frames_from_video(self, video_file, pts_list):
        decoder = create_from_file(video_file)
        add_video_stream(
            decoder,
            num_threads=self._num_threads,
            color_conversion_library=self._color_conversion_library,
        )
        frames = []
        times = []
        for pts in pts_list:
            start = timeit.default_timer()
            seek_to_pts(decoder, pts)
            frame = get_next_frame(decoder)
            end = timeit.default_timer()
            times.append(round(end - start, 3))
            frames.append(frame)
        if self._print_each_iteration_time:
            print("torchcodec times=", times, sum(times))
        return frames

    def get_consecutive_frames_from_video(self, video_file, numFramesToDecode):
        create_time = timeit.default_timer()
        decoder = create_from_file(video_file)
        add_stream_time = timeit.default_timer()
        add_video_stream(
            decoder,
            num_threads=self._num_threads,
            color_conversion_library=self._color_conversion_library,
        )
        frames = []
        times = []
        frames_time = timeit.default_timer()
        for _ in range(numFramesToDecode):
            start = timeit.default_timer()
            frame = get_next_frame(decoder)
            end = timeit.default_timer()
            times.append(round(end - start, 3))
            frames.append(frame)
        del_time = timeit.default_timer()
        if self._print_each_iteration_time:
            del decoder
        done_time = timeit.default_timer()
        create_duration = 1000 * round(add_stream_time - create_time, 3)
        add_stream_duration = 1000 * round(frames_time - add_stream_time, 3)
        frames_duration = 1000 * round(del_time - frames_time, 3)
        del_duration = 1000 * round(done_time - del_time, 3)
        total_duration = 1000 * round(done_time - create_time, 3)
        if self._print_each_iteration_time:
            print(
                f"{numFramesToDecode=} {create_duration=} {add_stream_duration=} {frames_duration=} {del_duration=} {total_duration=} {frames[0][0].shape=}"
            )
            print("torchcodec times=", times, sum(times))
        return frames


class TorchCodecNonCompiledBatch(AbstractDecoder):
    def __init__(self, num_threads=None, color_conversion_library=None):
        self._print_each_iteration_time = False
        self._num_threads = int(num_threads) if num_threads else None
        self._color_conversion_library = color_conversion_library

    def get_frames_from_video(self, video_file, pts_list):
        decoder = create_from_file(video_file)
        scan_all_streams_to_update_metadata(decoder)
        add_video_stream(
            decoder,
            num_threads=self._num_threads,
            color_conversion_library=self._color_conversion_library,
        )
        metadata = json.loads(get_json_metadata(decoder))
        average_fps = metadata["averageFps"]
        best_video_stream = metadata["bestVideoStreamIndex"]
        indexes_list = [int(pts * average_fps) for pts in pts_list]
        frames = []
        frames = get_frames_at_indices(
            decoder, stream_index=best_video_stream, frame_indices=indexes_list
        )
        return frames

    def get_consecutive_frames_from_video(self, video_file, numFramesToDecode):
        decoder = create_from_file(video_file)
        scan_all_streams_to_update_metadata(decoder)
        add_video_stream(
            decoder,
            num_threads=self._num_threads,
            color_conversion_library=self._color_conversion_library,
        )
        metadata = json.loads(get_json_metadata(decoder))
        best_video_stream = metadata["bestVideoStreamIndex"]
        frames = []
        indices_list = list(range(numFramesToDecode))
        frames = get_frames_at_indices(
            decoder, stream_index=best_video_stream, frame_indices=indices_list
        )
        return frames


@torch.compile(fullgraph=True, backend="eager")
def compiled_seek_and_next(decoder, pts):
    seek_to_pts(decoder, pts)
    return get_next_frame(decoder)


@torch.compile(fullgraph=True, backend="eager")
def compiled_next(decoder):
    return get_next_frame(decoder)


class TorchcodecCompiled(AbstractDecoder):
    def __init__(self):
        pass

    def get_frames_from_video(self, video_file, pts_list):
        decoder = create_from_file(video_file)
        add_video_stream(decoder)
        frames = []
        for pts in pts_list:
            frame = compiled_seek_and_next(decoder, pts)
            frames.append(frame)
        return frames

    def get_consecutive_frames_from_video(self, video_file, numFramesToDecode):
        decoder = create_from_file(video_file)
        add_video_stream(decoder)
        frames = []
        for _ in range(numFramesToDecode):
            frame = compiled_next(decoder)
            frames.append(frame)
        return frames


class TorchAudioDecoder(AbstractDecoder):
    def __init__(self):
        import torchaudio  # noqa: F401

        self.torchaudio = torchaudio

        pass

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


def get_test_resource_path(filename: str) -> str:
    if in_fbcode():
        resource = importlib.resources.files(__package__).joinpath(filename)
        with importlib.resources.as_file(resource) as path:
            return os.fspath(path)
    return os.path.join(
        os.path.dirname(__file__), "..", "..", "test", "resources", filename
    )


def create_torchcodec_decoder_from_file(video_file):
    video_decoder = create_from_file(video_file)
    add_video_stream(video_decoder)
    get_next_frame(video_decoder)
    return video_decoder


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
        default="decord,torchcodec,torchvision,torchaudio,torchcodec1,torchcodec_compiled,torchcodec_filtergraph,torchcodec_swsscale",
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
                k, v = item.split("=")
                kwargs_dict[k] = v
            decoder_dict["TorchCodecNonCompiledBatch:" + options] = (
                TorchCodecNonCompiledBatch(**kwargs_dict)
            )
        elif decoder.startswith("tcoptions:"):
            options = decoder[len("tcoptions:") :]
            kwargs_dict = {}
            for item in options.split("+"):
                k, v = item.split("=")
                kwargs_dict[k] = v
            decoder_dict["TorchcodecNonCompiled:" + options] = (
                TorchcodecNonCompiledWithOptions(**kwargs_dict)
            )

    decoder_dict["TVNewAPIDecoderWithBackendVideoReader"]

    results = []
    for decoder_name, decoder in decoder_dict.items():
        for video_path in args.bm_video_paths.split(","):
            # We only use the SimpleVideoDecoder to get the metadata and get
            # the list of PTS values to seek to.
            simple_decoder = SimpleVideoDecoder(video_path)
            duration = simple_decoder.metadata.duration_seconds
            pts_list = [
                i * duration / num_uniform_samples for i in range(num_uniform_samples)
            ]
            metadata = simple_decoder.metadata
            metadata_string = f"{metadata.codec} {metadata.width}x{metadata.height}, {metadata.duration_seconds}s {metadata.average_fps}fps"
            if args.verbose:
                print(
                    f"video={video_path}, decoder={decoder_name}, pts_list={pts_list}"
                )
            seeked_result = benchmark.Timer(
                stmt="decoder.get_frames_from_video(video_file, pts_list)",
                globals={
                    "video_file": video_path,
                    "pts_list": pts_list,
                    "decoder": decoder,
                },
                label=f"video={video_path} {metadata_string}",
                sub_label=decoder_name,
                description=f"{num_uniform_samples} seek()+next()",
            )
            results.append(
                seeked_result.blocked_autorange(
                    min_run_time=args.bm_video_speed_min_run_seconds
                )
            )
            for num_consecutive_nexts in [1, 10]:
                consecutive_frames_result = benchmark.Timer(
                    stmt="decoder.get_consecutive_frames_from_video(video_file, consecutive_frames_to_extract)",
                    globals={
                        "video_file": video_path,
                        "consecutive_frames_to_extract": num_consecutive_nexts,
                        "decoder": decoder,
                    },
                    label=f"video={video_path} {metadata_string}",
                    sub_label=decoder_name,
                    description=f"{num_consecutive_nexts} next()",
                )
                results.append(
                    consecutive_frames_result.blocked_autorange(
                        min_run_time=args.bm_video_speed_min_run_seconds
                    )
                )

    first_video_path = args.bm_video_paths.split(",")[0]
    if args.bm_video_creation:
        creation_result = benchmark.Timer(
            stmt="create_torchcodec_decoder_from_file(video_file)",
            globals={
                "video_file": first_video_path,
                "create_torchcodec_decoder_from_file": create_torchcodec_decoder_from_file,
            },
            label=f"video={first_video_path} {metadata_string}",
            sub_label="TorchcodecNonCompiled",
            description="create()+next()",
        )
        results.append(
            creation_result.blocked_autorange(
                min_run_time=2.0,
            )
        )
    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":
    main()
