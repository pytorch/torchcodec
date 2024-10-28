import abc
import json
import timeit

import torch
import torch.utils.benchmark as benchmark
from torchcodec.decoders import VideoDecoder

from torchcodec.decoders._core import (
    _add_video_stream,
    create_from_file,
    get_frames_at_indices,
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
            create_duration = 1000 * round(create_done - start, 3)
            frames_duration = 1000 * round(frames_done - create_done, 3)
            total_duration = 1000 * round(frames_done - start, 3)
            print(f"TV: {create_duration=} {frames_duration=} {total_duration=}")
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
            create_duration = 1000 * round(create_done - start, 3)
            frames_duration = 1000 * round(frames_done - create_done, 3)
            total_duration = 1000 * round(frames_done - start, 3)
            print(
                f"TV: consecutive: {create_duration=} {frames_duration=} {total_duration=} {frames[0].shape=}"
            )
        return frames


class TorchcodecNonCompiledWithOptions(AbstractDecoder):
    def __init__(self, num_threads=None, color_conversion_library=None):
        self._print_each_iteration_time = False
        self._num_threads = int(num_threads) if num_threads else None
        self._color_conversion_library = color_conversion_library

    def get_frames_from_video(self, video_file, pts_list):
        decoder = create_from_file(video_file)
        _add_video_stream(
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
        _add_video_stream(
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

        if self._print_each_iteration_time:
            done_time = timeit.default_timer()
            create_duration = 1000 * round(add_stream_time - create_time, 3)
            add_stream_duration = 1000 * round(frames_time - add_stream_time, 3)
            frames_duration = 1000 * round(done_time - frames_time, 3)
            total_duration = 1000 * round(done_time - create_time, 3)
            print(
                f"{numFramesToDecode=} {create_duration=} {add_stream_duration=} {frames_duration=} {total_duration=} {frames[0][0].shape=}"
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
        _add_video_stream(
            decoder,
            num_threads=self._num_threads,
            color_conversion_library=self._color_conversion_library,
        )
        metadata = json.loads(get_json_metadata(decoder))
        average_fps = metadata["averageFps"]
        best_video_stream = metadata["bestVideoStreamIndex"]
        indices_list = [int(pts * average_fps) for pts in pts_list]
        frames = []
        frames, *_ = get_frames_at_indices(
            decoder, stream_index=best_video_stream, frame_indices=indices_list
        )
        return frames

    def get_consecutive_frames_from_video(self, video_file, numFramesToDecode):
        decoder = create_from_file(video_file)
        scan_all_streams_to_update_metadata(decoder)
        _add_video_stream(
            decoder,
            num_threads=self._num_threads,
            color_conversion_library=self._color_conversion_library,
        )
        metadata = json.loads(get_json_metadata(decoder))
        best_video_stream = metadata["bestVideoStreamIndex"]
        frames = []
        indices_list = list(range(numFramesToDecode))
        frames, *_ = get_frames_at_indices(
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


def create_torchcodec_decoder_from_file(video_file):
    video_decoder = create_from_file(video_file)
    _add_video_stream(video_decoder)
    get_next_frame(video_decoder)
    return video_decoder


def run_benchmarks(
    decoder_dict,
    video_paths,
    num_uniform_samples,
    min_runtime_seconds,
    benchmark_video_creation,
):
    results = []
    verbose = False
    for decoder_name, decoder in decoder_dict.items():
        for video_path in video_paths.split(","):
            print(f"video={video_path}, decoder={decoder_name}")
            # We only use the VideoDecoder to get the metadata and get
            # the list of PTS values to seek to.
            simple_decoder = VideoDecoder(video_path)
            duration = simple_decoder.metadata.duration_seconds
            pts_list = [
                i * duration / num_uniform_samples for i in range(num_uniform_samples)
            ]
            metadata = simple_decoder.metadata
            metadata_string = f"{metadata.codec} {metadata.width}x{metadata.height}, {metadata.duration_seconds}s {metadata.average_fps}fps"
            if verbose:
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
                seeked_result.blocked_autorange(min_run_time=min_runtime_seconds)
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
                        min_run_time=min_runtime_seconds
                    )
                )

        first_video_path = video_paths.split(",")[0]
        if benchmark_video_creation:
            simple_decoder = VideoDecoder(first_video_path)
            metadata = simple_decoder.metadata
            metadata_string = f"{metadata.codec} {metadata.width}x{metadata.height}, {metadata.duration_seconds}s {metadata.average_fps}fps"
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
