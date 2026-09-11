# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# FFmpeg-dependent ops, thin Python wrappers, and torch.compile abstract impls.
#
# This module is split out from ``ops.py`` and imported by it when FFmpeg is
# available.

import io
import json
import warnings

import torch
from torch.library import get_ctx, register_fake
from torchcodec._core._ffmpeg_op_names import FFMPEG_OP_NAMES
from torchcodec._internally_replaced_utils import load_pybind_ops

__all__ = sorted(FFMPEG_OP_NAMES)

_pybind_ops = load_pybind_ops()


# Note: We use disallow_in_graph because PyTorch does constant propagation of
# factory functions.
create_from_file = torch._dynamo.disallow_in_graph(
    torch.ops.torchcodec_ns.create_from_file.default
)
create_from_tensor = torch._dynamo.disallow_in_graph(
    torch.ops.torchcodec_ns.create_from_tensor.default
)
_create_from_file_like = torch._dynamo.disallow_in_graph(
    torch.ops.torchcodec_ns._create_from_file_like.default
)
_add_video_stream_raw = torch.ops.torchcodec_ns.add_video_stream.default
_add_video_stream = torch.ops.torchcodec_ns._add_video_stream.default


def add_video_stream(
    decoder: torch.Tensor,
    *,
    num_threads: int | None = None,
    dimension_order: str | None = None,
    stream_index: int | None = None,
    device: str = "cpu",
    device_variant: str = "default",
    transform_specs: str = "",
    custom_frame_mappings: (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    ) = None,
    output_dtype: str = "uint8",
) -> None:
    custom_frame_mappings_pts: torch.Tensor | None = None
    custom_frame_mappings_keyframe_indices: torch.Tensor | None = None
    custom_frame_mappings_duration: torch.Tensor | None = None
    if custom_frame_mappings is not None:
        (
            custom_frame_mappings_pts,
            custom_frame_mappings_keyframe_indices,
            custom_frame_mappings_duration,
        ) = custom_frame_mappings
    _add_video_stream_raw(
        decoder,
        num_threads=num_threads,
        dimension_order=dimension_order,
        stream_index=stream_index,
        device=device,
        device_variant=device_variant,
        transform_specs=transform_specs,
        custom_frame_mappings_pts=custom_frame_mappings_pts,
        custom_frame_mappings_duration=custom_frame_mappings_duration,
        custom_frame_mappings_keyframe_indices=custom_frame_mappings_keyframe_indices,
        output_dtype=output_dtype,
    )


add_audio_stream = torch.ops.torchcodec_ns.add_audio_stream.default
seek_to_pts = torch.ops.torchcodec_ns.seek_to_pts.default
get_next_frame = torch.ops.torchcodec_ns.get_next_frame.default
get_frame_at_pts = torch.ops.torchcodec_ns.get_frame_at_pts.default
get_frame_at_index = torch.ops.torchcodec_ns.get_frame_at_index.default
_get_frames_at_indices_tensor_input = (
    torch.ops.torchcodec_ns.get_frames_at_indices.default
)
_get_frames_by_pts_tensor_input = torch.ops.torchcodec_ns.get_frames_by_pts.default
get_frames_in_range = torch.ops.torchcodec_ns.get_frames_in_range.default
get_frames_by_pts_in_range = torch.ops.torchcodec_ns.get_frames_by_pts_in_range.default
get_frames_by_pts_in_range_audio = (
    torch.ops.torchcodec_ns.get_frames_by_pts_in_range_audio.default
)
get_json_metadata = torch.ops.torchcodec_ns.get_json_metadata.default

_blocks_create_demuxer_from_file = (
    torch.ops.torchcodec_ns._blocks_create_demuxer_from_file.default
)
_blocks_create_demuxer_from_tensor = (
    torch.ops.torchcodec_ns._blocks_create_demuxer_from_tensor.default
)
_blocks_create_demuxer_from_file_like_context = (
    torch.ops.torchcodec_ns._blocks_create_demuxer_from_file_like.default
)
_blocks_demuxer_add_stream = torch.ops.torchcodec_ns._blocks_demuxer_add_stream.default
_blocks_demuxer_get_audio_video_stream_indices = (
    torch.ops.torchcodec_ns._blocks_demuxer_get_audio_video_stream_indices.default
)
_blocks_demuxer_container_json_metadata = (
    torch.ops.torchcodec_ns._blocks_demuxer_container_json_metadata.default
)
_blocks_demuxer_stream_json_metadata = (
    torch.ops.torchcodec_ns._blocks_demuxer_stream_json_metadata.default
)
_blocks_demuxer_next_packet = (
    torch.ops.torchcodec_ns._blocks_demuxer_next_packet.default
)
_blocks_demuxer_seek = torch.ops.torchcodec_ns._blocks_demuxer_seek.default
_blocks_demuxer_scan = torch.ops.torchcodec_ns._blocks_demuxer_scan.default
_blocks_create_packet_decoder = (
    torch.ops.torchcodec_ns._blocks_create_packet_decoder.default
)
_blocks_packet_decoder_send_packet = (
    torch.ops.torchcodec_ns._blocks_packet_decoder_send_packet.default
)
_blocks_packet_decoder_send_eof = (
    torch.ops.torchcodec_ns._blocks_packet_decoder_send_eof.default
)
_blocks_packet_decoder_reset = (
    torch.ops.torchcodec_ns._blocks_packet_decoder_reset.default
)
_blocks_packet_decoder_receive_frame = (
    torch.ops.torchcodec_ns._blocks_packet_decoder_receive_frame.default
)
_blocks_audio_packet_decoder_receive_frame = (
    torch.ops.torchcodec_ns._blocks_audio_packet_decoder_receive_frame.default
)
_blocks_create_color_converter = (
    torch.ops.torchcodec_ns._blocks_create_color_converter.default
)
_blocks_convert_frame = torch.ops.torchcodec_ns._blocks_convert_frame.default
_blocks_create_audio_converter = (
    torch.ops.torchcodec_ns._blocks_create_audio_converter.default
)
_blocks_audio_converter_convert = (
    torch.ops.torchcodec_ns._blocks_audio_converter_convert.default
)
_blocks_audio_converter_drain = (
    torch.ops.torchcodec_ns._blocks_audio_converter_drain.default
)
_blocks_audio_converter_reset = (
    torch.ops.torchcodec_ns._blocks_audio_converter_reset.default
)
_blocks_frame_metadata = torch.ops.torchcodec_ns._blocks_frame_metadata.default
_blocks_frame_planes = torch.ops.torchcodec_ns._blocks_frame_planes.default

_test_frame_pts_equality = torch.ops.torchcodec_ns._test_frame_pts_equality.default
_get_container_json_metadata = (
    torch.ops.torchcodec_ns.get_container_json_metadata.default
)
_get_key_frame_indices = torch.ops.torchcodec_ns._get_key_frame_indices.default
scan_all_streams_to_update_metadata = (
    torch.ops.torchcodec_ns.scan_all_streams_to_update_metadata.default
)
_get_stream_json_metadata = torch.ops.torchcodec_ns.get_stream_json_metadata.default
_get_json_ffmpeg_library_versions = (
    torch.ops.torchcodec_ns._get_json_ffmpeg_library_versions.default
)
_get_backend_details = torch.ops.torchcodec_ns._get_backend_details.default
create_streaming_encoder = torch._dynamo.disallow_in_graph(
    torch.ops.torchcodec_ns.create_streaming_encoder.default
)
streaming_encoder_close = torch.ops.torchcodec_ns.streaming_encoder_close.default
streaming_encoder_add_video_stream = (
    torch.ops.torchcodec_ns.streaming_encoder_add_video_stream.default
)
streaming_encoder_add_audio_stream = (
    torch.ops.torchcodec_ns.streaming_encoder_add_audio_stream.default
)
streaming_encoder_open_file = (
    torch.ops.torchcodec_ns.streaming_encoder_open_file.default
)
_streaming_encoder_open_file_like = (
    torch.ops.torchcodec_ns.streaming_encoder_open_file_like.default
)
streaming_encoder_add_frames = (
    torch.ops.torchcodec_ns.streaming_encoder_add_frames.default
)
streaming_encoder_add_samples = (
    torch.ops.torchcodec_ns.streaming_encoder_add_samples.default
)
set_nvdec_cache_capacity = torch.ops.torchcodec_ns.set_nvdec_cache_capacity.default
get_nvdec_cache_capacity = torch.ops.torchcodec_ns.get_nvdec_cache_capacity.default
_get_nvdec_cache_size = torch.ops.torchcodec_ns._get_nvdec_cache_size.default
_set_cpp_log_level = torch.ops.torchcodec_ns._set_cpp_log_level.default
_get_log_level = torch.ops.torchcodec_ns._get_log_level.default
create_wav_decoder_from_file = (
    torch.ops.torchcodec_ns.create_wav_decoder_from_file.default
)
create_wav_decoder_from_tensor = (
    torch.ops.torchcodec_ns.create_wav_decoder_from_tensor.default
)
_create_wav_decoder_from_file_like = (
    torch.ops.torchcodec_ns._create_wav_decoder_from_file_like.default
)
get_wav_samples_in_range = torch.ops.torchcodec_ns.get_wav_samples_in_range.default
get_wav_metadata_from_decoder = (
    torch.ops.torchcodec_ns.get_wav_metadata_from_decoder.default
)


# =============================
# Functions not related to custom ops, but similar implementation to c++ ops
# =============================
def _bytes_to_tensor(data: bytes) -> torch.Tensor:
    with warnings.catch_warnings():
        # Ignore warning stating that the underlying buffer is non-writable.
        warnings.filterwarnings("ignore", category=UserWarning)
        return torch.frombuffer(data, dtype=torch.uint8)


def create_from_bytes(video_bytes: bytes, seek_mode: str | None = None) -> torch.Tensor:
    return create_from_tensor(_bytes_to_tensor(video_bytes), seek_mode)


def create_from_file_like(
    file_like: io.RawIOBase | io.BufferedReader, seek_mode: str | None = None
) -> torch.Tensor:
    assert _pybind_ops is not None
    return _create_from_file_like(
        _pybind_ops.create_file_like_context(
            file_like, False  # False means not for writing
        ),
        seek_mode,
    )


def _blocks_create_demuxer_from_bytes(video_bytes: bytes) -> torch.Tensor:
    return _blocks_create_demuxer_from_tensor(_bytes_to_tensor(video_bytes))


def _blocks_create_demuxer_from_file_like(
    file_like: io.RawIOBase | io.BufferedReader,
) -> torch.Tensor:
    assert _pybind_ops is not None
    return _blocks_create_demuxer_from_file_like_context(
        _pybind_ops.create_file_like_context(
            file_like, False  # False means not for writing
        ),
    )


def create_wav_decoder_from_bytes(wav_bytes: bytes) -> torch.Tensor:
    return create_wav_decoder_from_tensor(_bytes_to_tensor(wav_bytes))


def create_wav_decoder_from_file_like(
    file_like: io.RawIOBase | io.BufferedReader,
) -> torch.Tensor:
    assert _pybind_ops is not None
    return _create_wav_decoder_from_file_like(
        _pybind_ops.create_file_like_context(
            file_like, False  # False means not for writing
        ),
    )


def streaming_encoder_open_file_like(
    encoder: torch.Tensor,
    format: str,
    file_like: io.RawIOBase | io.BufferedIOBase,
) -> None:
    assert _pybind_ops is not None
    _streaming_encoder_open_file_like(
        encoder,
        format,
        _pybind_ops.create_file_like_context(file_like, True),
    )


def get_frames_at_indices(
    decoder: torch.Tensor, *, frame_indices: torch.Tensor | list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(frame_indices, torch.Tensor):
        # Ensure indices is the correct dtype (int64)
        frame_indices = frame_indices.to(torch.int64)
    else:
        # Convert list to tensor for dispatch
        frame_indices = torch.tensor(frame_indices)
    return _get_frames_at_indices_tensor_input(decoder, frame_indices=frame_indices)


def get_frames_by_pts(
    decoder: torch.Tensor, *, timestamps: torch.Tensor | list[float]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(timestamps, torch.Tensor):
        # Ensure indices is the correct dtype (float64)
        timestamps = timestamps.to(torch.float64)
    else:
        # Convert list to tensor for dispatch
        try:
            timestamps = torch.tensor(timestamps, dtype=torch.float64)
        except Exception as e:
            raise ValueError("Couldn't convert timestamps input to a tensor") from e
    return _get_frames_by_pts_tensor_input(decoder, timestamps=timestamps)


# ==============================
# Abstract impl for the operators. Needed by torch.compile.
# ==============================
@register_fake("torchcodec_ns::create_from_file")
def create_from_file_abstract(filename: str, seek_mode: str | None) -> torch.Tensor:
    return torch.empty([], dtype=torch.long)


@register_fake("torchcodec_ns::_create_from_file_like")
def _create_from_file_like_abstract(
    file_like: int, seek_mode: str | None
) -> torch.Tensor:
    return torch.empty([], dtype=torch.long)


@register_fake("torchcodec_ns::create_from_tensor")
def create_from_tensor_abstract(
    video_tensor: torch.Tensor, seek_mode: str | None
) -> torch.Tensor:
    return torch.empty([], dtype=torch.long)


@register_fake("torchcodec_ns::_add_video_stream")
def _add_video_stream_abstract(
    decoder: torch.Tensor,
    *,
    num_threads: int | None = None,
    dimension_order: str | None = None,
    stream_index: int | None = None,
    device: str = "cpu",
    device_variant: str = "default",
    transform_specs: str = "",
    custom_frame_mappings_pts: torch.Tensor | None = None,
    custom_frame_mappings_duration: torch.Tensor | None = None,
    custom_frame_mappings_keyframe_indices: torch.Tensor | None = None,
    color_conversion_library: str | None = None,
    output_dtype: str = "uint8",
) -> None:
    return


@register_fake("torchcodec_ns::add_video_stream")
def add_video_stream_abstract(
    decoder: torch.Tensor,
    *,
    num_threads: int | None = None,
    dimension_order: str | None = None,
    stream_index: int | None = None,
    device: str = "cpu",
    device_variant: str = "default",
    transform_specs: str = "",
    custom_frame_mappings_pts: torch.Tensor | None = None,
    custom_frame_mappings_duration: torch.Tensor | None = None,
    custom_frame_mappings_keyframe_indices: torch.Tensor | None = None,
    output_dtype: str = "uint8",
) -> None:
    return


@register_fake("torchcodec_ns::add_audio_stream")
def add_audio_stream_abstract(
    decoder: torch.Tensor,
    *,
    stream_index: int | None = None,
    sample_rate: int | None = None,
    num_channels: int | None = None,
) -> None:
    return


@register_fake("torchcodec_ns::seek_to_pts")
def seek_abstract(decoder: torch.Tensor, seconds: float) -> None:
    return


@register_fake("torchcodec_ns::get_next_frame")
def get_next_frame_abstract(
    decoder: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Images are 3 dimensions: height, width, channels.
    # The exact permutation depends on the constructor options passed in.
    image_size = [get_ctx().new_dynamic_size() for _ in range(3)]
    return (
        torch.empty(image_size),
        torch.empty([], dtype=torch.float),
        torch.empty([], dtype=torch.float),
    )


@register_fake("torchcodec_ns::get_frame_at_pts")
def get_frame_at_pts_abstract(
    decoder: torch.Tensor, seconds: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_size = [get_ctx().new_dynamic_size() for _ in range(3)]
    return (
        torch.empty(image_size),
        torch.empty([], dtype=torch.float),
        torch.empty([], dtype=torch.float),
    )


@register_fake("torchcodec_ns::get_frames_by_pts")
def get_frames_by_pts_abstract(
    decoder: torch.Tensor,
    *,
    timestamps: torch.Tensor | list[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_size = [get_ctx().new_dynamic_size() for _ in range(4)]
    return (
        torch.empty(image_size),
        torch.empty([], dtype=torch.float),
        torch.empty([], dtype=torch.float),
    )


@register_fake("torchcodec_ns::get_frame_at_index")
def get_frame_at_index_abstract(
    decoder: torch.Tensor, *, frame_index: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_size = [get_ctx().new_dynamic_size() for _ in range(3)]
    return (
        torch.empty(image_size),
        torch.empty([], dtype=torch.float),
        torch.empty([], dtype=torch.float),
    )


@register_fake("torchcodec_ns::get_frames_at_indices")
def get_frames_at_indices_abstract(
    decoder: torch.Tensor, *, frame_indices: torch.Tensor | list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_size = [get_ctx().new_dynamic_size() for _ in range(4)]
    return (
        torch.empty(image_size),
        torch.empty([], dtype=torch.float),
        torch.empty([], dtype=torch.float),
    )


@register_fake("torchcodec_ns::get_frames_in_range")
def get_frames_in_range_abstract(
    decoder: torch.Tensor,
    *,
    start: int,
    stop: int,
    step: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_size = [get_ctx().new_dynamic_size() for _ in range(4)]
    return (
        torch.empty(image_size),
        torch.empty([], dtype=torch.float),
        torch.empty([], dtype=torch.float),
    )


@register_fake("torchcodec_ns::get_frames_by_pts_in_range")
def get_frames_by_pts_in_range_abstract(
    decoder: torch.Tensor,
    *,
    start_seconds: float,
    stop_seconds: float,
    fps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_size = [get_ctx().new_dynamic_size() for _ in range(4)]
    return (
        torch.empty(image_size),
        torch.empty([], dtype=torch.float),
        torch.empty([], dtype=torch.float),
    )


@register_fake("torchcodec_ns::get_frames_by_pts_in_range_audio")
def get_frames_by_pts_in_range_audio_abstract(
    decoder: torch.Tensor,
    *,
    start_seconds: float,
    stop_seconds: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_size = [get_ctx().new_dynamic_size() for _ in range(4)]
    return (torch.empty(image_size), torch.empty([], dtype=torch.float))


@register_fake("torchcodec_ns::_get_key_frame_indices")
def get_key_frame_indices_abstract(decoder: torch.Tensor) -> torch.Tensor:
    return torch.empty([], dtype=torch.int)


@register_fake("torchcodec_ns::get_json_metadata")
def get_json_metadata_abstract(decoder: torch.Tensor) -> str:
    return ""


@register_fake("torchcodec_ns::get_container_json_metadata")
def get_container_json_metadata_abstract(decoder: torch.Tensor) -> str:
    return ""


@register_fake("torchcodec_ns::get_stream_json_metadata")
def get_stream_json_metadata_abstract(decoder: torch.Tensor, stream_idx: int) -> str:
    return ""


@register_fake("torchcodec_ns::_test_frame_pts_equality")
def _test_frame_pts_equality_abstract(
    decoder: torch.Tensor,
    *,
    frame_index: int,
    pts_seconds_to_test: float,
) -> bool:
    return False


@register_fake("torchcodec_ns::_get_json_ffmpeg_library_versions")
def _get_json_ffmpeg_library_versions_abstract() -> str:
    return ""


@register_fake("torchcodec_ns::scan_all_streams_to_update_metadata")
def scan_all_streams_to_update_metadata_abstract(decoder: torch.Tensor) -> None:
    return


def get_ffmpeg_library_versions():
    versions_json = _get_json_ffmpeg_library_versions()
    return json.loads(versions_json)


@register_fake("torchcodec_ns::_get_backend_details")
def _get_backend_details_abstract(decoder: torch.Tensor) -> str:
    return ""


@register_fake("torchcodec_ns::create_streaming_encoder")
def _create_streaming_encoder_abstract() -> torch.Tensor:
    return torch.empty([], dtype=torch.long)


@register_fake("torchcodec_ns::streaming_encoder_close")
def streaming_encoder_close_abstract(encoder: torch.Tensor) -> None:
    return


@register_fake("torchcodec_ns::streaming_encoder_add_video_stream")
def streaming_encoder_add_video_stream_abstract(
    encoder: torch.Tensor,
    height: int,
    width: int,
    frame_rate: float,
    device: str = "cpu",
    codec: str | None = None,
    pixel_format: str | None = None,
    crf: float | None = None,
    preset: str | None = None,
    extra_options: list[str] | None = None,
) -> int:
    return 0


@register_fake("torchcodec_ns::streaming_encoder_add_audio_stream")
def streaming_encoder_add_audio_stream_abstract(
    encoder: torch.Tensor,
    sample_rate: int,
    num_channels: int,
    bit_rate: int | None = None,
    output_num_channels: int | None = None,
    output_sample_rate: int | None = None,
) -> int:
    return 0


@register_fake("torchcodec_ns::streaming_encoder_open_file")
def streaming_encoder_open_file_abstract(encoder: torch.Tensor, filename: str) -> None:
    return


@register_fake("torchcodec_ns::streaming_encoder_open_file_like")
def streaming_encoder_open_file_like_abstract(
    encoder: torch.Tensor, format: str, file_like_context: int
) -> None:
    return


@register_fake("torchcodec_ns::streaming_encoder_add_frames")
def streaming_encoder_add_frames_abstract(
    encoder: torch.Tensor, frames: torch.Tensor, stream_index: int
) -> None:
    return


@register_fake("torchcodec_ns::streaming_encoder_add_samples")
def streaming_encoder_add_samples_abstract(
    encoder: torch.Tensor, samples: torch.Tensor, stream_index: int
) -> None:
    return


@register_fake("torchcodec_ns::set_nvdec_cache_capacity")
def set_nvdec_cache_capacity_abstract(capacity: int) -> None:
    return


@register_fake("torchcodec_ns::get_nvdec_cache_capacity")
def get_nvdec_cache_capacity_abstract() -> int:
    return 0


@register_fake("torchcodec_ns::_get_nvdec_cache_size")
def _get_nvdec_cache_size_abstract(device_index: int) -> int:
    return 0


@register_fake("torchcodec_ns::_set_cpp_log_level")
def _set_cpp_log_level_abstract(level: int) -> None:
    return


@register_fake("torchcodec_ns::_get_log_level")
def _get_log_level_abstract() -> int:
    return 0


@register_fake("torchcodec_ns::create_wav_decoder_from_file")
def create_wav_decoder_from_file_abstract(filename: str) -> torch.Tensor:
    return torch.empty([], dtype=torch.long)


@register_fake("torchcodec_ns::create_wav_decoder_from_tensor")
def create_wav_decoder_from_tensor_abstract(data: torch.Tensor) -> torch.Tensor:
    return torch.empty([], dtype=torch.long)


@register_fake("torchcodec_ns::_create_wav_decoder_from_file_like")
def _create_wav_decoder_from_file_like_abstract(
    file_like_context: int,
) -> torch.Tensor:
    return torch.empty([], dtype=torch.long)


@register_fake("torchcodec_ns::get_wav_samples_in_range")
def get_wav_samples_in_range_abstract(
    decoder: torch.Tensor, start_seconds: float, stop_seconds: float | None
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_size = [
        get_ctx().new_dynamic_size() for _ in range(2)
    ]  # [channels, samples]
    frames = torch.empty(sample_size)
    pts = torch.empty([], dtype=torch.float64)
    return frames, pts


@register_fake("torchcodec_ns::get_wav_metadata_from_decoder")
def get_wav_metadata_from_decoder_abstract(decoder: torch.Tensor) -> str:
    return ""
