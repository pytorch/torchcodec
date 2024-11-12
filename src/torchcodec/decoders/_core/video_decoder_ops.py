# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import warnings
from typing import List, Optional, Tuple

import torch
from torch.library import get_ctx, register_fake

from torchcodec._internally_replaced_utils import (  # @manual=//pytorch/torchcodec/src:internally_replaced_utils
    _get_extension_path,
)


def load_torchcodec_extension():
    # Successively try to load libtorchcodec7.so, libtorchcodec6.so,
    # libtorchcodec5.so, and libtorchcodec4.so. Each of these correspond to an
    # ffmpeg major version. This should cover all potential ffmpeg versions
    # installed on the user's machine.
    #
    # On fbcode, _get_extension_path() is overridden and directly points to the
    # correct .so file, so this for-loop succeeds on the first iteration.

    exceptions = []
    for ffmpeg_major_version in (7, 6, 5, 4):
        library_name = f"libtorchcodec{ffmpeg_major_version}"
        try:
            torch.ops.load_library(_get_extension_path(library_name))
            return
        except Exception as e:
            # TODO: recording and reporting exceptions this way is OK for now as  it's just for debugging,
            # but we should probably handle that via a proper logging mechanism.
            exceptions.append(e)

    traceback = (
        "\n[start of libtorchcodec loading traceback]\n"
        + "\n".join(str(e) for e in exceptions)
        + "\n[end of libtorchcodec loading traceback]."
    )
    raise RuntimeError(
        f"""Could not load libtorchcodec. Likely causes:
          1. FFmpeg is not properly installed in your environment. We support
             versions 4, 5, 6 and 7.
          2. The PyTorch version ({torch.__version__}) is not compatible with
             this version of TorchCodec. Refer to the version compatibility
             table:
             https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec.
          3. Another runtime dependency; see exceptions below.
        The following exceptions were raised as we tried to load libtorchcodec:
        """
        f"{traceback}"
    )


load_torchcodec_extension()


# Note: We use disallow_in_graph because PyTorch does constant propagation of
# factory functions.
create_from_file = torch._dynamo.disallow_in_graph(
    torch.ops.torchcodec_ns.create_from_file.default
)
create_from_tensor = torch._dynamo.disallow_in_graph(
    torch.ops.torchcodec_ns.create_from_tensor.default
)
add_video_stream = torch.ops.torchcodec_ns.add_video_stream.default
_add_video_stream = torch.ops.torchcodec_ns._add_video_stream.default
seek_to_pts = torch.ops.torchcodec_ns.seek_to_pts.default
get_next_frame = torch.ops.torchcodec_ns.get_next_frame.default
get_frame_at_pts = torch.ops.torchcodec_ns.get_frame_at_pts.default
get_frame_at_index = torch.ops.torchcodec_ns.get_frame_at_index.default
get_frames_at_indices = torch.ops.torchcodec_ns.get_frames_at_indices.default
get_frames_by_pts = torch.ops.torchcodec_ns.get_frames_by_pts.default
get_frames_in_range = torch.ops.torchcodec_ns.get_frames_in_range.default
get_frames_by_pts_in_range = torch.ops.torchcodec_ns.get_frames_by_pts_in_range.default
get_json_metadata = torch.ops.torchcodec_ns.get_json_metadata.default
_test_frame_pts_equality = torch.ops.torchcodec_ns._test_frame_pts_equality.default
_get_container_json_metadata = (
    torch.ops.torchcodec_ns.get_container_json_metadata.default
)
scan_all_streams_to_update_metadata = (
    torch.ops.torchcodec_ns.scan_all_streams_to_update_metadata.default
)
_get_stream_json_metadata = torch.ops.torchcodec_ns.get_stream_json_metadata.default
_get_json_ffmpeg_library_versions = (
    torch.ops.torchcodec_ns._get_json_ffmpeg_library_versions.default
)


# =============================
# Functions not related to custom ops, but similar implementation to c++ ops
# =============================
def create_from_bytes(video_bytes: bytes) -> torch.Tensor:
    with warnings.catch_warnings():
        # Ignore warning stating that the underlying video_bytes buffer is
        # non-writable.
        warnings.filterwarnings("ignore", category=UserWarning)
        buffer = torch.frombuffer(video_bytes, dtype=torch.uint8)
    return create_from_tensor(buffer)


# ==============================
# Abstract impl for the operators. Needed by torch.compile.
# ==============================
@register_fake("torchcodec_ns::create_from_file")
def create_from_file_abstract(filename: str) -> torch.Tensor:
    return torch.empty([], dtype=torch.long)


@register_fake("torchcodec_ns::create_from_tensor")
def create_from_tensor_abstract(video_tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty([], dtype=torch.long)


@register_fake("torchcodec_ns::_add_video_stream")
def _add_video_stream_abstract(
    decoder: torch.Tensor,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_threads: Optional[int] = None,
    dimension_order: Optional[str] = None,
    stream_index: Optional[int] = None,
    device: Optional[str] = None,
    color_conversion_library: Optional[str] = None,
) -> None:
    return


@register_fake("torchcodec_ns::add_video_stream")
def add_video_stream_abstract(
    decoder: torch.Tensor,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_threads: Optional[int] = None,
    dimension_order: Optional[str] = None,
    stream_index: Optional[int] = None,
    device: Optional[str] = None,
) -> None:
    return


@register_fake("torchcodec_ns::seek_to_pts")
def seek_abstract(decoder: torch.Tensor, seconds: float) -> None:
    return


@register_fake("torchcodec_ns::get_next_frame")
def get_next_frame_abstract(
    decoder: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    stream_index: int,
    timestamps: List[float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_size = [get_ctx().new_dynamic_size() for _ in range(4)]
    return (
        torch.empty(image_size),
        torch.empty([], dtype=torch.float),
        torch.empty([], dtype=torch.float),
    )


@register_fake("torchcodec_ns::get_frame_at_index")
def get_frame_at_index_abstract(
    decoder: torch.Tensor, *, stream_index: int, frame_index: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_size = [get_ctx().new_dynamic_size() for _ in range(3)]
    return (
        torch.empty(image_size),
        torch.empty([], dtype=torch.float),
        torch.empty([], dtype=torch.float),
    )


@register_fake("torchcodec_ns::get_frames_at_indices")
def get_frames_at_indices_abstract(
    decoder: torch.Tensor,
    *,
    stream_index: int,
    frame_indices: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    stream_index: int,
    start: int,
    stop: int,
    step: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    stream_index: int,
    start_seconds: float,
    stop_seconds: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_size = [get_ctx().new_dynamic_size() for _ in range(4)]
    return (
        torch.empty(image_size),
        torch.empty([], dtype=torch.float),
        torch.empty([], dtype=torch.float),
    )


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
    stream_index: int,
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
