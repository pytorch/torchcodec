from typing import List, Optional

import torch
from torch.library import get_ctx, register_fake

from torchcodec._internally_replaced_utils import (  # @manual=//pytorch/torchcodec/src:internally_replaced_utils
    _get_extension_path,
)

# TODO: cleanup, don't print stuff, raise better error messages.
# TODO: I think we need `""` for fbcode, need to check
for ffmpeg_version in (6, 5, 4, ""):
    libtorchcodec = f"libtorchcodec{ffmpeg_version}"
    print(f"Trying to load {libtorchcodec}")
    try:
        torch.ops.load_library(_get_extension_path(libtorchcodec))
        print("SUCCESS!!!")
        break
    except Exception as e:
        print(e)
        continue  # Try lower version of ffmpeg
    
    raise RuntimeError("Could not load libtorchcodec.")

# TODO: PyTorch team needs to figure out how to not constant prop factory functions
create_from_file = torch._dynamo.disallow_in_graph(
    torch.ops.torchcodec_ns.create_from_file.default
)
create_from_tensor = torch._dynamo.disallow_in_graph(
    torch.ops.torchcodec_ns.create_from_tensor.default
)
add_video_stream = torch.ops.torchcodec_ns.add_video_stream.default
seek_to_pts = torch.ops.torchcodec_ns.seek_to_pts.default
get_next_frame = torch.ops.torchcodec_ns.get_next_frame.default
get_frame_at_pts = torch.ops.torchcodec_ns.get_frame_at_pts.default
get_frame_at_index = torch.ops.torchcodec_ns.get_frame_at_index.default
get_frames_at_indices = torch.ops.torchcodec_ns.get_frames_at_indices.default
get_json_metadata = torch.ops.torchcodec_ns.get_json_metadata.default


# =============================
# Functions not related to custom ops, but similar implementation to c++ ops
# =============================
def create_from_bytes(video_bytes: bytes) -> torch.Tensor:
    return create_from_tensor(torch.frombuffer(video_bytes, dtype=torch.uint8))


# ==============================
# Abstract impl for the operators. Needed by torch.compile.
# ==============================
@register_fake("torchcodec_ns::create_from_file")
def create_from_file_abstract(filename: str) -> torch.Tensor:
    return torch.empty([], dtype=torch.long)


@register_fake("torchcodec_ns::create_from_tensor")
def create_from_tensor_abstract(video_tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty([], dtype=torch.long)


@register_fake("torchcodec_ns::add_video_stream")
def add_video_stream_abstract(
    decoder: torch.Tensor,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_threads: Optional[int] = None,
    shape: Optional[str] = None,
    stream_index: Optional[int] = None
) -> None:
    return


@register_fake("torchcodec_ns::seek_to_pts")
def seek_abstract(decoder: torch.Tensor, seconds: float) -> torch.Tensor:
    return


@register_fake("torchcodec_ns::get_next_frame")
def get_next_frame_abstract(decoder: torch.Tensor) -> torch.Tensor:
    # Images are 3 dimensions: height, width, channels.
    # The exact permutation depends on the constructor options passed in.
    image_size = [get_ctx().new_dynamic_size() for _ in range(3)]
    return torch.empty(image_size)


@register_fake("torchcodec_ns::get_frame_at_pts")
def get_frame_at_pts_abstract(decoder: torch.Tensor, seconds: float) -> torch.Tensor:
    image_size = [get_ctx().new_dynamic_size() for _ in range(3)]
    return torch.empty(image_size)


@register_fake("torchcodec_ns::get_frame_at_index")
def get_frame_at_index_abstract(
    decoder: torch.Tensor, *, frame_index: int, stream_index: Optional[int] = None
) -> torch.Tensor:
    image_size = [get_ctx().new_dynamic_size() for _ in range(3)]
    return torch.empty(image_size)


@register_fake("torchcodec_ns::get_frames_at_indices")
def get_frames_at_indices_abstract(
    decoder: torch.Tensor,
    *,
    frame_indices: List[int],
    stream_index: Optional[int] = None
) -> torch.Tensor:
    image_size = [get_ctx().new_dynamic_size() for _ in range(4)]
    return torch.empty(image_size)


@register_fake("torchcodec_ns::get_json_metadata")
def get_json_metadata_abstract(decoder: torch.Tensor) -> str:
    return torch.empty_like("")
