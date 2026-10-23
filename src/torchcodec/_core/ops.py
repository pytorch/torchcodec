# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torchcodec._core._ffmpeg_op_names import FFMPEG_OP_NAMES
from torchcodec._core._file_like import create_file_like_context  # noqa: F401
from torchcodec._internally_replaced_utils import (  # @manual=//pytorch/torchcodec/src:internally_replaced_utils
    load_core_libraries,
    load_heic_library,
    load_image_library,
)

expose_ffmpeg_dlls = nullcontext
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    # On windows we try to locate the FFmpeg DLLs and temporarily add them to
    # the DLL search path. This seems to be needed on some users machine, but
    # not on our CI. We don't know why.
    if ffmpeg_path := shutil.which("ffmpeg"):

        def expose_ffmpeg_dlls():  # noqa: F811
            ffmpeg_dir = Path(ffmpeg_path).parent.absolute()
            return os.add_dll_directory(str(ffmpeg_dir))  # that's the actual CM


load_image_library()

decode_jpeg = torch.ops.torchcodec_ns.decode_jpeg.default
decode_jpegs_cuda = torch.ops.torchcodec_ns.decode_jpegs_cuda.default
decode_png = torch.ops.torchcodec_ns.decode_png.default
decode_webp = torch.ops.torchcodec_ns.decode_webp.default
decode_gif = torch.ops.torchcodec_ns.decode_gif.default
decode_avif = torch.ops.torchcodec_ns.decode_avif.default
encode_png_to_file = torch.ops.torchcodec_ns.encode_png_to_file.default
encode_png_to_file_like = torch.ops.torchcodec_ns.encode_png_to_file_like.default
encode_jpeg_to_file = torch.ops.torchcodec_ns.encode_jpeg_to_file.default
encode_jpeg_to_file_like = torch.ops.torchcodec_ns.encode_jpeg_to_file_like.default
encode_jpeg_to_tensor_cuda = torch.ops.torchcodec_ns.encode_jpeg_to_tensor_cuda.default


def get_decode_heic():
    # decode_heic lives in the separate, lazily-loaded libtorchcodec_heic (see
    # load_heic_library), so its op isn't registered until that library loads,
    # and loading fails if the user hasn't installed libheif. Turn that into an
    # actionable error.
    try:
        load_heic_library()
    except Exception as e:
        raise ImportError(
            "Failed to load the HEIC decoding library. HEIC decoding requires "
            "libheif to be installed and discoverable at runtime; TorchCodec "
            "does not bundle it (it's LGPL). Install it, e.g. with "
            "`conda install -c conda-forge libheif`, `apt install libheif1`, or "
            "`brew install libheif`."
        ) from e
    return torch.ops.torchcodec_ns.decode_heic.default


# FFmpeg is now an optional dependency, since the image decoders above don't
# need it, and we want users to be able to use them without having FFmpeg
# installed.  We try to load the ffmpeg-dependent ops at import time and bind
# their names to the actual implementation. If it doesn't work
# (_FFMPEG_AVAILABLE is then False), we bind the names to a dummy lambda that
# will raise a proper error when called.
try:
    with expose_ffmpeg_dlls():
        ffmpeg_major_version, core_library_path = load_core_libraries()
    _FFMPEG_AVAILABLE = True
except Exception:
    _FFMPEG_AVAILABLE = False
    ffmpeg_major_version = None
    core_library_path = None


if _FFMPEG_AVAILABLE:
    from torchcodec._core._ffmpeg_ops import *  # noqa: F401,F403
else:

    def __getattr__(name):
        # Mocks all names in FFMPEG_OP_NAMES to raise an error when called, if
        # FFmpeg is not available.
        if name in FFMPEG_OP_NAMES:
            return lambda *args, **kwargs: load_core_libraries()
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
