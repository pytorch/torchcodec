# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import importlib
import importlib.machinery
import sys
import traceback
from pathlib import Path

import torch


# Copy pasted from torchvision
# https://github.com/pytorch/vision/blob/947ae1dc71867f28021d5bc0ff3a19c249236e2a/torchvision/_internally_replaced_utils.py#L25
def _get_extension_path(lib_name: str) -> str:
    extension_suffixes = []
    if sys.platform == "linux" or sys.platform.startswith("freebsd"):
        extension_suffixes = importlib.machinery.EXTENSION_SUFFIXES
    elif sys.platform == "darwin":
        extension_suffixes = importlib.machinery.EXTENSION_SUFFIXES + [".dylib"]
    elif sys.platform in ("win32", "cygwin"):
        extension_suffixes = importlib.machinery.EXTENSION_SUFFIXES + [".dll", ".pyd"]
    else:
        raise NotImplementedError(f"{sys.platform = } is not not supported")
    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        extension_suffixes,
    )

    extfinder = importlib.machinery.FileFinder(
        str(Path(__file__).parent), loader_details
    )
    ext_specs = extfinder.find_spec(lib_name)
    if ext_specs is None:
        raise ImportError(f"No spec found for {lib_name}")

    if ext_specs.origin is None:
        raise ImportError(f"Existing spec found for {lib_name} does not have an origin")

    return ext_specs.origin


def load_image_library() -> None:
    """Load the FFmpeg-free image library"""
    image_library_path = _get_extension_path("libtorchcodec_image")
    torch.ops.load_library(image_library_path)


@functools.cache
def load_heic_library() -> None:
    """Load the HEIC library (libtorchcodec_heic).

    It's loaded lazily, only when HEIC decoding is requested, never at import:
    because it links libheif, an optional runtime dependency we don't bundle.
    importing torchcodec must not require libheif to be installed.
    """
    torch.ops.load_library(_get_extension_path("libtorchcodec_heic"))


@functools.cache
def load_core_libraries() -> tuple[int, str]:
    """Load the FFmpeg-dependent shared libraries, memoizing the result.

      This raises if the libraries cannot be loaded, typically because FFmpeg
      couldn't be found. This is called:
      - at import time (import torchcodec) within a try/except, to determine
        whether FFmpeg is available, and then load the corresponding ops if it is.
        If FFmpeg isn't available, we don't propagate the exception and
        `import torchcodec` still works: we still want to support the image
        decoders, which don't need FFmpeg.
      - at runtime in all FFmpeg-dependent public entry-points like VideoDecoder,
        so that if FFmpeg isn't available the user gets a clear error message.

      Since this is @functools.cache'd, the libraries are only ever loaded once on
      success. Failures are *not* cached (functools.cache only caches return
      values), so when FFmpeg is missing the load is retried and re-raises on each
      call. That's fine, it only happens on the error path.



    We successively try to load the shared libraries for each version of FFmpeg
    that we support, starting with the highest version and working our way down.
    Once we can load the core and custom-ops libraries for a version of FFmpeg,
    we have succeeded and we stop.

    We load them via torch.ops.load_library(), which registers the custom ops
    with PyTorch's runtime so they're accessible through torch.ops.
    """
    exceptions = []
    for ffmpeg_major_version in (9, 8, 7, 6, 5, 4):
        core_library_name = f"libtorchcodec_core{ffmpeg_major_version}"
        custom_ops_library_name = f"libtorchcodec_custom_ops{ffmpeg_major_version}"
        try:
            core_library_path = _get_extension_path(core_library_name)
            torch.ops.load_library(core_library_path)
            torch.ops.load_library(_get_extension_path(custom_ops_library_name))
            return (ffmpeg_major_version, core_library_path)
        except Exception:
            # Capture the full traceback for this exception
            exc_traceback = traceback.format_exc()
            exceptions.append((ffmpeg_major_version, exc_traceback))

    traceback_info = (
        "\n[start of libtorchcodec loading traceback]\n"
        + "\n".join(f"FFmpeg version {v}:\n{tb}" for v, tb in exceptions)
        + "[end of libtorchcodec loading traceback]."
    )
    raise RuntimeError(
        f"""Could not load libtorchcodec. Likely causes:
          1. FFmpeg is not properly installed in your environment. We support
             versions 4, 5, 6, 7, 8, and 9, and we attempt to load libtorchcodec
             for each of those versions. Errors for versions not installed on
             your system are expected; only the error for your installed FFmpeg
             version is relevant. On Windows, ensure you've installed the
             "full-shared" version which ships DLLs.
          2. The PyTorch version ({torch.__version__}) is not compatible with
             this version of TorchCodec. Refer to the version compatibility
             table:
             https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec.
          3. Another runtime dependency; see exceptions below.

        The following exceptions were raised as we tried to load libtorchcodec:
        """
        f"{traceback_info}"
    )
