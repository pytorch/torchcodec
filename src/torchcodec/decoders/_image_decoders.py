# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from enum import Enum
from pathlib import Path
from typing import Literal

import torch
from torchcodec._core.ops import (
    decode_avif as _decode_avif,
    decode_gif as _decode_gif,
    decode_jpeg as _decode_jpeg,
    decode_jpegs_cuda as _decode_jpegs_cuda,
    decode_png as _decode_png,
    decode_webp as _decode_webp,
    get_decode_heic as _get_decode_heic,
)


class ImageReadMode(Enum):
    """Color mode for image decoding.

    You don't have to use this, you can just pass strings like "RGB" or "GRAY" instead.
    """

    UNCHANGED = 0
    GRAY = 1
    GRAY_ALPHA = 2
    RGB = 3
    RGB_ALPHA = 4
    RGBA = 4  # undocumented alias for RGB_ALPHA


def _normalize_mode(
    mode: (
        Literal["UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"] | ImageReadMode
    ),
) -> ImageReadMode:
    # Normalize the public `mode` argument (a case-insensitive string, or an
    # ImageReadMode for BC) to an ImageReadMode, which is what the rest of the
    # decoding code works with.
    if isinstance(mode, ImageReadMode):
        return mode
    if isinstance(mode, str):
        try:
            return ImageReadMode[mode.upper()]
        except KeyError:
            valid = ", ".join(repr(m.name) for m in ImageReadMode)
            raise ValueError(
                f"Invalid mode {mode!r}. Supported modes are {valid} "
                "(case-insensitive)."
            ) from None
    raise TypeError(f"mode must be a str (or ImageReadMode), got {type(mode)}.")


def _source_to_tensor(source: str | Path | bytes | torch.Tensor) -> torch.Tensor:
    # Turn any supported source into a 1-D uint8 tensor of encoded bytes.
    if isinstance(source, torch.Tensor):
        # dtype is validated in cpp.
        return source
    if isinstance(source, (str, Path)):
        # We keep the file reading in pure Python (rather than a C++ read_file
        # op like in torchvision): benchmarked against a C++ op, the
        # read is only ~1-1.5% of total decode time.
        source = Path(source).read_bytes()
    if isinstance(source, (bytes, bytearray)):
        with warnings.catch_warnings():
            # torch.frombuffer warns that the underlying buffer is non-writable;
            # we only read from the resulting tensor, so this is safe to ignore.
            warnings.filterwarnings("ignore", category=UserWarning)
            return torch.frombuffer(source, dtype=torch.uint8)

    raise TypeError(
        f"Unknown source type: {type(source)}. "
        "Supported types are str, Path, bytes and torch.Tensor."
    )


# Output modes each native codec can produce directly, for any input. Any output
# mode not listed here is obtained via a post-decode conversion.
_JPEG_NATIVE_OUTPUT_MODES = frozenset(
    (ImageReadMode.UNCHANGED, ImageReadMode.GRAY, ImageReadMode.RGB)
)
_PNG_NATIVE_OUTPUT_MODES = frozenset(ImageReadMode)
_WEBP_NATIVE_OUTPUT_MODES = _GIF_NATIVE_OUTPUT_MODES = _AVIF_NATIVE_OUTPUT_MODES = (
    _HEIC_NATIVE_OUTPUT_MODES
) = frozenset((ImageReadMode.UNCHANGED, ImageReadMode.RGB, ImageReadMode.RGB_ALPHA))


def _append_opaque_alpha(img: torch.Tensor) -> torch.Tensor:
    # Append a fully-opaque alpha channel on the channel dim
    # works on CHW and NCHW tensors.
    alpha_shape = list(img.shape)
    alpha_shape[-3] = 1
    alpha = torch.full(
        alpha_shape, torch.iinfo(img.dtype).max, dtype=img.dtype, device=img.device
    )
    return torch.cat([img, alpha], dim=-3)


def _rgb_to_gray(img: torch.Tensor) -> torch.Tensor:
    # ITU-R 601-2 luma weights, matching torchvision's rgb_to_grayscale.
    # works on CHW and NCHW tensors.
    weights = torch.tensor([0.2989, 0.587, 0.114])
    gray = (img.to(torch.float32) * weights[:, None, None]).sum(dim=-3, keepdim=True)
    return gray.round().clamp(0, torch.iinfo(img.dtype).max).to(img.dtype)


def _decode_to_mode(decode_fn, data, mode, native_output_modes) -> torch.Tensor:
    if mode in native_output_modes:
        return decode_fn(data, mode.value)

    if mode is ImageReadMode.GRAY:
        return _rgb_to_gray(decode_fn(data, ImageReadMode.RGB.value))
    elif mode is ImageReadMode.RGB_ALPHA:
        return _append_opaque_alpha(decode_fn(data, ImageReadMode.RGB.value))
    elif mode is ImageReadMode.GRAY_ALPHA:
        if ImageReadMode.RGB_ALPHA in native_output_modes:
            # Real alpha available (e.g. webp): decode RGBA and reduce the color
            # channels to luma while preserving the alpha channel.
            rgba = decode_fn(data, ImageReadMode.RGB_ALPHA.value)
            rgb, alpha = rgba[..., :3, :, :], rgba[..., 3:, :, :]
            return torch.cat([_rgb_to_gray(rgb), alpha], dim=-3)
        elif ImageReadMode.GRAY in native_output_modes:
            return _append_opaque_alpha(decode_fn(data, ImageReadMode.GRAY.value))
        else:
            gray = _rgb_to_gray(decode_fn(data, ImageReadMode.RGB.value))
            return _append_opaque_alpha(gray)
    else:
        raise RuntimeError(
            f"Reached an unexpected code path while decoding to mode {mode}. "
            "This should never happen, please report a bug to the TorchCodec repo."
        )


def _validate_output_dtype(output_dtype) -> int:
    # Validates output_dtype and returns the integer code understood by the C++
    # decoders. Must be kept in-sync with the OutputDtype enum in ImageCommon.h.
    output_dtype_to_code = {torch.uint8: 0, torch.uint16: 1, "auto": 2}
    if output_dtype not in output_dtype_to_code:
        raise ValueError(
            f"Invalid output_dtype ({output_dtype}). "
            "Supported values are torch.uint8, torch.uint16, and 'auto'."
        )
    return output_dtype_to_code[output_dtype]


def _to_output_dtype(
    decoded: torch.Tensor, output_dtype: torch.dtype | str
) -> torch.Tensor:
    if output_dtype == "auto" or decoded.dtype == output_dtype:
        return decoded
    elif output_dtype == torch.uint16:
        if decoded.dtype != torch.uint8:
            raise RuntimeError("Oops, please report a bug to the TorchCodec repo.")
        return (decoded.to(torch.int32) * 257).to(torch.uint16)
    elif output_dtype == torch.uint8:
        if decoded.dtype != torch.uint16:
            raise RuntimeError("Oops, please report a bug to the TorchCodec repo.")
        return (decoded.to(torch.float32) / 257).round().clamp(0, 255).to(torch.uint8)
    else:
        raise RuntimeError(
            "This should never happen, please report a bug to the TorchCodec repo."
        )


def _decode_jpegs_cuda_with_mode(
    tensors: list[torch.Tensor], mode: ImageReadMode, device: torch.device
) -> list[torch.Tensor]:
    # Batched GPU equivalent of _decode_to_mode for JPEG: decode the whole
    # batch in one nvJPEG call using a native mode, then emulate the alpha modes
    # per-image in Python (nvJPEG natively supports UNCHANGED, GRAY and RGB, same
    # as libjpeg on CPU).
    if mode in _JPEG_NATIVE_OUTPUT_MODES:
        return _decode_jpegs_cuda(tensors, mode.value, device)
    if mode is ImageReadMode.RGB_ALPHA:
        decoded = _decode_jpegs_cuda(tensors, ImageReadMode.RGB.value, device)
        return [_append_opaque_alpha(img) for img in decoded]
    if mode is ImageReadMode.GRAY_ALPHA:
        decoded = _decode_jpegs_cuda(tensors, ImageReadMode.GRAY.value, device)
        return [_append_opaque_alpha(img) for img in decoded]
    raise RuntimeError(
        f"Reached an unexpected code path while decoding to mode {mode}. "
        "This should never happen, please report a bug to the TorchCodec repo."
    )


# TODO_ROCM: How do we document ROCm support? "CUDA" isn't enough to know ROCm is supported.
# TODO_ROCM: Install instructions on README etc.
def decode_jpeg(
    source: str | Path | bytes | torch.Tensor | list,
    *,
    mode: (
        Literal["UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"] | ImageReadMode
    ) = "RGB",
    output_dtype: torch.dtype | Literal["auto"] = torch.uint8,
    device: str | torch.device = "cpu",
) -> torch.Tensor | list[torch.Tensor]:
    """Decode a JPEG image into a ``CHW`` tensor, on CPU or CUDA.

    .. note::

        For CUDA decoding, prefer passing a batch (a list of sources) in a
        single call: the whole batch is decoded in one nvJPEG/rocJPEG call, which
        is much faster than decoding images one at a time.
        Passing a batch of sources is supported on CPU too, but it won't be
        faster than decoding them one at a time.

    Example:

        .. code-block:: python

            from torchcodec.decoders import decode_jpeg

            img = decode_jpeg("image.jpg")
            img = decode_jpeg("image.jpg", device="cuda")  # decode on GPU

    Args:
        source (str, ``pathlib.Path``, bytes, ``torch.Tensor``, or list of these):
            The encoded JPEG data: a path (``str`` or ``pathlib.Path``), a
            ``bytes`` object, or a 1-D uint8 ``torch.Tensor`` of the raw encoded
            bytes. Pass a list (or tuple) to decode a batch, in which case a list of
            tensors is returned instead of a single tensor. The encoded bytes must
            live on CPU, even when decoding to a CUDA device.
        mode (str or ImageReadMode, optional): Desired color mode of the output
            image. Can be one of ``"UNCHANGED"``, ``"GRAY"``, ``"GRAY_ALPHA"``,
            ``"RGB"``, or ``"RGB_ALPHA"``. Default is ``"RGB"``.
        output_dtype (torch.dtype or ``"auto"``, optional): desired dtype of the
            output image tensor. Accepted values are ``torch.uint8`` (default),
            ``torch.uint16``, and ``"auto"``. Since JPEG is an 8-bit format,
            ``"auto"`` and ``torch.uint8`` are equivalent. ``torch.uint16``
            emulates a 16-bit output by scaling the 8-bit values to the full
            16-bit range (0-255 -> 0-65535).
        device (str or torch.device, optional): Device to decode on, ``"cpu"``
            (default) or a CUDA device. On a CUDA device, decoding uses nvJPEG on
            NVIDIA GPUs and rocJPEG on AMD/ROCm GPUs (both exposed under the
            ``"cuda"`` device string). We recommend passing a batch of sources
            when decoding on CUDA, for speed.

    Returns:
        torch.Tensor or list of torch.Tensor of shape ``C, H, W``: The decoded
        image(s). A single tensor for a single source, or a list of tensors for
        a batch.
    """
    _validate_output_dtype(output_dtype)
    mode = _normalize_mode(mode)
    device = torch.device(device)

    is_batch = isinstance(source, (list, tuple))
    sources: list = list(source) if is_batch else [source]  # type: ignore[arg-type]

    if device.type == "cpu":
        decoded_list = [
            _to_output_dtype(
                _decode_to_mode(
                    _decode_jpeg,
                    _source_to_tensor(s),
                    mode,
                    _JPEG_NATIVE_OUTPUT_MODES,
                ),
                output_dtype,
            )
            for s in sources
        ]
    else:
        tensors = [_source_to_tensor(s) for s in sources]
        decoded_list = _decode_jpegs_cuda_with_mode(tensors, mode, device)
        decoded_list = [_to_output_dtype(img, output_dtype) for img in decoded_list]

    return decoded_list if is_batch else decoded_list[0]


def decode_png(
    source: str | Path | bytes | torch.Tensor,
    *,
    mode: (
        Literal["UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"] | ImageReadMode
    ) = "RGB",
    output_dtype: torch.dtype | Literal["auto"] = torch.uint8,
) -> torch.Tensor:
    """Decode a PNG image into a ``CHW`` tensor.

    Example:

        .. code-block:: python

            from torchcodec.decoders import decode_png

            img = decode_png("image.png")

    Args:
        source (str, ``pathlib.Path``, bytes, or ``torch.Tensor``):
            The encoded PNG data: a path (``str`` or ``pathlib.Path``), a
            ``bytes`` object, or a 1-D uint8 ``torch.Tensor`` of the raw encoded
            bytes.
        mode (str or ImageReadMode, optional): Desired color mode of the output
            image. Can be one of ``"UNCHANGED"``, ``"GRAY"``, ``"GRAY_ALPHA"``,
            ``"RGB"``, or ``"RGB_ALPHA"``. Default is ``"RGB"``.
        output_dtype (torch.dtype or ``"auto"``, optional): desired dtype of the
            output image tensor. Accepted values are ``torch.uint8`` (default),
            ``torch.uint16``, and ``"auto"``. PNG images can natively store
            16-bit samples: ``torch.uint16`` preserves that precision (8-bit
            sources are scaled up, 0-255 -> 0-65535), while ``torch.uint8``
            scales 16-bit sources down. ``"auto"`` keeps the source's native bit
            depth, yielding uint8 for 8-bit PNGs and uint16 for 16-bit ones.

    Returns:
        torch.Tensor: The decoded image, of shape ``(C, H, W)``.
    """
    output_dtype_code = _validate_output_dtype(output_dtype)
    mode = _normalize_mode(mode)
    data = _source_to_tensor(source)
    return _decode_to_mode(
        lambda d, m: _decode_png(d, m, output_dtype_code),
        data,
        mode,
        _PNG_NATIVE_OUTPUT_MODES,
    )


def decode_webp(
    source: str | Path | bytes | torch.Tensor,
    *,
    mode: (
        Literal["UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"] | ImageReadMode
    ) = "RGB",
    output_dtype: torch.dtype | Literal["auto"] = torch.uint8,
) -> torch.Tensor:
    """Decode a WebP image into a ``[N]CHW`` tensor.

    The output shape is ``(C, H, W)`` for a still WebP and ``(N, C, H, W)`` for
    an animated one (N frames).

    Example:

        .. code-block:: python

            from torchcodec.decoders import decode_webp

            img = decode_webp("image.webp")

    Args:
        source (str, ``pathlib.Path``, bytes, or ``torch.Tensor``):
            The encoded WebP data: a path (``str`` or ``pathlib.Path``), a
            ``bytes`` object, or a 1-D uint8 ``torch.Tensor`` of the raw encoded
            bytes.
        mode (str or ImageReadMode, optional): Desired color mode of the output
            image. Can be one of ``"UNCHANGED"``, ``"GRAY"``, ``"GRAY_ALPHA"``,
            ``"RGB"``, or ``"RGB_ALPHA"``. Default is ``"RGB"``.
        output_dtype (torch.dtype or ``"auto"``, optional): desired dtype of the
            output image tensor. Accepted values are ``torch.uint8`` (default),
            ``torch.uint16``, and ``"auto"``. Since WebP is an 8-bit format,
            ``"auto"`` and ``torch.uint8`` are equivalent. ``torch.uint16``
            emulates a 16-bit output by scaling the 8-bit values to the full
            16-bit range (0-255 -> 0-65535).

    Returns:
        torch.Tensor: The decoded image, of shape ``(C, H, W)`` (still) or
        ``(N, C, H, W)`` (animated).
    """
    _validate_output_dtype(output_dtype)
    mode = _normalize_mode(mode)
    data = _source_to_tensor(source)
    decoded = _decode_to_mode(_decode_webp, data, mode, _WEBP_NATIVE_OUTPUT_MODES)
    return _to_output_dtype(decoded, output_dtype)


def decode_gif(
    source: str | Path | bytes | torch.Tensor,
    *,
    mode: (
        Literal["UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"] | ImageReadMode
    ) = "RGB",
    output_dtype: torch.dtype | Literal["auto"] = torch.uint8,
) -> torch.Tensor:
    """Decode a GIF image into a ``[N]CHW`` tensor.

    The output shape is ``(C, H, W)`` for a still GIF and ``(N, C, H, W)`` for
    an animated one (N frames).

    Example:

        .. code-block:: python

            from torchcodec.decoders import decode_gif

            img = decode_gif("image.gif")

    Args:
        source (str, ``pathlib.Path``, bytes, or ``torch.Tensor``):
            The encoded GIF data: a path (``str`` or ``pathlib.Path``), a
            ``bytes`` object, or a 1-D uint8 ``torch.Tensor`` of the raw encoded
            bytes.
        mode (str or ImageReadMode, optional): Desired color mode of the output
            image. Can be one of ``"UNCHANGED"``, ``"GRAY"``, ``"GRAY_ALPHA"``,
            ``"RGB"``, or ``"RGB_ALPHA"``. Default is ``"RGB"``.
        output_dtype (torch.dtype or ``"auto"``, optional): desired dtype of the
            output image tensor. Accepted values are ``torch.uint8`` (default),
            ``torch.uint16``, and ``"auto"``. Since GIF is an 8-bit format,
            ``"auto"`` and ``torch.uint8`` are equivalent. ``torch.uint16``
            emulates a 16-bit output by scaling the 8-bit values to the full
            16-bit range (0-255 -> 0-65535).

    Returns:
        torch.Tensor: The decoded image, of shape ``(C, H, W)`` (still) or
        ``(N, C, H, W)`` (animated).
    """
    _validate_output_dtype(output_dtype)
    mode = _normalize_mode(mode)
    data = _source_to_tensor(source)
    decoded = _decode_to_mode(_decode_gif, data, mode, _GIF_NATIVE_OUTPUT_MODES)
    return _to_output_dtype(decoded, output_dtype)


def decode_avif(
    source: str | Path | bytes | torch.Tensor,
    *,
    mode: (
        Literal["UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"] | ImageReadMode
    ) = "RGB",
    output_dtype: torch.dtype | Literal["auto"] = torch.uint8,
    num_threads: int = 1,
) -> torch.Tensor:
    """Decode an AVIF image into a ``[N]CHW`` tensor.

    The output shape is ``(C, H, W)`` for a still AVIF and ``(N, C, H, W)`` for
    an animated one (N frames).

    Example:

        .. code-block:: python

            from torchcodec.decoders import decode_avif

            img = decode_avif("image.avif")

    Args:
        source (str, ``pathlib.Path``, bytes, or ``torch.Tensor``):
            The encoded AVIF data: a path (``str`` or ``pathlib.Path``), a
            ``bytes`` object, or a 1-D uint8 ``torch.Tensor`` of the raw encoded
            bytes.
        mode (str or ImageReadMode, optional): Desired color mode of the output
            image. Can be one of ``"UNCHANGED"``, ``"GRAY"``, ``"GRAY_ALPHA"``,
            ``"RGB"``, or ``"RGB_ALPHA"``. Default is ``"RGB"``.
        output_dtype (torch.dtype or ``"auto"``, optional): desired dtype of the
            output image tensor. Accepted values are ``torch.uint8`` (default),
            ``torch.uint16``, and ``"auto"``. AVIF can store more than 8 bits per
            channel (e.g. 10- or 12-bit sources). ``torch.uint16`` always scales
            the samples up to fill the full 16-bit range ``[0, 65535]`` (8-bit
            0-255, 10-bit 0-1023 and 12-bit 0-4095 sources are all upscaled),
            while ``torch.uint8`` scales higher-bit sources down. ``"auto"``
            yields uint8 for 8-bit AVIFs and uint16 (again filling ``[0, 65535]``)
            for higher-bit ones.
        num_threads (int, optional): Number of threads to use for decoding,
            directly passed to libavif. Default is 1.

    Returns:
        torch.Tensor: The decoded image, of shape ``(C, H, W)`` (still) or
        ``(N, C, H, W)`` (animated).
    """
    output_dtype_code = _validate_output_dtype(output_dtype)
    mode = _normalize_mode(mode)
    data = _source_to_tensor(source)
    return _decode_to_mode(
        lambda d, m: _decode_avif(d, m, output_dtype_code, num_threads),
        data,
        mode,
        _AVIF_NATIVE_OUTPUT_MODES,
    )


def decode_heic(
    source: str | Path | bytes | torch.Tensor,
    *,
    mode: (
        Literal["UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"] | ImageReadMode
    ) = "RGB",
    output_dtype: torch.dtype | Literal["auto"] = torch.uint8,
) -> torch.Tensor:
    """Decode an HEIC/HEIF image into a ``[N]CHW`` tensor - requires ``libheif``!

    The output shape is ``(C, H, W)`` for a single-image HEIC and
    ``(N, C, H, W)`` for a multi-image one. All images must share the same
    dimensions and bit depth.

    .. important::

        HEIC decoding requires **libheif** to be installed and discoverable at
        runtime. TorchCodec does not bundle it (libheif is LGPL): install it via
        e.g. ``conda install -c conda-forge libheif``.

    Example:

        .. code-block:: python

            from torchcodec.decoders import decode_heic

            img = decode_heic("image.heic")

    Args:
        source (str, ``pathlib.Path``, bytes, or ``torch.Tensor``):
            The encoded HEIC/HEIF data: a path (``str`` or ``pathlib.Path``), a
            ``bytes`` object, or a 1-D uint8 ``torch.Tensor`` of the raw encoded
            bytes.
        mode (str or ImageReadMode, optional): Desired color mode of the output
            image. Can be one of ``"UNCHANGED"``, ``"GRAY"``, ``"GRAY_ALPHA"``,
            ``"RGB"``, or ``"RGB_ALPHA"``. Default is ``"RGB"``.
        output_dtype (torch.dtype or ``"auto"``, optional): desired dtype of the
            output image tensor. Accepted values are ``torch.uint8`` (default),
            ``torch.uint16``, and ``"auto"``. HEIC can store more than 8 bits per
            channel (e.g. 10- or 12-bit sources). ``torch.uint16`` always scales
            the samples up to fill the full 16-bit range ``[0, 65535]`` (8-bit
            0-255, 10-bit 0-1023 and 12-bit 0-4095 sources are all upscaled),
            while ``torch.uint8`` scales higher-bit sources down. ``"auto"``
            yields uint8 for 8-bit HEICs and uint16 (again filling ``[0, 65535]``)
            for higher-bit ones.

    Returns:
        torch.Tensor: The decoded image, of shape ``(C, H, W)`` (single-image)
        or ``(N, C, H, W)`` (multi-image).
    """
    output_dtype_code = _validate_output_dtype(output_dtype)
    mode = _normalize_mode(mode)
    data = _source_to_tensor(source)
    decode_heic_op = _get_decode_heic()
    decoded = _decode_to_mode(
        lambda d, m: decode_heic_op(d, m, output_dtype_code),
        data,
        mode,
        _HEIC_NATIVE_OUTPUT_MODES,
    )
    return _to_output_dtype(decoded, output_dtype)


def _detect_image_format(data: torch.Tensor) -> str:
    # Sniff the codec from the leading "magic" bytes of the encoded data.
    # This used to be implemented in C++ in torchvision, but benchmarks show
    # this is negligible in Python
    header = bytes(data[:64].tolist())
    if header[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if header[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "webp"
    if header[4:8] == b"ftyp":
        # ISOBMFF container (AVIF/HEIC/...). The major brand is at [8:12], with
        # compatible brands following. AVIF uses the "avif" (still) or "avis"
        # (animated) brands; HEIC/HEIF uses "heic"/"heix" (HEVC-coded) and the
        # generic "mif1"/"msf1" brands. We check avif first (it can also carry
        # mif1/msf1), then fall back to the HEIC-specific brands.
        brands = header[8:]
        if b"avif" in brands or b"avis" in brands:
            return "avif"
        if (
            b"heic" in brands
            or b"heix" in brands
            or b"heim" in brands
            or b"heis" in brands
            or b"hevc" in brands
            or b"hevx" in brands
            or b"mif1" in brands
            or b"msf1" in brands
        ):
            return "heic"
    raise ValueError(
        "Unsupported or unrecognized image format. Supported formats are "
        "JPEG, PNG, WebP, GIF, AVIF and HEIC. If you know you have a valid "
        "image, try using the dedicated decode_* functions like decode_jpeg() "
        "instead."
    )


# Design note: the parameters of decode_image must apply to *all* codecs
# uniformly. That's why all modes are supported by decode_image even though not
# all codec would natively support all mode - e.g. jpeg has no alpha support, so
# we prepend an opaque alpha channel as a post-processing step. As a resut, all
# codec-specific entry points like decode_jpeg, decode_png etc. must still
# expose the same parameters that decode_image exposes. The codec-specific
# parameters should live in the codec-specific entry points, e.g. decode_avif
# has its `num_threads`, decode_jpeg has `device`, etc.
def decode_image(
    source: str | Path | bytes | torch.Tensor,
    *,
    mode: (
        Literal["UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"] | ImageReadMode
    ) = "RGB",
    output_dtype: torch.dtype | Literal["auto"] = torch.uint8,
) -> torch.Tensor:
    """Decode an image into a ``[N]CHW`` tensor, detecting the format automatically.

    The format is detected from the encoded data (not the file extension), and
    decoding is delegated to the matching format-specific decoder. Supported
    formats are JPEG, PNG, WebP, GIF, AVIF and HEIC (requires ``libheif``). The
    output shape is ``(C, H, W)`` for a single image and ``(N, C, H, W)`` for
    animated or multi-image formats (WebP, GIF, AVIF, HEIC).

    For finer control, or for format-specific options (e.g. ``device`` for CUDA
    JPEG decoding, ``num_threads`` for AVIF), use the dedicated decoders
    directly: :func:`decode_jpeg`, :func:`decode_png`, :func:`decode_webp`,
    :func:`decode_gif`, :func:`decode_avif`, :func:`decode_heic`.

    Example:

        .. code-block:: python

            from torchcodec.decoders import decode_image

            jpeg_img = decode_image("image.jpg")
            png_img = decode_image("image.png")

    Args:
        source (str, ``pathlib.Path``, bytes, or ``torch.Tensor``):
            The encoded image data: a path (``str`` or ``pathlib.Path``), a
            ``bytes`` object, or a 1-D uint8 ``torch.Tensor`` of the raw encoded
            bytes.
        mode (str or ImageReadMode, optional): Desired color mode of the output
            image. Can be one of ``"UNCHANGED"``, ``"GRAY"``, ``"GRAY_ALPHA"``,
            ``"RGB"``, or ``"RGB_ALPHA"``. Default is ``"RGB"``.
        output_dtype (torch.dtype or ``"auto"``, optional): desired dtype of the
            output image tensor. Accepted values are ``torch.uint8`` (default),
            ``torch.uint16``, and ``"auto"``. Formats that can carry more than 8
            bits per channel (PNG, AVIF, HEIC) preserve that precision with
            ``torch.uint16`` and ``"auto"``. See the format-specific decoders
            for details.

    Returns:
        torch.Tensor: The decoded image, of shape ``[N]CHW``.
    """
    data = _source_to_tensor(source)
    format_to_decoder = {
        "jpeg": decode_jpeg,
        "png": decode_png,
        "webp": decode_webp,
        "gif": decode_gif,
        "avif": decode_avif,
        "heic": decode_heic,
    }
    fmt = _detect_image_format(data)
    return format_to_decoder[fmt](data, mode=mode, output_dtype=output_dtype)  # type: ignore[operator]
