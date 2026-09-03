import functools
import importlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from torchcodec import ffmpeg_major_version
from torchcodec._core import get_ffmpeg_library_versions
from torchcodec.decoders import (
    decode_avif,
    decode_heic,
    decode_jpeg,
    decode_png,
    decode_webp,
    set_cuda_backend,
    VideoDecoder,
)
from torchcodec.decoders._video_decoder import _read_custom_frame_mappings

IS_WINDOWS = sys.platform in ("win32", "cygwin")
IN_GITHUB_CI = bool(os.getenv("GITHUB_ACTIONS"))


def call_ffprobe(args):
    # We write ffprobe's output to a temp file instead of capturing via
    # subprocess pipe, to avoid sporadic JSON truncation on Windows.
    with tempfile.NamedTemporaryFile(mode="r", suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        with open(tmp_path, "w") as stdout_file:
            subprocess.run(
                ["ffprobe", "-v", "error", "-hide_banner"] + args + ["-of", "json"],
                check=True,
                stdout=stdout_file,
                stderr=subprocess.DEVNULL,
            )
        with open(tmp_path) as f:
            return json.loads(f.read())
    finally:
        os.unlink(tmp_path)


# Decorator for skipping CUDA tests when CUDA isn't available. The tests are
# effectively marked to be skipped in pytest_collection_modifyitems() of
# conftest.py
def needs_cuda(test_item):
    return pytest.mark.needs_cuda(test_item)


# Decorator for skipping ffmpeg tests when ffmpeg cli isn't available. The tests are
# effectively marked to be skipped in pytest_collection_modifyitems() of
# conftest.py
def needs_ffmpeg_cli(test_item):
    return pytest.mark.needs_ffmpeg_cli(test_item)


# Decorator for skipping tests that need libjpeg (torchcodec may be built without
# it). Handled in pytest_collection_modifyitems() of conftest.py.
def needs_jpeg(test_item):
    return pytest.mark.needs_jpeg(test_item)


# Decorator for skipping tests that need libpng (torchcodec may be built without
# it). Handled in pytest_collection_modifyitems() of conftest.py.
def needs_png(test_item):
    return pytest.mark.needs_png(test_item)


# Decorator for skipping tests that need libwebp (torchcodec may be built
# without it). Handled in pytest_collection_modifyitems() of conftest.py.
def needs_webp(test_item):
    return pytest.mark.needs_webp(test_item)


# Decorator for skipping tests that need libavif (torchcodec may be built
# without it). Handled in pytest_collection_modifyitems() of conftest.py.
def needs_avif(test_item):
    return pytest.mark.needs_avif(test_item)


# Decorator for skipping tests that need libheif. Unlike the other image libs,
# libheif is never bundled: it's an optional user-supplied runtime dependency,
# so it may simply be absent. Handled in pytest_collection_modifyitems() of
# conftest.py.
def needs_heic(test_item):
    return pytest.mark.needs_heic(test_item)


# This is a special device string that we use to test the legacy "ffmpeg" CUDA
# backend. It only exists here, in this test utils file. Public and core APIs
# have no idea that this is how we're testing them. That is, that's not a
# supported `device` parameter for the VideoDecoder or for the _core APIs.
# Tests using all_supported_devices() will get this device string, and the test
# need to clean it up by calling either make_video_decoder for VideoDecoder, or
# unsplit_device_str for core APIs.
_CUDA_FFMPEG_DEVICE_STR = "cuda:ffmpeg"


def all_supported_devices():
    return (
        "cpu",
        pytest.param("cuda", marks=pytest.mark.needs_cuda),
        pytest.param(_CUDA_FFMPEG_DEVICE_STR, marks=pytest.mark.needs_cuda),
    )


def cuda_devices():
    return (
        pytest.param("cuda", marks=pytest.mark.needs_cuda),
        pytest.param(_CUDA_FFMPEG_DEVICE_STR, marks=pytest.mark.needs_cuda),
    )


def unsplit_device_str(device_str: str) -> str:
    # helper meant to be used as
    # device, device_variant = unsplit_device_str(device)
    # when `device` comes from all_supported_devices() and may be _CUDA_FFMPEG_DEVICE_STR.
    # It is used:
    # - before calling `.to(device)` where device can't be _CUDA_FFMPEG_DEVICE_STR.
    # - before calling add_video_stream(device=device, device_variant=device_variant)
    if device_str == _CUDA_FFMPEG_DEVICE_STR:
        return "cuda", "ffmpeg"
    else:
        return device_str, "default"


def make_video_decoder(*args, **kwargs) -> tuple[VideoDecoder, str]:
    # Helper to create a VideoDecoder with the right cuda backend if needed.
    # kwargs is expected to have a "device" key which comes from
    # all_supported_devices(), and can be _CUDA_FFMPEG_DEVICE_STR.
    device = kwargs.pop("device", "cpu")
    if device == _CUDA_FFMPEG_DEVICE_STR:
        clean_device, backend = "cuda", "ffmpeg"
    else:
        clean_device, backend = device, "nvdec"

    # set_cuda_backend is a no-op if the device is "cpu", so we can use it
    # unconditionally.
    with set_cuda_backend(backend):
        dec = VideoDecoder(*args, **kwargs, device=clean_device)

    return dec, clean_device


def get_ffmpeg_minor_version():
    ffmpeg_version = get_ffmpeg_library_versions()["ffmpeg_version"]
    # When building FFmpeg from source there can be a `n` prefix in the version
    # string.  This is quite brittle as we're using av_version_info(), which has
    # no stable format. See https://github.com/pytorch/torchcodec/issues/100
    if ffmpeg_version.startswith("n"):
        ffmpeg_version = ffmpeg_version.removeprefix("n")
    return int(ffmpeg_version.split(".")[1])


def get_python_version() -> tuple[int, int]:
    return (sys.version_info.major, sys.version_info.minor)


def cuda_version_used_for_building_torch() -> tuple[int, int | None]:
    # Return the CUDA version that was used to build PyTorch. That's not always
    # the same as the CUDA version that is currently installed on the running
    # machine, which is what we actually want. On the CI though, these are the
    # same.
    if torch.version.cuda is None:
        return None
    else:
        return tuple(int(x) for x in torch.version.cuda.split("."))


def psnr(a, b, max_val=255) -> float:
    # Return Peak Signal-to-Noise Ratio (PSNR) between two tensors a and b. The
    # higher, the better.
    # According to https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio,
    # typical values for the PSNR in lossy image and video compression are
    # between 30 and 50 dB.
    # Acceptable values for wireless transmission quality loss are considered to
    # be about 20 dB to 25 dB
    mse = torch.mean((a.float() - b.float()) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()


# For use with decoded data frames. On CPU Linux, we expect exact, bit-for-bit
# equality. On CUDA Linux, we expect a small tolerance.
# On other platforms (e.g. MacOS), we also allow a small tolerance. FFmpeg does
# not guarantee bit-for-bit equality across systems and architectures, so we
# also cannot. We currently use Linux on x86_64 as our reference system.
def assert_frames_equal(*args, **kwargs):
    if sys.platform == "linux" and "x86" in platform.machine().lower():
        if args[0].device.type == "cuda":
            atol = 3 if cuda_version_used_for_building_torch() >= (13, 0) else 2
            if ffmpeg_major_version == 4:
                assert_tensor_close_on_at_least(
                    args[0], args[1], percentage=95, atol=atol
                )
            else:
                torch.testing.assert_close(*args, **kwargs, atol=atol, rtol=0)
        else:
            torch.testing.assert_close(*args, **kwargs, atol=0, rtol=0)
    else:
        # Here: Windows, MacOS, and Linux for non-x86 architectures like aarch64
        torch.testing.assert_close(*args, **kwargs, atol=3, rtol=0)


# Asserts that at least `percentage`% of the values are within the absolute tolerance.
# Percentage is expected in [0, 100] (actually, [60, 100])
def assert_tensor_close_on_at_least(
    actual_tensor, ref_tensor, *, percentage, atol, **kwargs
):
    # In theory lower bound should be 0, but we want to make sure we don't
    # mistakenly pass percentage in [0, 1]
    assert 60 < percentage <= 100, (
        f"Percentage must be in [60, 100], got {percentage}. "
        "Are you sure setting such a low tolerance is desired?"
    )
    assert (
        actual_tensor.device == ref_tensor.device
    ), f"Devices don't match: {actual_tensor.device} vs {ref_tensor.device}"

    abs_diff = (ref_tensor.float() - actual_tensor.float()).abs()
    valid_percentage = (abs_diff <= atol).float().mean() * 100
    if valid_percentage < percentage:
        raise AssertionError(
            f"Expected at least {percentage}% of values to be within atol={atol}, "
            f"but only {valid_percentage}% were."
        )


# We embed filtergraph expressions in filenames, but they contain characters that
# some filesystems don't like. We turn all special characters into underscores.
def sanitize_filtergraph_expression(expression: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in expression)


def in_fbcode() -> bool:
    return os.environ.get("IN_FBCODE_TORCHCODEC") == "1"


def _get_file_path(filename: str) -> pathlib.Path:
    if in_fbcode():
        resource = (
            importlib.resources.files(__spec__.parent)
            .joinpath("resources")
            .joinpath(filename)
        )
        with importlib.resources.as_file(resource) as path:
            return path
    else:
        return pathlib.Path(__file__).parent / "resources" / filename


@dataclass
class TestImage:
    __test__ = False  # prevents pytest from thinking this is a test class

    filename: str
    width: int
    height: int
    num_channels: int

    @property
    def path(self) -> pathlib.Path:
        return _get_file_path(self.filename)


# 720p RGB gradient JPEG. Generated with:
# h, w = 720, 1280
# r = np.linspace(0, 255, w, dtype=np.uint8)[None, :].repeat(h, 0)
# g = np.linspace(0, 255, h, dtype=np.uint8)[:, None].repeat(w, 1)
# b = ((r.astype(int) + g.astype(int)) // 2).astype(np.uint8)
# Image.fromarray(np.stack([r, g, b], axis=-1)).save("gradient.jpg", quality=90)
GRADIENT_JPEG = TestImage(
    filename="gradient.jpg", width=1280, height=720, num_channels=3
)

# 720p grayscale gradient JPEG. Generated with:
# h, w = 720, 1280
# r = np.linspace(0, 255, w, dtype=np.uint8)[None, :].repeat(h, 0)
# g = np.linspace(0, 255, h, dtype=np.uint8)[:, None].repeat(w, 1)
# gray = ((r.astype(int) + g.astype(int)) // 2).astype(np.uint8)
# Image.fromarray(gray, mode="L").save("grayscale.jpg", quality=90)
GRAYSCALE_JPEG = TestImage(
    filename="grayscale.jpg", width=1280, height=720, num_channels=1
)

# 720p CMYK JPEG, same gradient as GRADIENT_JPEG but stored as CMYK. Generated
# with the GRADIENT_JPEG recipe above, then:
# Image.fromarray(rgb).convert("CMYK").save("cmyk.jpg", quality=90)
CMYK_JPEG = TestImage(filename="cmyk.jpg", width=1280, height=720, num_channels=4)

# JPEG with a bad Huffman table (damaged but still decodable). Taken from
# torchvision's test assets (test/assets/damaged_jpeg/bad_huffman.jpg).
BAD_HUFFMAN_JPEG = TestImage(
    filename="bad_huffman.jpg", width=1024, height=768, num_channels=3
)

# Malformed images ported from torchvision's fuzzer-derived test assets. They
# are never decoded successfully - they only exist to check that the decoders
# raise cleanly instead of crashing, so only their `.path` matters (the
# width/height/num_channels are the nominal header values, or 0 when the header
# itself is unreadable).
#
# CORRUPT_JPEG has a valid header but a corrupt entropy stream, which trips an
# "Unsupported marker type" error late in the decode (during
# jpeg_finish_decompress).
CORRUPT_JPEG = TestImage(filename="corrupt.jpg", width=120, height=90, num_channels=3)
# PNG crashers found by fuzzing libpng (out-of-bound reads).
SIGSEGV_PNG = TestImage(filename="sigsegv.png", width=0, height=0, num_channels=0)
HEAPBOF_PNG = TestImage(filename="heapbof.png", width=0, height=0, num_channels=0)

# Adam7-interlaced version of a small RGB gradient. It exercises the decoder's
# multi-pass interlace-handling path (png_set_interlace_handling), which no other
# asset covers. PIL can't write interlaced PNGs, so this was authored once with
# ImageMagick: `magick gradient.png -interlace PNG gradient_interlaced.png`.
GRADIENT_INTERLACED_PNG = TestImage(
    filename="gradient_interlaced.png", width=64, height=48, num_channels=3
)


@functools.cache
def jpeg_is_available() -> bool:
    try:
        decode_jpeg(GRADIENT_JPEG.path)
    except RuntimeError as e:
        if "libjpeg" in str(e):
            return False
        raise
    return True


# 720p RGB gradient PNG, same gradient as GRADIENT_JPEG. Generated with:
# h, w = 720, 1280
# r = np.linspace(0, 255, w, dtype=np.uint8)[None, :].repeat(h, 0)
# g = np.linspace(0, 255, h, dtype=np.uint8)[:, None].repeat(w, 1)
# b = ((r.astype(int) + g.astype(int)) // 2).astype(np.uint8)
# Image.fromarray(np.stack([r, g, b], axis=-1)).save("gradient.png")
GRADIENT_PNG = TestImage(
    filename="gradient.png", width=1280, height=720, num_channels=3
)

# 720p grayscale gradient PNG. Generated with the GRADIENT_PNG recipe above, then:
# gray = ((r.astype(int) + g.astype(int)) // 2).astype(np.uint8)
# Image.fromarray(gray, mode="L").save("grayscale.png")
GRAYSCALE_PNG = TestImage(
    filename="grayscale.png", width=1280, height=720, num_channels=1
)

# 720p RGBA PNG: same gradient as GRADIENT_PNG with a diagonal alpha ramp.
# Generated with the GRADIENT_PNG recipe above, then:
# a = ((r.astype(int) + (255 - g.astype(int))) // 2).astype(np.uint8)
# rgba = np.concatenate([np.stack([r, g, b], axis=-1), a[..., None]], axis=-1)
# Image.fromarray(rgba, mode="RGBA").save("rgba.png")
RGBA_PNG = TestImage(filename="rgba.png", width=1280, height=720, num_channels=4)

# 720p grayscale-alpha (LA) PNG: grayscale gradient with the same diagonal alpha
# ramp as RGBA_PNG. Generated with the GRADIENT_PNG recipe above, then:
# gray = ((r.astype(int) + g.astype(int)) // 2).astype(np.uint8)
# a = ((r.astype(int) + (255 - g.astype(int))) // 2).astype(np.uint8)
# la = np.stack([gray, a], axis=-1)
# Image.fromarray(la, mode="LA").save("grayscale_alpha.png")
GRAYSCALE_ALPHA_PNG = TestImage(
    filename="grayscale_alpha.png", width=1280, height=720, num_channels=2
)

# 48x64 16-bit grayscale PNG (a full-range gradient, so the low byte carries real
# information). Exercises the decoder's 16-bit path. Generated with:
# h, w = 48, 64
# gray = np.linspace(0, 65535, h * w).reshape(h, w).astype(np.uint16)
# Image.fromarray(gray).save("grayscale_16bit.png")  # PIL infers the I;16 mode
GRAYSCALE_16BIT_PNG = TestImage(
    filename="grayscale_16bit.png", width=64, height=48, num_channels=1
)

# 48x64 16-bit RGB PNG with smooth full-range per-channel gradients. PIL can't
# write 16-bit RGB, so it's authored with ffmpeg from raw rgb48 samples:
# h, w = 48, 64
# r = np.linspace(0, 65535, w, dtype=np.uint16)[None].repeat(h, 0)
# g = np.linspace(0, 65535, h, dtype=np.uint16)[:, None].repeat(w, 1)
# b = np.linspace(65535, 0, w, dtype=np.uint16)[None].repeat(h, 0)
# np.stack([r, g, b], -1).astype("<u2").tofile("rgb48.bin")
# ffmpeg -f rawvideo -pixel_format rgb48le -video_size 64x48 -i rgb48.bin \
#     -frames:v 1 -pix_fmt rgb48be gradient_16bit.png
GRADIENT_16BIT_PNG = TestImage(
    filename="gradient_16bit.png", width=64, height=48, num_channels=3
)


@functools.cache
def png_is_available() -> bool:
    try:
        decode_png(GRADIENT_PNG.path)
    except RuntimeError as e:
        if "libpng" in str(e):
            return False
        raise
    return True


# 720p RGB gradient WebP (lossless), same gradient as GRADIENT_JPEG. Generated
# with the GRADIENT_JPEG recipe above, then:
# Image.fromarray(np.stack([r, g, b], axis=-1)).save(
#     "gradient.webp", "WEBP", lossless=True)
GRADIENT_WEBP = TestImage(
    filename="gradient.webp", width=1280, height=720, num_channels=3
)

# 720p RGBA WebP (lossless): same gradient as GRADIENT_WEBP with the diagonal
# alpha ramp of RGBA_PNG. Generated with the GRADIENT_PNG recipe above, then:
# a = ((r.astype(int) + (255 - g.astype(int))) // 2).astype(np.uint8)
# rgba = np.concatenate([np.stack([r, g, b], axis=-1), a[..., None]], axis=-1)
# Image.fromarray(rgba, mode="RGBA").save("rgba.webp", "WEBP", lossless=True)
RGBA_WEBP = TestImage(filename="rgba.webp", width=1280, height=720, num_channels=4)


@functools.cache
def webp_is_available() -> bool:
    try:
        decode_webp(GRADIENT_WEBP.path)
    except RuntimeError as e:
        if "libwebp" in str(e):
            return False
        raise
    return True


# 720p RGB gradient AVIF (lossy, libavif defaults), same gradient as
# GRADIENT_JPEG. Generated with the GRADIENT_JPEG recipe above, then:
# Image.fromarray(np.stack([r, g, b], axis=-1)).save("gradient.avif")
GRADIENT_AVIF = TestImage(
    filename="gradient.avif", width=1280, height=720, num_channels=3
)

# 720p RGBA AVIF (lossy): same gradient as GRADIENT_AVIF with the diagonal alpha
# ramp of RGBA_PNG. Generated with the GRADIENT_PNG recipe above, then:
# a = ((r.astype(int) + (255 - g.astype(int))) // 2).astype(np.uint8)
# rgba = np.concatenate([np.stack([r, g, b], axis=-1), a[..., None]], axis=-1)
# Image.fromarray(rgba, mode="RGBA").save("rgba.avif")
RGBA_AVIF = TestImage(filename="rgba.avif", width=1280, height=720, num_channels=4)

# 48x64 10- and 12-bit AVIFs (genuine >8-bit sources). PIL only writes 8-bit
# AVIF, so they're authored with ffmpeg from an 8-bit RGB gradient. Generated
# with:
# h, w = 48, 64
# r = np.linspace(0, 255, w)[None].repeat(h, 0)
# g = np.linspace(0, 255, h)[:, None].repeat(w, 1)
# b = np.linspace(255, 0, w)[None].repeat(h, 0)
# Image.fromarray(np.stack([r, g, b], -1).astype(np.uint8), "RGB").save("src.png")
# ffmpeg -i src.png -c:v libaom-av1 -pix_fmt yuv444p10le -still-picture 1 \
#     gradient_10bit.avif    # (yuv444p12le for the 12-bit one)
GRADIENT_10BIT_AVIF = TestImage(
    filename="gradient_10bit.avif", width=64, height=48, num_channels=3
)
GRADIENT_12BIT_AVIF = TestImage(
    filename="gradient_12bit.avif", width=64, height=48, num_channels=3
)


@functools.cache
def avif_is_available() -> bool:
    try:
        decode_avif(GRADIENT_AVIF.path)
    except RuntimeError as e:
        if "libavif" in str(e):
            return False
        raise
    return True


# 720p RGB gradient HEIC, the SAME gradient as GRADIENT_PNG, saved losslessly
# (4:4:4, no chroma subsampling) so decode is exact up to rounding. Generated
# with the GRADIENT_PNG recipe above, then (needs pillow-heif):
# import pillow_heif; pillow_heif.register_heif_opener()
# Image.fromarray(np.stack([r, g, b], axis=-1)).save(
#     "gradient.heic", quality=-1, chroma=444)
GRADIENT_HEIC = TestImage(
    filename="gradient.heic", width=1280, height=720, num_channels=3
)

# GRADIENT_HEIC saved with orientation metadata (stored by pillow-heif as the
# HEIF irot/imir transform properties, which libheif applies on decode). Used to
# check we respect orientation. width/height are the DECODED (post-orientation)
# dimensions. Generated with the GRADIENT_PNG recipe above, then (needs
# pillow-heif):
# import pillow_heif; pillow_heif.register_heif_opener()
# for orientation, fname in ((6, "gradient_rotated.heic"),
#                            (2, "gradient_mirrored.heic")):
#     img = Image.fromarray(np.stack([r, g, b], axis=-1))
#     exif = img.getexif(); exif[0x0112] = orientation  # EXIF orientation tag
#     img.save(fname, exif=exif.tobytes(), quality=-1, chroma=444)
# 6 is a 90-degree rotation (exercises irot); 2 is a horizontal mirror (imir).
GRADIENT_ROTATED_HEIC = TestImage(
    filename="gradient_rotated.heic", width=720, height=1280, num_channels=3
)
GRADIENT_MIRRORED_HEIC = TestImage(
    filename="gradient_mirrored.heic", width=1280, height=720, num_channels=3
)

# 720p RGBA HEIC (lossless 4:4:4): same gradient as GRADIENT_HEIC with the
# diagonal alpha ramp of RGBA_PNG. Generated with the GRADIENT_PNG recipe, then:
# a = ((r.astype(int) + (255 - g.astype(int))) // 2).astype(np.uint8)
# rgba = np.concatenate([np.stack([r, g, b], axis=-1), a[..., None]], axis=-1)
# Image.fromarray(rgba, mode="RGBA").save("rgba.heic", quality=-1, chroma=444)
RGBA_HEIC = TestImage(filename="rgba.heic", width=1280, height=720, num_channels=4)

# 48x64 genuine 10-bit HEIC (a real >8-bit source). PIL only writes 8-bit HEIC,
# so it's authored from raw 10-bit samples via pillow-heif's low-level API:
# h, w = 48, 64
# r = np.linspace(0, 1023, w)[None].repeat(h, 0)
# g = np.linspace(0, 1023, h)[:, None].repeat(w, 1)
# b = np.linspace(1023, 0, w)[None].repeat(h, 0)
# data = (np.stack([r, g, b], -1).astype(np.uint16) << 6).astype("<u2").tobytes()
# heif = pillow_heif.from_bytes(mode="RGB;16", size=(w, h), data=data)
# heif.save("gradient_10bit.heic", quality=-1, chroma=444)
GRADIENT_10BIT_HEIC = TestImage(
    filename="gradient_10bit.heic", width=64, height=48, num_channels=3
)


# Small 3-frame HEIC image sequence with full-canvas solid-color frames, saved
# losslessly (4:4:4) so the solid colors survive the YUV round-trip. Each frame
# is a distinct color so frame ordering is verifiable. Used to test the
# (N, C, H, W) multi-image output. Generated (needs pillow-heif):
# import pillow_heif; pillow_heif.register_heif_opener()
# colors = [(200, 30, 30), (30, 200, 30), (30, 30, 200)]
# frames = [Image.fromarray(np.full((48, 64, 3), c, np.uint8)) for c in colors]
# frames[0].save("animated.heic", save_all=True, append_images=frames[1:],
#                quality=-1, chroma=444)
ANIMATED_HEIC = TestImage(filename="animated.heic", width=64, height=48, num_channels=3)


@functools.cache
def heic_is_available() -> bool:
    # "Available" means we can actually DECODE a HEIC here. We probe with a real
    # decode (not just a library load): a libheif can load fine yet fail to
    # decode with "Unsupported codec" when it lacks an HEVC decoder (libde265).
    # This must never raise -- it's called from conftest's collection hook, so
    # any exception would abort the whole session -- and every failure mode
    # (missing libheif, stub build, missing codec) just means "skip".
    try:
        decode_heic(GRADIENT_HEIC.path)
    except Exception as e:
        print(f"heic_is_available() -> False: {type(e).__name__}: {e}")
        return False
    return True


# 720p RGB gradient GIF, same gradient as GRADIENT_JPEG.
# Generated with the GRADIENT_JPEG recipe above, then:
# Image.fromarray(np.stack([r, g, b], axis=-1)).save("gradient.gif")
GRADIENT_GIF = TestImage(
    filename="gradient.gif", width=1280, height=720, num_channels=3
)

# Small 4-frame animated GIF with full-canvas opaque frames (no partial frames
# or transparency, so giflib and PIL composite identically). Used to test the
# (N, C, H, W) animated output. Generated with:
# ah, aw = 48, 64
# frames = []
# for i in range(4):
#     fr = np.zeros((ah, aw, 3), dtype=np.uint8)
#     fr[..., i % 3] = 40 + 60 * i
#     fr[:, i * 12 : i * 12 + 12, :] = 255
#     frames.append(Image.fromarray(fr).convert("P", palette=Image.ADAPTIVE))
# frames[0].save("animated.gif", save_all=True, append_images=frames[1:],
#                duration=100, loop=0, disposal=1)
ANIMATED_GIF = TestImage(filename="animated.gif", width=64, height=48, num_channels=3)

# Palette GIF with a transparent index over a (non-zero) red background, so it
# exercises the RGBA transparency path and the "welcome2" background-vs-
# transparency case. num_channels=4: UNCHANGED decodes it to RGBA. Generated
# with:
# palette = [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]  # 0=red(bg), ...
# idx = np.full((48, 64), 2, np.uint8)  # index 2 will be transparent
# idx[8:40, 8:32] = 1; idx[20:28, 40:56] = 3  # opaque green + white rectangles
# im = Image.fromarray(idx, "P"); im.putpalette(palette)
# im.save("transparent.gif", transparency=2)
TRANSPARENT_GIF = TestImage(
    filename="transparent.gif", width=64, height=48, num_channels=4
)

# Hand-crafted GIF whose logical screen is 4x4 but whose single first frame is
# 8x8 (larger than the screen), so the output is sized to the frame (8x8). The
# out-of-screen border is transparent, which regression-tests that those pixels
# are initialized (transparent) rather than left as uninitialized memory.
# Palette 0=red(bg), 1=green, 2=blue(transparent), 3=white; the top-left 4x4 is
# opaque green and the rest is the transparent index. See git history for the
# raw GIF89a builder used to author it (PIL can't emit a frame > logical screen).
FRAME_EXCEEDS_SCREEN_GIF = TestImage(
    filename="frame_exceeds_screen.gif", width=8, height=8, num_channels=4
)


@dataclass
class TestFrameInfo:
    pts_seconds: float
    duration_seconds: float


@dataclass
class TestVideoStreamInfo:
    width: int
    height: int
    num_color_channels: int


@dataclass
class TestAudioStreamInfo:
    sample_rate: int
    num_channels: int
    duration_seconds: float
    num_frames: int
    sample_format: str


@dataclass
class TestContainerFile:
    __test__ = False  # prevents pytest from thinking this is a test class

    filename: str

    default_stream_index: int
    stream_infos: dict[int, TestVideoStreamInfo | TestAudioStreamInfo]
    frames: dict[int, dict[int, TestFrameInfo]]
    _custom_frame_mappings_data: dict[
        int, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
    ] = field(default_factory=dict)

    def __post_init__(self):
        # We load the .frames attribute from the checked-in json files, if needed.
        # These frame info files are dumped with ffprobe, e.g.:
        # ```
        # ffprobe -v error -hide_banner -select_streams v:1 -show_frames -of json test/resources/nasa_13013.mp4 | jq '[.frames[] | {duration_time, pts_time}]'
        # ```
        # This will output the metadata for the frames of the second video
        # stream (v:1). First audio stream would be a:0.
        # Note that we are using the absolute stream index in the file. But
        # ffprobe uses a relative stream for that media type.
        for stream_index in self.stream_infos:
            if stream_index in self.frames:
                # .frames may be manually set: for some streams, we don't need
                # the info for all frames. We don't need to load anything in
                # this case
                continue

            frames_info_path = _get_file_path(
                f"{self.filename}.stream{stream_index}.all_frames_info.json"
            )

            if not frames_info_path.exists():
                raise ValueError(
                    f"Couldn't find {frames_info_path} for {self.filename}. "
                    "You need to submit this file, or specify the `frames` field manually."
                )

            with open(frames_info_path) as f:
                frames_info = json.loads(f.read())
            self.frames[stream_index] = {
                frame_index: TestFrameInfo(
                    pts_seconds=float(frame_info["pts_time"]),
                    duration_seconds=float(frame_info["duration_time"]),
                )
                for frame_index, frame_info in enumerate(frames_info)
            }

    @property
    def path(self) -> pathlib.Path:
        return _get_file_path(self.filename)

    def to_tensor(self) -> torch.Tensor:
        arr = np.fromfile(self.path, dtype=np.uint8)
        return torch.from_numpy(arr)

    def get_frame_data_by_index(
        self, idx: int, *, stream_index: int | None = None
    ) -> torch.Tensor:
        raise NotImplementedError("Override in child classes")

    def get_frame_data_by_range(
        self,
        start: int,
        stop: int,
        step: int = 1,
        *,
        stream_index: int | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Override in child classes")

    def get_pts_seconds_by_range(
        self,
        start: int,
        stop: int,
        step: int = 1,
        *,
        stream_index: int | None = None,
    ) -> torch.Tensor:
        if stream_index is None:
            stream_index = self.default_stream_index

        all_pts = [
            self.frames[stream_index][i].pts_seconds for i in range(start, stop, step)
        ]
        return torch.tensor(all_pts, dtype=torch.float64)

    def get_duration_seconds_by_range(
        self,
        start: int,
        stop: int,
        step: int = 1,
        *,
        stream_index: int | None = None,
    ) -> torch.Tensor:
        if stream_index is None:
            stream_index = self.default_stream_index

        all_durations = [
            self.frames[stream_index][i].duration_seconds
            for i in range(start, stop, step)
        ]
        return torch.tensor(all_durations, dtype=torch.float64)

    def get_frame_info(
        self, idx: int, *, stream_index: int | None = None
    ) -> TestFrameInfo:
        if stream_index is None:
            stream_index = self.default_stream_index

        return self.frames[stream_index][idx]

    # This function is used to get the frame mappings for the custom_frame_mappings seek mode.
    def get_custom_frame_mappings(
        self, stream_index: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if stream_index is None:
            stream_index = self.default_stream_index
        if self._custom_frame_mappings_data.get(stream_index) is None:
            self._custom_frame_mappings_data[stream_index] = (
                _read_custom_frame_mappings(
                    self.generate_custom_frame_mappings(stream_index)
                )
            )
        return self._custom_frame_mappings_data[stream_index]

    def generate_custom_frame_mappings(self, stream_index: int) -> str:
        parsed = call_ffprobe(
            [
                "-i",
                f"{self.path}",
                "-select_streams",
                f"{stream_index}",
                "-show_frames",
            ],
        )
        return json.dumps(parsed)

    @property
    def empty_pts_seconds(self) -> torch.Tensor:
        return torch.empty([0], dtype=torch.float64)

    @property
    def empty_duration_seconds(self) -> torch.Tensor:
        return torch.empty([0], dtype=torch.float64)


@dataclass
class TestVideo(TestContainerFile):
    """Base class for the *video* streams of a video container"""

    def get_base_path_by_index(
        self, idx: int, *, stream_index: int, filters: str | None = None
    ) -> pathlib.Path:
        stream_and_frame = f"stream{stream_index}.frame{idx:06d}"
        if filters is not None:
            full_name = f"{self.filename}.{sanitize_filtergraph_expression(filters)}.{stream_and_frame}"
        else:
            full_name = f"{self.filename}.{stream_and_frame}"

        return _get_file_path(full_name)

    def get_frame_data_by_index(
        self,
        idx: int,
        *,
        stream_index: int | None = None,
        filters: str | None = None,
    ) -> torch.Tensor:
        if stream_index is None:
            stream_index = self.default_stream_index

        base_path = self.get_base_path_by_index(
            idx, stream_index=stream_index, filters=filters
        )
        tensor_file_path = f"{base_path}.pt"
        return torch.load(tensor_file_path, weights_only=True).permute(2, 0, 1)

    def get_frame_data_by_index_rgb48(
        self,
        idx: int,
        *,
        stream_index: int | None = None,
    ) -> torch.Tensor:
        if stream_index is None:
            stream_index = self.default_stream_index

        base_path = self.get_base_path_by_index(idx, stream_index=stream_index)
        tensor_file_path = f"{base_path}.rgb48.pt"
        return torch.load(tensor_file_path, weights_only=True).permute(2, 0, 1)

    def get_frame_data_by_range(
        self,
        start: int,
        stop: int,
        step: int = 1,
        *,
        stream_index: int | None = None,
    ) -> torch.Tensor:
        tensors = [
            self.get_frame_data_by_index(i, stream_index=stream_index)
            for i in range(start, stop, step)
        ]
        return torch.stack(tensors)

    @property
    def width(self) -> int:
        return self.stream_infos[self.default_stream_index].width

    @property
    def height(self) -> int:
        return self.stream_infos[self.default_stream_index].height

    @property
    def num_color_channels(self) -> int:
        return self.stream_infos[self.default_stream_index].num_color_channels

    @property
    def empty_chw_tensor(self) -> torch.Tensor:
        return torch.empty(
            [0, self.num_color_channels, self.height, self.width], dtype=torch.uint8
        )

    def get_width(self, *, stream_index: int | None = None) -> int:
        if stream_index is None:
            stream_index = self.default_stream_index

        return self.stream_infos[stream_index].width

    def get_height(self, *, stream_index: int | None = None) -> int:
        if stream_index is None:
            stream_index = self.default_stream_index

        return self.stream_infos[stream_index].height

    def get_num_color_channels(self, *, stream_index: int | None = None) -> int:
        if stream_index is None:
            stream_index = self.default_stream_index

        return self.stream_infos[stream_index].num_color_channels

    def get_empty_chw_tensor(self, *, stream_index: int) -> torch.Tensor:
        return torch.empty(
            [
                0,
                self.get_num_color_channels(stream_index=stream_index),
                self.get_height(stream_index=stream_index),
                self.get_width(stream_index=stream_index),
            ],
            dtype=torch.uint8,
        )


NASA_VIDEO = TestVideo(
    filename="nasa_13013.mp4",
    default_stream_index=3,
    stream_infos={
        0: TestVideoStreamInfo(width=320, height=180, num_color_channels=3),
        3: TestVideoStreamInfo(width=480, height=270, num_color_channels=3),
    },
    frames={},  # Automatically loaded from json file
)

NASA_VIDEO_ROTATED = TestVideo(
    filename="nasa_13013_rotated.mp4",
    default_stream_index=0,
    stream_infos={
        # Post-rotation dimensions: 90-degree rotation swaps width/height
        # This is a short video (~15 frames) extracted from nasa_13013.mp4 stream 3
        # with 90-degree rotation metadata added
        0: TestVideoStreamInfo(width=270, height=480, num_color_channels=3),
    },
    frames={},  # Automatically loaded from json file
)

# Video generated with:
# ffmpeg -f lavfi -i testsrc2=duration=1:size=200x200:rate=30 -c:v libx265 -pix_fmt yuv420p10le -preset fast -crf 23 h265_10bits.mp4
H265_10BITS = TestVideo(
    filename="h265_10bits.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=200, height=200, num_color_channels=3),
    },
    frames={0: {}},  # Not needed yet
)

# Video generated with:
# peg -f lavfi -i testsrc2=duration=1:size=200x200:rate=30 -c:v libx264 -pix_fmt yuv420p10le -preset fast -crf 23 h264_10bits.mp4
H264_10BITS = TestVideo(
    filename="h264_10bits.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=200, height=200, num_color_channels=3),
    },
    frames={0: {}},  # Not needed yet
)


H265_VIDEO = TestVideo(
    filename="h265_video.mp4",
    default_stream_index=0,
    # This metadata is extracted manually.
    #  $ ffprobe -v error -hide_banner -select_streams v:0 -show_frames -of json test/resources/h265_video.mp4 > out.json
    stream_infos={
        0: TestVideoStreamInfo(width=128, height=128, num_color_channels=3),
    },
    frames={
        0: {
            6: TestFrameInfo(pts_seconds=0.6, duration_seconds=0.1),
        },
    },
)

# Video whose leading GOP, including its first keyframe,  is trimmed by an
# mp4 edit list, so those packets are marked AV_PKT_FLAG_DISCARD. The decoder
# still has to decode that discarded keyframe to produce the first *output*
# frame (pts=0), but it is never itself emitted.
#
# Generated with:
#   $ ffmpeg -f lavfi -i testsrc2=duration=1.2:size=64x64:rate=25 \
#       -c:v libx264 -pix_fmt yuv420p -g 10 -keyint_min 10 -sc_threshold 0 src.mp4
#   $ ffmpeg -ss 0.2 -i src.mp4 -c copy discard_first_keyframe.mp4
#
# Then the first few packets look like this:
# # $ ffprobe -v error -select_streams \
#     v:0 -show_entries packet=pts_time,flags -of csv discard_first_keyframe.mp4
# packet,-0.200000,KD_
# packet,-0.040000,_D_
# packet,-0.120000,_D_
# packet,-0.160000,_D_
# packet,-0.080000,_D_
# packet,0.000000,___
# packet,0.080000,___
# packet,0.040000,___
# packet,0.160000,___
# packet,0.120000,___
# packet,0.200000,K__
# ...
DISCARD_FIRST_KEYFRAME_VIDEO = TestVideo(
    filename="discard_first_keyframe.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=64, height=64, num_color_channels=3),
    },
    frames={
        0: {
            0: TestFrameInfo(pts_seconds=0.0, duration_seconds=0.04),
        },
    },
)

AV1_VIDEO = TestVideo(
    filename="av1_video.mkv",
    default_stream_index=0,
    # This metadata is extracted manually.
    #  $ ffprobe -v error -hide_banner -select_streams v:0 -show_frames -of json test/resources/av1_video.mkv > out.json
    stream_infos={
        0: TestVideoStreamInfo(width=640, height=360, num_color_channels=3),
    },
    frames={
        0: {
            10: TestFrameInfo(pts_seconds=0.400000, duration_seconds=0.040000),
        },
    },
)


# This is a BT.709 full range video, generated with:
# ffmpeg -f lavfi -i testsrc2=duration=1:size=1920x720:rate=30 \
# -c:v libx264 -pix_fmt yuv420p -color_primaries bt709 -color_trc bt709 \
# -colorspace bt709 -color_range pc bt709_full_range.mp4
#
# We can confirm the color space and color range with:
# ffprobe -v quiet -select_streams v:0 -show_entries stream=color_space,color_transfer,color_primaries,color_range -of default=noprint_wrappers=1 test/resources/bt709_full_range.mp4
# color_range=pc
# color_space=bt709
# color_transfer=bt709
# color_primaries=bt709
BT709_FULL_RANGE = TestVideo(
    filename="bt709_full_range.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=1280, height=720, num_color_channels=3),
    },
    frames={0: {}},  # Not needed for now
)

# BT.2020 10-bit video with limited range (tv), generated with:
# ffmpeg -f lavfi -i testsrc2=duration=2:size=320x240:rate=30 -c:v libx265 \
# -pix_fmt yuv420p10le -color_primaries bt2020 -color_trc smpte2084 \
# -colorspace bt2020nc -color_range tv bt2020_10bit.mp4
#
# Confirm color space with:
# ffprobe -v quiet -select_streams v:0 -show_entries stream=color_space,color_transfer,color_primaries,color_range -of default=noprint_wrappers=1 test/resources/bt2020_10bit.mp4
# color_range=tv
# color_space=bt2020nc
# color_transfer=smpte2084
# color_primaries=bt2020
BT2020_LIMITED_RANGE_10BIT = TestVideo(
    filename="bt2020_10bit.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=320, height=240, num_color_channels=3),
    },
    frames={0: {}},  # Not needed for now
)

# Full range BT.601 video, generated with:
# ffmpeg -f lavfi -i testsrc2=duration=2:size=320x240:rate=30 -c:v libx264
# -profile:v high -pix_fmt yuv420p
# -vf "setparams=color_primaries=smpte170m:color_trc=smpte170m:colorspace=smpte170m:range=pc"
# bt601_full_range.mp4
#
# Confirm color space with:
# ffprobe -v quiet -select_streams v:0 -show_entries stream=color_space,color_transfer,color_primaries,color_range -of default=noprint_wrappers=1 test/resources/bt601_full_range.mp4
# color_range=pc
# color_space=smpte170m
# color_transfer=smpte170m
# color_primaries=smpte170m
BT601_FULL_RANGE = TestVideo(
    filename="bt601_full_range.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=320, height=240, num_color_channels=3),
    },
    frames={0: {}},  # Not needed for now
)

# Limited range BT.601 video, generated with:
# ffmpeg -f lavfi -i testsrc2=duration=2:size=320x240:rate=30 -c:v libx264
# -profile:v baseline -pix_fmt yuv420p
# -vf "setparams=color_primaries=smpte170m:color_trc=smpte170m:colorspace=smpte170m:range=tv"
# bt601_limited_range.mp4
#
# Confirm color space with:
# ffprobe -v quiet -select_streams v:0 -show_entries stream=color_space,color_transfer,color_primaries,color_range -of default=noprint_wrappers=1 test/resources/bt601_limited_range.mp4
# color_range=tv
# color_space=smpte170m
# color_transfer=smpte170m
# color_primaries=smpte170m
BT601_LIMITED_RANGE = TestVideo(
    filename="bt601_limited_range.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=320, height=240, num_color_channels=3),
    },
    frames={0: {}},  # Not needed for now
)

# HDR re-encode of NASA video (10-bit H265 with BT.2020 + PQ), generated with:
# ffmpeg -i test/resources/nasa_13013.mp4 -map 0:v:0 -c:v libx265 -pix_fmt yuv420p10le \
# -x265-params "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited" \
# -preset fast -crf 23 test/resources/nasa_13013_hdr.mp4
NASA_VIDEO_HDR = TestVideo(
    filename="nasa_13013_hdr.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=320, height=180, num_color_channels=3),
    },
    frames={0: {}},  # Not needed for now
)

# HDR re-encode of testsrc2 (10-bit H265 with BT.2020 + PQ), generated with:
# ffmpeg -i test/resources/testsrc2.mp4 -c:v libx265 -pix_fmt yuv420p10le \
# -x265-params "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited" \
# -preset fast -crf 23 test/resources/testsrc2_hdr.mp4
TEST_SRC_2_720P_HDR = TestVideo(
    filename="testsrc2_hdr.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=320, height=180, num_color_channels=3),
    },
    frames={0: {}},  # Not needed for now
)

# 12-bit HDR testsrc2 (H265 with BT.2020 + PQ), generated with:
# ffmpeg -f lavfi -i testsrc2=duration=2:size=320x180:rate=30 -c:v libx265
# -pix_fmt yuv420p12le -x265-params
# "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited"
# -preset fast -crf 23 test/resources/testsrc2_12bit_hdr.mp4
TEST_SRC_2_12BIT_HDR = TestVideo(
    filename="testsrc2_12bit_hdr.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=320, height=180, num_color_channels=3),
    },
    frames={0: {}},
)

# ffmpeg -f lavfi -i testsrc2=duration=2:size=1280x720:rate=30 -c:v libx264 -profile:v baseline -level 3.1 -pix_fmt yuv420p -b:v 2500k -r 30 -movflags +faststart output_720p_2s.mp4
TEST_SRC_2_720P = TestVideo(
    filename="testsrc2.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=1280, height=720, num_color_channels=3),
    },
    frames={0: {}},  # Not needed for now
)
# ffmpeg -f lavfi -i testsrc2=duration=10:size=1280x720:rate=30 -c:v libx265 -crf 23 -preset medium output.mp4
TEST_SRC_2_720P_H265 = TestVideo(
    filename="testsrc2_h265.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=1280, height=720, num_color_channels=3),
    },
    frames={0: {}},  # Not needed for now
)

# ffmpeg -f lavfi -i testsrc2=size=1280x720:rate=30:duration=1 -c:v libvpx-vp9 -b:v 1M output_vp9.webm
TEST_SRC_2_720P_VP9 = TestVideo(
    filename="testsrc2_vp9.webm",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=1280, height=720, num_color_channels=3),
    },
    frames={0: {}},  # Not needed for now
)

# ffmpeg -f lavfi -i testsrc2=size=1280x720:rate=30:duration=1 -c:v libvpx -b:v 1M output_vp8.webm
TEST_SRC_2_720P_VP8 = TestVideo(
    filename="testsrc2_vp8.webm",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=1280, height=720, num_color_channels=3),
    },
    frames={0: {}},  # Not needed for now
)

# ffmpeg -f lavfi -i testsrc2=size=1280x720:rate=30:duration=1 -c:v mpeg4 -q:v 5 output_mpeg4.avi
TEST_SRC_2_720P_MPEG4 = TestVideo(
    filename="testsrc2_mpeg4.avi",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=1280, height=720, num_color_channels=3),
    },
    frames={0: {}},  # Not needed for now
)

# ffmpeg -f lavfi -i color=c=black:s=64x64:d=0.034 -c:v mpeg4 -q:v 31 testsrc2_mpeg4.mp4
TEST_SRC_2_MPEG4_MP4 = TestVideo(
    filename="testsrc2_mpeg4.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=64, height=64, num_color_channels=3),
    },
    frames={0: {}},  # Not needed for now
)

# Video with non-zero start time (start_time ~8.333s)
# Used to test that PTS values are correctly reported for videos that don't
# start at time 0.
TEST_NON_ZERO_START = TestVideo(
    filename="test_non_zero_start.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=200, height=112, num_color_channels=3),
    },
    frames={},  # Automatically loaded from json file
)

# ffmpeg -f lavfi -i "testsrc2=size=321x240:rate=25:duration=1,format=rgb24" \
#  -c:v libx264 -pix_fmt yuv444p -profile:v high444 testsrc2_odd_width_444.mp4
TESTSRC2_ODD_WIDTH_444 = TestVideo(
    filename="testsrc2_odd_width_444.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=321, height=240, num_color_channels=3),
    },
    frames={0: {}},
)

# ffmpeg -f lavfi -i "testsrc2=size=320x241:rate=25:duration=1,format=rgb24" \
#  -c:v libx264 -pix_fmt yuv444p -profile:v high444 testsrc2_odd_height_444.mp4
TESTSRC2_ODD_HEIGHT_444 = TestVideo(
    filename="testsrc2_odd_height_444.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=320, height=241, num_color_channels=3),
    },
    frames={0: {}},
)

# ffmpeg -f lavfi -i "testsrc2=size=321x241:rate=25:duration=1,format=rgb24" \
#  -c:v libx264 -pix_fmt yuv444p -profile:v testsrc2_odd_height_and_width_444.mp4
TESTSRC2_ODD_HEIGHT_AND_WIDTH_444 = TestVideo(
    filename="testsrc2_odd_height_and_width_444.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=321, height=241, num_color_channels=3),
    },
    frames={0: {}},
)

# AV1 4:2:0 10-bit. NVDEC offers only a P016 output surface for this one, no
# NV12, which used to send it to the CPU fallback whenever uint8 was requested.
# ffmpeg -f lavfi -i "testsrc2=size=320x240:rate=25:duration=1" \
#  -c:v libsvtav1 -pix_fmt yuv420p10le testsrc2_av1_10bit.mp4
TESTSRC2_AV1_10BIT = TestVideo(
    filename="testsrc2_av1_10bit.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=320, height=240, num_color_channels=3),
    },
    frames={0: {}},
)

# ffmpeg -f lavfi -i "testsrc2=size=321x241:rate=25:duration=1,format=rgb24" \
#  -c:v libx264 -pix_fmt yuv444p10le -profile:v high444 \
#  testsrc2_odd_height_and_width_444_10bit.mp4
TESTSRC2_ODD_HEIGHT_AND_WIDTH_444_10BIT = TestVideo(
    filename="testsrc2_odd_height_and_width_444_10bit.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=321, height=241, num_color_channels=3),
    },
    frames={0: {}},
)

# HEVC 4:4:4, which NVDEC *can* decode natively (unlike H264 4:4:4 above), at
# 8, 10 and 12 bits. Odd dimensions, so they also cover the cropping NVDEC's
# even-aligned surfaces need. Encoded with, for DEPTH in 8/10/12:
# ffmpeg -f lavfi -i "testsrc2=size=321x241:rate=25:duration=1,format=rgb24" \
#  -c:v libx265 -pix_fmt yuv444pDEPTHle -tag:v hvc1 testsrc2_444_DEPTHbit_hevc.mp4
# (the 8-bit one uses -pix_fmt yuv444p)
TESTSRC2_444_8BIT_HEVC = TestVideo(
    filename="testsrc2_444_8bit_hevc.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=321, height=241, num_color_channels=3),
    },
    frames={0: {}},
)

TESTSRC2_444_10BIT_HEVC = TestVideo(
    filename="testsrc2_444_10bit_hevc.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=321, height=241, num_color_channels=3),
    },
    frames={0: {}},
)

TESTSRC2_444_12BIT_HEVC = TestVideo(
    filename="testsrc2_444_12bit_hevc.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=321, height=241, num_color_channels=3),
    },
    frames={0: {}},
)

# The sources whose frames don't come out as three YUV planes. libx264 accepts
# -pix_fmt gray but silently encodes 4:2:0 anyway, hence libx265 here. Even
# dimensions, because both encoders below round odd ones down.
# ffmpeg -f lavfi -i "testsrc2=size=320x240:rate=25:duration=1" \
#  -vf format=gray -c:v libx265 -tag:v hvc1 testsrc2_gray_hevc.mp4
TESTSRC2_GRAY_HEVC = TestVideo(
    filename="testsrc2_gray_hevc.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=320, height=240, num_color_channels=3),
    },
    frames={0: {}},
)

# ffmpeg -f lavfi -i "testsrc2=size=321x241:rate=25:duration=1,format=rgb24" \
#  -vf format=gbrp -c:v libx265 -tag:v hvc1 testsrc2_gbrp_hevc.mp4
TESTSRC2_GBRP_HEVC = TestVideo(
    filename="testsrc2_gbrp_hevc.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=321, height=241, num_color_channels=3),
    },
    frames={0: {}},
)

# FFV1 is lossless, so this one is a fifth of a second rather than a full one.
# VP9's alpha is not an option: it rides in a separate layer, and the frames the
# decoder produces are plain yuv420p.
# ffmpeg -f lavfi -i "testsrc2=size=320x240:rate=25:duration=0.2" \
#  -vf format=yuva420p -c:v ffv1 testsrc2_yuva420p_ffv1.mkv
TESTSRC2_YUVA420P_FFV1 = TestVideo(
    filename="testsrc2_yuva420p_ffv1.mkv",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=320, height=240, num_color_channels=3),
    },
    frames={0: {}},
)

# ffmpeg -f lavfi -i "testsrc2=size=321x240:rate=25:duration=1,format=rgb24" \
#  -c:v libvpx-vp9 -pix_fmt yuv420p -b:v 1M testsrc2_odd_width_vp9.mp4
TESTSRC2_ODD_WIDTH_VP9 = TestVideo(
    filename="testsrc2_odd_width_vp9.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=321, height=240, num_color_channels=3),
    },
    frames={0: {}},
)

# ffmpeg -f lavfi -i "testsrc2=size=320x241:rate=25:duration=1,format=rgb24" \
#  -c:v libvpx-vp9 -pix_fmt yuv420p -b:v 1M testsrc2_odd_height_vp9.mp4
TESTSRC2_ODD_HEIGHT_VP9 = TestVideo(
    filename="testsrc2_odd_height_vp9.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=320, height=241, num_color_channels=3),
    },
    frames={0: {}},
)

# ffmpeg -f lavfi -i "testsrc2=size=321x241:rate=25:duration=1,format=rgb24" \
#  -c:v libvpx-vp9 -pix_fmt yuv420p -b:v 1M testsrc2_odd_height_and_width_vp9.mp4
TESTSRC2_ODD_HEIGHT_AND_WIDTH_VP9 = TestVideo(
    filename="testsrc2_odd_height_and_width_vp9.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=321, height=241, num_color_channels=3),
    },
    frames={0: {}},
)

# ffmpeg -f lavfi -i "testsrc2=size=321x240:rate=25:duration=1,format=rgb24" \
#  -c:v libvpx-vp9 -pix_fmt yuv420p10le -b:v 1M testsrc2_odd_width_vp9_10bit.mp4
TESTSRC2_ODD_WIDTH_VP9_10BIT = TestVideo(
    filename="testsrc2_odd_width_vp9_10bit.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=321, height=240, num_color_channels=3),
    },
    frames={0: {}},
)

# ffmpeg -f lavfi -i "testsrc2=size=320x241:rate=25:duration=1,format=rgb24" \
#  -c:v libvpx-vp9 -pix_fmt yuv420p10le -b:v 1M testsrc2_odd_height_vp9_10bit.mp4
TESTSRC2_ODD_HEIGHT_VP9_10BIT = TestVideo(
    filename="testsrc2_odd_height_vp9_10bit.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=320, height=241, num_color_channels=3),
    },
    frames={0: {}},
)

# ffmpeg -f lavfi -i "testsrc2=size=321x241:rate=25:duration=1,format=rgb24" \
#  -c:v libvpx-vp9 -pix_fmt yuv420p10le -b:v 1M testsrc2_odd_height_and_width_vp9_10bit.mp4
TESTSRC2_ODD_HEIGHT_AND_WIDTH_VP9_10BIT = TestVideo(
    filename="testsrc2_odd_height_and_width_vp9_10bit.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=321, height=241, num_color_channels=3),
    },
    frames={0: {}},
)

# Odd dimensions with 4:2:0 chroma, in a codec NVDEC doesn't decode. That's the
# only combination that reaches the CPU fallback with a frame our 4:2:0 CUDA
# kernel can't consume as-is: it has to be padded to even dimensions before
# color conversion, and cropped back afterwards.
# ffmpeg -f lavfi -i "testsrc2=rate=25:duration=0.4:size=121x80,format=rgb24" \
#  -c:v mpeg2video -pix_fmt yuv420p testsrc2_odd_width_mpeg2.mp4
TESTSRC2_ODD_WIDTH_MPEG2 = TestVideo(
    filename="testsrc2_odd_width_mpeg2.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=121, height=80, num_color_channels=3),
    },
    frames={0: {}},
)

# Also odd in height, which additionally exercises the chroma plane's own
# rounding: an odd-height 4:2:0 frame has ceil(height / 2) chroma rows.
# ffmpeg -f lavfi -i "testsrc2=rate=25:duration=0.4:size=121x81,format=rgb24" \
#  -c:v mpeg2video -pix_fmt yuv420p testsrc2_odd_height_and_width_mpeg2.mp4
TESTSRC2_ODD_HEIGHT_AND_WIDTH_MPEG2 = TestVideo(
    filename="testsrc2_odd_height_and_width_mpeg2.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=121, height=81, num_color_channels=3),
    },
    frames={0: {}},
)


def supports_approximate_mode(asset: TestVideo) -> bool:
    # Those are missing the `duration` field so they fail in approximate mode (on all devices).
    # TODO: we should address this, see
    # https://github.com/meta-pytorch/torchcodec/issues/945
    return asset not in (AV1_VIDEO, TEST_SRC_2_720P_VP9, TEST_SRC_2_720P_VP8)


@dataclass
class TestAudio(TestContainerFile):
    """Base class for the *audio* streams of a container (potentially a video),
    or a pure audio file"""

    stream_infos: dict[int, TestAudioStreamInfo]
    # stream_index -> list of 2D frame tensors of shape (num_channels, num_samples_in_that_frame)
    # num_samples_in_that_frame isn't necessarily constant for a given stream.
    _reference_frames: dict[int, list[torch.Tensor]] = field(default_factory=dict)

    # Storing each individual frame is too expensive for audio, because there's
    # a massive overhead in the binary format saved by pytorch. Saving all the
    # frames in a single file uses 1.6MB while saving all frames in individual
    # files uses 302MB (yes).
    # So we store the reference frames in a single file, and load/cache those
    # when the TestAudio instance is created.
    def __post_init__(self):
        super().__post_init__()
        for stream_index in self.stream_infos:
            frames_data_path = _get_file_path(
                f"{self.filename}.stream{stream_index}.all_frames.pt"
            )

            if frames_data_path.exists():
                # To ease development, we allow for the reference frames not to
                # exist. It means the asset cannot be used to check validity of
                # decoded frames.
                self._reference_frames[stream_index] = torch.load(
                    frames_data_path, weights_only=True
                )

    def get_frame_data_by_index(
        self, idx: int, *, stream_index: int | None = None
    ) -> torch.Tensor:
        if stream_index is None:
            stream_index = self.default_stream_index

        return self._reference_frames[stream_index][idx]

    def get_frame_data_by_range(
        self,
        start: int,
        stop: int,
        step: int = 1,
        *,
        stream_index: int | None = None,
    ) -> torch.Tensor:
        tensors = [
            self.get_frame_data_by_index(i, stream_index=stream_index)
            for i in range(start, stop, step)
        ]
        return torch.cat(tensors, dim=-1)

    def get_frame_index(
        self, *, pts_seconds: float, stream_index: int | None = None
    ) -> int:
        if stream_index is None:
            stream_index = self.default_stream_index

        if pts_seconds <= self.frames[stream_index][0].pts_seconds:
            # Special case for e.g. NASA_AUDIO_MP3 whose first frame's pts is
            # 0.13~, not 0.
            return 0
        try:
            # Could use bisect() to maek this faster if needed
            return next(
                frame_index
                for (frame_index, frame_info) in self.frames[stream_index].items()
                if frame_info.pts_seconds
                <= pts_seconds
                < frame_info.pts_seconds + frame_info.duration_seconds
            )
        except StopIteration:
            return len(self.frames[stream_index]) - 1

    @property
    def sample_rate(self) -> int:
        return self.stream_infos[self.default_stream_index].sample_rate

    @property
    def num_channels(self) -> int:
        return self.stream_infos[self.default_stream_index].num_channels

    @property
    def duration_seconds(self) -> float:
        return self.stream_infos[self.default_stream_index].duration_seconds

    @property
    def num_frames(self) -> int:
        return self.stream_infos[self.default_stream_index].num_frames

    @property
    def sample_format(self) -> str:
        return self.stream_infos[self.default_stream_index].sample_format


# This file was generated with:
# ffmpeg -y -i test/resources/nasa_13013.mp4 -b:a 192K -vn test/resources/nasa_13013.mp4.audio.mp3"
NASA_AUDIO_MP3 = TestAudio(
    filename="nasa_13013.mp4.audio.mp3",
    default_stream_index=0,
    frames={},  # Automatically loaded from json file
    stream_infos={
        0: TestAudioStreamInfo(
            sample_rate=8_000,
            num_channels=2,
            duration_seconds=13.248,
            num_frames=183,
            sample_format="fltp",
        )
    },
)

# This file is the same as NASA_AUDIO_MP3, with a sample rate of 44_100. It was generated with:
# ffmpeg -i test/resources/nasa_13013.mp4.audio.mp3 -ar 44100 test/resources/nasa_13013.mp4.audio_44100.mp3
NASA_AUDIO_MP3_44100 = TestAudio(
    filename="nasa_13013.mp4.audio_44100.mp3",
    default_stream_index=0,
    frames={},  # Automatically loaded from json file
    stream_infos={
        0: TestAudioStreamInfo(
            sample_rate=44_100,
            num_channels=2,
            duration_seconds=13.09,
            num_frames=501,
            sample_format="fltp",
        )
    },
)

NASA_AUDIO = TestAudio(
    filename="nasa_13013.mp4",
    default_stream_index=4,
    frames={},  # Automatically loaded from json file
    stream_infos={
        4: TestAudioStreamInfo(
            sample_rate=16_000,
            num_channels=2,
            duration_seconds=13.056,
            num_frames=204,
            sample_format="fltp",
        )
    },
)

# Note that the file itself is s32 sample format, but the reference frames are
# stored as fltp. We can add the s32 original reference frames once we support
# decoding to non-fltp format, but for now we don't need to.
SINE_MONO_S32 = TestAudio(
    filename="sine_mono_s32.wav",
    default_stream_index=0,
    frames={},  # Automatically loaded from json file
    stream_infos={
        0: TestAudioStreamInfo(
            sample_rate=16_000,
            num_channels=1,
            duration_seconds=4,
            num_frames=63,
            sample_format="s32",
        )
    },
)

# This file is an upsampled version of SINE_MONO_S32, generated with:
# ffmpeg -i test/resources/sine_mono_s32.wav -ar 44100 -c:a pcm_s32le test/resources/sine_mono_s32_44100.wav
SINE_MONO_S32_44100 = TestAudio(
    filename="sine_mono_s32_44100.wav",
    default_stream_index=0,
    frames={},  # Automatically loaded from json file
    stream_infos={
        0: TestAudioStreamInfo(
            sample_rate=44_100,
            num_channels=1,
            duration_seconds=4,
            num_frames=173,
            sample_format="s32",
        )
    },
)

# This file is a downsampled version of SINE_MONO_S32, generated with:
# ffmpeg -i test/resources/sine_mono_s32.wav -ar 8000 -c:a pcm_s32le test/resources/sine_mono_s32_8000.wav
SINE_MONO_S32_8000 = TestAudio(
    filename="sine_mono_s32_8000.wav",
    default_stream_index=0,
    frames={},  # Automatically loaded from json file
    stream_infos={
        0: TestAudioStreamInfo(
            sample_rate=8000,
            num_channels=1,
            duration_seconds=4,
            num_frames=32,
            sample_format="s32",
        )
    },
)

# Same sample rate as SINE_MONO_S32, but encoded as s16 instead of s32. Generated with:
# ffmpeg -i test/resources/sine_mono_s32.wav -ar 16000 -c:a pcm_s16le test/resources/sine_mono_s16.wav
SINE_MONO_S16 = TestAudio(
    filename="sine_mono_s16.wav",
    default_stream_index=0,
    frames={},  # Automatically loaded from json file
    stream_infos={
        0: TestAudioStreamInfo(
            sample_rate=16_000,
            num_channels=1,
            duration_seconds=4,
            num_frames=63,
            sample_format="s16",
        )
    },
)

# Same sample rate as SINE_MONO_S32, but encoded as u8 instead of s32. Generated with:
# ffmpeg -i test/resources/sine_mono_s32.wav -c:a pcm_u8 test/resources/sine_mono_u8.wav
SINE_MONO_U8 = TestAudio(
    filename="sine_mono_u8.wav",
    default_stream_index=0,
    frames={0: {}},
    stream_infos={
        0: TestAudioStreamInfo(
            sample_rate=16_000,
            num_channels=1,
            duration_seconds=4,
            num_frames=63,
            sample_format="u8",
        )
    },
)

# Same sample rate as SINE_MONO_S32, but encoded as s24 instead of s32. Generated with:
# ffmpeg -i test/resources/sine_mono_s32.wav -c:a pcm_s24le test/resources/sine_mono_s24.wav
SINE_MONO_S24 = TestAudio(
    filename="sine_mono_s24.wav",
    default_stream_index=0,
    frames={0: {}},
    stream_infos={
        0: TestAudioStreamInfo(
            sample_rate=16_000,
            num_channels=1,
            duration_seconds=4,
            num_frames=63,
            sample_format="s32",
        )
    },
)

# Same sample rate as SINE_MONO_S32, but encoded as f32. Generated with:
# ffmpeg -i test/resources/sine_mono_s32.wav -c:a pcm_f32le test/resources/sine_mono_f32.wav
SINE_MONO_F32 = TestAudio(
    filename="sine_mono_f32.wav",
    default_stream_index=0,
    frames={0: {}},
    stream_infos={
        0: TestAudioStreamInfo(
            sample_rate=16_000,
            num_channels=1,
            duration_seconds=4,
            num_frames=63,
            sample_format="flt",
        )
    },
)

# Same sample rate as SINE_MONO_S32, but encoded as f64. Generated with:
# ffmpeg -i test/resources/sine_mono_s32.wav -c:a pcm_f64le test/resources/sine_mono_f64.wav
SINE_MONO_F64 = TestAudio(
    filename="sine_mono_f64.wav",
    default_stream_index=0,
    frames={0: {}},
    stream_infos={
        0: TestAudioStreamInfo(
            sample_rate=16_000,
            num_channels=1,
            duration_seconds=4,
            num_frames=63,
            sample_format="dbl",
        )
    },
)

# WAV file with an odd-sized data chunk and a trailing metadata chunk.
# This reproduces https://github.com/meta-pytorch/torchcodec/issues/1378 where
# FFmpeg seeks past EOF when scanning for trailing chunks, causing a crash
# when decoding from a bytes tensor (but not from a file path).
# Generated with:
#     import struct
#     path = "test/resources/reproduce_seek_bug.wav"
#     sample_rate = 48000
#     block_align = 3  # 24-bit mono
#     num_samples = 48001  # * 3 = 144003 bytes (odd!)
#     data_size = num_samples * block_align
#     sample_data = bytes(data_size)  # silence
#     trailing_data = b"\x00" * 256
#     with open(path, "wb") as f:
#         riff_size = 4 + (8 + 16) + (8 + data_size + 1) + (8 + len(trailing_data))
#         f.write(b"RIFF")
#         f.write(struct.pack("<I", riff_size))
#         f.write(b"WAVE")
#         f.write(b"fmt ")
#         f.write(struct.pack("<I", 16))
#         f.write(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * block_align, block_align, 24))
#         f.write(b"data")
#         f.write(struct.pack("<I", data_size))
#         f.write(sample_data)
#         f.write(b"\x00")  # RIFF padding byte for odd-sized chunk
#         f.write(b"_PMX")
#         f.write(struct.pack("<I", len(trailing_data)))
#         f.write(trailing_data)
WAV_ODD_DATA_TRAILING_CHUNK = TestAudio(
    filename="reproduce_seek_bug.wav",
    default_stream_index=0,
    frames={0: {}},
    stream_infos={
        0: TestAudioStreamInfo(
            sample_rate=48_000,
            num_channels=1,
            duration_seconds=1.000021,
            num_frames=12,
            sample_format="s32",
        )
    },
)

# MPEG program stream. Seeking into one lands on a container-level byte offset,
# so the parser that rebuilds MP2 frames out of the byte stream resumes
# mid-frame and emits undecodable packets until it resyncs. The 384k bitrate
# matters: it makes MP2 frames big enough relative to the 2048-byte PS packs
# that resyncing takes several packets, not just one. Generated with:
# ffmpeg -f lavfi -i "testsrc=size=128x128:rate=25:duration=4" -f lavfi -i "sine=frequency=440:sample_rate=44100:duration=4" -c:v mpeg1video -b:v 100k -c:a mp2 -b:a 384k -ac 2 test/resources/sine_stereo_mp2.mpg
SINE_STEREO_MP2_MPEG_PS = TestAudio(
    filename="sine_stereo_mp2.mpg",
    default_stream_index=1,
    frames={1: {}},
    stream_infos={
        1: TestAudioStreamInfo(
            sample_rate=44_100,
            num_channels=2,
            duration_seconds=3.996733,
            num_frames=154,
            sample_format="s16p",
        )
    },
)

# 16-channel audio for testing support for >8 channels. Generated with:
# ffmpeg -i test/resources/sine_mono_s32.wav -t 1 -filter_complex "[0]asplit=16[s0][s1][s2][s3][s4][s5][s6][s7][s8][s9][s10][s11][s12][s13][s14][s15];[s0][s1][s2][s3][s4][s5][s6][s7][s8][s9][s10][s11][s12][s13][s14][s15]amerge=inputs=16" -c:a pcm_s16le test/resources/sine_16ch_s16.wav
SINE_16_CHANNEL_S16 = TestAudio(
    filename="sine_16ch_s16.wav",
    default_stream_index=0,
    frames={},  # Automatically loaded from json file
    stream_infos={
        0: TestAudioStreamInfo(
            sample_rate=16_000,
            num_channels=16,
            duration_seconds=1,
            num_frames=16,
            sample_format="s16",
        )
    },
)

# Generated with:
# ffmpeg -y -f lavfi -i "sine=frequency=440:duration=2" -c:a mp3 -b:a 32k -ar 44100 -ac 1 test/resources/sine_mono_mp3.swf
UNSEEKABLE_SWF = TestAudio(
    filename="sine_mono_mp3.swf",
    default_stream_index=0,
    frames={0: {}},
    stream_infos={
        0: TestAudioStreamInfo(
            sample_rate=44_100,
            num_channels=1,
            duration_seconds=2.2739909297052154,
            num_frames=78,
            sample_format="fltp",
        )
    },
)
