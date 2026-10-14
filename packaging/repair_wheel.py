# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bundle binary dependencies into the wheel.

TorchCodec depends on non-Python libraries: FFmpeg, libjpeg, libtorch, etc.
"repairing" a wheel means bundling those binary dependencies into the wheel so
that the wheel runs standalone on a system that doesn't have those libraries
installed.

We bundle some third-party native libraries like libjpeg(-turbo), libpng, zlib,
libwebp (+libsharpyuv), libavif, libnvjpeg, while making sure we EXCLUDE FFmpeg
(user-provided at runtime) and torch/CUDA (provided by the torch wheel).

Because we redistribute those libraries as binaries inside the wheel, their
(permissive) licenses require us to also ship their copyright/license texts.
do that in bundle_third_party_licenses().
"""

import io
import json
import os
import platform
import re
import shutil
import site
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

DIST_DIR = Path("dist")
REPAIRED_DIR = Path("dist_repaired")


def _is_cuda_wheel(wheel):
    # Detect a CUDA wheel from its local-version tag (e.g. "+cu126") in the filename.
    return re.search(r"[+_]cu\d", Path(wheel).name) is not None


def run(cmd, **kwargs):
    cmd = [str(c) for c in cmd]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def _avif_lib_dir():
    # libavif isn't in conda like the other image libs; it's fetched from S3
    # into scikit-build's build dir.
    dirs = [p.resolve() for p in Path("build").glob("*/_deps/avif_s3-src/lib")]
    if len(dirs) != 1:
        raise RuntimeError(f"Expected exactly one S3 libavif dir, found: {dirs}")
    return dirs[0]


def _find_nvjpeg_libs():
    # Find the nvJPEG runtime lib(s) to bundle. Its location varies a lot across
    # CI setups so we search a wide set of CUDA roots recursively (+ ldconfig on
    # Linux).
    is_windows = platform.system() == "Windows"
    pattern = "nvjpeg64*.dll" if is_windows else "libnvjpeg.so*"

    roots = []
    for var in ("CUDA_HOME", "CUDA_PATH", "CUDAToolkit_ROOT", "CONDA_PREFIX"):
        if v := os.environ.get(var):
            roots.append(Path(v))
    # nvcc on PATH -> toolkit root (e.g. /usr/local/cuda/bin/nvcc -> /usr/local/cuda).
    if nvcc := shutil.which("nvcc"):
        roots.append(Path(nvcc).resolve().parent.parent)
    if is_windows:
        roots += list(
            Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA").glob("*")
        )
    else:
        roots += list(Path("/usr/local").glob("cuda*"))
    for site_dir in site.getsitepackages():
        roots.append(Path(site_dir) / "nvidia")  # pip nvidia-*-cu12 packages

    matches = []
    for root in roots:
        try:
            matches.extend(root.rglob(pattern))
        except OSError:
            pass
    if not is_windows:
        try:
            out = subprocess.run(
                ["ldconfig", "-p"], capture_output=True, text=True, check=False
            ).stdout
            for line in out.splitlines():
                if "libnvjpeg.so" in line and "=>" in line:
                    matches.append(Path(line.split("=>")[-1].strip()))
        except (OSError, subprocess.SubprocessError):
            pass

    found = set()
    for m in matches:
        # Skip the CUDA "stubs" libs: they're link-time placeholders (unversioned
        # libnvjpeg.so with no real code), not the runtime lib we must bundle.
        if "stubs" in m.parts:
            continue
        try:
            resolved = m.resolve()
        except OSError:
            continue
        if resolved.is_file():
            found.add(resolved)
    return list(found)


def _find_nvjpeg_license():
    # Try to find EULA.txt, fallback to LICENSE
    dirs = []
    for lib in _find_nvjpeg_libs():
        dirs.extend(lib.parents)
    for var in ("CUDA_HOME", "CUDA_PATH", "CUDAToolkit_ROOT"):
        if v := os.environ.get(var):
            dirs.append(Path(v))
    for filename in ("EULA.txt", "LICENSE"):
        for d in dirs:
            candidate = d / filename
            if candidate.is_file():
                return candidate
    return None


def repair_linux(wheels):
    run([sys.executable, "-m", "pip", "install", "--upgrade", "auditwheel"])
    run(["auditwheel", "--version"])
    env = os.environ.copy()
    # for auditwheel to graft libs, it must be able to find them, so we set
    # LD_LIBRARY_PATH: jpeg/png/webp are from conda, libavif is from the S3
    # build dir, and (for CUDA wheels) libnvjpeg is from the CUDA toolkit.
    lib_dirs = [str(_avif_lib_dir())]
    if conda_prefix := env.get("CONDA_PREFIX"):
        lib_dirs.append(str(Path(conda_prefix) / "lib"))
    if any(_is_cuda_wheel(w) for w in wheels):
        lib_dirs.extend(sorted({str(f.parent) for f in _find_nvjpeg_libs()}))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [*lib_dirs, env.get("LD_LIBRARY_PATH", "")]
    )

    excludes = []
    for pattern in (
        # FFmpeg libs, spelled out rather than "libav*" so we don't match libavif.
        "libavcodec*",
        "libavdevice*",
        "libavfilter*",
        "libavformat*",
        "libavutil*",
        "libavresample*",
        "libsw*",
        "libpostproc*",
        "libtorch*",
        "libc10*",
        "libcu*",
        "libcupti*",
        "libheif*",
        "libde265*",
        "libx265*",
        "libnvrtc*",
        "libnvToolsExt*",
        "libnvtx*",
        "libnvjitlink*",
        "libnvshmem*",
        "libnvfatbin*",
        "libnvcuvid*",
        # rocJPEG: the GPU JPEG decoder our image lib links on ROCm. Unlike
        # nvJPEG (which we bundle), rocJPEG is not shipped by the torch-ROCm
        # wheel, and bundling it would drag in torch's ROCm libs under mismatched
        # (hashed) sonames. So we treat it as a runtime dependency provided by the
        # ROCm install, like FFmpeg. decode_jpeg(device='cuda') therefore needs
        # ROCm (with rocJPEG) present at runtime.
        # TODO_ROCM: Should we still try to ship librocjpeg?
        "librocjpeg*",
        # ROCm/HIP runtime and its system deps: provided by the torch-ROCm wheel
        # (torch/lib/) at runtime, exactly like the CUDA libs above. Never bundle
        # them, they'd duplicate torch's copies and bloat the wheel.
        "libamdhip64*",
        "libamd_comgr*",
        "libhsa-runtime64*",
        "libhiprtc*",
        "librocm-core*",
        "librocprofiler-register*",
        "libroctx64*",
        "librocroller*",
        "libMIOpen*",
        "libhipblas*",
        "libhipfft*",
        "libhiprand*",
        "libhipsolver*",
        "libhipsparse*",
        "librocblas*",
        "librocfft*",
        "librocrand*",
        "librocsolver*",
        "librocsparse*",
        "librccl*",
        "libnuma*",
        "libdrm*",
        "libelf*",
        "libbz2*",
        "liblzma*",
    ):
        excludes += ["--exclude", pattern]
    for wheel in wheels:
        run(
            ["auditwheel", "repair", *excludes, "--wheel-dir", REPAIRED_DIR, wheel],
            env=env,
        )


def repair_macos(wheels):
    run([sys.executable, "-m", "pip", "install", "--upgrade", "delocate"])
    run(["delocate-wheel", "--version"])

    # Same search path as for linux: the libavif install dir for libavif and
    # CONDA_PREFIX for the other image libs.
    search = os.pathsep.join(
        [str(_avif_lib_dir())]
        + ([str(Path(p) / "lib")] if (p := os.environ.get("CONDA_PREFIX")) else [])
    )
    excludes = " ".join(
        f"--exclude {p}"
        for p in (
            "libavcodec",
            "libavdevice",
            "libavfilter",
            "libavformat",
            "libavutil",
            "libavresample",
            "libsw",
            "libpostproc",
            "libtorch.",
            "libtorch_",
            "libc10",
            "libomp",
            "libheif",
            "libde265",
            "libx265",
        )
    )

    for wheel in wheels:
        run(
            [
                "bash",
                "-c",
                # DYLD_LIBRARY_PATH must be set inline on the command ($0=search,
                # $1=wheel): macOS SIP strips it from inherited env on CI (see
                # cibuildwheel #816).
                #
                # --no-sanitize-rpaths: delocate otherwise deletes every absolute
                # LC_RPATH from each lib. They'd lose the
                # /opt/homebrew/opt/ffmpeg/lib rpath that lets a user's Homebrew
                # FFmpeg be found at runtime (see the INSTALL_RPATH logic in
                # _core/CMakeLists.txt).
                f'DYLD_LIBRARY_PATH="$0" delocate-wheel -v --no-sanitize-rpaths '
                f'--ignore-missing-dependencies {excludes} -w "{REPAIRED_DIR}" "$1"',
                search,
                str(wheel),
            ]
        )


def repair_windows(wheels):
    # We do what torchvision does on Windows: copy the libjpeg/libpng/zlib etc.
    # DLLs next to our libs inside the wheel. At load time Windows resolves a
    # DLL's dependencies from the DLL's own directory, so they are found. We
    # repack with `wheel` so the RECORD is regenerated.
    run([sys.executable, "-m", "pip", "install", "-U", "wheel"])
    bin_dir = Path(os.environ.get("CONDA_PREFIX", "")) / "Library" / "bin"

    jpeg_dlls = set(bin_dir.glob("jpeg*.dll")) | set(bin_dir.glob("libjpeg*.dll"))
    if not jpeg_dlls:
        raise FileNotFoundError(f"No libjpeg DLL found under {bin_dir}")
    png_dlls = set(bin_dir.glob("libpng*.dll")) | set(bin_dir.glob("png*.dll"))
    if not png_dlls:
        raise FileNotFoundError(f"No libpng DLL found under {bin_dir}")
    # libpng depends on zlib; bundle it too so libpng can resolve it at load time.
    zlib_dlls = set(bin_dir.glob("zlib*.dll")) | set(bin_dir.glob("libz*.dll"))
    if not zlib_dlls:
        raise FileNotFoundError(f"No zlib DLL found under {bin_dir}")
    # libwebp depends on libsharpyuv; bundle both.
    webp_dlls = set(bin_dir.glob("libwebp*.dll")) | set(bin_dir.glob("webp*.dll"))
    if not webp_dlls:
        raise FileNotFoundError(f"No libwebp DLL found under {bin_dir}")
    sharpyuv_dlls = set(bin_dir.glob("libsharpyuv*.dll")) | set(
        bin_dir.glob("sharpyuv*.dll")
    )
    if not sharpyuv_dlls:
        raise FileNotFoundError(f"No libsharpyuv DLL found under {bin_dir}")
    # libavif comes from our S3 build (not conda): its DLL is in the FetchContent
    # build dir's bin/.
    avif_dlls = set(Path("build").glob("*/_deps/avif_s3-src/bin/libavif*.dll"))
    if not avif_dlls:
        raise FileNotFoundError("No libavif DLL under build/*/_deps/avif_s3-src/bin")

    dlls = jpeg_dlls | png_dlls | zlib_dlls | webp_dlls | sharpyuv_dlls | avif_dlls

    if any(_is_cuda_wheel(w) for w in wheels):
        nvjpeg_dlls = set(_find_nvjpeg_libs())
        # Also check the conda Library\bin next to the other image DLLs.
        nvjpeg_dlls |= set(bin_dir.glob("nvjpeg64*.dll"))
        if not nvjpeg_dlls:
            raise FileNotFoundError(
                "No nvjpeg64*.dll found for a CUDA build. See the CUDA bundling "
                "debug above for the roots searched."
            )
        dlls |= nvjpeg_dlls

    dlls = sorted(dlls)

    for wheel in wheels:
        unpack_dir = REPAIRED_DIR / "unpack"
        if unpack_dir.is_dir():
            shutil.rmtree(unpack_dir)
        run([sys.executable, "-m", "wheel", "unpack", wheel, "-d", unpack_dir])
        pkg_dirs = list(unpack_dir.glob("*/torchcodec"))
        if not pkg_dirs:
            raise FileNotFoundError("torchcodec/ package dir not found in wheel")
        pkg_dir = pkg_dirs[0]
        for dll in dlls:
            print(f"bundling {dll} -> {pkg_dir}", flush=True)
            shutil.copy(dll, pkg_dir)
        run([sys.executable, "-m", "wheel", "pack", pkg_dir.parent, "-d", REPAIRED_DIR])
        shutil.rmtree(unpack_dir)


def bundle_third_party_licenses():
    """Inject the license/copyright texts of the bundled third-party libraries
    into each wheel's .dist-info/licenses/third_party/ dir.

    We redistribute libjpeg-turbo, libpng, zlib, libwebp and libavif (which
    statically embeds dav1d and libyuv) as binaries inside the wheel. Their
    permissive licenses (IJG/BSD/zlib) require reproducing the copyright notice
    and license text in binary redistributions, so we ship them next to our own
    LICENSE. CUDA wheels additionally bundle libnvjpeg, redistributed under the
    NVIDIA CUDA Toolkit EULA, which we ship as well.
    """

    def _resolve_conda_licenses():
        """Map dest filename -> source path for the conda-provided image libs.

        conda ships each package's upstream license text under
        <extracted_package_dir>/info/licenses/, and
        CONDA_PREFIX/conda-meta/<pkg>.json records where that dir is. We resolve
        from there so the text always matches the exact binary we bundle.
        """
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if not conda_prefix:
            raise RuntimeError(
                "CONDA_PREFIX not set; cannot locate conda license files."
            )
        meta_dir = Path(conda_prefix) / "conda-meta"

        # logical lib -> (dest filename stem, candidate conda package names).
        # Some libs are packaged under more than one name across
        # channels/versions (e.g. zlib vs libzlib), so we try each candidate and
        # take the first that resolves.
        wanted = {
            "libjpeg-turbo": ("LICENSE.libjpeg-turbo", ["libjpeg-turbo"]),
            "libpng": ("LICENSE.libpng", ["libpng"]),
            "zlib": ("LICENSE.zlib", ["libzlib", "zlib"]),
            "libwebp": ("LICENSE.libwebp", ["libwebp", "libwebp-base"]),
        }

        collected = {}
        for logical, (dest_stem, candidates) in wanted.items():
            src_files = None
            for pkg in candidates:
                metas = sorted(meta_dir.glob(f"{pkg}-*.json"))
                if not metas:
                    continue
                info = json.loads(metas[0].read_text())
                lic_dir = Path(info["extracted_package_dir"]) / "info" / "licenses"
                if lic_dir.is_dir():
                    src_files = sorted(f for f in lic_dir.iterdir() if f.is_file())
                    break
            if not src_files:
                raise RuntimeError(
                    f"Could not find license files for {logical} (tried conda "
                    f"packages {candidates} under {meta_dir})."
                )
            # A package usually ships a single license file; if it ships several,
            # keep them all, suffixed with their original name.
            if len(src_files) == 1:
                collected[dest_stem] = src_files[0]
            else:
                for f in src_files:
                    collected[f"{dest_stem}.{f.name}"] = f
        return collected

    def _resolve_avif_licenses():
        """Map dest filename -> source path for the libavif stack (libavif
        itself, plus dav1d and libyuv which are statically embedded inside
        libavif). These are collected into licenses/ by
        packaging/build_libavif.sh and shipped in the S3 artifact that
        fetch_avif_from_s3.cmake unpacks into scikit-build's build dir.
        """
        dirs = [
            p for p in Path("build").glob("*/_deps/avif_s3-src/licenses") if p.is_dir()
        ]
        if not dirs:
            raise RuntimeError(
                "libavif licenses dir not found under "
                "build/*/_deps/avif_s3-src/licenses"
            )
        # Multiple build dirs (one per ABI) may exist; the license texts are
        # identical, so pick any.
        return {f.name: f for f in sorted(dirs[0].iterdir()) if f.is_file()}

    run([sys.executable, "-m", "pip", "install", "-U", "wheel"])
    base_licenses = {**_resolve_conda_licenses(), **_resolve_avif_licenses()}
    print("Third-party license files to bundle:")
    for name, src in sorted(base_licenses.items()):
        print(f"  {name} <- {src}")

    scratch = Path("dist_licenses")
    if scratch.is_dir():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    for wheel in sorted(DIST_DIR.glob("*.whl")):
        licenses = dict(base_licenses)
        if _is_cuda_wheel(wheel):
            if (nvjpeg_license := _find_nvjpeg_license()) is None:
                raise RuntimeError(
                    f"{wheel.name} bundles libnvjpeg but the NVIDIA CUDA EULA "
                    "could not be located to ship alongside it."
                )
            licenses["LICENSE.libnvjpeg-NVIDIA-CUDA-EULA.txt"] = nvjpeg_license
            print(f"  LICENSE.libnvjpeg-NVIDIA-CUDA-EULA.txt <- {nvjpeg_license}")

        unpack_dir = scratch / "unpack"
        if unpack_dir.is_dir():
            shutil.rmtree(unpack_dir)
        run([sys.executable, "-m", "wheel", "unpack", wheel, "-d", unpack_dir])
        dist_info_dirs = list(unpack_dir.glob("*/*.dist-info"))
        if len(dist_info_dirs) != 1:
            raise RuntimeError(
                f"Expected exactly one .dist-info in {wheel.name}, "
                f"found: {dist_info_dirs}"
            )
        dest = dist_info_dirs[0] / "licenses" / "third_party"
        dest.mkdir(parents=True, exist_ok=True)
        for name, src in licenses.items():
            shutil.copy(src, dest / name)
        # Repack: `wheel pack` regenerates RECORD so the new files are recorded.
        run(
            [
                sys.executable,
                "-m",
                "wheel",
                "pack",
                dist_info_dirs[0].parent,
                "-d",
                scratch,
            ]
        )
        shutil.rmtree(unpack_dir)

    for wheel in DIST_DIR.glob("*.whl"):
        wheel.unlink()
    for wheel in scratch.glob("*.whl"):
        shutil.move(str(wheel), str(DIST_DIR))
    shutil.rmtree(scratch)


def check_bundling():
    """Raise if:
    - a wheel bundles a lib that's not in the allowlist. This would raise if we
      ever try to bundle FFmpeg or torch/CUDA.
    - a wheel does NOT bundle libjpeg, libpng, libwebp, libwebpdemux or libavif.
    - a wheel is missing the license/copyright text of any bundled third-party
      lib under .dist-info/licenses/third_party/ (see
      bundle_third_party_licenses).
    - the wheel bundles an AV1 encoder library: our libavif is decode-only, so
      encoders (aom/rav1e/svtav1) must never ship (all platforms). This is not
      for licensing concern, this is to keep wheel size low.
    - the wheel bundles libheif or its HEVC codecs (libde265/libx265): these are
      LGPL/GPL and must NEVER ship. libtorchcodec_heic links libheif at build
      time but the user supplies it at runtime (like FFmpeg).
    - a CUDA wheel does NOT bundle libnvjpeg (the GPU JPEG decoder lib), or a
      non-CUDA wheel DOES bundle it.
    - the compressed wheel is larger than MAX_WHEEL_BYTES: the slim decode-only
      libavif should keep us under it.
    - (Linux only) the bundled libjpeg isn't libjpeg-turbo.
    - (Linux only) libtorchcodec_image.so or libtorchcodec_pybind_ops.so links
      FFmpeg.
    - (MacOS only) a bundled libtorchcodec_core*.dylib is missing the Homebrew
      FFmpeg rpath
    """

    def _is_shared_lib(name):
        base = name.rsplit("/", 1)[-1]
        return ".so" in base or ".dylib" in base or base.endswith((".dll", ".pyd"))

    def _is_jpeg(lib):
        return lib.startswith("libjpeg") or (
            lib.startswith("jpeg") and lib.endswith(".dll")
        )

    def _is_png(lib):
        return lib.startswith("libpng") or (
            lib.startswith("png") and lib.endswith(".dll")
        )

    def _is_zlib(lib):
        return lib.startswith(("libz", "zlib"))

    def _is_webp(lib):
        return lib.startswith(("libwebp", "libsharpyuv")) or (
            lib.startswith(("webp", "sharpyuv")) and lib.endswith(".dll")
        )

    def _is_avif(lib):
        stem = lib.lower()
        return stem.startswith("libavif") or (
            stem.startswith("avif") and stem.endswith(".dll")
        )

    def _is_nvjpeg(lib):
        return lib.startswith("libnvjpeg") or (
            lib.startswith("nvjpeg") and lib.endswith(".dll")
        )

    def _is_avif_encoder(lib):
        stem = lib.lower()
        return stem.startswith(("libaom", "librav1e", "libsvtav1", "libdav1d")) or (
            stem.startswith(("aom", "rav1e", "svtav1", "dav1d"))
            and stem.endswith(".dll")
        )

    def _is_forbidden_lgpl(lib):
        # libheif and its HEVC codecs are LGPL/GPL and must never be bundled.
        stem = lib.lower()
        return stem.startswith(("libheif", "libde265", "libx265")) or (
            stem.startswith(("heif", "de265", "x265")) and stem.endswith(".dll")
        )

    def _is_webp_demux(lib):
        # libwebpdemux is a separate lib from the base libwebp; it provides the
        # WebPAnimDecoder API used to decode animated webp files.
        return lib.startswith("libwebpdemux") or (
            lib.startswith("webpdemux") and lib.endswith(".dll")
        )

    def _is_allowed(lib):
        if (
            lib.startswith("libtorchcodec_")
            or _is_jpeg(lib)
            or _is_png(lib)
            or _is_zlib(lib)
            or _is_webp(lib)
            or _is_avif(lib)
            or _is_nvjpeg(lib)
        ):
            return True
        if platform.system() == "Darwin" and lib.startswith(("libc++", "libpython")):
            # I can attest libc++ is there, but I'm not entirely sure about
            # libpython. I used to be there when `delocate` was run from the
            # `test-infra` job, but now that we run it here it doesn't seem to
            # be there anymore. I guess it doesn't hurt.
            return True
        return False

    _FFMPEG_SONAME_PREFIXES = (
        "libavcodec",
        "libavdevice",
        "libavfilter",
        "libavformat",
        "libavutil",
        "libavresample",
        "libsw",
        "libpostproc",
    )

    def _assert_linux_lib_no_ffmpeg(zf, lib_name):
        """Enforce that `lib_name` does NOT link FFmpeg (no FFmpeg soname in
        DT_NEEDED; see _FFMPEG_SONAME_PREFIXES).

        Both libtorchcodec_image.so (the image decoders/encoders) and
        libtorchcodec_pybind_ops.so (the Python file-like bridge) are built
        separately from the FFmpeg-dependent core{4,5,6,7,8,9}.so libraries and
        must stay FFmpeg-free:
        - the image lib, to avoid symbol interposition between the bundled image
          codec libs (libjpeg/libpng/libwebp) and the user's FFmpeg, which may
          come with its own libjpeg/libpng too;
        - the pybind lib, so it can be loaded (and image encoding used) even when
          FFmpeg isn't installed.

        This check ensures we didn't accidentally link FFmpeg into them, which
        would defeat the purpose of building them separately.
        """
        from elftools.elf.elffile import ELFFile

        members = [n for n in zf.namelist() if n.rsplit("/", 1)[-1] == lib_name]
        if not members:
            raise RuntimeError(
                f"{lib_name} not found in wheel; it's expected to live in its "
                "own shared library."
            )
        elf = ELFFile(io.BytesIO(zf.read(members[0])))
        dynamic = elf.get_section_by_name(".dynamic")
        needed = [t.needed for t in dynamic.iter_tags("DT_NEEDED")] if dynamic else []
        ffmpeg_needed = [n for n in needed if n.startswith(_FFMPEG_SONAME_PREFIXES)]
        if ffmpeg_needed:
            raise RuntimeError(
                f"{lib_name} must not link FFmpeg, but its DT_NEEDED lists: "
                + " ".join(ffmpeg_needed)
            )

    def _assert_linux_libjpeg_is_turbo(zf):
        jpeg_members = [
            n
            for n in zf.namelist()
            if _is_shared_lib(n) and _is_jpeg(n.rsplit("/", 1)[-1])
        ]
        assert len(jpeg_members) == 1
        jpeg_member = jpeg_members[0]

        from elftools.elf.elffile import ELFFile

        elf = ELFFile(io.BytesIO(zf.read(jpeg_member)))
        verdefs = elf.get_section_by_name(".gnu.version_d")
        is_turbo = verdefs is not None and any(
            aux.name.startswith("LIBJPEGTURBO")
            for _, auxes in verdefs.iter_versions()
            for aux in auxes
        )
        if not is_turbo:
            raise RuntimeError(
                f"Bundled {jpeg_member.rsplit('/', 1)[-1]} is not libjpeg-turbo (no "
                "LIBJPEGTURBO version node). Ensure libjpeg-turbo is the libjpeg "
                "found at build time."
            )

    def _assert_macos_homebrew_rpath_is_present(zf):
        def get_rpaths(dylib_bytes):
            """return LC_RPATH entries of a thin 64-bit little-endian Mach-O image."""
            LC_RPATH = 0x8000001C
            (magic,) = struct.unpack_from("<I", dylib_bytes, 0)
            if magic != 0xFEEDFACF:
                raise RuntimeError(
                    f"Expected a thin 64-bit Mach-O, got magic {magic:#x}"
                )
            (num_commands,) = struct.unpack_from("<I", dylib_bytes, 16)
            offset = 32  # mach_header_64 size
            rpaths = []
            for _ in range(num_commands):
                command, command_size = struct.unpack_from("<II", dylib_bytes, offset)
                if command == LC_RPATH:
                    (path_offset,) = struct.unpack_from("<I", dylib_bytes, offset + 8)
                    path = dylib_bytes[offset + path_offset : offset + command_size]
                    rpaths.append(path.split(b"\0")[0].decode())
                offset += command_size
            return rpaths

        # Mirrors the CMake logic.
        homebrew_prefix = os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew")
        expected = f"{homebrew_prefix}/opt/ffmpeg/lib"
        members = [
            n
            for n in zf.namelist()
            if re.fullmatch(
                r"libtorchcodec_(core|custom_ops)\d+\.dylib", n.rsplit("/", 1)[-1]
            )
        ]
        if not members:
            raise RuntimeError("No libtorchcodec_core*.dylib found in wheel.")
        for member in members:
            rpaths = get_rpaths(zf.read(member))
            if expected not in rpaths:
                raise RuntimeError(
                    f"{member} is missing the {expected} rpath needed to find a "
                    f"Homebrew-installed FFmpeg. Its rpaths are: {rpaths}"
                )

    def _assert_third_party_licenses(zf, is_cuda):
        """Every bundled third-party lib must ship its license text under
        .dist-info/licenses/third_party/ (see bundle_third_party_licenses)."""
        license_files = [
            n
            for n in zf.namelist()
            if "/licenses/third_party/" in n and not n.endswith("/")
        ]
        # keyword each bundled lib's license file must be identifiable by. CUDA
        # wheels also bundle libnvjpeg, whose NVIDIA CUDA EULA must ship too.
        keywords = ["jpeg", "png", "zlib", "webp", "avif", "dav1d", "yuv"]
        if is_cuda:
            keywords.append("nvjpeg")
        for keyword in keywords:
            if not any(keyword in n.lower() for n in license_files):
                raise RuntimeError(
                    f"No third-party license file matching '{keyword}' found in "
                    f".dist-info/licenses/third_party/. Found: {license_files}"
                )

    for wheel in DIST_DIR.glob("*.whl"):
        print(f"Checking bundled libraries in {wheel.name}")
        with zipfile.ZipFile(wheel) as zf:
            _assert_third_party_licenses(zf, _is_cuda_wheel(wheel))
            names = zf.namelist()
            libs = sorted({n.rsplit("/", 1)[-1] for n in names if _is_shared_lib(n)})
            if unexpected := [lib for lib in libs if not _is_allowed(lib)]:
                raise RuntimeError(
                    f"Unexpected libraries bundled in {wheel.name}: "
                    + " ".join(unexpected)
                )
            if not any(_is_jpeg(lib) for lib in libs):
                raise RuntimeError(f"{wheel.name} does not bundle libjpeg.")
            if not any(_is_png(lib) for lib in libs):
                raise RuntimeError(f"{wheel.name} does not bundle libpng.")
            if not any(_is_webp(lib) for lib in libs):
                raise RuntimeError(f"{wheel.name} does not bundle libwebp.")
            if not any(lib.lower().startswith(("libavif", "avif")) for lib in libs):
                raise RuntimeError(f"{wheel.name} does not bundle libavif.")
            if not any(_is_webp_demux(lib) for lib in libs):
                raise RuntimeError(
                    f"{wheel.name} does not bundle libwebpdemux (needed for "
                    "animated webp decoding)."
                )
            is_cuda = _is_cuda_wheel(wheel)
            bundles_nvjpeg = any(_is_nvjpeg(lib) for lib in libs)
            if is_cuda and not bundles_nvjpeg:
                raise RuntimeError(
                    f"{wheel.name} is a CUDA wheel but does not bundle libnvjpeg. "
                    "GPU JPEG decoding (decode_jpeg(..., device='cuda')) needs it, "
                    "and torch does not ship it. Check that libnvjpeg is findable "
                    "at repair time (see _find_nvjpeg_libs) and not excluded."
                )
            if not is_cuda and bundles_nvjpeg:
                raise RuntimeError(
                    f"{wheel.name} is not a CUDA wheel but bundles libnvjpeg."
                )
            if encoders := [lib for lib in libs if _is_avif_encoder(lib)]:
                raise RuntimeError(
                    f"{wheel.name} bundles AV1 codec libraries that must not "
                    "ship with our decode-only libavif (they should be "
                    "statically embedded or absent): " + " ".join(encoders)
                )
            if lgpl := [lib for lib in libs if _is_forbidden_lgpl(lib)]:
                raise RuntimeError(
                    f"{wheel.name} bundles LGPL/GPL libraries that must NEVER "
                    "ship (libheif is a user-supplied runtime dependency, like "
                    "FFmpeg): " + " ".join(lgpl)
                )
            MAX_WHEEL_BYTES = (17 if is_cuda else 10) * 1024 * 1024
            wheel_bytes = wheel.stat().st_size
            if wheel_bytes > MAX_WHEEL_BYTES:
                raise RuntimeError(
                    f"{wheel.name} is {wheel_bytes / 1024 / 1024:.1f} MB "
                    "compressed, over the "
                    f"{MAX_WHEEL_BYTES / 1024 / 1024:.0f} MB limit. "
                    "Bump MAX_WHEEL_BYTES if a legitimate dependency growth pushes us over. "
                )
            if platform.system() == "Linux":
                _assert_linux_libjpeg_is_turbo(zf)
                _assert_linux_lib_no_ffmpeg(zf, "libtorchcodec_image.so")
                _assert_linux_lib_no_ffmpeg(zf, "libtorchcodec_pybind_ops.so")
            elif platform.system() == "Darwin":
                _assert_macos_homebrew_rpath_is_present(zf)
        print("OK: only libjpeg (and allowed libs) bundled.")


def main():
    wheels = list(DIST_DIR.glob("*.whl"))
    if not wheels:
        raise FileNotFoundError("No wheels found in dist/.")

    if REPAIRED_DIR.is_dir():
        shutil.rmtree(REPAIRED_DIR)
    REPAIRED_DIR.mkdir(parents=True)

    system = platform.system()
    if system == "Linux":
        repair_linux(wheels)
    elif system == "Darwin":
        repair_macos(wheels)
    elif system == "Windows":
        repair_windows(wheels)
    else:
        raise RuntimeError(f"Unknown platform {system!r}.")

    # Replace the original wheels with the repaired ones.
    for wheel in wheels:
        wheel.unlink()
    for wheel in REPAIRED_DIR.glob("*.whl"):
        shutil.move(str(wheel), str(DIST_DIR))
    shutil.rmtree(REPAIRED_DIR)

    bundle_third_party_licenses()

    print("Repaired wheels:")
    for wheel in DIST_DIR.glob("*.whl"):
        print(f"  {wheel}")

    check_bundling()


if __name__ == "__main__":
    sys.exit(main())
