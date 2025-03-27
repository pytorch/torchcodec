# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Build / install instructions:

- use a virtual env (conda or whatever you want)
- install pytorch nightly (https://pytorch.org/get-started/locally/)
- pip install -e . --no-build-isolation


Note:
The "torch" package is not just a runtime dependency but also a *build time*
dependency, since we are including pytorch's headers. We are however not
specifying either of these dependencies in our pyproject.toml file.

Why we don't specify torch as a runtime dep: I'm not 100% sure, all I know is
that no project does it and those who tried had tons of problems. I think it has
to do with the fact that there are different flavours of torch (cpu, cuda, etc.)
and the pyproject.toml system does not allow a fine-grained enough control over
that.

Why we don't specify torch as a build time dep: because really developers need
to rely on torch-nightly, not on the stable version of torch. And the only way
to install torch nightly is to specify a custom `--index-url` and sadly
pyproject.toml does not allow that.

To be perfeclty honest I'm not 110% sure about the above, but this is definitely
fine for now. Basically what that means is that we expect developers and users
to install the correct version of torch before they install / build torchcodec.
This is what all other libraries expect as well.

Oh, and by default, doing `pip install -e .` would try to build the package in
an isolated virtual environment, not in the current one. But because we're not
specifying torch as a build-time dependency, this fails loudly as torch can't be
found. That's why we're passing `--no-build-isolation`: this tells pip to build
the package within the current virtual env, where torch would have already been
installed.
"""

import os
import subprocess
import sys
from pathlib import Path

import torch
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


_ROOT_DIR = Path(__file__).parent.resolve()


class CMakeBuild(build_ext):

    def __init__(self, *args, **kwargs):
        self._install_prefix = None
        super().__init__(*args, **kwargs)

    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake is not available.") from None
        super().run()

    def build_extension(self, ext):
        """Call our CMake build system to build libtorchcodec*.so"""
        # Setuptools was designed to build one extension (.so file) at a time,
        # calling this method for each Extension object. We're using a
        # CMake-based build where all our extensions are built together at once.
        # If we were to declare one Extension object per .so file as in a
        # standard setup, a) we'd have to keep the Extensions names in sync with
        # the CMake targets, and b) we would be calling into CMake for every
        # single extension: that's overkill and inefficient, since CMake builds
        # all the extensions at once. To avoid all that we create a *single*
        # fake Extension which triggers the CMake build only once.
        assert ext.name == "FAKE_NAME", f"Unexpected extension name: {ext.name}"
        # The price to pay for our non-standard setup is that we have to tell
        # setuptools *where* those extensions are expected to be within the
        # source tree (for sdists or editable installs) or within the wheel.
        # Normally, setuptools relies on the extension's name to figure that
        # out, e.g. an extension named `torchcodec.libtorchcodec.so` would be
        # placed in `torchcodec/` and importable from `torchcodec.`. From that,
        # setuptools knows how to move the extensions from their temp build
        # directories back into the proper dir.
        # Our fake extension's name is just a placeholder, so we have to handle
        # that relocation logic ourselves.
        # _install_prefix is the temp directory where the built extension(s)
        # will be "installed" by CMake. Once they're copied to install_prefix,
        # the built .so files still need to be copied back into:
        # - the source tree (for editable installs) - this is handled in
        #   copy_extensions_to_source()
        # - the (temp) wheel directory (when building a wheel). I cannot tell
        #   exactly *where* this is handled, but for this to work we must
        #   prepend the "/torchcodec" folder to _install_prefix: this tells
        #   setuptools to eventually move those .so files into `torchcodec/`.
        # It may seem overkill to 'cmake install' the extensions in a temp
        # directory and move them back to another dir, but this is what
        # setuptools would do and expect even in a standard build setup.
        self._install_prefix = (
            Path(self.get_ext_fullpath(ext.name)).parent.absolute() / "torchcodec"
        )
        self._build_all_extensions_with_cmake()

    def _build_all_extensions_with_cmake(self):
        # Note that self.debug is True when you invoke setup.py like this:
        # python setup.py build_ext --debug install
        torch_dir = Path(torch.utils.cmake_prefix_path) / "Torch"
        cmake_build_type = os.environ.get("CMAKE_BUILD_TYPE", "Release")
        enable_cuda = os.environ.get("ENABLE_CUDA", "")
        enable_xpu = os.environ.get("ENABLE_XPU", "")
        python_version = sys.version_info
        cmake_args = [
            f"-DCMAKE_INSTALL_PREFIX={self._install_prefix}",
            f"-DTorch_DIR={torch_dir}",
            "-DCMAKE_VERBOSE_MAKEFILE=ON",
            f"-DCMAKE_BUILD_TYPE={cmake_build_type}",
            f"-DPYTHON_VERSION={python_version.major}.{python_version.minor}",
            f"-DENABLE_CUDA={enable_cuda}",
            f"-DENABLE_XPU={enable_xpu}",
        ]

        Path(self.build_temp).mkdir(parents=True, exist_ok=True)

        subprocess.check_call(
            ["cmake", str(_ROOT_DIR)] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(["cmake", "--build", "."], cwd=self.build_temp)
        subprocess.check_call(["cmake", "--install", "."], cwd=self.build_temp)

    def copy_extensions_to_source(self):
        """Copy built extensions from temporary folder back into source tree.

        This is called by setuptools at the end of .run() during editable installs.
        """
        self.get_finalized_command("build_py")
        extensions = []
        if sys.platform == "linux":
            extensions = ["so"]
        elif sys.platform == "darwin":
            # Mac has BOTH .dylib and .so as library extensions. Short version
            # is that a .dylib is a shared library that can be both dynamically
            # loaded and depended on by other libraries; a .so can only be a
            # dynamically loaded module. For more, see:
            #   https://stackoverflow.com/a/2339910
            extensions = ["dylib", "so"]
        else:
            raise NotImplementedError(
                "Platforms other than linux/darwin are not supported yet"
            )

        for ext in extensions:
            for lib_file in self._install_prefix.glob(f"*.{ext}"):
                assert "libtorchcodec" in lib_file.name
                destination = Path("src/torchcodec/") / lib_file.name
                print(f"Copying {lib_file} to {destination}")
                self.copy_file(lib_file, destination, level=self.verbose)


NOT_A_LICENSE_VIOLATION_VAR = "I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION"
BUILD_AGAINST_ALL_FFMPEG_FROM_S3_VAR = "BUILD_AGAINST_ALL_FFMPEG_FROM_S3"
not_a_license_violation = os.getenv(NOT_A_LICENSE_VIOLATION_VAR) is not None
build_against_all_ffmpeg_from_s3 = (
    os.getenv(BUILD_AGAINST_ALL_FFMPEG_FROM_S3_VAR) is not None
)
if "bdist_wheel" in sys.argv and not (
    build_against_all_ffmpeg_from_s3 or not_a_license_violation
):
    raise ValueError(
        "It looks like you're trying to build a wheel. "
        f"You probably want to set {BUILD_AGAINST_ALL_FFMPEG_FROM_S3_VAR}. "
        f"If you have a good reason *not* to, then set {NOT_A_LICENSE_VIOLATION_VAR}."
    )

# See `CMakeBuild.build_extension()`.
fake_extension = Extension(name="FAKE_NAME", sources=[])


def _write_version_files():
    if version := os.getenv("BUILD_VERSION"):
        # BUILD_VERSION is set by the `test-infra` build jobs. It typically is
        # the content of `version.txt` plus some suffix like "+cpu" or "+cu112".
        # See
        # https://github.com/pytorch/test-infra/blob/61e6da7a6557152eb9879e461a26ad667c15f0fd/tools/pkg-helpers/pytorch_pkg_helpers/version.py#L113
        with open(_ROOT_DIR / "version.txt", "w") as f:
            f.write(f"{version}")
    else:
        with open(_ROOT_DIR / "version.txt") as f:
            version = f.readline().strip()
        try:
            sha = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=str(_ROOT_DIR)
                )
                .decode("ascii")
                .strip()
            )
            version += "+" + sha[:7]
        except Exception:
            print("INFO: Didn't find sha. Is this a git repo?")

    with open(_ROOT_DIR / "src/torchcodec/version.py", "w") as f:
        f.write("# Note that this file is generated during install.\n")
        f.write(f"__version__ = '{version}'\n")


_write_version_files()

setup(
    ext_modules=[fake_extension],
    cmdclass={"build_ext": CMakeBuild},
)
