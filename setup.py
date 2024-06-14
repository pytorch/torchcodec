import os
import subprocess
from pathlib import Path

import torch
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

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


_ROOT_DIR = Path(__file__).parent.resolve()


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake is not available.") from None
        super().run()

    def build_extension(self, ext):
        install_prefix = Path(self.get_ext_fullpath(ext.name)).parent.absolute()
        # Note that self.debug is True when you invoke setup.py like this:
        # python setup.py build_ext --debug install
        build_type = "Debug" if self.debug else "Release"
        torch_dir = Path(torch.utils.cmake_prefix_path) / "Torch"
        cmake_args = [
            f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
            f"-DTorch_DIR={torch_dir}",
            "-DCMAKE_VERBOSE_MAKEFILE=ON",
            f"-DCMAKE_BUILD_TYPE={build_type}",
        ]

        Path(self.build_temp).mkdir(parents=True, exist_ok=True)

        subprocess.check_call(
            ["cmake", str(_ROOT_DIR)] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(["cmake", "--build", "."], cwd=self.build_temp)
        subprocess.check_call(["cmake", "--install", "."], cwd=self.build_temp)

    def get_ext_filename(self, fullname):
        # When copying the .so files from the build tmp dir to the actual
        # package dir, this tells setuptools to look for a .so file without the
        # Python ABI suffix, i.e. "libtorchcodec4.so" instead of e.g.
        # "libtorchcodec4.cpython-38-x86_64-linux-gnu.so", which is what
        # setuptools looks for by default.
        ext_filename = super().get_ext_filename(fullname)
        ext_filename_parts = ext_filename.split(".")
        without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
        ext_filename = ".".join(without_abi)
        return ext_filename


# If BUILD_AGAINST_INSTALLED_FFMPEG is set, we only build against the installed
# FFmpeg. We don't know what FFmpeg version that is, so we build
# `libtorchcodec.so` without any version suffix.
# If BUILD_AGAINST_INSTALLED_FFMPEG is not set then we want to build against all
# ffmpeg major version that are available on our S3 bucket.
FFMPEG_MAJOR_VERSIONS = (
    ("",) if os.getenv("BUILD_AGAINST_INSTALLED_FFMPEG") is not None else (4, 5, 6, 7)
)
extensions = [
    Extension(
        # The names here must be kept in sync with the target names in the
        # CMakeLists file. Grep for [ LIBTORCHCODEC_KEEP_IN_SYNC ].  The name
        # parameter specifies not just the name but mainly *where* the .so file
        # should be copied to. By setting the name this way, we're telling
        # `libtorchcodec{ffmpeg_major_version}.so` to be copied at the root of
        # the torchcodec package (next to the __init__.py file). This is where
        # it's expected to be when we call torch.ops.load_library().
        name=f"torchcodec.libtorchcodec{ffmpeg_major_version}",
        # sources is a mandatory parameter so we have to pass it, but we leave
        # it empty because we'll be building the extensions with our custom
        # CMakeBuild class, and the sources are specified within the
        # CMakeLists.txt file.
        sources=[],
    )
    for ffmpeg_major_version in FFMPEG_MAJOR_VERSIONS
]
setup(ext_modules=extensions, cmdclass={"build_ext": CMakeBuild})
