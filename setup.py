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
        # Setuptools was designed to build one extension at a time, calling this
        # method for each Extension object. We're using a CMake-based build
        # where all our extensions are built together at once, so we only need a
        # fake extension to trigger the build.
        assert ext.name == "FAKE_NAME", f"Unexpected extension name: {ext.name}"
        # install_prefix is a temp directory where the built extension(s) will
        # be "installed" by CMake. Once they're copied to install_prefix, the
        # built .so files still need to be copied back into:
        # - the source tree (for editable installs) - this is handled in
        #   copy_extensions_to_source()
        # - the (temp) wheel directory (when building a wheel). I cannot tell
        #   exactly *where* this is handled, but for this to work we must
        #   prepend the "/torchcodec" folder to install_prefix: this tells
        #   setuptools to eventually move those .so files into `torchcodec/`.
        self._install_prefix = Path(self.get_ext_fullpath(ext.name)).parent.absolute() / "torchcodec"
        self._build_all_extensions_with_cmake()

    def _build_all_extensions_with_cmake(self):
        # Note that self.debug is True when you invoke setup.py like this:
        # python setup.py build_ext --debug install
        build_type = "Debug" if self.debug else "Release"
        torch_dir = Path(torch.utils.cmake_prefix_path) / "Torch"
        cmake_args = [
            f"-DCMAKE_INSTALL_PREFIX={self._install_prefix}",
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

    def copy_extensions_to_source(self):
        """Copy built extensions from temporary folder back into source tree.

        This is called by setuptools during editable (-e) install.
        """
        self.get_finalized_command('build_py')

        for so_file in self._install_prefix.glob("*.so"):
            assert "libtorchcodec" in so_file.name
            destination = Path("src/torchcodec/") / so_file.name
            print(f"Copying {so_file} to {destination}")
            self.copy_file(so_file, destination, level=self.verbose)


# See `CMakeBuild.build_extension()`.
fake_extension = Extension(name="FAKE_NAME", sources=[])
setup(ext_modules=[fake_extension], cmdclass={"build_ext": CMakeBuild})
