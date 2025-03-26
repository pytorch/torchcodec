# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import sys
from pathlib import Path
from types import ModuleType


# Copy pasted from torchvision
# https://github.com/pytorch/vision/blob/947ae1dc71867f28021d5bc0ff3a19c249236e2a/torchvision/_internally_replaced_utils.py#L25
def _get_extension_path(lib_name: str) -> str:
    extension_suffixes = []
    if sys.platform == "linux":
        extension_suffixes = importlib.machinery.EXTENSION_SUFFIXES
    elif sys.platform == "darwin":
        extension_suffixes = importlib.machinery.EXTENSION_SUFFIXES + [".dylib"]
    else:
        raise NotImplementedError(
            "Platforms other than linux/darwin are not supported yet"
        )
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


def _load_pybind11_module(module_name: str, library_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        module_name,
        library_path,
    )
    if spec is None:
        raise ImportError(
            f"Unable to load spec for module {module_name} from path {library_path}"
        )

    return importlib.util.module_from_spec(spec)
