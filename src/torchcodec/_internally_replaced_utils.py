# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import sys
from pathlib import Path


# Copy pasted from torchvision
# https://github.com/pytorch/vision/blob/947ae1dc71867f28021d5bc0ff3a19c249236e2a/torchvision/_internally_replaced_utils.py#L25
def _get_extension_path(lib_name):
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
        raise ImportError

    return ext_specs.origin
