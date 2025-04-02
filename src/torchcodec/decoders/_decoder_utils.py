# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from typing import Union

from torch import Tensor
from torchcodec import _core as core

ERROR_REPORTING_INSTRUCTIONS = """
This should never happen. Please report an issue following the steps in
https://github.com/pytorch/torchcodec/issues/new?assignees=&labels=&projects=&template=bug-report.yml.
"""


def create_decoder(
    *, source: Union[str, Path, bytes, Tensor], seek_mode: str
) -> Tensor:
    if isinstance(source, str):
        return core.create_from_file(source, seek_mode)
    elif isinstance(source, Path):
        return core.create_from_file(str(source), seek_mode)
    elif isinstance(source, bytes):
        return core.create_from_bytes(source, seek_mode)
    elif isinstance(source, Tensor):
        return core.create_from_tensor(source, seek_mode)

    raise TypeError(
        f"Unknown source type: {type(source)}. "
        "Supported types are str, Path, bytes and Tensor."
    )
