# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Note: usort wants to put Frame and FrameBatch after decoders and samplers,
# but that results in circular import.
from ._frame import Frame, FrameBatch  # usort:skip # noqa
from . import decoders, samplers  # noqa

try:
    # Note that version.py is generated during install.
    from .version import __version__  # noqa: F401
except Exception:
    pass
