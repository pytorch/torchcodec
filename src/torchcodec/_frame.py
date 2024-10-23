# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from dataclasses import dataclass
from typing import Iterable, Iterator, Union

from torch import Tensor


def _frame_repr(self):
    # Utility to replace Frame and FrameBatch __repr__ method. This prints the
    # shape of the .data tensor rather than printing the (potentially very long)
    # data tensor itself.
    s = self.__class__.__name__ + ":\n"
    spaces = "  "
    for field in dataclasses.fields(self):
        field_name = field.name
        field_val = getattr(self, field_name)
        if field_name == "data":
            field_name = "data (shape)"
            field_val = field_val.shape
        s += f"{spaces}{field_name}: {field_val}\n"
    return s


@dataclass
class Frame(Iterable):
    """A single video frame with associated metadata."""

    data: Tensor
    """The frame data as (3-D ``torch.Tensor``)."""
    pts_seconds: float
    """The :term:`pts` of the frame, in seconds (float)."""
    duration_seconds: float
    """The duration of the frame, in seconds (float)."""

    def __iter__(self) -> Iterator[Union[Tensor, float]]:
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)

    def __repr__(self):
        return _frame_repr(self)


@dataclass
class FrameBatch(Iterable):
    """Multiple video frames with associated metadata."""

    data: Tensor
    """The frames data as (4-D ``torch.Tensor``)."""
    pts_seconds: Tensor
    """The :term:`pts` of the frame, in seconds (1-D ``torch.Tensor`` of floats)."""
    duration_seconds: Tensor
    """The duration of the frame, in seconds (1-D ``torch.Tensor`` of floats)."""

    def __iter__(self) -> Iterator[Union[Tensor, float]]:
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)

    def __getitem__(self, key):
        return FrameBatch(
            self.data[key],
            self.pts_seconds[key],
            self.duration_seconds[key],
        )

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return _frame_repr(self)
