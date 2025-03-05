# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Literal, Optional, Union

from torch import Tensor

from torchcodec.decoders import _core as core
from torchcodec.decoders._decoder_utils import (
    create_decoder,
    get_and_validate_stream_metadata,
    validate_seek_mode,
)


class AudioDecoder:
    """TODO-audio docs"""

    def __init__(
        self,
        source: Union[str, Path, bytes, Tensor],
        *,
        stream_index: Optional[int] = None,
        seek_mode: Literal["exact", "approximate"] = "exact",
    ):
        validate_seek_mode(seek_mode)
        self._decoder = create_decoder(source=source, seek_mode=seek_mode)

        core.add_audio_stream(self._decoder, stream_index=stream_index)

        (
            self.metadata,
            self.stream_index,
            self._num_frames,
            self._begin_stream_seconds,
            self._end_stream_seconds,
        ) = get_and_validate_stream_metadata(
            decoder=self._decoder, stream_index=stream_index, media_type="audio"
        )
