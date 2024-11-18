# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torchcodec
from torchvision.io.image import write_png

print(f"{torchcodec.__version__ = }")

decoder = torchcodec.decoders._core.create_from_file(
    str(Path(__file__).parent / "../resources/nasa_13013.mp4")
)
torchcodec.decoders._core.scan_all_streams_to_update_metadata(decoder)
torchcodec.decoders._core.add_video_stream(decoder, stream_index=3)
frame, _, _ = torchcodec.decoders._core.get_frame_at_index(
    decoder, stream_index=3, frame_index=180
)
write_png(frame, "frame180.png")
