# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from ._core import VideoStreamMetadata
from ._video_decoder import VideoDecoder  # noqa

# from ._audio_decoder import AudioDecoder  # Will be public when more stable

SimpleVideoDecoder = VideoDecoder
