# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

import numpy as np

import torch
from PIL import Image

if __name__ == "__main__":
    img_file = sys.argv[1]
    # Get base filename without extension
    base_filename = os.path.splitext(img_file)[0]
    extension = os.path.splitext(img_file)[1]
    if extension == ".pt":
        img_tensor = torch.load(img_file)
        print(img_tensor.shape)
        print(img_tensor.dtype)
        # Save tensor to disk
        output_file = base_filename + ".bmp"
        if len(sys.argv) > 2:
            output_file = sys.argv[2]
        if img_tensor.shape[0] == 3:
            img_tensor = img_tensor.permute(1, 2, 0)
        img_array = img_tensor.cpu().numpy()
        img = Image.fromarray(img_array)
        img.save(output_file, format="BMP")
        print(f"Saved BMP to {output_file}")
    else:
        pil_image = Image.open(img_file)
        img_tensor = torch.from_numpy(np.asarray(pil_image))
        print(img_tensor.shape)
        print(img_tensor.dtype)
        # Save tensor to disk
        output_file = base_filename + ".pt"
        if len(sys.argv) > 2:
            output_file = sys.argv[2]
        torch.save(img_tensor, output_file, _use_new_zipfile_serialization=True)
        print(f"Saved tensor to {output_file}")
