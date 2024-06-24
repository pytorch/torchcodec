import os
import sys

import numpy as np

import torch
from PIL import Image

if __name__ == "__main__":
    img_file = sys.argv[1]
    # Get base filename without extension
    base_filename = os.path.splitext(img_file)[0]
    pil_image = Image.open(img_file)
    img_tensor = torch.from_numpy(np.asarray(pil_image))
    print(img_tensor.shape)
    print(img_tensor.dtype)
    # Save tensor to disk
    torch.save(img_tensor, base_filename + ".pt", _use_new_zipfile_serialization=True)
