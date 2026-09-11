# Encoding images

In this example, we'll learn how to encode an image tensor to JPEG or PNG using
the [`JpegEncoder`](../../generated/torchcodec.encoders.JpegEncoder.html#torchcodec.encoders.JpegEncoder) and
[`PngEncoder`](../../generated/torchcodec.encoders.PngEncoder.html#torchcodec.encoders.PngEncoder) classes.

Note

These encoders supersede the ones from `torchvision.io`: they are more
robust and support more features. See
[Migrating from TorchVision to TorchCodec](../migration/torchvision_migration.html#sphx-glr-generated-examples-migration-torchvision-migration-py) for a
migration guide.

First, a bit of boilerplate: we'll download an image from the web and define a
plotting utility. You can ignore that part and jump right below to
Encoding an image.

```
import requests
import torch

from torchcodec.decoders import decode_image

url = "https://raw.githubusercontent.com/meta-pytorch/torchcodec/refs/heads/main/docs/source/_static/thumbnails/pigeon_encoding.jpeg"
response = requests.get(url, headers={"User-Agent": ""})
if response.status_code != 200:
 raise RuntimeError(f"Failed to download image. {response.status_code = }.")

# The image to encode, a CHW uint8 tensor. It could come from anywhere (e.g. a
# model output); here we just decode one.
image = decode_image(response.content)

def plot(image: torch.Tensor):
 try:
 import matplotlib.pyplot as plt
 from torchvision.transforms.v2.functional import to_pil_image
 except ImportError:
 print("Cannot plot, please run `pip install torchvision matplotlib`")
 return

 pil_image = to_pil_image(image)
 fig = plt.figure(figsize=(pil_image.width / 100, pil_image.height / 100))
 ax = fig.add_axes([0, 0, 1, 1])
 ax.imshow(pil_image)
 ax.axis("off")
```

## Encoding an image

Encoders expect a 3D uint8 tensor in CHW layout (1 or 3 channels), which is
exactly what our image is:

```
print(f"{image.shape = }, {image.dtype = }")
plot(image)
```

![image encoding](../../_images/sphx_glr_image_encoding_001.png)

```
image.shape = torch.Size([3, 288, 300]), image.dtype = torch.uint8
```

We instantiate a [`JpegEncoder`](../../generated/torchcodec.encoders.JpegEncoder.html#torchcodec.encoders.JpegEncoder) with the image, and
encode it. Three destinations are supported: a file with
[`to_file()`](../../generated/torchcodec.encoders.JpegEncoder.html#torchcodec.encoders.JpegEncoder.to_file), a file-like object with
[`to_file_like()`](../../generated/torchcodec.encoders.JpegEncoder.html#torchcodec.encoders.JpegEncoder.to_file_like), or a 1D uint8 tensor of
raw bytes with [`to_tensor()`](../../generated/torchcodec.encoders.JpegEncoder.html#torchcodec.encoders.JpegEncoder.to_tensor).

```
import io

from torchcodec.encoders import JpegEncoder

encoder = JpegEncoder(image)

encoder.to_file("image.jpg") # to a file
encoder.to_file_like(io.BytesIO()) # to a file-like object
encoded = encoder.to_tensor() # to a tensor

print(f"{encoded.shape = }, {encoded.dtype = }")
```

```
encoded.shape = torch.Size([12146]), encoded.dtype = torch.uint8
```

That's it! We can decode the encoded bytes back to make sure everything worked:

```
from torchcodec.decoders import decode_jpeg

decoded = decode_jpeg(encoded)
print(f"{decoded.shape = }")
plot(decoded)
```

![image encoding](../../_images/sphx_glr_image_encoding_002.png)

```
decoded.shape = torch.Size([3, 288, 300])
```

[`PngEncoder`](../../generated/torchcodec.encoders.PngEncoder.html#torchcodec.encoders.PngEncoder) works exactly the same way, and PNG is
lossless (unlike JPEG):

```
from torchcodec.encoders import PngEncoder

encoded = PngEncoder(image).to_tensor()
print(f"{encoded.shape = }")
```

```
encoded.shape = torch.Size([104350])
```

Both encoders support encoding options: `JpegEncoder` takes a `quality`
(1-100), and `PngEncoder` takes a `compression_level` (0-9). For example, a
lower JPEG quality yields a smaller output:

```
small = JpegEncoder(image).to_tensor(quality=10)
large = JpegEncoder(image).to_tensor(quality=95)
print(f"{small.numel() = }, {large.numel() = }")
```

```
small.numel() = 3824, large.numel() = 28251
```

## Encoding JPEGs on GPU

`JpegEncoder` can encode directly on a CUDA device with nvJPEG: just pass it
an image that already lives on the GPU, and the encoding happens there. Only
3-channel RGB images are supported on CUDA. With [`to_tensor`](../../generated/torchcodec.encoders.JpegEncoder.html#torchcodec.encoders.JpegEncoder.to_tensor), the encoded bytes stay on the GPU
(call `.cpu()` to bring them back to the host).

```
from torchcodec.encoders import JpegEncoder

encoded = JpegEncoder(image.cuda()).to_tensor() # encoded bytes on the GPU
# you can still use to_file and to_file_like, but the encoded bytes will
# be copied back to the CPU first.
```

PNG encoding is CPU-only.

Check the docstrings of the encoding methods to learn about the different
encoding options.

**Total running time of the script:** (0 minutes 0.129 seconds)

[`Download Jupyter notebook: image_encoding.ipynb`](../../_downloads/071d0183e16c5e6ef85b625a2f126867/image_encoding.ipynb)

[`Download Python source code: image_encoding.py`](../../_downloads/d9aa94f540367c75e341e60118bada71/image_encoding.py)

[`Download zipped: image_encoding.zip`](../../_downloads/1689ad89cb6adaf1b06ea28f74ae69c9/image_encoding.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)