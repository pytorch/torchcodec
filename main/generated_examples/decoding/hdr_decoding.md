# Decoding HDR videos

In this example, we'll learn how to decode HDR (High Dynamic Range) videos
using the `output_dtype` parameter of the
[`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) class.

Note

The `output_dtype` parameter is in beta. Its behavior may change in future
versions.

HDR videos typically encode pixel data with more than 8 bits per channel (e.g.
10 or 12 bits). This allows them to represent a wider range of colors and
brightness levels. When decoding such content, it is generally desirable to
preserve that extra precision by decoding into `float32` tensors rather than
the default `uint8`.

## Generating test videos with FFmpeg

First, we'll use FFmpeg to create two short synthetic videos: an SDR video
(standard 8-bit H.264) and an HDR video (10-bit H.265 with BT.2020 color
primaries and SMPTE ST 2084 / PQ transfer characteristics, which is a common
HDR format).

```
import subprocess
import tempfile
from pathlib import Path

import torch

temp_dir = tempfile.mkdtemp()
sdr_video_path = Path(temp_dir) / "sdr_video.mp4"
hdr_video_path = Path(temp_dir) / "hdr_video.mp4"

# Generate a short SDR video (standard 8-bit H.264)
subprocess.run(
 [
 "ffmpeg", "-y",
 "-f", "lavfi", "-i", "testsrc2=duration=2:size=320x180:rate=30",
 "-c:v", "libx264", "-pix_fmt", "yuv420p",
 "-preset", "fast", "-crf", "23",
 str(sdr_video_path),
 ],
 check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)

# Generate a short HDR video (10-bit H.265 with BT.2020 + PQ)
subprocess.run(
 [
 "ffmpeg", "-y",
 "-f", "lavfi", "-i", "testsrc2=duration=2:size=320x180:rate=30",
 "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
 "-x265-params",
 "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited",
 "-preset", "fast", "-crf", "23",
 str(hdr_video_path),
 ],
 check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)
```

```
CompletedProcess(args=['ffmpeg', '-y', '-f', 'lavfi', '-i', 'testsrc2=duration=2:size=320x180:rate=30', '-c:v', 'libx265', '-pix_fmt', 'yuv420p10le', '-x265-params', 'colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited', '-preset', 'fast', '-crf', '23', '/tmp/tmp8mttw1w3/hdr_video.mp4'], returncode=0, stdout=b'', stderr=b"ffmpeg version 7.1.1 Copyright (c) 2000-2025 the FFmpeg developers\n built with gcc 14.3.0 (conda-forge gcc 14.3.0-5)\n configuration: --prefix=/__w/_temp/conda_environment_34606743767 --cc=/home/conda/feedstock_root/build_artifacts/ffmpeg_1758923917305/_build_env/bin/x86_64-conda-linux-gnu-cc --cxx=/home/conda/feedstock_root/build_artifacts/ffmpeg_1758923917305/_build_env/bin/x86_64-conda-linux-gnu-c++ --nm=/home/conda/feedstock_root/build_artifacts/ffmpeg_1758923917305/_build_env/bin/x86_64-conda-linux-gnu-nm --ar=/home/conda/feedstock_root/build_artifacts/ffmpeg_1758923917305/_build_env/bin/x86_64-conda-linux-gnu-ar --disable-doc --enable-openssl --enable-demuxer=dash --enable-hardcoded-tables --enable-libfreetype --enable-libharfbuzz --enable-libfontconfig --enable-libopenh264 --enable-libdav1d --disable-gnutls --enable-libvpx --enable-libass --enable-pthreads --enable-alsa --enable-libpulse --enable-vaapi --enable-libvpl --enable-libopenvino --enable-gpl --enable-libx264 --enable-libx265 --enable-libmp3lame --enable-libaom --enable-libsvtav1 --enable-libxml2 --enable-pic --enable-shared --disable-static --enable-version3 --enable-zlib --enable-libvorbis --enable-libopus --enable-librsvg --enable-ffplay --pkg-config=/home/conda/feedstock_root/build_artifacts/ffmpeg_1758923917305/_build_env/bin/pkg-config\n libavutil 59. 39.100 / 59. 39.100\n libavcodec 61. 19.101 / 61. 19.101\n libavformat 61. 7.100 / 61. 7.100\n libavdevice 61. 3.100 / 61. 3.100\n libavfilter 10. 4.100 / 10. 4.100\n libswscale 8. 3.100 / 8. 3.100\n libswresample 5. 3.100 / 5. 3.100\n libpostproc 58. 3.100 / 58. 3.100\nInput #0, lavfi, from 'testsrc2=duration=2:size=320x180:rate=30':\n Duration: N/A, start: 0.000000, bitrate: N/A\n Stream #0:0: Video: wrapped_avframe, yuv420p, 320x180 [SAR 1:1 DAR 16:9], 30 fps, 30 tbr, 30 tbn\nStream mapping:\n Stream #0:0 -> #0:0 (wrapped_avframe (native) -> hevc (libx265))\nPress [q] to stop, [?] for help\nx265 [info]: HEVC encoder version 3.5+1-f0c1022b6\nx265 [info]: build info [Linux][GCC 15.3.0][64 bit] 10bit\nx265 [info]: using cpu capabilities: MMX2 SSE2Fast LZCNT SSSE3 SSE4.2 AVX FMA3 BMI2 AVX2\nx265 [info]: Main 10 profile, Level-2 (Main tier)\nx265 [info]: Thread pool created using 16 threads\nx265 [info]: Slices : 1\nx265 [info]: frame threads / pool features : 4 / wpp(3 rows)\nx265 [warning]: Source height < 720p; disabling lookahead-slices\nx265 [info]: Coding QT: max CU size, min CU size : 64 / 8\nx265 [info]: Residual QT: max TU size, max depth : 32 / 1 inter / 1 intra\nx265 [info]: ME / range / subpel / merge : hex / 57 / 2 / 2\nx265 [info]: Keyframe min / max / scenecut / bias : 25 / 250 / 40 / 5.00 \nx265 [info]: Lookahead / bframes / badapt : 15 / 4 / 0\nx265 [info]: b-pyramid / weightp / weightb : 1 / 1 / 0\nx265 [info]: References / ref-limit cu / depth : 3 / on / on\nx265 [info]: AQ: mode / str / qg-size / cu-tree : 2 / 1.0 / 32 / 1\nx265 [info]: Rate Control / qCompress : CRF-23.0 / 0.60\nx265 [info]: tools: rd=2 psy-rd=2.00 rskip mode=1 signhide tmvp fast-intra\nx265 [info]: tools: strong-intra-smoothing deblock sao\nOutput #0, mp4, to '/tmp/tmp8mttw1w3/hdr_video.mp4':\n Metadata:\n encoder : Lavf61.7.100\n Stream #0:0: Video: hevc (hev1 / 0x31766568), yuv420p10le(tv, progressive), 320x180 [SAR 1:1 DAR 16:9], q=2-31, 30 fps, 15360 tbn\n Metadata:\n encoder : Lavc61.19.101 libx265\n Side data:\n cpb: bitrate max/min/avg: 0/0/0 buffer size: 0 vbv_delay: N/A\n[out#0/mp4 @ 0x55e9ed8a5980] video:67KiB audio:0KiB subtitle:0KiB other streams:0KiB global headers:2KiB muxing overhead: 5.737264%\nframe= 60 fps=0.0 q=30.9 Lsize= 71KiB time=00:00:01.93 bitrate= 300.9kbits/s speed=8.02x \nx265 [info]: frame I: 1, Avg QP:21.56 kb/s: 966.00 \nx265 [info]: frame P: 12, Avg QP:24.23 kb/s: 550.02 \nx265 [info]: frame B: 47, Avg QP:30.60 kb/s: 188.91 \nx265 [info]: Weighted P-Frames: Y:0.0% UV:0.0%\nx265 [info]: consecutive B-frames: 7.7% 0.0% 0.0% 7.7% 84.6% \n\nencoded 60 frames in 0.22s (268.48 fps), 274.08 kb/s, Avg QP:29.17\n")
```

## Default behavior: `uint8` output

By default, the decoder outputs frames as `torch.uint8` tensors with
values in [0, 255]. This works well for SDR content, but for HDR videos
the 10-bit (or 12-bit) pixel values get quantized down to 8 bits, losing
precision:

```
from torchcodec.decoders import VideoDecoder

sdr_decoder = VideoDecoder(sdr_video_path)
sdr_frame = sdr_decoder[0]

print(f"SDR pixel format: {sdr_decoder.metadata.pixel_format}")
print(f"SDR frame dtype: {sdr_frame.dtype}")
print(f"SDR frame value range: [{sdr_frame.min()}, {sdr_frame.max()}]")
```

```
SDR pixel format: yuv420p
SDR frame dtype: torch.uint8
SDR frame value range: [0, 255]
```

```
hdr_decoder = VideoDecoder(hdr_video_path)
hdr_frame = hdr_decoder[0]

print(f"HDR pixel format: {hdr_decoder.metadata.pixel_format}")
print(f"HDR frame dtype: {hdr_frame.dtype}")
print(f"HDR frame value range: [{hdr_frame.min()}, {hdr_frame.max()}]")
```

```
HDR pixel format: yuv420p10le
HDR frame dtype: torch.uint8
HDR frame value range: [0, 255]
```

Both SDR and HDR videos produce `uint8` frames. For the HDR video, this
means precision is lost: the original 10-bit values (0-1023) are squeezed
into 8-bit (0-255).

## Using `output_dtype=torch.float32`

To preserve the full precision of HDR content, set
`output_dtype=torch.float32`. This produces frames with values in [0, 1].
This can also be used on SDR content if you want normalized float values:

```
sdr_decoder_float = VideoDecoder(sdr_video_path, output_dtype=torch.float32)
sdr_frame_float = sdr_decoder_float[0]

print(f"SDR frame as float32: dtype={sdr_frame_float.dtype}, "
 f"range=[{sdr_frame_float.min():.4f}, {sdr_frame_float.max():.4f}]")
```

```
SDR frame as float32: dtype=torch.float32, range=[0.0000, 1.0000]
```

```
hdr_decoder_float = VideoDecoder(hdr_video_path, output_dtype=torch.float32)
hdr_frame_float = hdr_decoder_float[0]

print(f"HDR frame as float32: dtype={hdr_frame_float.dtype}, "
 f"range=[{hdr_frame_float.min():.4f}, {hdr_frame_float.max():.4f}]")
```

```
HDR frame as float32: dtype=torch.float32, range=[0.0000, 1.0000]
```

## Using `output_dtype="auto"`

When working with a mix of SDR and HDR videos, you can use
`output_dtype="auto"` to let the decoder choose the output dtype
automatically. SDR content will be decoded as `uint8`, and HDR content
(i.e. videos with more than 8 bits per channel) will be decoded as
`float32`:

```
auto_sdr_decoder = VideoDecoder(sdr_video_path, output_dtype="auto")
auto_hdr_decoder = VideoDecoder(hdr_video_path, output_dtype="auto")

print(f"SDR video with 'auto': {auto_sdr_decoder[0].dtype}")
print(f"HDR video with 'auto': {auto_hdr_decoder[0].dtype}")
```

```
SDR video with 'auto': torch.uint8
HDR video with 'auto': torch.float32
```

## Inspecting HDR metadata

You can inspect color-related metadata to understand the HDR characteristics
of a video. Key fields are `pixel_format`, `color_primaries`,
`color_space`, and `color_transfer_characteristic`:

```
print(f"Pixel format: {auto_hdr_decoder.metadata.pixel_format}")
print(f"Color primaries: {auto_hdr_decoder.metadata.color_primaries}")
print(f"Color space: {auto_hdr_decoder.metadata.color_space}")
print(f"Transfer characteristic: {auto_hdr_decoder.metadata.color_transfer_characteristic}")
```

```
Pixel format: yuv420p10le
Color primaries: bt2020
Color space: bt2020nc
Transfer characteristic: smpte2084
```

We can verify that these match the raw stream properties reported by
`ffprobe`:

```
result = subprocess.run(
 [
 "ffprobe", "-v", "quiet",
 "-select_streams", "v:0",
 "-show_entries", "stream=pix_fmt,color_primaries,color_space,color_transfer",
 "-of", "default=noprint_wrappers=1",
 str(hdr_video_path),
 ],
 capture_output=True, text=True, check=True,
)
print(result.stdout)
```

```
pix_fmt=yuv420p10le
color_space=bt2020nc
color_transfer=smpte2084
color_primaries=bt2020
```

```
import shutil
shutil.rmtree(temp_dir)
```

**Total running time of the script:** (0 minutes 0.439 seconds)

[`Download Jupyter notebook: hdr_decoding.ipynb`](../../_downloads/01dc406b3f53981468974c53f6e40c46/hdr_decoding.ipynb)

[`Download Python source code: hdr_decoding.py`](../../_downloads/80c7717989dc5b33fb1880951c8b5de7/hdr_decoding.py)

[`Download zipped: hdr_decoding.zip`](../../_downloads/c7e85613b63b3c44335352321c761eaf/hdr_decoding.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)