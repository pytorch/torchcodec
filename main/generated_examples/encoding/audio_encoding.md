# Encoding audio samples with AudioEncoder

In this example, we'll learn how to encode audio samples to a file or to raw
bytes using the [`AudioEncoder`](../../generated/torchcodec.encoders.AudioEncoder.html#torchcodec.encoders.AudioEncoder) class.

Note

This is a convenience class for simple, one-shot audio encoding. For
multi-stream encoding (e.g. video + audio), incremental encoding, or
encoding multiple audio streams, use
[`Encoder`](../../generated/torchcodec.encoders.Encoder.html#torchcodec.encoders.Encoder) instead. See
[Encoding audio and video streams with the Encoder](multi_stream_encoding.html#sphx-glr-generated-examples-encoding-multi-stream-encoding-py) for
a tutorial.

Let's first generate some samples to be encoded. The data to be encoded could
also just come from an [`AudioDecoder`](../../generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder)!

```
import torch
from IPython.display import Audio as play_audio

def make_sinewave() -> tuple[torch.Tensor, int]:
 freq_A = 440 # Hz
 sample_rate = 16000 # Hz
 duration_seconds = 3 # seconds
 t = torch.linspace(0, duration_seconds, int(sample_rate * duration_seconds), dtype=torch.float32)
 return torch.sin(2 * torch.pi * freq_A * t), sample_rate

samples, sample_rate = make_sinewave()

print(f"Encoding samples with {samples.shape = } and {sample_rate = }")
play_audio(samples, rate=sample_rate)
```

```
Encoding samples with samples.shape = torch.Size([48000]) and sample_rate = 16000
```

 Your browser does not support the audio element.
 

We first instantiate an [`AudioEncoder`](../../generated/torchcodec.encoders.AudioEncoder.html#torchcodec.encoders.AudioEncoder). We pass it
the samples to be encoded. The samples must be a 2D tensors of shape
`(num_channels, num_samples)`, or in this case, a 1D tensor where
`num_channels` is assumed to be 1. The values must be float values
normalized in `[-1, 1]`: this is also what the
[`AudioDecoder`](../../generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder) would return.

Note

The `sample_rate` parameter corresponds to the sample rate of the
*input*, not the desired encoded sample rate.

```
from torchcodec.encoders import AudioEncoder

encoder = AudioEncoder(samples=samples, sample_rate=sample_rate)
```

[`AudioEncoder`](../../generated/torchcodec.encoders.AudioEncoder.html#torchcodec.encoders.AudioEncoder) supports encoding samples into a
file via the [`to_file()`](../../generated/torchcodec.encoders.AudioEncoder.html#torchcodec.encoders.AudioEncoder.to_file) method, or to
raw bytes via [`to_tensor()`](../../generated/torchcodec.encoders.AudioEncoder.html#torchcodec.encoders.AudioEncoder.to_tensor). For the
purpose of this tutorial we'll use
[`to_tensor()`](../../generated/torchcodec.encoders.AudioEncoder.html#torchcodec.encoders.AudioEncoder.to_tensor), so that we can easily
re-decode the encoded samples and check their properies. The
[`to_file()`](../../generated/torchcodec.encoders.AudioEncoder.html#torchcodec.encoders.AudioEncoder.to_file) method works very similarly.

```
encoded_samples = encoder.to_tensor(format="mp3")
print(f"{encoded_samples.shape = }, {encoded_samples.dtype = }")
```

```
encoded_samples.shape = torch.Size([9512]), encoded_samples.dtype = torch.uint8
```

That's it!

Now that we have our encoded data, we can decode it back, to make sure it
looks and sounds as expected:

```
from torchcodec.decoders import AudioDecoder

samples_back = AudioDecoder(encoded_samples).get_all_samples()

print(samples_back)
play_audio(samples_back.data, rate=samples_back.sample_rate)
```

```
AudioSamples:
 data (shape): torch.Size([1, 48000])
 pts_seconds: 0.0690625
 duration_seconds: 3.0
 sample_rate: 16000
```

 Your browser does not support the audio element.
 

The encoder supports some encoding options that allow you to change how to
data is encoded. For example, we can decide to encode our mono data (1
channel) into stereo data (2 channels), and to specify an output sample rate:

```
desired_sample_rate = 32000
encoded_samples = encoder.to_tensor(format="wav", num_channels=2, sample_rate=desired_sample_rate)

stereo_samples_back = AudioDecoder(encoded_samples).get_all_samples()

print(stereo_samples_back)
play_audio(stereo_samples_back.data, rate=desired_sample_rate)
```

```
AudioSamples:
 data (shape): torch.Size([2, 96000])
 pts_seconds: 0.0
 duration_seconds: 3.0
 sample_rate: 32000
```

 Your browser does not support the audio element.
 

Check the docstring of the encoding methods to learn about the different
encoding options.

**Total running time of the script:** (0 minutes 0.030 seconds)

[`Download Jupyter notebook: audio_encoding.ipynb`](../../_downloads/11ef1d93158a89ea05a303d1d7c2cc02/audio_encoding.ipynb)

[`Download Python source code: audio_encoding.py`](../../_downloads/ff115efad635d10ef95973df4f1ef183/audio_encoding.py)

[`Download zipped: audio_encoding.zip`](../../_downloads/83beac0daacb23697f2a04db723d6132/audio_encoding.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)