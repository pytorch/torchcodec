# Decoding audio streams with AudioDecoder

In this example, we'll learn how to decode an audio file using the
[`AudioDecoder`](../../generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder) class. If you're decoding WAV files,
also check out the WavDecoder for WAV files section below.

First, a bit of boilerplate: we'll download an audio file from the web and
define an audio playing utility. You can ignore that part and jump right
below to Creating a decoder.

```
import requests
from IPython.display import Audio

def play_5s(samples):
 # Play 5 seconds of the audio. Playing the entire file would take too much
 # space in our docs (~40Mb!).
 return Audio(samples.data[:, :5 * samples.sample_rate], rate=samples.sample_rate)

# Audio source is CC0: https://opengameart.org/content/town-theme-rpg
# Attribution: cynicmusic.com pixelsphere.org
url = "https://opengameart.org/sites/default/files/TownTheme.mp3"
response = requests.get(url, headers={"User-Agent": ""})
if response.status_code != 200:
 raise RuntimeError(f"Failed to download video. {response.status_code = }.")

raw_audio_bytes = response.content
```

## Creating a decoder

We can now create a decoder from the raw (encoded) audio bytes. You can of
course use a local audio file and pass the path as input. You can also decode
audio streams from videos!

```
from torchcodec.decoders import AudioDecoder

decoder = AudioDecoder(raw_audio_bytes)
```

The has not yet been decoded by the decoder, but we already have access to
some metadata via the `metadata` attribute which is an
[`AudioStreamMetadata`](../../generated/torchcodec.decoders.AudioStreamMetadata.html#torchcodec.decoders.AudioStreamMetadata) object.

```
print(decoder.metadata)
```

```
AudioStreamMetadata:
 duration_seconds_from_header: 97.48897959183674
 begin_stream_seconds_from_header: 0.02505668934240363
 bit_rate: 108039
 codec: mp3
 stream_index: 0
 sample_rate: 44100
 num_channels: 2
 sample_format: fltp
 duration_seconds: 97.48897959183674
 begin_stream_seconds: 0.02505668934240363
```

## Decoding samples

To get decoded samples, we just need to call the
[`get_all_samples()`](../../generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder.get_all_samples) method,
which returns an [`AudioSamples`](../../generated/torchcodec.AudioSamples.html#torchcodec.AudioSamples) object:

```
samples = decoder.get_all_samples()

print(samples)
play_5s(samples)
```

```
AudioSamples:
 data (shape): torch.Size([2, 4297722])
 pts_seconds: 0.02505668934240363
 duration_seconds: 97.45401360544217
 sample_rate: 44100
```

 Your browser does not support the audio element.
 

The `.data` field is a tensor of shape `(num_channels, num_samples)` and
of float dtype with values in [-1, 1].

The `.pts_seconds` field indicates the starting time of the output samples.
Here it's 0.025 seconds, even though we asked for samples starting from 0. Not
all streams start exactly at 0! This is not a bug in TorchCodec, this is a
property of the file that was defined when it was encoded.

## Specifying a range

If we don't need all the samples, we can use
[`get_samples_played_in_range()`](../../generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder.get_samples_played_in_range) to
decode the samples within a custom range:

```
samples = decoder.get_samples_played_in_range(start_seconds=10, stop_seconds=70)

print(samples)
play_5s(samples)
```

```
AudioSamples:
 data (shape): torch.Size([2, 2646000])
 pts_seconds: 10.0
 duration_seconds: 60.0
 sample_rate: 44100
```

 Your browser does not support the audio element.
 

## Custom sample rate

We can also decode the samples into a desired sample rate using the
`sample_rate` parameter of [`AudioDecoder`](../../generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder). The
ouput will sound similar, but note that the number of samples greatly
decreased:

```
decoder = AudioDecoder(raw_audio_bytes, sample_rate=16_000)
samples = decoder.get_all_samples()

print(samples)
play_5s(samples)
```

```
AudioSamples:
 data (shape): torch.Size([2, 1559264])
 pts_seconds: 0.02505668934240363
 duration_seconds: 97.454
 sample_rate: 16000
```

 Your browser does not support the audio element.
 

## WavDecoder for WAV files

If your audio source is a WAV file and you don't need resampling or channel
remixing, you can use [`WavDecoder`](../../generated/torchcodec.decoders.WavDecoder.html#torchcodec.decoders.WavDecoder) for
significantly faster decoding. It has the same
[`get_all_samples()`](../../generated/torchcodec.decoders.WavDecoder.html#torchcodec.decoders.WavDecoder.get_all_samples) and
[`get_samples_played_in_range()`](../../generated/torchcodec.decoders.WavDecoder.html#torchcodec.decoders.WavDecoder.get_samples_played_in_range) methods
as [`AudioDecoder`](../../generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder). See
[TorchCodec Performance Tips and Best Practices](performance_tips.html#sphx-glr-generated-examples-decoding-performance-tips-py) for more
details.

**Total running time of the script:** (0 minutes 0.957 seconds)

[`Download Jupyter notebook: audio_decoding.ipynb`](../../_downloads/f6b4925719fd3f116259f89a204c6888/audio_decoding.ipynb)

[`Download Python source code: audio_decoding.py`](../../_downloads/32a6aeee5f4f47b42a94840af3078059/audio_decoding.py)

[`Download zipped: audio_decoding.zip`](../../_downloads/7375647204934d2bdc05a60eb1fa134f/audio_decoding.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)