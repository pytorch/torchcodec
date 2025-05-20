Audio encoding design
=====================

Let's talk about the design of our audio encoding capabilities.  This design doc
is not meant to be merged into the repo. I'm creating a PR to start a discussion
and enable comments on the design proposal. The PR will eventually be closed
without merging.


Feature space and requirements
------------------------------

When users give us the samples to be encoded, they have to provide:

- the FLTP tensor of decoded samples (akin to what we decode)
- the sample rate of the samples. That's needed for FFmpeg to know when each
  sample should be played, and it cannot be inferred.

Those are naturally supplied as 2 separate parameters (1 for the tensor, 1 for
the sample rate), but if our APIs also allowed users to pass a single
[AudioSamples](https://docs.pytorch.org/torchcodec/stable/generated/torchcodec.AudioSamples.html#torchcodec.AudioSamples)
object as parameter, that could be a good UX.

We want to enable users to encode these samples:

- to a file, like "output.mp3". When encoding to a file, we automatically infer
  the format (mp3) from the filename.
- to a file-like (NYI, will come eventually). When encoding to a file-like, we
  can't infer the format, so users have to specify it to us.
- to a tensor. Same here, users have to specify the output format.

We want to allow users to specify additional encoding options:

- The encoded bit rate, for compressed formats like mp3
- The number of channels, to automatically encode to audio or to stereo,
  potentially different from that of the input (NYI).
- The encoded sample rate, to automatically encode into a given sample rate,
  potentially different from that of the input (NYI).
- (Maybe) other parameters, like codec-specific stuff.

API proposal
------------


### Option 1

A natural option is to create 3 separate stateless functions: one for each kind
of output we want to support.

```py
def encode_audio_to_file(
    samples: torch.Tensor,
    sample_rate: int,
    filename: Union[str, Path],
    bit_rate: Optional[int] = None,
    num_channels: Optional[int] = None,
    output_sample_rate: Optional[int] = None,
) -> None:
    pass


def encode_audio_to_file_like(
    samples: torch.Tensor,
    sample_rate: int,
    file_like: object,
    format: str,
    bit_rate: Optional[int] = None,
    num_channels: Optional[int] = None,
    output_sample_rate: Optional[int] = None,
) -> None:
    pass


def encode_audio_to_tensor(
    samples: torch.Tensor,
    sample_rate: int,
    format: str,
    bit_rate: Optional[int] = None,
    num_channels: Optional[int] = None,
    output_sample_rate: Optional[int] = None,
) -> torch.Tensor:
    pass
```

A few notes:

- I'm not attempting to define where the keyword-only parameters should start,
  that can be discussed later or on the PRs.
- Both `to_file_like` and `to_tensor` need an extra `format` parameter, because
  it cannot be inferred. In `to_file`, it is inferred from `filename`.
- To avoid collision between the input sample rate and the optional desired
  output sample rate, we have to use `output_sample_rate`. That's a bit meh.
  Technically, all of `format`, `bit_rate` and `num_channel` could also qualify
  for the `output_` prefix, but that would be very heavy.

### Option 2

Another option is to expose each of these functions as methods on a stateless
object.

```py
class AudioEncoder:
    def __init__(
        self,
        samples: torch.Tensor,
        sample_rate: int,
    ):
        pass

    def to_file(
        self,
        filename: Union[str, Path],
        bit_rate: Optional[int] = None,
        num_channels: Optional[int] = None,
        sample_rate: Optional[int] = None,
    ) -> None:
        pass

    def to_file_like(
        self,
        file_like: object,
        bit_rate: Optional[int] = None,
        num_channels: Optional[int] = None,
        sample_rate: Optional[int] = None,
    ) -> None:
        pass

    def to_tensor(
        format: str,
        bit_rate: Optional[int] = None,
        num_channels: Optional[int] = None,
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        pass
```

Usually, we like to expose objects (instead of stateless functions) when there
is a clear state to be managed. That's not the case here: the `AudioEncoder` is
largely stateless.
Instead, we can justify exposing an object by noting that it allows us to
cleanly separate unrelated blocks of parameters:
- the parameters relating to the **input** are in `__init__()`
- the parameters relating to the **output** are in the `to_*` methods.

A nice consequence of that is that we do not have a collision between the 2
`sample_rate` parameters anymore, and their respective purpose is clear from the
methods they are exposed in.

### Option 2.1

A natural extension of option 2 is to allow users to pass an `AudioSample`
object to `__init__()`, instead of passing 2 separate parameters:

```py
samples = # ... AudioSamples e.g. coming from the decoder
AudioEncoder(samples).to_file("output.wav")
```

This can be enabled via this kind of logic:

```py
class AudioEncoder:
    def __init__(
        self,
        samples: Union[torch.Tensor, AudioSamples],
        sample_rate: Optional[int] = None,
    ):
        assert (
            isinstance(samples, torch.Tensor) and sample_rate is not None) or (
            isinstance(sample, AudioSamples) and sample_rate is None
        )
```


### Thinking ahead

I don't want to be prescriptive on what the video decoder should look like, but I
suspect that we will soon need to expose a **multistream** encoder, i.e. an
encoder that can encode both an audio and a video stream at the same time (think
of video generation models). I suspect the API of such encoder will look
something like this (a bit similar to what TorchAudio exposes):

```py
Encoder().add_audio(...).add_video(...).to_file(filename)
Encoder().add_audio(...).add_video(...).to_file_like(filelike)
encoded_bytes = Encoder().add_audio(...).add_video(...).to_tensor()
```

This too will involve exposing an object, despite the actual managed "state"
being very limited.
