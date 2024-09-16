
Feature space
-------------

At a high level, a sampler returns a sequence of "clips" where a clip is a batch
of (non-necessarily contiguous) frames. Based on an analysis of the sampling
options of other libraries (torchvision, torchmultimodal, and various internal
libraries), we want to support the following use-cases:


**Clip location** (and number thereof)

- Get `num_clips_per_video` random clips: typically called "Random" sampler
- Get `num_clips_per_video` equally-spaced clips: "Uniform" sampler. Space is
  index-based.
- Get `num_clips_per_second` clips per seconds [within the [`start_seconds`,
  `stop_seconds`] sampling range]: "Periodic" sampler.

**Clip content**

- Clip must be `num_frames_per_clip` frames spaced by `dilation` frames.

Notes:

- There are 3 sampling strategies defining the clip locations: Random, Uniform
  and Periodic. The names are prone to confusion. "Random" is in fact uniformly
  random while "Uniform" isn't random. "Uniform" is periodic and "Periodic" is
  uniformly spaced too.
- The parametrization of the *Clip Content* dimension is limited and
  reduces to just `num_frames_per_clip` and the index-based `dilation`.
- The *Clip Content* dimension is never random. `num_frames_per_clip` is always
  passed by the user. This makes sense: training typically requires fixed-sized
  inputs so `num_frames_per_clip` is expected to be constant. The content of a
  clip (space between frames) is always regular, never random.


Proposal
--------

```py
from torchcodec.decoders import SimpleVideoDecoder
from torchcodec import samplers

decoder = SimpleVideoDecoder("./cool_vid")


def samplers.clips_at_random_timestamps(
    decoder,
    *,
    num_clips,
    num_frames_per_clip=1,
    seconds_between_frames=None,
) -> FrameBatch

def samplers.clips_at_regular_indices(
    decoder,
    *,
    num_clips,
    num_frames_per_clip=1,
    num_indices_between_frames=1,  # in indices (dilation)
) -> FrameBatch

def samplers.clips_at_regular_timestamps(
    decoder,
    *,
    num_clips_per_second,
    num_frames_per_clip=1,
    seconds_between_frames=None,
    clip_range_start_seconds: float = 0,
    clip_range_end_seconds: float = float("inf"),
) -> FrameBatch
```

Discussion / Notes
------------------

The sampler is agnostic to decoding options: the decoder object is passed to the
sampler by the user. This is because we don't want to duplicate the decoder
parametrization in the samplers, and also because it's not up to the sampler to
decide on some frame properties like their shape.

`clips_at_regular_indices` and `clips_at_regular_timestamps` are dual of
one-another, and have a slightly different parametrization: `num_clips` vs
`num_clips_per_second`. This is to respect the existing parametrization of the
surveyed libraries.

The samplers are explicitly time-based or index-based, from their function name.

We believe that in general, a time-based sampler is closer to the user's mental
model. It is also potentially more "correct" in the sense that it accounts for
variable framerates. This is why the "random" sampler is explicitly implemented
as a time-based sampler. Note that existing "random" samplers are implemented
as index-based, which is likely a result of the existing decoders not having
fine-grained pts-based APIs, rather than a deliberate design choice. We may
implement a "random" index-based sampler in the future if there are feature
requests.

The "dilation" parameter has different semantic depending on whether the sampler
is time-based or index-based. It is named either `seconds_between_frames` or
`num_indices_between_frames`. We don't allow `num_indices_between_frames` to be
used within a time-based sampler. The rationale is that if a user uses a
time-based API, it means they care about variable framerates for the clip start
locations, and we infer they also care about variable framerate for the clip
content. We may allow mix-matching in the future depending on user feedback,
either via additional functions, or via mutually exclusive parameters.

The term "clip" is being used to denote a sequence of frames in presentation
order. The frames aren't necessarily consecutive. We should add this term to the
glossary. I don't feel too strongly about using the name "clip", but it's used
in all existing libraries, and I don't know of an alternative that would be
universally understood.

Eventually we'll want to implement a "fast and approximate" version of the
random sampler as some users are happy to sacrifice entropy for speed.
Basically, prioritize key-frames in a smart way. This isn't part of this
proposal to limit the scope of discussions. Eventually, this sampler will likely
be exposed as a parameter to the exi sting random sampler(s), or as a new
function with the same parametriation as the random sampler.

The output of the samplers is a sequence of clips where a clips is a batch of
frames. Such output can be represented as:
  - a `List[FrameBatch]` where the `FrameBatch.data` tensor is 4D
  - a `FrameBatch` where the `FrameBatch.data` tensor is 5D. This is the current
    proposal. This can be done as long as `num_frames_per_clip` is not random
    (it is currently never random).
