**DISCLAIMER** This RFC is currently lacking (at least):

- a "time-based" equivalent of dilation
- a "time-based" equivalent of random uniform sampling.

Main design principles
----------------------

- The existing feature space of different libraries (torchvision, torchmm,
  internal stuff) is supported. In particular "Random", "Uniform" and
  "Periodic" samplers are supported, with hopefully more descriptive names.
- Sampler is agnostic to decoding options: the decoder object is passed to the
  sampler by the user.
- Explicit **non goal**: to support arbitrary strategy that we haven't
  observed or heard usage of. For example, in all existing libraries the clip
  content is never random and just determined by 2 parameters:
  `num_frames_per_clip` and `step_between_frames` (dilation). We allow for
  future extensibility, but we're not explicitly enabling additional
  clip-content strategies.
- The term "clip" is being used to denote a sequence of frames in presentation
  order. The frames aren't necessarily consecutive (when
  `step_between_frames > 1`). We should add this term to the glossary. I don't
  feel too strongly about using the name "clip", but it's used in all existing
  libraries, and I don't know of an alternative that would be universally
  understood.


Option 1
--------

- One function per sampling strategy
- Note: this is not 100% stateless, the decoder object is seeked so there are
  side effects.

```py
from torchcodec.decoders import SimpleVideoDecoder
from torchcodec import samplers

video = "cool_vid"
decoder = SimpleVideoDecoder(video)

clips = samplers.get_uniformly_random_clips(
    decoder,
    num_clips=12,
    # clip content params:
    num_frames_per_clip=4,
    step_between_frames=2,  # often called "dilation"
    # sampler-specific params:
    prioritize_keyframes=False,  # to implement "fast" sampling.
                                 # Might need to be a separate function.
)

clips = samplers.get_evenly_spaced_clips(  # often called "Uniform"
    decoder,
    num_clips=12,
    # clip content params:
    num_frames_per_clip=4,
    step_between_frames=2,
)

clips = samplers.get_evenly_timed_clips(  # often called "Periodic"
    decoder,
    num_clips_per_second=3,
    # clip content params:
    num_frames_per_clip=4,
    step_between_frames=2,
    # sampler-specific params:
    clip_range_start_seconds=3,
    clip_range_end_seconds=4,
)
```

Option 2
--------

- One ClipSampler object where each sampling strategy is a method.
- Bonus: the underlying `decoder` object can be re-initilized by the sampler
  in-between calls, possibly improving seek time??

```py
from torchcodec.sampler import ClipSampler

sampler = ClipSampler(
    # Here: parameters that apply to all sampling strategies
    decoder,
    num_frames_per_clip=4,
    step_between_frames=2,
)
# One method per sampling strat
sampler.get_uniformly_random_clips(num_clips=12, prioritize_keyframes=True)
sampler.get_evenly_spaced_clips(num_clips=12)
sampler.get_evenly_timed_clips(
    num_clips_per_second=3,
    clip_range_start_seconds=3,
    cip_range_end_seconds=4
)
```


Questions
---------

- should the returned `clips` be...?
  - a List[FrameBatch] where the FrameBatch.data is 4D
  - a FrameBatch where the FrameBatch.data is 5D
    - This is OK as long as num_frames_per_clip is not random (it is currently
      never random)
