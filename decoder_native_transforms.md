# Decoder Native Transforms

## API
We want to support this user-facing API:

 ```python
    decoder = VideoDecoder(
        "vid.mp4",
        transforms=[
            torchcodec.transforms.FPS(
                fps=30,
            ),
            torchvision.transforms.v2.Resize(
                width=640,
                height=480,
            ),
            torchvision.transforms.v2.RandomCrop(
                width=32,
                height=32,
            ),
        ]
    )
```

What the user is asking for, in English:

1. I want to decode frames from the file `"vid.mp4".`
2. For each decoded frame, I want each frame to pass through the following transforms:
   1. Add or remove frames as necessary to ensure a constant 30 frames per second.
   2. Resize the frame to 640x480. Use the algorithm that is TorchVision's default.
   3. Inside the resized frame, crop the image to 32x32. The x and y coordinates are
      chosen randomly upon the creation of the Python `VideoDecoder` object. All decoded
      frames use the same values for x and y.

## Design Considerations
These three transforms are instructive, as they force us to consider:

1. How "easy" TorchVision transforms will be handled, where all values are
   static. `Resize` is such an example.
2. Transforms that involve randomness. The main question we need to resolve
   is when the random value is resolved. I think this comes down to: once
   upon Python `VideoDecoder` creation, or different for each frame decoded?
   I made the call above that it should be once upon Python `VideoDecoder`
   creation, but we need to make sure that lines up with what we think
   users will want.
3. Transforms that are supported by FFmpeg but not supported by
   TorchVision. In particular, FPS is something that multiple users have
   asked for.

## Implementation Sketch
First let's consider implementing the "easy" case of `Resize`.

1. We add an optional `transforms` parameter to the initialization of
   `VideoDecoder`. It is a sequence of TorchVision Transforms.
2. During `VideoDecoder` object creation, we walk the list, capturing two
   pieces of information:
   1. The transform name that the C++ layer will understand. (We will
      have to decide if we want to just use the FFmpeg filter name
      here, the fully resolved Transform name, or introduce a new
      naming layer.)
   2. The parameters in a format that the C++ layer will understand. We
      obtain them by calling `make_params()` on the Transform object.
3. We add an optional transforms parameter to `core.add_video_stream()`. This
   parameter will be a vector, but whether the vector contains strings,
   tensors, or some combination of them is TBD.
4. The `custom_ops.cpp` and `pybind_ops.cpp` layer is responsible for turning
   the values passed from the Python layer into transform objects that the
   C++ layer knows about. We will have one class per transform we support.
   Each class will have:
   1. A name which matches the FFmpeg filter name.
   2. One member for each supported parameter.
   3. A virtual member function that knows how to produce a string that
      can be passed to FFmpeg's filtergraph.
5. We add a vector of such transforms to
   `SingleStreamDecoder::addVideoStream`. We store the vector as a field in
   `SingleStreamDecoder`.
6. We need to reconcile `FilterGraph`, `FiltersContext` and this vector of
   transforms. They are all related, but it's not clear to me what the
   exact relationship should be.
7. The actual string we pass to FFmepg's filtergraph comes from calling
   the virtual member function on each transform object.

For the transforms that do not exist in TorchVision, we can build on the above:

1. We define a new module, `torchcodec.decoders.transforms`.
2. All transforms we define in there inherit from
   `torchvision.transforms.v2.Transform`.
3. We implement the mimimum needed to hook the new transforms into the
   machinery defined above.

## Open questions:

1. Is `torchcodec.transforms` the right namespace?
2. For random transforms, when should the value be fixed?
3. Transforms such as Resize don't actually implement a `make_params()`
   method. How does TorchVision get their parameters? How will TorchCodec?
4. Should the name at the bridge layer between Python and C++ just be the FFmpeg filter name?
5. How do we communicate the transformation names and parameters to the C++
   layer? We need to support transforms with an arbitrary number of parameters.
6. How does this generalize to `AudioDecoder`? Ideally we would be able to
   support TorchAudio's transforms in a similar way.
7. What is the relationship between the C++ transform objects, `FilterGraph`
   and `FiltersContext`?
