# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod
from collections.abc import Sequence
from types import ModuleType

import torch
from torch import nn


class DecoderTransform(ABC):
    """Base class for all decoder transforms.

    A *decoder transform* is a transform that is applied by the decoder before
    returning the decoded frame.  Applying decoder transforms to frames
    should be both faster and more memory efficient than receiving normally
    decoded frames and applying the same kind of transform.

    Most ``DecoderTransform`` objects have a complementary transform in TorchVision,
    specificially in `torchvision.transforms.v2 <https://docs.pytorch.org/vision/stable/transforms.html>`_.
    For such transforms, we ensure that:

      1. The names are the same.
      2. Default behaviors are the same.
      3. The parameters for the ``DecoderTransform`` object are a subset of the
         TorchVision :class:`~torchvision.transforms.v2.Transform` object.
      4. Parameters with the same name control the same behavior and accept a
         subset of the same types.
      5. The difference between the frames returned by a decoder transform and
         the complementary TorchVision transform are such that a model should
         not be able to tell the difference.
    """

    @abstractmethod
    def _make_transform_spec(self, input_dims: tuple[int | None, int | None]) -> str:
        """Makes the transform spec that is used by the `VideoDecoder`.

        Args:
            input_dims (tuple[int | None, int | None]): The dimensions of
                the input frame in the form (height, width). We cannot know the
                dimensions at object construction time because it's dependent on
                the video being decoded and upstream transforms in the same
                transform pipeline. Not all transforms need to know this; those
                that don't will ignore it. The individual values in the tuple are
                optional because the original values come from file metadata which
                may be missing. We maintain the optionality throughout the APIs so
                that we can decide as late as possible that it's necessary for the
                values to exist. That is, if the values are missing from the
                metadata and we have transforms which ignore the input dimensions,
                we want that to still work.

                Note: This method is the moral equivalent of TorchVision's
                `Transform.make_params()`.

        Returns:
            str: A string which contains the spec for the transform that the
                `VideoDecoder` knows what to do with.
        """
        pass

    def _get_output_dims(
        self, input_dims: tuple[int | None, int | None]
    ) -> tuple[int | None, int | None] | None:
        """Get the dimensions of the output frame.

        Transforms that change the frame dimensions need to override this
        method. Transforms that don't change the frame dimensions can rely on
        this default implementation.

        Args:
            input_dims (tuple[int | None, int | None]): The dimensions of
                the input frame in the form (height, width).

        Returns:
            tuple[int | None, int | None] | None: The output dimensions.
                - None: The output dimensions are the same as the input dimensions.
                - (int, int): The (height, width) of the output frame.
        """
        return None


def import_torchvision_transforms_v2() -> ModuleType:
    try:
        from torchvision.transforms import v2
    except ImportError as e:
        raise RuntimeError(
            "Cannot import TorchVision; this should never happen, please report a bug."
        ) from e
    return v2


class Resize(DecoderTransform):
    """Resize the decoded frame to a given size.

    Complementary TorchVision transform: :class:`~torchvision.transforms.v2.Resize`.
    Interpolation is always bilinear. Anti-aliasing is always on.

    Args:
        size (int, Sequence[int], or None): Desired output size. If ``size`` is
            a sequence of the form (height, width), the output will have that
            exact size. If it is an int or a one-element sequence, the shorter
            edge will be matched to that value while preserving the aspect
            ratio. If it is None, the output size is determined by
            ``max_size``.
        max_size (int, optional): Maximum size of the longer edge. This is only
            supported when ``size`` is an int, a one-element sequence, or None.
            When ``size`` is None, the longer edge is matched to ``max_size``
            exactly, which may upscale the input.
    """

    def __init__(
        self,
        size: int | Sequence[int] | None,
        *,
        max_size: int | None = None,
    ):
        if not (
            isinstance(size, int)
            or size is None
            or (isinstance(size, Sequence) and len(size) in (1, 2))
        ):
            raise ValueError(
                "Resize size can be an integer, a sequence of one or two "
                f"integers, or None, but got {size}."
            )

        if size is None and not isinstance(max_size, int):
            raise ValueError(
                "max_size must be an integer when size is None, "
                f"but got {max_size} instead."
            )

        if max_size is not None and not isinstance(max_size, int):
            raise ValueError(
                f"max_size must be an integer or None, but got {max_size} instead."
            )

        if max_size is not None and isinstance(size, Sequence) and len(size) == 2:
            raise ValueError(
                "max_size should only be passed if size is None or specifies "
                "the length of the smaller edge."
            )

        self.size = size
        self.max_size = max_size

    def _make_transform_spec(self, input_dims: tuple[int | None, int | None]) -> str:
        output_height, output_width = self._get_output_dims(input_dims)
        return f"resize, {output_height}, {output_width}"

    def _get_output_dims(
        self, input_dims: tuple[int | None, int | None]
    ) -> tuple[int, int]:
        if isinstance(self.size, Sequence) and len(self.size) == 2:
            return (self.size[0], self.size[1])

        height, width = input_dims
        if height is None or width is None:
            raise ValueError(
                "Resize with an aspect-ratio-preserving size can only be used "
                "when input frame dimensions are known."
            )

        short, long = (width, height) if width <= height else (height, width)
        if self.size is None:
            assert self.max_size is not None
            new_short = int(self.max_size * short / long)
            new_long = self.max_size
        else:
            requested_new_short = (
                self.size if isinstance(self.size, int) else self.size[0]
            )
            new_short = requested_new_short
            new_long = int(requested_new_short * long / short)

            if self.max_size is not None:
                if self.max_size <= requested_new_short:
                    raise ValueError(
                        f"max_size = {self.max_size} must be strictly greater "
                        "than the requested size for the smaller edge "
                        f"size = {self.size}."
                    )
                if new_long > self.max_size:
                    new_short = int(self.max_size * new_short / new_long)
                    new_long = self.max_size

        new_width, new_height = (
            (new_short, new_long) if width <= height else (new_long, new_short)
        )
        return (new_height, new_width)

    @classmethod
    def _from_torchvision(cls, tv_resize: nn.Module):
        v2 = import_torchvision_transforms_v2()

        assert isinstance(tv_resize, v2.Resize)

        if tv_resize.interpolation not in (v2.InterpolationMode.BILINEAR, "bilinear"):
            raise ValueError(
                "TorchVision Resize transform must use bilinear interpolation."
            )
        if tv_resize.antialias is False:
            raise ValueError(
                "TorchVision Resize transform must have antialias enabled."
            )
        return cls(size=tv_resize.size, max_size=tv_resize.max_size)


class CenterCrop(DecoderTransform):
    """Crop the decoded frame to a given size in the center of the frame.

    Complementary TorchVision transform: :class:`~torchvision.transforms.v2.CenterCrop`.

    Args:
        size (Sequence[int]): Desired output size. Must be a sequence of
            the form (height, width).
    """

    def __init__(self, size: Sequence[int]):
        if len(size) != 2:
            raise ValueError(
                "CenterCrop transform must have a (height, width) "
                f"pair for the size, got {size}."
            )
        self.size = size

    def _make_transform_spec(self, input_dims: tuple[int | None, int | None]) -> str:
        return f"center_crop, {self.size[0]}, {self.size[1]}"

    def _get_output_dims(
        self, input_dims: tuple[int | None, int | None]
    ) -> tuple[int | None, int | None] | None:
        return (self.size[0], self.size[1])

    @classmethod
    def _from_torchvision(
        cls,
        tv_center_crop: nn.Module,
    ):
        v2 = import_torchvision_transforms_v2()

        if not isinstance(tv_center_crop, v2.CenterCrop):
            raise ValueError(
                "Transform must be TorchVision's CenterCrop, "
                f"it is instead {type(tv_center_crop).__name__}. "
                "This should never happen, please report a bug."
            )

        if len(tv_center_crop.size) != 2:
            raise ValueError(
                "TorchVision CenterCrop transform must have a (height, width) "
                f"pair for the size, got {tv_center_crop.size}."
            )

        return cls(size=tv_center_crop.size)


class RandomCrop(DecoderTransform):
    """Crop the decoded frame to a given size at a random location in the frame.

    Complementary TorchVision transform: :class:`~torchvision.transforms.v2.RandomCrop`.
    Padding of all kinds is disabled. The random location within the frame is
    determined during the initialization of the
    :class:`~torchcodec.decoders.VideoDecoder` object that owns this transform.
    As a consequence, each decoded frame in the video will be cropped at the
    same location. Videos with variable resolution may result in undefined
    behavior.

    Args:
        size (Sequence[int]): Desired output size. Must be a sequence of
            the form (height, width).
    """

    def __init__(self, size: Sequence[int]):
        if len(size) != 2:
            raise ValueError(
                "RandomCrop transform must have a (height, width) "
                f"pair for the size, got {size}."
            )
        self.size = size

    def _make_transform_spec(self, input_dims: tuple[int | None, int | None]) -> str:
        height, width = input_dims
        if height is None:
            raise ValueError(
                "Video metadata has no height. "
                "RandomCrop can only be used when input frame dimensions are known."
            )
        if width is None:
            raise ValueError(
                "Video metadata has no width. "
                "RandomCrop can only be used when input frame dimensions are known."
            )

        # Note: This logic below must match the logic in
        #       torchvision.transforms.v2.RandomCrop.make_params(). Given
        #       the same seed, they should get the same result. This is an
        #       API guarantee with our users.
        if height < self.size[0] or width < self.size[1]:
            raise ValueError(
                f"Input dimensions {input_dims} are smaller than the crop size {self.size}."
            )

        top = int(torch.randint(0, height - self.size[0] + 1, size=()).item())
        left = int(torch.randint(0, width - self.size[1] + 1, size=()).item())

        return f"crop, {self.size[0]}, {self.size[1]}, {left}, {top}"

    def _get_output_dims(
        self, input_dims: tuple[int | None, int | None]
    ) -> tuple[int | None, int | None] | None:
        return (self.size[0], self.size[1])

    @classmethod
    def _from_torchvision(
        cls,
        tv_random_crop: nn.Module,
    ):
        v2 = import_torchvision_transforms_v2()

        if not isinstance(tv_random_crop, v2.RandomCrop):
            raise ValueError(
                "Transform must be TorchVision's RandomCrop, "
                f"it is instead {type(tv_random_crop).__name__}. "
                "This should never happen, please report a bug."
            )

        if tv_random_crop.padding is not None:
            raise ValueError(
                "TorchVision RandomCrop transform must not specify padding."
            )

        if tv_random_crop.pad_if_needed is True:
            raise ValueError(
                "TorchVision RandomCrop transform must not specify pad_if_needed."
            )

        if tv_random_crop.fill != 0:
            raise ValueError("TorchVision RandomCrop fill must be 0.")

        if tv_random_crop.padding_mode != "constant":
            raise ValueError("TorchVision RandomCrop padding_mode must be constant.")

        if len(tv_random_crop.size) != 2:
            raise ValueError(
                "TorchVision RandcomCrop transform must have a (height, width) "
                f"pair for the size, got {tv_random_crop.size}."
            )

        return cls(size=tv_random_crop.size)


def _make_transform_specs(
    transforms: Sequence[DecoderTransform | nn.Module] | None,
    input_dims: tuple[int | None, int | None],
) -> str:
    """Given a sequence of transforms, turn those into the specification string
       the core API expects.

    Args:
        transforms: Optional sequence of transform objects. The objects can be
            one of two types:
                1. torchcodec.transforms.DecoderTransform
                2. torchvision.transforms.v2.Transform, but our type annotation
                   only mentions its base, nn.Module. We don't want to take a
                   hard dependency on TorchVision.
        input_dims: Optional (height, width) pair. Note that only some
            transforms need to know the dimensions. If the user provides
            transforms that don't need to know the dimensions, and that metadata
            is missing, everything should still work. That means we assert their
            existence as late as possible.

    Returns:
        String of transforms in the format the core API expects: transform
        specifications separate by semicolons.
    """
    if transforms is None:
        return ""

    try:
        from torchvision.transforms import v2

        tv_available = True
    except ImportError:
        tv_available = False

    # The following loop accomplishes two tasks:
    #
    #     1. Converts the transform to a DecoderTransform, if necessary. We
    #        accept TorchVision transform objects and they must be converted
    #        to their matching DecoderTransform.
    #     2. Calculates what the input dimensions are to each transform.
    #
    # The order in our transforms list is semantically meaningful, as we
    # actually have a pipeline where the output of one transform is the input to
    # the next. For example, if we have the transforms list [A, B, C, D], then
    # we should understand that as:
    #
    #     A -> B -> C -> D
    #
    # Where the frame produced by A is the input to B, the frame produced by B
    # is the input to C, etc. This particularly matters for frame dimensions.
    # Transforms can both:
    #
    #     1. Produce frames with arbitrary dimensions.
    #     2. Rely on their input frame's dimensions to calculate ahead-of-time
    #        what their runtime behavior will be.
    #
    # The consequence of the above facts is that we need to statically track
    # frame dimensions in the pipeline while we pre-process it. The input
    # frame's dimensions to A, our first transform, is always what we know from
    # our metadata. For each transform, we always calculate its output
    # dimensions from its input dimensions. We store these with the converted
    # transform, to be all used together when we generate the specs.
    converted_transforms: list[
        tuple[
            DecoderTransform,
            # A (height, width) pair where the values may be missing.
            tuple[int | None, int | None],
        ]
    ] = []
    curr_input_dims = input_dims
    for transform in transforms:
        if not isinstance(transform, DecoderTransform):
            if not tv_available:
                raise ValueError(
                    f"The supplied transform, {transform}, is not a TorchCodec "
                    " DecoderTransform. TorchCodec also accepts TorchVision "
                    "v2 transforms, but TorchVision is not installed."
                )
            elif isinstance(transform, v2.Resize):
                transform = Resize._from_torchvision(transform)
            elif isinstance(transform, v2.CenterCrop):
                transform = CenterCrop._from_torchvision(transform)
            elif isinstance(transform, v2.RandomCrop):
                transform = RandomCrop._from_torchvision(transform)
            else:
                raise ValueError(
                    f"Unsupported transform: {transform}. Transforms must be "
                    "either a TorchCodec DecoderTransform or a TorchVision "
                    "v2 transform."
                )

        converted_transforms.append((transform, curr_input_dims))
        output_dims = transform._get_output_dims(curr_input_dims)
        curr_input_dims = output_dims if output_dims is not None else curr_input_dims

    return ";".join([t._make_transform_spec(dims) for t, dims in converted_transforms])
