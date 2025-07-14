
API Ref
=======

.. _torchcodec:

torchcodec
----------

.. currentmodule:: torchcodec


.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: dataclass.rst

    Frame
    FrameBatch
    AudioSamples


.. _decoders:

torchcodec.decoders
-------------------

.. currentmodule:: torchcodec.decoders


For a video decoder tutorial, see: :ref:`sphx_glr_generated_examples_decoding_basic_example.py`.
For an audio decoder tutorial, see: :ref:`sphx_glr_generated_examples_decoding_audio_decoding.py`.


.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    VideoDecoder
    AudioDecoder


.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: dataclass.rst

    VideoStreamMetadata
    AudioStreamMetadata

.. _encoders:

torchcodec.encoders
-------------------

.. currentmodule:: torchcodec.encoders


For an audio decoder tutorial, see: :ref:`sphx_glr_generated_examples_encoding_audio_encoding.py`.


.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    AudioEncoder


.. _samplers:

torchcodec.samplers
-------------------


.. currentmodule:: torchcodec.samplers

For a tutorial, see: :ref:`sphx_glr_generated_examples_decoding_sampling.py`.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst

    clips_at_regular_indices
    clips_at_random_indices
    clips_at_regular_timestamps
    clips_at_random_timestamps
