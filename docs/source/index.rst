Welcome to the TorchCodec documentation!
========================================

TorchCodec is a Python library for decoding video and audio data into PyTorch
tensors, on CPU and CUDA GPU. It aims to be fast, easy to use, and well
integrated into the PyTorch ecosystem. If you want to use PyTorch to train ML
models on videos and audio, TorchCodec is how you turn these into data.

We achieve these capabilities through:

* Pythonic APIs that mirror Python and PyTorch conventions.
* Relying on `FFmpeg <https://www.ffmpeg.org/>`_ to do the decoding. TorchCodec
  uses the version of FFmpeg you already have installed. FMPEG is a mature
  library with broad coverage available on most systems. It is, however, not
  easy to use. TorchCodec abstracts FFmpeg's complexity to ensure it is used
  correctly and efficiently.
* Returning data as PyTorch tensors, ready to be fed into PyTorch transforms
  or used directly to train models.

.. grid:: 3

     .. grid-item-card:: :octicon:`file-code;1em`
        Installation instructions
        :img-top: _static/img/card-background.svg
        :link: https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec
        :link-type: url

        How to install TorchCodec

     .. grid-item-card:: :octicon:`file-code;1em`
        Getting Started with TorchCodec
        :img-top: _static/img/card-background.svg
        :link: generated_examples/basic_example.html
        :link-type: url

        A simple video decoding example

     .. grid-item-card:: :octicon:`file-code;1em`
        Audio Decoding
        :img-top: _static/img/card-background.svg
        :link: generated_examples/audio_decoding.html
        :link-type: url

        A simple audio decoding example

     .. grid-item-card:: :octicon:`file-code;1em`
        GPU decoding
        :img-top: _static/img/card-background.svg
        :link: generated_examples/basic_cuda_example.html
        :link-type: url

        A simple example demonstrating CUDA GPU decoding

     .. grid-item-card:: :octicon:`file-code;1em`
        Streaming video
        :img-top: _static/img/card-background.svg
        :link: generated_examples/file_like.html
        :link-type: url

        How to efficiently decode videos from the cloud

     .. grid-item-card:: :octicon:`file-code;1em`
        Clip sampling
        :img-top: _static/img/card-background.svg
        :link: generated_examples/sampling.html
        :link-type: url

        How to sample regular and random clips from a video

.. note::

   TorchCodec is still in development stage and we are actively seeking
   feedback. If you have any suggestions or issues, please let us know by
   `opening an issue <https://github.com/pytorch/torchcodec/issues/new/choose>`_
   on our `GitHub repository <https://github.com/pytorch/torchcodec/>`_.

.. toctree::
   :maxdepth: 1
   :caption: TorchCodec documentation
   :hidden:

   Home <self>
   glossary

.. toctree::
   :maxdepth: 1
   :caption: Examples and tutorials
   :hidden:

   Installation instructions <https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec>
   generated_examples/index


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   api_ref_torchcodec
   api_ref_decoders
   api_ref_samplers
