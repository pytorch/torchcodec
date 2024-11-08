Installation Instructions
=========================

.. note::
    TorchCodec is only available on Linux for now. We plan to support other
    platforms in the future.

There are three steps to installing TorchCodec:

1. Install the latest stable version of PyTorch following the
   `official instructions <https://pytorch.org/get-started/locally/>`_. TorchCodec
   requires `PyTorch 2.4 <https://pytorch.org/docs/2.4/>`_.

2. Install FFmpeg, if it's not already installed. Your Linux distribution probably
   comes with FFmpeg pre-installed. TorchCodec supports all major FFmpeg versions
   in [4, 7]. If FFmpeg is not already installed, or you need a later version, install
   it with:

   .. code:: bash

      conda install ffmpeg
      # or
      conda install ffmpeg -c conda-forge
3. Install TorchCodec:

   .. code:: bash

      pip install torchcodec

Note that installation instructions may slightly change over time. The most
up-to-date instructions should be available from the `README
<https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec>`_.
