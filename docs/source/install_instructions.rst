Installation Instructions
=========================

.. note::
    TorchCodec is only available on Linux for now. We plan to support other
    platforms in the future.

Installing torchcodec should be as simple as:

.. code:: bash

    pip install torchcodec

You will need a working PyTorch installation, which you can install following
the `official instructions <https://pytorch.org/get-started/locally/>`_.

You will also need FFmpeg installed on your system, and TorchCodec decoding
capabilities are determined by your underlying FFmpeg installation. There are
different options to install FFmpeg e.g.:

.. code:: bash

    conda install ffmpeg
    # or
    conda install ffmpeg -c conda-forge

Your Linux distribution probably comes with FFmpeg pre-installed as well.
TorchCodec supports all major FFmpeg version in [4, 7].

.. .. TODO_UPDATE_LINK add link
.. For more advanced installation instructions and details, please refer to the guidelines in our GitHub repo.
