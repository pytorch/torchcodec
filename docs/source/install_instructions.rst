Installation Instructions
=========================

.. note::
    TorchCodec is only available on Linux for now. We plan to support other
    platforms in the future.

Installing torchcodec should be as simple as:

.. code:: bash

    pip install torchcodec


You will also need FFmpeg installed on your system, and TorchCodec decoding
capabilities are determined by your underlying FFmpeg installation. There are
different options to install FFmpeg e.g.:

.. code:: bash

    conda install ffmpeg
    # or
    conda install ffmpeg -c conda-forge

You Linux distribution probably comes with FFmpeg pre-installed as well.
TorchCodec supports all major FFmpeg version in [4, 7].

.. .. TODO add link
.. For more advanced installation instructions and details, please refer to the guidelines in our GitHub repo.
