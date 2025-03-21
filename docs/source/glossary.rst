Glossary
========

.. glossary::

    pts
       Presentation Time Stamp. The time at which a frame or audio sample should be played.
       In TorchCodec, pts are expressed in seconds.

    best stream
       The notion of "best" stream is determined by FFmpeg. Quoting the `FFmpeg docs
       <https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga757780d38f482deb4d809c6c521fbcc2>`_:

        *The best stream is determined according to various heuristics as the most likely to be what the user expects.*

    scan
       A scan corresponds to an entire pass over a video file, with the purpose
       of retrieving metadata about the different streams and frames. **It does
       not involve decoding**, so it is a lot cheaper than decoding the file.
       The :class:`~torchcodec.decoders.VideoDecoder` performs a scan when using
       ``seek_mode="exact"``, and doesn't scan when using
       ``seek_mode="approximate"``.

    clips
        A clip is a sequence of frames, usually in :term:`pts` order. The frames
        may not necessarily be consecutive. A clip is represented as a 4D
        :class:`~torchcodec.FrameBatch`. A group of clips, which is what the
        :ref:`samplers <samplers>` return, is represented as 5D
        :class:`~torchcodec.FrameBatch`.
