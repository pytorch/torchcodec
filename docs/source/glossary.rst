Glossary
========

.. glossary::

    pts
       Presentation Time Stamp. The time at which a frame should be displayed.
       In TorchCodec, pts are expressed in seconds.

    best stream
       The notion of "best" stream is determined by FFmpeg. Quoting the `FFmpeg docs
       <https://ffmpeg.org/doxygen/trunk/group__lavf__decoding.html#ga757780d38f482deb4d809c6c521fbcc2>`_:

        *The best stream is determined according to various heuristics as the most likely to be what the user expects.*
