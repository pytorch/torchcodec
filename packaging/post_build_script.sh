#!/bin/bash

conda install "ffmpeg=4.4.2" -c conda-forge

# LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH" python packaging/wheel/relocate.py
