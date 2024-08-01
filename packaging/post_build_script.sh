#!/bin/bash

conda install ffmpeg -c conda-forge

LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH" python packaging/wheel/relocate.py
