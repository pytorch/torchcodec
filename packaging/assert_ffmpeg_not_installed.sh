#!/usr/bin/env bash

if command -v "ffmpeg" &> /dev/null
then
    echo "ffmpeg is installed, but it shouldn't! Exiting!!"
    exit 1
fi
