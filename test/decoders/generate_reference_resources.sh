#!/bin/bash

# Run this script to update the resources used in unit tests. The resources are all derived
# from source media already checked into the repo.

# Fail loudly on errors.
set -x
set -e

# 1. Create a temporary directory to dump bitmaps into.
TEMP_DIR=$(mktemp -d)
echo "Creating all bitmaps in $TEMP_DIR"

TORCHCODEC_PATH=$HOME/fbsource/fbcode/pytorch/torchcodec
RESOURCES_DIR=$TORCHCODEC_PATH/test/decoders/resources
VIDEO_PATH=$RESOURCES_DIR/nasa_13013.mp4

# Important note: I used ffmpeg version 6.1.1 to generate these images. We
# must have the version that matches the one that we link against in the test.
ffmpeg -i "$VIDEO_PATH" -vf select='eq(n\,0)+eq(n\,1)' -vsync vfr -q:v 2 "$VIDEO_PATH.frame%06d.bmp"
ffmpeg -ss 6.0 -i "$VIDEO_PATH" -frames:v 1 "$VIDEO_PATH.time6.000000.bmp"
ffmpeg -ss 6.1 -i "$VIDEO_PATH" -frames:v 1 "$VIDEO_PATH.time6.100000.bmp"
ffmpeg -ss 10.0 -i "$VIDEO_PATH" -frames:v 1 "$VIDEO_PATH.time10.000000.bmp"
# This is the last frame of this video.
ffmpeg -ss 12.979633 -i "$VIDEO_PATH" -frames:v 1 "$VIDEO_PATH.time12.979633.bmp"
# Audio generation in the form of an mp3.
ffmpeg -i "$VIDEO_PATH" -b:a 192K -vn "$VIDEO_PATH.audio.mp3"

for bmp in "$RESOURCES_DIR"/*.bmp
do
  python convert_image_to_tensor.py "$bmp"
  rm -f "$bmp"
done
