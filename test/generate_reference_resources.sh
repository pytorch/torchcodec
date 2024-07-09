#!/bin/bash

# Run this script to update the resources used in unit tests. The resources are all derived
# from source media already checked into the repo.

# Fail loudly on errors.
set -x
set -e

TORCHCODEC_PATH=$HOME/fbsource/fbcode/pytorch/torchcodec
RESOURCES_DIR=$TORCHCODEC_PATH/test/resources
VIDEO_PATH=$RESOURCES_DIR/nasa_13013.mp4

# Last generated with ffmpeg version 4.3
#
# Note: The naming scheme used here must match the naming scheme used to load
# tensors in ./utils.py.
FRAMES=(0 1 2 3 4 5 6 7 8 9)
FRAMES+=(15 20 25 30 35)
FRAMES+=(386 387 388 389)
for frame in "${FRAMES[@]}"; do
  # Note that we are using 0-based index naming. Asking ffmpeg to number output
  # frames would result in 1-based index naming. We enforce 0-based index naming
  # so that the name of reference frames matches the index when accessing that
  # frame in the Python decoder.
  frame_name=$(printf "%06d" "$frame")
  ffmpeg -y -i "$VIDEO_PATH" -vf select="eq(n\,$frame)" -vsync vfr -q:v 2 "$VIDEO_PATH.frame$frame_name.bmp"
done
ffmpeg -y -ss 6.0 -i "$VIDEO_PATH" -frames:v 1 "$VIDEO_PATH.time6.000000.bmp"
ffmpeg -y -ss 6.1 -i "$VIDEO_PATH" -frames:v 1 "$VIDEO_PATH.time6.100000.bmp"
ffmpeg -y -ss 10.0 -i "$VIDEO_PATH" -frames:v 1 "$VIDEO_PATH.time10.000000.bmp"
# This is the last frame of this video.
ffmpeg -y -ss 12.979633 -i "$VIDEO_PATH" -frames:v 1 "$VIDEO_PATH.time12.979633.bmp"
# Audio generation in the form of an mp3.
ffmpeg -y -i "$VIDEO_PATH" -b:a 192K -vn "$VIDEO_PATH.audio.mp3"

for bmp in "$RESOURCES_DIR"/*.bmp
do
  python3 convert_image_to_tensor.py "$bmp"
  rm -f "$bmp"
done
