#!/bin/bash

# Run this script to update the resources used in unit tests. The resources are all derived
# from source media already checked into the repo.

# Fail loudly on errors.
set -x
set -e

TORCHCODEC_PATH=$HOME/fbsource/fbcode/pytorch/torchcodec
RESOURCES_DIR=$TORCHCODEC_PATH/test/resources
VIDEO_PATH=$RESOURCES_DIR/nasa_13013.mp4

# Important note: I used ffmpeg version 6.1.1 to generate these images. We
# must have the version that matches the one that we link against in the test.
ffmpeg -y -i "$VIDEO_PATH" -vf select='eq(n\,0)+eq(n\,1)+eq(n\,2)+eq(n\,3)+eq(n\,4)+eq(n\,5)+eq(n\,6)+eq(n\,7)+eq(n\,8)+eq(n\,9)' -vsync vfr -q:v 2 "$VIDEO_PATH.frame%06d.bmp"
ffmpeg -y -i "$VIDEO_PATH" -vf select='eq(n\,15)' -vsync vfr -q:v 2 "$VIDEO_PATH.frame000015.bmp"
ffmpeg -y -i "$VIDEO_PATH" -vf select='eq(n\,20)' -vsync vfr -q:v 2 "$VIDEO_PATH.frame000020.bmp"
ffmpeg -y -i "$VIDEO_PATH" -vf select='eq(n\,25)' -vsync vfr -q:v 2 "$VIDEO_PATH.frame000025.bmp"
ffmpeg -y -i "$VIDEO_PATH" -vf select='eq(n\,30)' -vsync vfr -q:v 2 "$VIDEO_PATH.frame000030.bmp"
ffmpeg -y -i "$VIDEO_PATH" -vf select='eq(n\,35)' -vsync vfr -q:v 2 "$VIDEO_PATH.frame000035.bmp"
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
