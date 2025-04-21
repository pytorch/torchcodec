#!/bin/bash

set -ex

source packaging/helpers.sh

wheel_path=$(pwd)/$(find dist -type f -name "*.whl")
echo "Wheel content:"
unzip -l $wheel_path

unamestr=$(uname)
if [[ "$unamestr" == 'Linux' ]]; then
    ext="so"
elif [[ "$unamestr" == 'Darwin' ]]; then
    ext="dylib"
else
    echo "Unknown operating system: $unamestr"
    exit 1
fi

# TODO: Make ffmpeg4 work with nvcc.
if [[ "$ENABLE_CUDA" -eq 1 ]]; then
  ffmpeg_versions=(5 6 7)
fi

for ffmpeg_major_version in ${ffmpeg_versions[@]}; do
    assert_in_wheel $wheel_path torchcodec/libtorchcodec${ffmpeg_major_version}.${ext}
done
assert_not_in_wheel $wheel_path libtorchcodec.${ext}

for ffmpeg_ext in libavcodec.${ext} libavfilter.${ext} libavformat.${ext} libavutil.${ext} libavdevice.${ext} ; do
    assert_not_in_wheel $wheel_path $ffmpeg_ext
done

assert_not_in_wheel $wheel_path "^test"
assert_not_in_wheel $wheel_path "^doc"
assert_not_in_wheel $wheel_path "^benchmarks"
assert_not_in_wheel $wheel_path "^packaging"

echo "ls dist"
ls dist
