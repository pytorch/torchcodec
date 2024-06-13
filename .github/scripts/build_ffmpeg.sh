#!/usr/bin/env bash

# This is taken and adapated from torchaudio, only keeping the parts relevant to
# linux.
# ref: https://github.com/pytorch/audio/blob/main/.github/scripts/ffmpeg/build.sh
#
# This script builds FFmpeg libraries without any functional features.
#
# IMPORTANT:
# The resulting library files have to be LGPL version of FFmpeg libraries.
# - Do not enable `--enable-nonfree` and `--enable-gpl`.
# - Do not enable third party library integrations like x264.
#
# This script is not meant to build useful FFmpeg libraries, but to build
# a skeleton of FFmpeg libraries that are use only during the build process of
# torchcodec.
#
# The resulting FFmpeg libraries should not be shipped either.

set -eux

prefix="${FFMPEG_ROOT}"
archive="https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n${FFMPEG_VERSION}.tar.gz"

build_dir=$(mktemp -d -t ffmpeg-build.XXXXXXXXXX)
cleanup() {
    rm -rf "${build_dir}"
}
trap 'cleanup $?' EXIT

cd "${build_dir}"
curl -LsS -o ffmpeg.tar.gz "${archive}"
tar -xf ffmpeg.tar.gz --strip-components 1
./configure \
    --prefix="${prefix}" \
    --disable-all \
    --disable-everything \
    --disable-programs \
    --disable-doc \
    --disable-debug \
    --disable-autodetect \
    --disable-x86asm \
    --disable-iconv \
    --disable-encoders \
    --disable-decoders \
    --disable-hwaccels \
    --disable-muxers \
    --disable-demuxers \
    --disable-parsers \
    --disable-bsfs \
    --disable-protocols \
    --disable-devices \
    --disable-filters \
    --disable-asm \
    --disable-static \
    --enable-shared \
    --enable-rpath \
    --enable-pic \
    --enable-avcodec \
    --enable-avdevice \
    --enable-avfilter \
    --enable-avformat \
    --enable-avutil

make -j install
ls ${prefix}/*
