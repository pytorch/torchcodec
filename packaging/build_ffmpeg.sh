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
    --enable-avutil \
    --enable-swscale \
    --enable-swresample

make -j install
ls ${prefix}/*

# macos: Fix rpath so that the libraries are searched dynamically in user environment.
# In Linux, this is handled by `--enable-rpath` flag.
if [[ "$(uname)" == Darwin ]]; then
    ffmpeg_version="${FFMPEG_VERSION:-4.1.8}"
    major_ver=${ffmpeg_version:0:1}
    if [[ ${major_ver} == 4 ]]; then
        avutil=libavutil.56
        avcodec=libavcodec.58
        avformat=libavformat.58
        avdevice=libavdevice.58
        avfilter=libavfilter.7
        swscale=libswscale.5
        swresample=libswresample.3
    elif [[ ${major_ver} == 5 ]]; then
        avutil=libavutil.57
        avcodec=libavcodec.59
        avformat=libavformat.59
        avdevice=libavdevice.59
        avfilter=libavfilter.8
        swscale=libswscale.6
        swresample=libswresample.4
    elif [[ ${major_ver} == 6 ]]; then
        avutil=libavutil.58
        avcodec=libavcodec.60
        avformat=libavformat.60
        avdevice=libavdevice.60
        avfilter=libavfilter.9
        swscale=libswscale.7
        swresample=libswresample.4
    elif [[ ${major_ver} == 7 ]]; then
        avutil=libavutil.59
        avcodec=libavcodec.61
        avformat=libavformat.61
        avdevice=libavdevice.61
        avfilter=libavfilter.10
        swscale=libswscale.8
        swresample=libswresample.5
    else
        printf "Error: unexpected FFmpeg major version: %s\n"  ${major_ver}
        exit 1;
    fi

    otool="/usr/bin/otool"
    # NOTE: miniconda has a version of otool and install_name_tool installed and we want
    #       to use the default sytem version instead of the miniconda version since the miniconda
    #       version can produce inconsistent results

    # Attempt to use /usr/bin/otool as our default otool
    if [[ ! -e ${otool} ]]; then
        otool="$(which otool)"
    fi
    install_name_tool="/usr/bin/install_name_tool"
    # Attempt to use /usr/bin/install_name_tool as our default install_name_tool
    if [[ ! -e ${install_name_tool} ]]; then
        install_name_tool="$(which install_name_tool)"
    fi

    # list up the paths to fix
    for lib in ${avcodec} ${avdevice} ${avfilter} ${avformat} ${avutil} ${swscale} ${swresample}; do
        ${otool} -l ${prefix}/lib/${lib}.dylib | grep -B2 ${prefix}
    done

    # Replace the hardcoded paths to @rpath
    ${install_name_tool} \
        -change ${prefix}/lib/${avutil}.dylib @rpath/${avutil}.dylib \
        -delete_rpath ${prefix}/lib \
        -id @rpath/${avcodec}.dylib \
        ${prefix}/lib/${avcodec}.dylib
    ${otool} -l ${prefix}/lib/${avcodec}.dylib | grep -B2 ${prefix}

    ${install_name_tool} \
        -change ${prefix}/lib/${avformat}.dylib @rpath/${avformat}.dylib \
        -change ${prefix}/lib/${avcodec}.dylib @rpath/${avcodec}.dylib \
        -change ${prefix}/lib/${avutil}.dylib @rpath/${avutil}.dylib \
        -delete_rpath ${prefix}/lib \
        -id @rpath/${avdevice}.dylib \
        ${prefix}/lib/${avdevice}.dylib
    ${otool} -l ${prefix}/lib/${avdevice}.dylib | grep -B2 ${prefix}

    ${install_name_tool} \
        -change ${prefix}/lib/${avutil}.dylib @rpath/${avutil}.dylib \
        -delete_rpath ${prefix}/lib \
        -id @rpath/${avfilter}.dylib \
        ${prefix}/lib/${avfilter}.dylib
    ${otool} -l ${prefix}/lib/${avfilter}.dylib | grep -B2 ${prefix}

    ${install_name_tool} \
        -change ${prefix}/lib/${avutil}.dylib @rpath/${avutil}.dylib \
        -delete_rpath ${prefix}/lib \
        -id @rpath/${swscale}.dylib \
        ${prefix}/lib/${swscale}.dylib
    ${otool} -l ${prefix}/lib/${swscale}.dylib | grep -B2 ${prefix}

    ${install_name_tool} \
        -change ${prefix}/lib/${avutil}.dylib @rpath/${avutil}.dylib \
        -delete_rpath ${prefix}/lib \
        -id @rpath/${swresample}.dylib \
        ${prefix}/lib/${swresample}.dylib
    ${otool} -l ${prefix}/lib/${swresample}.dylib | grep -B2 ${prefix}

    ${install_name_tool} \
        -change ${prefix}/lib/${avcodec}.dylib @rpath/${avcodec}.dylib \
        -change ${prefix}/lib/${avutil}.dylib @rpath/${avutil}.dylib \
        -delete_rpath ${prefix}/lib \
        -id @rpath/${avformat}.dylib \
        ${prefix}/lib/${avformat}.dylib
    ${otool} -l ${prefix}/lib/${avformat}.dylib | grep -B2 ${prefix}

    ${install_name_tool} \
        -delete_rpath ${prefix}/lib \
        -id @rpath/${avutil}.dylib \
        ${prefix}/lib/${avutil}.dylib
    ${otool} -l ${prefix}/lib/${avutil}.dylib | grep -B2 ${prefix}
fi
