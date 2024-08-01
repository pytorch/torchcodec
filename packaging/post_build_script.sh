#!/bin/bash

echo "LS"
ls
echo "LS dist"
ls dist
echo "auditwheel"
auditwheel repair --plat manylinux_2_17_x86_64 dist/* --exclude libtorch.so --exclude libtorch_cpu.so --exclude libc10.so --exclude libavutil.so.56 --exclude libavcodec.so.58 --exclude libavformat.so.58 --exclude libavdevice.so.58 --exclude libavfilter.so.7 --exclude libavutil.so.57 --exclude libavcodec.so.59 --exclude libavformat.so.59 --exclude libavdevice.so.59 --exclude libavfilter.so.8 --exclude libavutil.so.58 --exclude libavcodec.so.60 --exclude libavformat.so.60 --exclude libavdevice.so.60 --exclude libavfilter.so.9 --exclude libavutil.so.59 --exclude libavcodec.so.61 --exclude libavformat.so.61 --exclude libavdevice.so.61 --exclude libavfilter.so.10

echo "find dist"
find dist
