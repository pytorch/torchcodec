#!/bin/bash

echo "LS"
ls
echo "LS dist"
ls dist
echo "auditwheel"
auditwheel repair --plat manylinux_2_17_x86_64 dist/*
