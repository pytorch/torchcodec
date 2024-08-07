#!/bin/bash

echo "LS"
ls
echo "LS dist"
ls dist


echo "MVVVV"
mv dist/*linux_x86_64*.whl $(echo dist/*linux_x86_64*.whl | sed 's/linux_x86_64/manylinux_2_17_x86_64.manylinux2014_x86_64/')

echo "LS dist"
ls dist
