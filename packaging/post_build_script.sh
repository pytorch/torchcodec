#!/bin/bash

echo "LS"
ls
echo "LS dist"
ls dist


echo "MVVVV"
mv dist/*linux*.whl $(echo dist/*linux*.whl | sed 's/linux/manylinux2014/')

echo "LS dist"
ls dist