#!/bin/bash

sudo yum install epel-release
sudo yum localinstall --nogpgcheck https://download1.rpmfusion.org/free/el/rpmfusion-free-release-7.noarch.rpm
sudo yum install ffmpeg

# LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH" python packaging/wheel/relocate.py
