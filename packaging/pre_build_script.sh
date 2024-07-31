
#!/bin/bash

echo "Installing build pre-requisites from pre_build_script.sh"

python -m pip install --upgrade pip
python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
python -m pip install auditwheel
python -m pip install --upgrade setuptools
