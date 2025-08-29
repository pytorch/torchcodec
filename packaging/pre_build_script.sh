#!/bin/bash

set -ex

# We need to install pybind11 because we need its CMake helpers in order to
# compile correctly on Mac. Pybind11 is actually a C++ header-only library,
# and PyTorch actually has it included. PyTorch, however, does not have the
# CMake helpers.
conda install -y pybind11 -c conda-forge

# Search for nvcuvid library in various locations for debugging CI build issues
echo "[NVCUVID-SEARCH] === Searching for nvcuvid library ==="

# Standard library search paths
SEARCH_PATHS=(
    "/usr/lib"
    "/usr/lib64" 
    "/usr/lib/x86_64-linux-gnu"
    "/usr/local/lib"
    "/usr/local/lib64"
    "/lib"
    "/lib64"
    "/opt/cuda/lib64"
    "/usr/local/cuda/lib64"
    "/usr/local/cuda/lib"
    "/usr/local/cuda-*/lib64"
    "/usr/local/cuda-*/lib"
)

# Library name variations to search for
LIB_PATTERNS=(
    "libnvcuvid.so*"
    "nvcuvid.so*" 
    "libnvcuvid.a"
    "nvcuvid.a"
    "libnvcuvid*"
    "nvcuvid*"
)

found_libraries=()

for search_path in "${SEARCH_PATHS[@]}"; do
    if [ -d "$search_path" ]; then
        echo "[NVCUVID-SEARCH] Searching in: $search_path"
        for pattern in "${LIB_PATTERNS[@]}"; do
            # Use find with error suppression to avoid permission errors
            found_files=$(find "$search_path" -maxdepth 3 -name "$pattern" 2>/dev/null || true)
            if [ -n "$found_files" ]; then
                echo "[NVCUVID-SEARCH]   Found: $found_files"
                found_libraries+=($found_files)
            fi
        done
    else
        echo "[NVCUVID-SEARCH] Directory not found: $search_path"
    fi
done

# Also try using ldconfig to find the library
echo "[NVCUVID-SEARCH] Checking ldconfig cache for nvcuvid..."
if command -v ldconfig >/dev/null 2>&1; then
    ldconfig_result=$(ldconfig -p 2>/dev/null | grep -i nvcuvid || echo "Not found in ldconfig cache")
    echo "[NVCUVID-SEARCH] ldconfig result: $ldconfig_result"
fi

# Try pkg-config if available
echo "[NVCUVID-SEARCH] Checking pkg-config for cuda libraries..."
if command -v pkg-config >/dev/null 2>&1; then
    pkg_result=$(pkg-config --list-all 2>/dev/null | grep -i cuda || echo "No CUDA packages found in pkg-config")
    echo "[NVCUVID-SEARCH] pkg-config cuda packages: $pkg_result"
fi

# Summary
if [ ${#found_libraries[@]} -gt 0 ]; then
    echo "[NVCUVID-SEARCH] === SUMMARY: Found ${#found_libraries[@]} nvcuvid library files ==="
    for lib in "${found_libraries[@]}"; do
        echo "[NVCUVID-SEARCH]   $lib"
        # Show file info if possible
        if [ -f "$lib" ]; then
            ls -la "$lib" 2>/dev/null || true
        fi
    done
else
    echo "[NVCUVID-SEARCH] === SUMMARY: No nvcuvid libraries found ==="
fi

echo "[NVCUVID-SEARCH] === End nvcuvid library search ==="
