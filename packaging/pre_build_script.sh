#!/bin/bash

set -ex

# We need to install pybind11 because we need its CMake helpers in order to
# compile correctly on Mac. Pybind11 is actually a C++ header-only library,
# and PyTorch actually has it included. PyTorch, however, does not have the
# CMake helpers.
conda install -y pybind11 -c conda-forge


# Search for nvcuvid library in various locations for debugging CI build issues
echo "[NVCUVID-SEARCH] === Searching for nvcuvid library ==="

# First, let's find where CUDA nppi libraries are located
echo "[NVCUVID-SEARCH] Looking for CUDA nppi libraries to find potential nvcuvid location..."
NPPI_LOCATIONS=()

# Search for CUDA nppi libraries that CMake should find
for nppi_lib in "libnppi.so*" "libnppicc.so*" "nppi.so*" "nppicc.so*" "libnppi*" "libnppicc*"; do
    found_nppi=$(find /usr -name "$nppi_lib" 2>/dev/null | head -5)
    if [ -n "$found_nppi" ]; then
        echo "[NVCUVID-SEARCH] Found CUDA nppi library: $found_nppi"
        while IFS= read -r lib_path; do
            lib_dir=$(dirname "$lib_path")
            if [[ ! " ${NPPI_LOCATIONS[@]} " =~ " $lib_dir " ]]; then
                NPPI_LOCATIONS+=("$lib_dir")
            fi
        done <<< "$found_nppi"
    fi
done

# Add these locations to our search paths
for nppi_dir in "${NPPI_LOCATIONS[@]}"; do
    echo "[NVCUVID-SEARCH] Adding CUDA library directory to search: $nppi_dir"
done

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

# Add the CUDA nppi library directories to our search paths
for nppi_dir in "${NPPI_LOCATIONS[@]}"; do
    SEARCH_PATHS+=("$nppi_dir")
done

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