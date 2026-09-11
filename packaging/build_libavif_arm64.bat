:: Copyright (c) Meta Platforms, Inc. and affiliates.
:: All rights reserved.
::
:: This source code is licensed under the BSD-style license found in the
:: LICENSE file in the root directory of this source tree.

:: Windows ARM64 variant of build_libavif.bat. Uses MSYS2's CLANGARM64
:: subsystem (native aarch64 clang toolchain) instead of MINGW64. nasm is
:: intentionally NOT installed: build_libavif.sh only requires nasm on
:: x86_64/i686 (for dav1d's hand-written x86 SIMD); on aarch64 dav1d's NEON
:: kernels are plain C intrinsics compiled by the C compiler, and
:: build_libavif.sh already asserts NEON symbols are present post-build.
@echo off

set PROJ_FOLDER=%cd%

choco install -y --no-progress msys2 --package-parameters "/NoUpdate" || exit /b 1
call :retry_pacman "pacman -S --noconfirm --needed base-devel mingw-w64-clang-aarch64-toolchain mingw-w64-clang-aarch64-cmake mingw-w64-clang-aarch64-ninja mingw-w64-clang-aarch64-meson diffutils" || exit /b 1
C:\tools\msys64\usr\bin\env MSYSTEM=CLANGARM64 /bin/bash -l -c "cd \"${PROJ_FOLDER}\" && packaging/build_libavif.sh" || exit /b 1

:end
goto :eof

:retry_pacman
for /L %%I in (1,1,3) do (
    C:\tools\msys64\usr\bin\env MSYSTEM=CLANGARM64 /bin/bash -l -c "%~1" && exit /b 0
    if %%I LSS 3 (
        echo pacman attempt %%I failed, retrying after 15 seconds...
        timeout /t 15 /nobreak >nul
    )
)
echo ERROR: pacman failed after 3 attempts. >&2
exit /b 1
