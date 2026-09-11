:: Copyright (c) Meta Platforms, Inc. and affiliates.
:: All rights reserved.
::
:: This source code is licensed under the BSD-style license found in the
:: LICENSE file in the root directory of this source tree.

:: Windows ARM64 variant of build_ffmpeg.bat. Uses MSYS2's CLANGARM64
:: subsystem (native aarch64 clang toolchain) instead of MINGW64, since
:: mingw-w64-x86_64-toolchain only ever targets x86_64. build_ffmpeg.sh
:: detects MSYSTEM=CLANGARM64 itself and passes --cc=clang --cxx=clang++ to
:: FFmpeg's ./configure (its cc_default is a hardcoded "gcc", which
:: CLANGARM64's toolchain package does not provide, and configure does not
:: read $CC/$CXX from the environment -- only the --cc=/--cxx= flags).
::
:: Unlike build_ffmpeg.bat (x86_64), this does NOT go through
:: vc_env_helper_arm64.bat / vcvarsall.bat first: doing so (confirmed on a
:: real windows-11-arm run) sets INCLUDE/LIB to MSVC's own UCRT headers,
:: which the CLANGARM64 clang driver picks up ahead of/alongside its own
:: mingw-w64 sysroot headers and fails to parse (e.g.
:: "ucrt\stdlib.h:1184:28: error: expected identifier or '('"). gcc-based
:: MINGW64 builds don't read %INCLUDE%, so build_ffmpeg.bat's use of
:: vcvarsall is harmless there; clang does, so it must be skipped here. This
:: matches build_libavif_arm64.bat, which never called vc_env_helper either.
@echo off

set PROJ_FOLDER=%cd%

choco install -y --no-progress msys2 --package-parameters "/NoUpdate" || exit /b 1
call :retry_pacman "pacman -S --noconfirm --needed base-devel mingw-w64-clang-aarch64-toolchain diffutils" || exit /b 1
C:\tools\msys64\usr\bin\env MSYSTEM=CLANGARM64 /bin/bash -l -c "cd \"${PROJ_FOLDER}\" && packaging/build_ffmpeg.sh" || exit /b 1

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
