:: Copyright (c) Meta Platforms, Inc. and affiliates.
:: All rights reserved.
::
:: This source code is licensed under the BSD-style license found in the
:: LICENSE file in the root directory of this source tree.

:: Same as vc_env_helper.bat, but initializes the VC toolchain for a native
:: Windows ARM64 build (`vcvarsall.bat arm64`) instead of x64. CUDA/XPU setup is
:: intentionally omitted: Windows ARM64 is CPU-only for torchcodec (no CUDA
:: toolkit ships for this target), matching the boundaries of this workstream.
::
:: NOT currently invoked by build_ffmpeg_arm64.bat or build_libavif_arm64.bat:
:: running those under an MSVC-initialized environment leaks INCLUDE/LIB
:: (MSVC UCRT headers) into the MSYS2 CLANGARM64 clang build and breaks it
:: (confirmed on real windows-11-arm hardware). This helper is kept for the
:: future torchcodec Arm64 wheel-build job (see implementation-status.md),
:: which does need MSVC's cl.exe/link.exe on PATH to build the C++ extension
:: itself, once its S3/libheif blockers are resolved.
@echo on

set VC_VERSION_LOWER=17
set VC_VERSION_UPPER=18

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -legacy -products * -version [%VC_VERSION_LOWER%^,%VC_VERSION_UPPER%^) -property installationPath`) do (
    if exist "%%i" if exist "%%i\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VS15INSTALLDIR=%%i"
        set "VS15VCVARSALL=%%i\VC\Auxiliary\Build\vcvarsall.bat"
        goto vswhere
    )
)

:vswhere
if not defined VS15VCVARSALL (
    echo ERROR: Could not locate a Visual Studio installation with vcvarsall.bat >&2
    echo Checked VS versions [%VC_VERSION_LOWER%,%VC_VERSION_UPPER%^) via vswhere.exe >&2
    exit /b 1
)

if "%VSDEVCMD_ARGS%" == "" (
    call "%VS15VCVARSALL%" arm64 || exit /b 1
) else (
    call "%VS15VCVARSALL%" arm64 %VSDEVCMD_ARGS% || exit /b 1
)

@echo on

set DISTUTILS_USE_SDK=1
set BUILD_AGAINST_ALL_FFMPEG_FROM_S3=1

if "%*" == "" (
    echo Usage: vc_env_helper_arm64.bat [command] [args]
    echo e.g. vc_env_helper_arm64.bat cl /c test.cpp
) else (
    %* || exit /b 1
)
