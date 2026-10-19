cmake_minimum_required(VERSION 3.18)
project(TorchCodec)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Guard against accidentally building a distributable wheel that dynamically
# links against a GPL build of FFmpeg (e.g. from conda/system), which would be a
# license violation. Wheels are expected to link against our pre-built non-GPL
# FFmpeg (BUILD_AGAINST_ALL_FFMPEG_FROM_S3).
if(SKBUILD_STATE STREQUAL "wheel"
   AND NOT DEFINED ENV{BUILD_AGAINST_ALL_FFMPEG_FROM_S3}
   AND NOT DEFINED ENV{I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION})
    message(FATAL_ERROR
        "It looks like you're trying to build a wheel. You probably want to set "
        "BUILD_AGAINST_ALL_FFMPEG_FROM_S3. If you have a good reason *not* to, "
        "then set I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION.")
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development Interpreter)

if(APPLE)
    # Some CI runners ship a Mono.framework which contains headers for libjpeg,
    # libpng, etc. We do not want those headers to be found - they're old and
    # incompatible with what we do here (e.g. missing eXIf support in libpng).
    # On the CI we want the conda-forge-provided libraries to be found first. So
    # we tell CMake to search for 'frameworks' last.
    set(CMAKE_FIND_FRAMEWORK LAST)
endif()

# Image decoder build toggles. By default all image decoders are built and
# *required*: if a codec's lib can't be found/fetched the build fails loudly,
# telling the user how to opt out. This is controlled by:
#   - TORCHCODEC_BUILD_IMAGE (catch-all, default ON)
#   - TORCHCODEC_BUILD_{JPEG,PNG,WEBP,AVIF,GIF} (default AUTO = inherit catch-all)
# A codec set to OFF is not built; its decode_*() then raises an actionable error
# at call time (see e.g. DecodeJpeg.cpp's !TORCHCODEC_ENABLE_JPEG stub), while importing
# torchcodec still works. We default the flags here too so that direct `cmake`
# invocations (e.g. the C++ tests build) behave like the scikit-build path,
# which passes them via pyproject.toml.
if(NOT DEFINED TORCHCODEC_BUILD_IMAGE)
    set(TORCHCODEC_BUILD_IMAGE "ON")
endif()
foreach(_codec JPEG PNG WEBP AVIF GIF HEIC NVJPEG ROCJPEG)
    if(NOT DEFINED TORCHCODEC_BUILD_${_codec})
        set(TORCHCODEC_BUILD_${_codec} "AUTO")
    endif()
endforeach()

# Returns true (via out_want) if TORCHCODEC_BUILD_IMAGE is on, or if that
# specific codec flag is explicitly on, and false otherwise.
function(resolve_image_codec codec_flag out_want)
    if(codec_flag STREQUAL "AUTO")
        set(_resolved "${TORCHCODEC_BUILD_IMAGE}")
    else()
        set(_resolved "${codec_flag}")
    endif()
    if(_resolved)
        set(${out_want} TRUE PARENT_SCOPE)
    else()
        set(${out_want} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(discover_image_codecs)
    resolve_image_codec("${TORCHCODEC_BUILD_JPEG}" want_jpeg)
    if(want_jpeg)
        if(NOT DEFINED JPEG_ROOT AND DEFINED ENV{CONDA_PREFIX})
            # JPEG_ROOT will automatically be passed to find_package(JPEG) by CMake
            # if it is set. We fallback to the conda prefix if it is not set.
            if(WIN32)
                set(JPEG_ROOT "$ENV{CONDA_PREFIX}/Library")
            else()
                set(JPEG_ROOT "$ENV{CONDA_PREFIX}")
            endif()
        endif()
        find_package(JPEG)  # Not REQUIRED: we emit our own actionable error below.
        if(JPEG_FOUND)
            message(STATUS "libjpeg found: building torchcodec with JPEG image decoding support.")
        else()
            message(FATAL_ERROR
                "libjpeg not found, but JPEG image decoding is enabled. Install "
                "libjpeg[-turbo] and its development headers, or set "
                "TORCHCODEC_BUILD_JPEG=0 to build without JPEG support (decode_jpeg "
                "will raise at runtime), or TORCHCODEC_BUILD_IMAGE=0 to disable all "
                "image decoders.")
        endif()
    else()
        message(STATUS "Not building torchcodec with JPEG image decoding support (disabled via TORCHCODEC_BUILD_JPEG/TORCHCODEC_BUILD_IMAGE): decode_jpeg will raise at runtime.")
    endif()

    resolve_image_codec("${TORCHCODEC_BUILD_PNG}" want_png)
    if(want_png)
        if(NOT DEFINED PNG_ROOT AND DEFINED ENV{CONDA_PREFIX})
            if(WIN32)
                set(PNG_ROOT "$ENV{CONDA_PREFIX}/Library")
            else()
                set(PNG_ROOT "$ENV{CONDA_PREFIX}")
            endif()
        endif()
        find_package(PNG)  # Transitively pulls in ZLIB.
        if(PNG_FOUND)
            message(STATUS "libpng found: building torchcodec with PNG image decoding support.")
        else()
            message(FATAL_ERROR
                "libpng not found, but PNG image decoding is enabled. Install libpng "
                "and its development headers, or set TORCHCODEC_BUILD_PNG=0 to build "
                "without PNG support (decode_png will raise at runtime), or "
                "TORCHCODEC_BUILD_IMAGE=0 to disable all image decoders.")
        endif()
    else()
        message(STATUS "Not building torchcodec with PNG image decoding support (disabled via TORCHCODEC_BUILD_PNG/TORCHCODEC_BUILD_IMAGE): decode_png will raise at runtime.")
    endif()

    resolve_image_codec("${TORCHCODEC_BUILD_WEBP}" want_webp)
    if(want_webp)
        if(NOT DEFINED WebP_ROOT AND DEFINED ENV{CONDA_PREFIX})
            if(WIN32)
                set(WebP_ROOT "$ENV{CONDA_PREFIX}/Library")
            else()
                set(WebP_ROOT "$ENV{CONDA_PREFIX}")
            endif()
        endif()
        find_package(WebP)  # Provides the WebP::webp and WebP::webpdemux targets.
        if(WebP_FOUND)
            message(STATUS "libwebp found: building torchcodec with WebP image decoding support.")
        else()
            message(FATAL_ERROR
                "libwebp not found, but WebP image decoding is enabled. Install "
                "libwebp and its development headers, or set TORCHCODEC_BUILD_WEBP=0 "
                "to build without WebP support (decode_webp will raise at runtime), "
                "or TORCHCODEC_BUILD_IMAGE=0 to disable all image decoders.")
        endif()
    else()
        message(STATUS "Not building torchcodec with WebP image decoding support (disabled via TORCHCODEC_BUILD_WEBP/TORCHCODEC_BUILD_IMAGE): decode_webp will raise at runtime.")
    endif()

    resolve_image_codec("${TORCHCODEC_BUILD_AVIF}" want_avif)
    if(want_avif)
        include(${CMAKE_CURRENT_SOURCE_DIR}/fetch_avif_from_s3.cmake)
        set(libavif_FOUND TRUE)
        message(STATUS "Building torchcodec with AVIF image decoding support (libavif from S3).")
    else()
        message(STATUS "Not building torchcodec with AVIF image decoding support (disabled via TORCHCODEC_BUILD_AVIF/TORCHCODEC_BUILD_IMAGE): decode_avif will raise at runtime.")
    endif()

    resolve_image_codec("${TORCHCODEC_BUILD_GIF}" want_gif)
    if(want_gif)
        message(STATUS "Building torchcodec with GIF image decoding support (vendored giflib).")
    else()
        message(STATUS "Not building torchcodec with GIF image decoding support (disabled via TORCHCODEC_BUILD_GIF/TORCHCODEC_BUILD_IMAGE): decode_gif will raise at runtime.")
    endif()

    # Re-export the resolved want_<codec> booleans and (for Windows) the avif
    # runtime lib path to make_torchcodec_image_library, our caller.
    foreach(_codec jpeg png webp avif gif)
        set(want_${_codec} "${want_${_codec}}" PARENT_SCOPE)
    endforeach()
    set(avif_runtime_lib "${avif_runtime_lib}" PARENT_SCOPE)
endfunction()

# Read a list variable from sources.bzl, the single source of truth for the
# _core C++ source files shared with the internal Buck build. The .bzl file is
# valid Python, so we exec it and print the requested list; configure_file
# registers it as a dependency so re-running CMake picks up edits.
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/sources.bzl
    ${CMAKE_CURRENT_BINARY_DIR}/sources.bzl
    COPYONLY)

function(read_sources_from_bzl output_var list_name)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
            "exec(open('${CMAKE_CURRENT_SOURCE_DIR}/sources.bzl').read()); print(';'.join(${list_name}))"
        OUTPUT_VARIABLE _sources
        RESULT_VARIABLE _retval
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _retval EQUAL 0)
        message(FATAL_ERROR "Failed to read '${list_name}' from sources.bzl")
    endif()
    set(${output_var} ${_sources} PARENT_SCOPE)
endfunction()

read_sources_from_bzl(TORCHCODEC_DECODER_CORE_SOURCES decoder_core_sources)
read_sources_from_bzl(TORCHCODEC_FFMPEG_COMMON_SOURCES ffmpeg_common_sources)
read_sources_from_bzl(TORCHCODEC_IO_SOURCES io_sources)
read_sources_from_bzl(TORCHCODEC_DECODER_CORE_CUDA_SOURCES decoder_core_cuda_sources)
read_sources_from_bzl(TORCHCODEC_FILE_LIKE_CONTEXT_SOURCES file_like_context_sources)
read_sources_from_bzl(TORCHCODEC_CUSTOM_OPS_SOURCES custom_ops_sources)
read_sources_from_bzl(TORCHCODEC_PYBIND_OPS_SOURCES pybind_ops_sources)
read_sources_from_bzl(TORCHCODEC_IMAGE_SOURCES image_sources)
read_sources_from_bzl(TORCHCODEC_IMAGE_OPS_SOURCES image_ops_sources)
read_sources_from_bzl(TORCHCODEC_GIFLIB_SOURCES giflib_sources)
read_sources_from_bzl(TORCHCODEC_HEIC_SOURCES heic_sources)
read_sources_from_bzl(TORCHCODEC_HEIC_OPS_SOURCES heic_ops_sources)

if(DEFINED TORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR AND TORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR)
    set(TORCHCODEC_WERROR_OPTION "")
else()
    if (WIN32)
        # TODO set warnings as errors on Windows as well.
        # set(TORCHCODEC_WERROR_OPTION "/WX")
    else()
        set(TORCHCODEC_WERROR_OPTION "-Werror")
    endif()
endif()

if (WIN32)
  # Avoid warnings about non-ASCII characters in source files.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4819")
  # Required for Unicode support in fmt library (bundled with PyTorch).
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
  # Important for when we add Windows CUDA: exporting all symbols is limited to
  # 65535 symbols, which (apparently) will not work for CUDA.
  # https://github.com/pytorch/pytorch/pull/3650
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 ${TORCHCODEC_WERROR_OPTION} ${TORCH_CXX_FLAGS}")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic ${TORCHCODEC_WERROR_OPTION} ${TORCH_CXX_FLAGS}")
endif()


function(make_torchcodec_sublibrary
    library_name
    type
    sources
    library_dependencies)

    add_library(${library_name} ${type} ${sources})
    set_target_properties(${library_name} PROPERTIES CXX_STANDARD 20)
    target_include_directories(${library_name}
        PRIVATE
        ./../../../
        "${TORCH_INSTALL_PREFIX}/include"
        ${Python3_INCLUDE_DIRS}
    )
    # On Windows, the pybind ops library links the Python import library
    # (python3XX.lib) by bare name via pybind11::module, so the Python library
    # directory must be on the linker search path or linking fails with
    # "LNK1104: cannot open file 'python3XX.lib'". This is a no-op on
    # Linux/macOS, where extension modules resolve Python symbols dynamically.
    if(WIN32)
        target_link_directories(${library_name} PRIVATE ${Python3_LIBRARY_DIRS})
    endif()
    # Use fmt library in header-only mode (bundled with PyTorch): the fmt
    # library is bundled with PyTorch and the headers are available in
    # TORCH_INSTALL_PREFIX/include/, which we add above. To be compatible with
    # torch, we want to rely on the "header-only" mode of these headers and hit
    # all the `#ifdef FMT_HEADER_ONLY` paths, so we define FMT_HEADER_ONLY here.```
    target_compile_definitions(${library_name} PRIVATE FMT_HEADER_ONLY)
    # TC is ABI-stable w.r.t. torch since torch 2.11
    target_compile_definitions(${library_name} PRIVATE TORCH_TARGET_VERSION=0x020b000000000000)

    # Avoid adding the "lib" prefix which we already add explicitly.
    set_target_properties(${library_name} PROPERTIES PREFIX "")

    # On macOS, for wheels, add Homebrew's FFmpeg library path to the rpath so
    # that users with Homebrew-installed FFmpeg can use torchcodec without
    # setting DYLD_LIBRARY_PATH. See
    # https://github.com/pytorch/torchcodec/issues/570.
    # That's the behavior we enable by default when building TorchCodec wheels
    # which we'll ship on PyPI.
    # Note that this means homebrew-FFmpeg will always take precedence over
    # conda-FFmpeg. For this reason, this behavior can (and should!) be disabled
    # when building TorchCodec for conda, by setting
    # TORCHCODEC_DISABLE_HOMEBREW_RPATH.
    # We should have the following behavior for users:
    # - For TorchCodec installed from PyPI (pip or uv): Users don't have to use
    #   a conda env, and by default the homebrew-ffmpeg should be found. For
    #   users who do use a conda env, they should be able to use the
    #   conda-installed FFmpeg, but only if HomeBrew FFmpeg doesn't exist since
    #   homebrew will take precedence.
    # - For TorchCodec installed from conda, users must install FFmpeg from
    #   conda as well (it should be a dependency of the TorchCodec conda
    #   package). Homebrew FFmpeg shouldn't be used.
    if(APPLE AND NOT (DEFINED TORCHCODEC_DISABLE_HOMEBREW_RPATH AND TORCHCODEC_DISABLE_HOMEBREW_RPATH))
        if(DEFINED ENV{HOMEBREW_PREFIX})
            set(HOMEBREW_FFMPEG_LIB "$ENV{HOMEBREW_PREFIX}/opt/ffmpeg/lib")
        else()
            # Default Homebrew location on Apple Silicon
            set(HOMEBREW_FFMPEG_LIB "/opt/homebrew/opt/ffmpeg/lib")
        endif()
        set_target_properties(${library_name} PROPERTIES
            INSTALL_RPATH "${HOMEBREW_FFMPEG_LIB}"
        )
    endif()

    target_link_libraries(
        ${library_name}
        PUBLIC
        ${library_dependencies}
    )

endfunction()

function(make_torchcodec_libraries
    ffmpeg_major_version
    ffmpeg_target)

    # We create three shared libraries per version of FFmpeg, where the version
    # is denoted by N:
    #
    # 1. libtorchcodec_coreN.{ext}: Base library which contains the
    #    implementation of VideoDecoder and everything VideoDecoder needs. On
    #    Linux, {ext} is so. On Mac, it is dylib. On Windows it's dll.
    #
    # 2. libtorchcodec_custom_opsN.{ext}: Implementation of the PyTorch custom
    #    ops. Depends on libtorchcodec_coreN.{ext}. On Linux, {ext} is so.
    #    On Mac, it is dylib. On Windows it's dll.
    #
    # 3. libtorchcodec_pybind_opsN.{ext}: Implementation of the pybind11 ops. We
    #    keep these separate from the PyTorch custom ops because we have to
    #    load these libraries separately on the Python side. Depends on
    #    libtorchcodec_coreN.{ext}. On BOTH Linux and Mac {ext} is so. On
    #    Windows, it's pyd.

    # 1. Create libtorchcodec_coreN.{ext}.
    set(core_library_name "libtorchcodec_core${ffmpeg_major_version}")
    set(core_sources
        ${TORCHCODEC_DECODER_CORE_SOURCES}
        ${TORCHCODEC_FFMPEG_COMMON_SOURCES}
        ${TORCHCODEC_IO_SOURCES}
    )

    if(ENABLE_CUDA)
        enable_language(CUDA)
        list(APPEND core_sources ${TORCHCODEC_DECODER_CORE_CUDA_SOURCES})
    endif()

    set(core_library_dependencies
        ${ffmpeg_target}
        ${TORCH_LIBRARIES}
    )

    make_torchcodec_sublibrary(
        "${core_library_name}"
        SHARED
        "${core_sources}"
        "${core_library_dependencies}"
    )

    if(ENABLE_CUDA)
        # We define USE_CUDA to guard CUDA-specific code paths (e.g.
        # NVDECCache usage in NVDECCacheConfig.cpp) and because some torch
        # APIs like aoti_torch_get_current_cuda_stream are only exposed when
        # USE_CUDA is defined.
        # https://github.com/pytorch/pytorch/blob/98e36864e640023a716e058d894ea2d20e76e5f7/torch/csrc/inductor/aoti_torch/c/shim.h#L573-L602
        target_compile_definitions(${core_library_name} PRIVATE USE_CUDA)
        set_property(TARGET ${core_library_name} PROPERTY CUDA_STANDARD 17)
        if(WIN32)
            # CUDA 13.2+ CCCL headers require MSVC's standard-conforming
            # preprocessor when compiling .cu files.
            target_compile_options(${core_library_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
        endif()
    endif()

    # 2. Create libtorchcodec_custom_opsN.{ext}.
    set(custom_ops_library_name "libtorchcodec_custom_ops${ffmpeg_major_version}")
    set(custom_ops_sources
        ${TORCHCODEC_CUSTOM_OPS_SOURCES}
    )
    set(custom_ops_dependencies
        ${core_library_name}
        pybind11::module # Links Python correctly per-platform (nothing extra on
                         # Linux, dynamic-lookup on Mac, import lib on Windows).
                         # Using ${Python3_LIBRARIES} instead adds a DT_NEEDED on
                         # a shared libpython (e.g. free-threaded builds ship
                         # libpython3.XXt.so), which auditwheel then wrongly
                         # bundles into the wheel.
    )
    make_torchcodec_sublibrary(
        "${custom_ops_library_name}"
        SHARED
        "${custom_ops_sources}"
        "${custom_ops_dependencies}"
    )

    # pybind11 limits the visibility of symbols in the shared library to prevent
    # stray initialization of py::objects. The rest of the object code must
    # match. See:
    #   https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes
    #
    # Note that this is Linux only. It's not necessary on Windows, and on Mac
    # hiding visibility can actually break dyanmic casts across share libraries
    # because the type infos don't get exported.
    if(LINUX)
        target_compile_options(
            ${custom_ops_library_name}
            PUBLIC
            "-fvisibility=hidden"
        )
    endif()

    # We disable the "attributes" warning in the core library.
    # This warning is triggered when, e.g., a class has "default" (i.e.
    # public) visibility but a member of that class has "hidden" visibility.
    # We have such a pattern in the core library with the VideoStreamOptions
    # class and its StableDevice field:
    # - VideoStreamOptions has "default" (public) visibility because it is part
    #   of the core library, and the core library must have public symbols
    #   because it is depended on by other libraries (the pybind ops library and
    #   the custom ops library)
    # - the StableDevice field has "hidden" visibility because this is how
    #   torch::stable exports it.
    # This creates this mismatch where a class has "higher" visibility than its
    # member, hence the warning.
    #
    # We choose to silence this warning here. A possibly better solution would
    # be to have a more fine-grained visibility control where each
    # class/function would be "hidden" unless explicitly marked as public, e.g.
    # through a TORCHCODEC_API annotation. In this case, it is likely that
    # VideoStreamOptions could be in fact "hidden" as well.
    # TODO_STABLE_ABI: do that!
    if (LINUX)
        target_compile_options(
            ${core_library_name}
            PRIVATE
            "-Wno-attributes"
        )
    endif()

    # Install all libraries.
    set(
        all_libraries
        ${core_library_name}
        ${custom_ops_library_name}
    )

    # We want the built libraries to end up in [src/]torchcodec so that the
    # Python runtime loader can find them. This is done in two mutually exclusive
    # ways depending on the build mode. Crucially, scikit-build-core doesn't
    # trigger cmake's install() step on editable installs:
    #
    # - Editable installs (`pip install -e .`, SKBUILD_STATE=="editable"): build
    #   the compiled libraries directly into the source package dir
    #   (src/torchcodec).
    # - Wheel builds and `pip install .`: install() copies the libraries into
    #   scikit-build-core's staging dir (which becomes the wheel).
    #  scikit-build-core manages the prefix, so we give a destination *relative* to it: "torchcodec/".
    if(SKBUILD_STATE STREQUAL "editable")
        # src/torchcodec, the parent of this _core dir.
        get_filename_component(
            TORCHCODEC_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
        set_target_properties(${all_libraries} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${TORCHCODEC_PACKAGE_DIR}"
            RUNTIME_OUTPUT_DIRECTORY "${TORCHCODEC_PACKAGE_DIR}"  # Windows
        )
    else()
        install(
            TARGETS ${all_libraries}
            LIBRARY DESTINATION torchcodec
            RUNTIME DESTINATION torchcodec  # For Windows
        )
    endif()

endfunction()

# Create libtorchcodec_image, the shared library holding the CPU image decoders
# (JPEG/PNG/WebP/GIF) and their custom-op registration.
#
# This is deliberately SEPARATE from the FFmpeg-linked core libraries and links
# NO FFmpeg. It is loaded at runtime via its own torch.ops.load_library call (see
# load_torchcodec_shared_libraries), which puts it in its own RTLD_LOCAL symbol
# group. That isolation is the whole point: our bundled image codec libs
# (libjpeg/libpng/libwebp) and the codec libs pulled in by the user's FFmpeg
# (which also links libpng/libjpeg/... with the same sonames) then live in
# separate scopes and cannot interpose on each other.
function(make_torchcodec_image_library)
    # This will expose the want_<codec> booleans and (for Windows) the avif
    # runtime lib path
    discover_image_codecs()

    set(image_library_name "libtorchcodec_image")
    set(image_library_sources
        ${TORCHCODEC_IMAGE_SOURCES}
        ${TORCHCODEC_IMAGE_OPS_SOURCES}
        ${TORCHCODEC_IO_SOURCES}
    )
    if(want_gif)
        list(APPEND image_library_sources ${TORCHCODEC_GIFLIB_SOURCES})
    endif()

    make_torchcodec_sublibrary(
        "${image_library_name}"
        SHARED
        "${image_library_sources}"
        "${TORCH_LIBRARIES}"
    )

    resolve_image_codec("${TORCHCODEC_BUILD_NVJPEG}" want_nvjpeg)
    if(ENABLE_CUDA AND want_nvjpeg)
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(${image_library_name} PRIVATE CUDA::nvjpeg CUDA::cudart)
        target_compile_definitions(${image_library_name} PRIVATE TORCHCODEC_ENABLE_NVJPEG=1 USE_CUDA)
        message(STATUS "Building torchcodec with nvJPEG GPU JPEG decoding/encoding support.")
    elseif(ENABLE_CUDA)
        message(STATUS "Not building torchcodec with nvJPEG support (disabled via TORCHCODEC_BUILD_NVJPEG/TORCHCODEC_BUILD_IMAGE): decode_jpeg(device='cuda') and encode_jpeg() on CUDA tensors will raise at runtime.")
    endif()

    resolve_image_codec("${TORCHCODEC_BUILD_ROCJPEG}" want_rocjpeg)
    if(ENABLE_ROCM AND want_rocjpeg)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c
                "from torch.utils.cpp_extension import ROCM_HOME; print(ROCM_HOME or '')"
            OUTPUT_VARIABLE ROCM_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(NOT ROCM_PATH)
            message(FATAL_ERROR
                "ROCM_HOME is not set. Install ROCm via the TheRock/pip-wheel layout "
                "(_rocm_sdk_core and _rocm_sdk_devel packages), which sets ROCM_HOME "
                "through torch.utils.cpp_extension.")
        endif()
        message(STATUS "Using ROCM_PATH=${ROCM_PATH}")

        # So find_package(hip) locates ${ROCM_PATH}/lib/cmake/hip/hip-config.cmake.
        list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
        find_package(hip REQUIRED)
        find_path(ROCJPEG_INCLUDE_DIR
            NAMES rocjpeg/rocjpeg.h
            HINTS "${ROCM_PATH}/include" "${ROCM_PATH}")
        find_library(ROCJPEG_LIBRARY
            NAMES rocjpeg
            HINTS "${ROCM_PATH}/lib" "${ROCM_PATH}/lib64" "${ROCM_PATH}")
        if(NOT ROCJPEG_INCLUDE_DIR OR NOT ROCJPEG_LIBRARY)
            message(FATAL_ERROR
                "rocJPEG not found under ROCM_PATH=${ROCM_PATH}. "
                "include=${ROCJPEG_INCLUDE_DIR} lib=${ROCJPEG_LIBRARY}. "
                "Install the _rocm_sdk_core and _rocm_sdk_devel pip packages "
                "(TheRock/pip-wheel layout), or set TORCHCODEC_BUILD_ROCJPEG=0 to build "
                "without it (decode_jpeg(device='cuda') will raise at runtime).")
        endif()
        target_include_directories(${image_library_name} PRIVATE ${ROCJPEG_INCLUDE_DIR})
        target_link_libraries(${image_library_name} PRIVATE ${ROCJPEG_LIBRARY} hip::host)
        target_compile_definitions(${image_library_name} PRIVATE
            TORCHCODEC_ENABLE_ROCJPEG=1 USE_ROCM __HIP_PLATFORM_AMD__)
        message(STATUS "Building torchcodec with rocJPEG GPU JPEG decoding support (${ROCJPEG_LIBRARY}).")
    elseif(ENABLE_ROCM)
        message(STATUS "Not building torchcodec with rocJPEG support (disabled via TORCHCODEC_BUILD_ROCJPEG/TORCHCODEC_BUILD_IMAGE): decode_jpeg(device='cuda') will raise at runtime.")
    endif()

    if(want_jpeg)
        target_compile_definitions(${image_library_name} PRIVATE TORCHCODEC_ENABLE_JPEG=1)
        target_link_libraries(${image_library_name} PRIVATE JPEG::JPEG)
    endif()
    if(want_png)
        target_compile_definitions(${image_library_name} PRIVATE TORCHCODEC_ENABLE_PNG=1)
        target_link_libraries(${image_library_name} PRIVATE PNG::PNG)
    endif()
    if(want_webp)
        target_compile_definitions(${image_library_name} PRIVATE TORCHCODEC_ENABLE_WEBP=1)
        # WebP::webpdemux provides the WebPAnimDecoder API used for animated
        # webp files.
        target_link_libraries(${image_library_name} PRIVATE WebP::webp WebP::webpdemux)
    endif()
    if(want_avif)
        target_compile_definitions(${image_library_name} PRIVATE TORCHCODEC_ENABLE_AVIF=1)
        target_link_libraries(${image_library_name} PRIVATE avif)
    endif()
    if(want_gif)
        target_compile_definitions(${image_library_name} PRIVATE TORCHCODEC_ENABLE_GIF=1)
    endif()

    # Hide our own (and statically linked giflib's) symbols so the only thing
    # exported is the torch op registration. Belt-and-suspenders on top of the
    # RTLD_LOCAL isolation.
    if(LINUX)
        target_compile_options(${image_library_name} PRIVATE "-fvisibility=hidden")
    endif()

    if(SKBUILD_STATE STREQUAL "editable")
        get_filename_component(
            TORCHCODEC_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
        set_target_properties(${image_library_name} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${TORCHCODEC_PACKAGE_DIR}"
            RUNTIME_OUTPUT_DIRECTORY "${TORCHCODEC_PACKAGE_DIR}"  # Windows
        )
        if(WIN32 AND want_avif)
            # Windows editable build needs the libavif DLL next to the image
            # lib. Note that the other image codec lib don't, because they are
            # already on the DLL search path. This is speculative and untested,
            # we're just trying to be nice to Windows editable builds.
            file(COPY "${avif_runtime_lib}" DESTINATION "${TORCHCODEC_PACKAGE_DIR}")
        endif()
    else()
        install(
            TARGETS ${image_library_name}
            LIBRARY DESTINATION torchcodec
            RUNTIME DESTINATION torchcodec  # For Windows
        )
    endif()
endfunction()

# Create libtorchcodec_pybind_ops, the single pybind11 module providing file-like
# support (create_file_like_context). It links NO FFmpeg (only pybind11 + torch)
# and is version-independent, so it's built once and shared by both the
# FFmpeg-free image encoders and the FFmpeg encoders/decoders.
function(make_torchcodec_pybind_library)
    set(pybind_ops_library_name "libtorchcodec_pybind_ops")
    set(pybind_ops_sources
        ${TORCHCODEC_IO_SOURCES}
        ${TORCHCODEC_FILE_LIKE_CONTEXT_SOURCES}
        ${TORCHCODEC_PYBIND_OPS_SOURCES}
    )
    make_torchcodec_sublibrary(
        "${pybind_ops_library_name}"
        MODULE # Not SHARED. On Mac,
               # MODULE becomes .so (importable), SHARED becomes .dylib.
        "${pybind_ops_sources}"
        "pybind11::module;${TORCH_LIBRARIES}"
    )

    if(WIN32)
        set_target_properties(
            ${pybind_ops_library_name} PROPERTIES SUFFIX ".pyd")
    endif()

    # See the matching comment in make_torchcodec_libraries: pybind11 hides
    # symbols, and the rest of the object code must match.
    if(LINUX)
        target_compile_options(
            ${pybind_ops_library_name} PUBLIC "-fvisibility=hidden")
    endif()

    # Must match _PYBIND_OPS_MODULE_NAME in _internally_replaced_utils.py.
    target_compile_definitions(
        ${pybind_ops_library_name}
        PRIVATE
        PYBIND_OPS_MODULE_NAME=core_pybind_ops)

    if(APPLE)
        target_link_options(
            ${pybind_ops_library_name}
            PUBLIC
            "LINKER:-undefined,dynamic_lookup")
    endif()

    if(SKBUILD_STATE STREQUAL "editable")
        get_filename_component(
            TORCHCODEC_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
        set_target_properties(${pybind_ops_library_name} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${TORCHCODEC_PACKAGE_DIR}"
            RUNTIME_OUTPUT_DIRECTORY "${TORCHCODEC_PACKAGE_DIR}"  # Windows
        )
    else()
        install(
            TARGETS ${pybind_ops_library_name}
            LIBRARY DESTINATION torchcodec
            RUNTIME DESTINATION torchcodec  # For Windows
        )
    endif()
endfunction()

# Create libtorchcodec_heic, a SEPARATE library holding just the HEIC decoder.
#
# HEIC links libheif (+ its HEVC decoder libde265), which are LGPL and therefore
# NOT bundled in our wheels: we treat libheif like FFmpeg, an optional
# user-supplied runtime dependency, and most of torchcodec should just work
# without libheif - we should only fail at runtime if it can't be found *and* if
# it's needed.
function(make_torchcodec_heic_library)
    set(heic_library_name "libtorchcodec_heic")
    set(heic_library_sources
        ${TORCHCODEC_HEIC_SOURCES}
        ${TORCHCODEC_HEIC_OPS_SOURCES}
    )

    make_torchcodec_sublibrary(
        "${heic_library_name}"
        SHARED
        "${heic_library_sources}"
        "${TORCH_LIBRARIES}"
    )

    resolve_image_codec("${TORCHCODEC_BUILD_HEIC}" want_heic)
    if(want_heic)
        # conda-forge's libheif ships a CMake package config (libheif-config.cmake)
        # exposing the `heif` imported target. Fall back to pkg-config otherwise.
        if(NOT DEFINED libheif_ROOT AND DEFINED ENV{CONDA_PREFIX})
            if(WIN32)
                set(libheif_ROOT "$ENV{CONDA_PREFIX}/Library")
            else()
                set(libheif_ROOT "$ENV{CONDA_PREFIX}")
            endif()
        endif()
        set(heic_found FALSE)
        find_package(libheif CONFIG QUIET)
        if(libheif_FOUND)
            set(heic_found TRUE)
            target_link_libraries(${heic_library_name} PRIVATE heif)
        else()
            find_package(PkgConfig QUIET)
            if(PkgConfig_FOUND)
                pkg_check_modules(HEIF QUIET IMPORTED_TARGET libheif)
                if(HEIF_FOUND)
                    set(heic_found TRUE)
                    target_link_libraries(${heic_library_name} PRIVATE PkgConfig::HEIF)
                endif()
            endif()
        endif()

        if(heic_found)
            target_compile_definitions(${heic_library_name} PRIVATE TORCHCODEC_ENABLE_HEIC=1)
            message(STATUS "libheif found: building torchcodec with HEIC image decoding support.")
        else()
            message(FATAL_ERROR
                "libheif not found, but HEIC image decoding is enabled. Install "
                "libheif and its development headers (e.g. `conda install -c "
                "conda-forge libheif`), or set TORCHCODEC_BUILD_HEIC=0 to build "
                "without HEIC support (decode_heic will raise at runtime), or "
                "TORCHCODEC_BUILD_IMAGE=0 to disable all image decoders.")
        endif()
    else()
        message(STATUS "Not building torchcodec with HEIC image decoding support (disabled via TORCHCODEC_BUILD_HEIC/TORCHCODEC_BUILD_IMAGE): decode_heic will raise at runtime.")
    endif()

    # Hide everything except the torch op registration, like the image library.
    if(LINUX)
        target_compile_options(${heic_library_name} PRIVATE "-fvisibility=hidden")
    endif()

    if(SKBUILD_STATE STREQUAL "editable")
        get_filename_component(
            TORCHCODEC_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
        set_target_properties(${heic_library_name} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${TORCHCODEC_PACKAGE_DIR}"
            RUNTIME_OUTPUT_DIRECTORY "${TORCHCODEC_PACKAGE_DIR}"  # Windows
        )
    else()
        install(
            TARGETS ${heic_library_name}
            LIBRARY DESTINATION torchcodec
            RUNTIME DESTINATION torchcodec  # For Windows
        )
    endif()
endfunction()

make_torchcodec_image_library()
make_torchcodec_pybind_library()
make_torchcodec_heic_library()

if(DEFINED ENV{BUILD_AGAINST_ALL_FFMPEG_FROM_S3})
    message(
        STATUS
        "Building and dynamically linking libtorchcodec against our pre-built
        non-GPL FFmpeg libraries. These libraries are only used at build time,
        you still need a different FFmpeg to be installed for run time!"
    )

    # This will expose the torchcodec::ffmpeg{N} (N=4,5,6,7,8) targets
    include(
        ${CMAKE_CURRENT_SOURCE_DIR}/fetch_and_expose_non_gpl_ffmpeg_libs.cmake
    )

    make_torchcodec_libraries(8 torchcodec::ffmpeg8)
    make_torchcodec_libraries(7 torchcodec::ffmpeg7)
    make_torchcodec_libraries(6 torchcodec::ffmpeg6)
    make_torchcodec_libraries(4 torchcodec::ffmpeg4)
    make_torchcodec_libraries(5 torchcodec::ffmpeg5)
else()
    message(
        STATUS
        "Building and dynamically linking libtorchcodec against the installed
        FFmpeg libraries. This require pkg-config to be installed. If you have
        installed FFmpeg from conda, make sure pkg-config is installed from
        conda as well."
    )

    # This will expose `add_ffmpeg_target_with_pkg_config`
    include("${CMAKE_CURRENT_SOURCE_DIR}/../share/cmake/TorchCodec/ffmpeg_versions.cmake")

    add_ffmpeg_target_with_pkg_config(ffmpeg_major_version)
    make_torchcodec_libraries(${ffmpeg_major_version} torchcodec::ffmpeg${ffmpeg_major_version})

    # Expose these values updwards so that the test compilation does not need
    # to re-figure it out. FIXME: it's not great that we just copy-paste the
    # library names.
    set(libtorchcodec_library_name "libtorchcodec_core${ffmpeg_major_version}" PARENT_SCOPE)
    set(libtorchcodec_custom_ops_name "libtorchcodec_custom_ops${ffmpeg_major_version}" PARENT_SCOPE)
    set(libav_include_dirs ${LIBAV_INCLUDE_DIRS} PARENT_SCOPE)
endif()
