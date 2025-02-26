cmake_minimum_required(VERSION 3.18)
project(TorchCodec)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
find_package(Python3 ${PYTHON_VERSION} EXACT COMPONENTS Development)

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

if(ENABLE_CUDA)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_CUDA")
endif()
if(ENABLE_XPU)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(L0 REQUIRED IMPORTED_TARGET level-zero)
    pkg_check_modules(LIBVA REQUIRED IMPORTED_TARGET libva)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_XPU")
endif()

function(make_torchcodec_sublibrary
    library_name
    type
    sources
    library_dependencies)

    add_library(${library_name} ${type} ${sources})
    set_target_properties(${library_name} PROPERTIES CXX_STANDARD 17)
    target_include_directories(${library_name}
        PRIVATE
        ./../../../
        "${TORCH_INSTALL_PREFIX}/include"
        ${Python3_INCLUDE_DIRS}
    )

    # Avoid adding the "lib" prefix which we already add explicitly.
    set_target_properties(${library_name} PROPERTIES PREFIX "")

    if(WIN32)
        # On Windows, the built artifacts are put in Release/Debug
        # subdirectories by default. We want to avoid that, otherwise our
        # install() step would not know where to find those.
        set_target_properties(${library_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}
            RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}
            LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}
            LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}
            ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}
            ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}
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
    #    Linux, {ext} is so. On Mac, it is dylib.
    #
    # 2. libtorchcodec_custom_opsN.{ext}: Implementation of the PyTorch custom
    #    ops. Depends on libtorchcodec_coreN.{ext}. On Linux, {ext} is so.
    #    On Mac, it is dylib.
    #
    # 3. libtorchcodec_pybind_opsN.{ext}: Implementation of the pybind11 ops. We
    #    keep these separate from the PyTorch custom ops because we have to
    #    load these libraries separately on the Python side. Depends on
    #    libtorchcodec_coreN.{ext}. On BOTH Linux and Mac {ext} is so.

    # 1. Create libtorchcodec_coreN.{ext}.
    set(core_library_name "libtorchcodec_core${ffmpeg_major_version}")
    set(core_sources
        AVIOContextHolder.cpp
        AVIOTensorContext.cpp
        FFMPEGCommon.cpp
        Frame.cpp
        DeviceInterface.cpp
        CpuDeviceInterface.cpp
        SingleStreamDecoder.cpp
        Encoder.cpp
    )

    if(ENABLE_CUDA)
        list(APPEND core_sources CudaDeviceInterface.cpp)
    endif()

    if(ENABLE_XPU)
        list(APPEND core_sources XpuDeviceInterface.cpp)
    endif()

    set(core_library_dependencies
        ${ffmpeg_target}
        ${TORCH_LIBRARIES}
    )

    if(ENABLE_CUDA)
        list(APPEND core_library_dependencies
            ${CUDA_nppi_LIBRARY}
            ${CUDA_nppicc_LIBRARY}
        )
    endif()

    if(ENABLE_XPU)
        list(APPEND core_library_dependencies
            PkgConfig::L0 PkgConfig::LIBVA)
    endif()

    make_torchcodec_sublibrary(
        "${core_library_name}"
        SHARED
        "${core_sources}"
        "${core_library_dependencies}"
    )

    # 2. Create libtorchcodec_custom_opsN.{ext}.
    set(custom_ops_library_name "libtorchcodec_custom_ops${ffmpeg_major_version}")
    set(custom_ops_sources
        AVIOTensorContext.cpp
        custom_ops.cpp
    )
    set(custom_ops_dependencies
        ${core_library_name}
        ${Python3_LIBRARIES}
    )
    make_torchcodec_sublibrary(
        "${custom_ops_library_name}"
        SHARED
        "${custom_ops_sources}"
        "${custom_ops_dependencies}"
    )

    # 3. Create libtorchcodec_pybind_opsN.so.
    set(pybind_ops_library_name "libtorchcodec_pybind_ops${ffmpeg_major_version}")
    set(pybind_ops_sources
        AVIOFileLikeContext.cpp
        pybind_ops.cpp
    )
    set(pybind_ops_dependencies
       ${core_library_name}
       pybind11::module # This library dependency makes sure we have the right
                        # Python libraries included as well as all of the right
                        # settings so that we can successfully load the shared
                        # library as a Python module on Mac. If we instead use
                        # ${Python3_LIBRARIES}, it works on Linux but not on
                        # Mac.
    )
    make_torchcodec_sublibrary(
        "${pybind_ops_library_name}"
        MODULE # Note that this not SHARED; otherwise we build the wrong kind
               # of library on Mac. On Mac, SHARED becomes .dylib and MODULE becomes
               # a .so. We want pybind11 libraries to become .so. If this is
               # changed to SHARED, we will be able to succesfully compile a
               # .dylib, but we will not be able to succesfully import that as
               # a Python module on Mac.
        "${pybind_ops_sources}"
        "${pybind_ops_dependencies}"
    )
    # pybind11 limits the visibility of symbols in the shared library to prevent
    # stray initialization of py::objects. The rest of the object code must
    # match. See:
    #   https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes
    if(NOT WIN32)
        target_compile_options(
            ${pybind_ops_library_name}
            PUBLIC
            "-fvisibility=hidden"
        )
    endif()

    # The value we use here must match the value we return from
    # _get_pybind_ops_module_name() on the Python side. If the values do not
    # match, then we will be unable to import the C++ shared library as a
    # Python module at runtime.
    target_compile_definitions(
        ${pybind_ops_library_name}
        PRIVATE
        PYBIND_OPS_MODULE_NAME=core_pybind_ops
    )

    if(APPLE)
        # If we don't make sure this flag is set, we run into segfauls at import
        # time on Mac. See:
        # https://github.com/pybind/pybind11/issues/3907#issuecomment-1170412764
        target_link_options(
            ${pybind_ops_library_name}
            PUBLIC
            "LINKER:-undefined,dynamic_lookup"
        )
    endif()

    # Install all libraries.
    set(
        all_libraries
        ${core_library_name}
        ${custom_ops_library_name}
        ${pybind_ops_library_name}
    )

    # The install step is invoked within CMakeBuild.build_library() in
    # setup.py and just copies the built files from the temp
    # cmake/setuptools build folder into the CMAKE_INSTALL_PREFIX folder. We
    # still need to manually pass "DESTINATION ..." for cmake to copy those
    # files in CMAKE_INSTALL_PREFIX instead of CMAKE_INSTALL_PREFIX/lib.
    install(
        TARGETS ${all_libraries}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}  # For Windows DLLs
    )

endfunction()

if(DEFINED ENV{BUILD_AGAINST_ALL_FFMPEG_FROM_S3})
    message(
        STATUS
        "Building and dynamically linking libtorchcodec against our pre-built
        non-GPL FFmpeg libraries. These libraries are only used at build time,
        you still need a different FFmpeg to be installed for run time!"
    )

    # This will expose the ffmpeg4, ffmpeg5, ffmpeg6, and ffmpeg7 targets
    include(
        ${CMAKE_CURRENT_SOURCE_DIR}/fetch_and_expose_non_gpl_ffmpeg_libs.cmake
    )

    make_torchcodec_libraries(7 ffmpeg7)
    make_torchcodec_libraries(6 ffmpeg6)
    make_torchcodec_libraries(4 ffmpeg4)
    make_torchcodec_libraries(5 ffmpeg5)
else()
    message(
        STATUS
        "Building and dynamically linking libtorchcodec against the installed
        FFmpeg libraries. This require pkg-config to be installed. If you have
        installed FFmpeg from conda, make sure pkg-config is installed from
        conda as well."
    )
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
        libavdevice
        libavfilter
        libavformat
        libavcodec
        libavutil
        libswresample
        libswscale
    )

    # Split libavcodec's version string by '.' and convert it to a list
    string(REPLACE "." ";" libavcodec_version_list ${LIBAV_libavcodec_VERSION})
    # Get the first element of the list, which is the major version
    list(GET libavcodec_version_list 0 libavcodec_major_version)

    if (${libavcodec_major_version} STREQUAL "58")
        set(ffmpeg_major_version "4")
    elseif (${libavcodec_major_version} STREQUAL "59")
        set(ffmpeg_major_version "5")
    elseif (${libavcodec_major_version} STREQUAL "60")
        set(ffmpeg_major_version "6")
    elseif (${libavcodec_major_version} STREQUAL "61")
        set(ffmpeg_major_version "7")
    else()
        message(
            FATAL_ERROR
            "Unsupported libavcodec version: ${libavcodec_major_version}"
        )
    endif()

    make_torchcodec_libraries(${ffmpeg_major_version} PkgConfig::LIBAV)

    # Expose these values updwards so that the test compilation does not need
    # to re-figure it out. FIXME: it's not great that we just copy-paste the
    # library names.
    set(libtorchcodec_library_name "libtorchcodec_core${ffmpeg_major_version}" PARENT_SCOPE)
    set(libtorchcodec_custom_ops_name "libtorchcodec_custom_ops${ffmpeg_major_version}" PARENT_SCOPE)
    set(libav_include_dirs ${LIBAV_INCLUDE_DIRS} PARENT_SCOPE)
endif()
