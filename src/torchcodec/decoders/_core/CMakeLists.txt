cmake_minimum_required(VERSION 3.18)
project(TorchCodec)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Werror ${TORCH_CXX_FLAGS}")
find_package(Python3 ${PYTHON_VERSION} EXACT COMPONENTS Development)

function(make_torchcodec_sublibrary
    library_name
    sources
    library_dependencies)

    add_library(${library_name} SHARED ${sources})
    set_target_properties(${library_name} PROPERTIES CXX_STANDARD 17)
    target_include_directories(${library_name}
        PRIVATE
        ./../../../../
        "${TORCH_INSTALL_PREFIX}/include"
        ${Python3_INCLUDE_DIRS}
    )

    # Avoid adding the "lib" prefix which we already add explicitly.
    set_target_properties(${library_name} PROPERTIES PREFIX "")

    target_link_libraries(
        ${library_name}
        PUBLIC
        ${library_dependencies}
    )
endfunction()

function(make_torchcodec_libraries
    ffmpeg_major_version
    ffmpeg_target)

    # TODO: List each library and its purpose.

    # Create libtorchcodec_decoderN.so
    set(decoder_library_name "libtorchcodec_decoder${ffmpeg_major_version}")
    set(decoder_sources
      AVIOContextHolder.cpp
      FFMPEGCommon.cpp
      VideoDecoder.cpp
    )

    if(ENABLE_CUDA)
        list(APPEND decoder_sources CudaDevice.cpp)
    else()
        list(APPEND decoder_sources CPUOnlyDevice.cpp)
    endif()

    set(decoder_library_dependencies
        ${ffmpeg_target}
        ${TORCH_LIBRARIES}
        ${Python3_LIBRARIES}
    )

    if(ENABLE_CUDA)
        list(APPEND decoder_library_dependencies
            ${CUDA_nppi_LIBRARY}
            ${CUDA_nppicc_LIBRARY}
        )
    endif()

    make_torchcodec_sublibrary(
        "${decoder_library_name}"
        "${decoder_sources}"
        "${decoder_library_dependencies}"
    )

    # Create libtorchcodec_custom_opsN.so
    set(custom_ops_library_name "libtorchcodec_custom_ops${ffmpeg_major_version}")
    set(custom_ops_sources
      AVIOBytesContext.cpp
      VideoDecoderOps.cpp
    )
    make_torchcodec_sublibrary(
        "${custom_ops_library_name}"
        "${custom_ops_sources}"
        "${decoder_library_name}"
    )

    # Create libtorchcodec_pybind_opsN.so
    set(pybind_ops_library_name "libtorchcodec_pybind_ops${ffmpeg_major_version}")
    set(pybind_ops_sources
      AVIOFileLikeContext.cpp
      PyBindOps.cpp
    )
    make_torchcodec_sublibrary(
        "${pybind_ops_library_name}"
        "${pybind_ops_sources}"
        "${decoder_library_name}"
    )
    # pybind11 quirk, see:
    #   https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes
    target_compile_options(
      ${pybind_ops_library_name}
      PUBLIC
      "-fvisibility=hidden"
    )

    # Install all libraries.
    set(
        all_libraries
        ${decoder_library_name}
        ${custom_ops_library_name}
        ${pybind_ops_library_name}
    )

    # The install step is invoked within CMakeBuild.build_library() in
    # setup.py and just copies the built .so files from the temp
    # cmake/setuptools build folder into the CMAKE_INSTALL_PREFIX folder. We
    # still need to manually pass "DESTINATION ..." for cmake to copy those
    # files in CMAKE_INSTALL_PREFIX instead of CMAKE_INSTALL_PREFIX/lib.
    install(
        TARGETS ${all_libraries}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
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
    set(libtorchcodec_library_name "libtorchcodec_decoder${ffmpeg_major_version}" PARENT_SCOPE)
    set(libtorchcodec_custom_ops_name "libtorchcodec_custom_ops${ffmpeg_major_version}" PARENT_SCOPE)
    set(libav_include_dirs ${LIBAV_INCLUDE_DIRS} PARENT_SCOPE)
endif()
