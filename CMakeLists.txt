# Enhanced CMakeLists.txt for BanditPAM with improved cross-platform support
cmake_minimum_required(VERSION 3.17)
project(BanditPAM VERSION 6.0.3 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Platform detection
if(WIN32)
    set(BANDITPAM_WINDOWS TRUE)
elseif(APPLE)
    set(BANDITPAM_MACOS TRUE)
else()
    set(BANDITPAM_LINUX TRUE)
endif()

# Detect if we're in GitHub Actions
if(DEFINED ENV{GITHUB_ACTIONS})
    set(GITHUB_ACTIONS TRUE)
    message(STATUS "Building in GitHub Actions environment")
else()
    set(GITHUB_ACTIONS FALSE)
endif()

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TESTS "Build tests" ON)
option(USE_OPENMP "Enable OpenMP support" ON)
option(BANDITPAM_VERBOSE "Enable verbose output" OFF)

# Verbose output helper
macro(banditpam_message msg)
    if(BANDITPAM_VERBOSE OR CMAKE_VERBOSE_MAKEFILE)
        message(STATUS "[BanditPAM] ${msg}")
    endif()
endmacro()

banditpam_message("Configuring BanditPAM ${PROJECT_VERSION}")
banditpam_message("Build type: ${CMAKE_BUILD_TYPE}")
banditpam_message("Platform: ${CMAKE_SYSTEM_NAME}")

# Find required dependencies
banditpam_message("Finding dependencies...")

# Find Armadillo
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(ARMADILLO armadillo)
endif()

if(NOT ARMADILLO_FOUND)
    find_package(Armadillo QUIET)
    if(Armadillo_FOUND)
        set(ARMADILLO_FOUND TRUE)
        set(ARMADILLO_INCLUDE_DIRS ${ARMADILLO_INCLUDE_DIRS})
        set(ARMADILLO_LIBRARIES ${ARMADILLO_LIBRARIES})
    endif()
endif()

# Manual Armadillo detection if not found
if(NOT ARMADILLO_FOUND)
    banditpam_message("Armadillo not found via CMake, trying manual detection...")

    # Search for Armadillo headers
    find_path(ARMADILLO_INCLUDE_DIR
        NAMES armadillo_bits/config.hpp armadillo
        PATHS
            /usr/include
            /usr/local/include
            /opt/homebrew/include  # M1 Mac
            /usr/local/Cellar/armadillo/*/include  # Intel Mac
            ${CMAKE_PREFIX_PATH}/include
            $ENV{VCPKG_ROOT}/installed/x64-windows/include
        PATH_SUFFIXES armadillo
    )

    # Search for Armadillo library
    find_library(ARMADILLO_LIBRARY
        NAMES armadillo
        PATHS
            /usr/lib
            /usr/local/lib
            /opt/homebrew/lib
            /usr/local/Cellar/armadillo/*/lib
            ${CMAKE_PREFIX_PATH}/lib
            $ENV{VCPKG_ROOT}/installed/x64-windows/lib
    )

    if(ARMADILLO_INCLUDE_DIR AND ARMADILLO_LIBRARY)
        set(ARMADILLO_FOUND TRUE)
        set(ARMADILLO_INCLUDE_DIRS ${ARMADILLO_INCLUDE_DIR})
        set(ARMADILLO_LIBRARIES ${ARMADILLO_LIBRARY})
        banditpam_message("Found Armadillo manually:")
        banditpam_message("  Include: ${ARMADILLO_INCLUDE_DIRS}")
        banditpam_message("  Library: ${ARMADILLO_LIBRARIES}")
    endif()
endif()

if(NOT ARMADILLO_FOUND)
    message(WARNING "Armadillo not found. Please install Armadillo:")
    if(BANDITPAM_LINUX)
        message(STATUS "  Ubuntu/Debian: sudo apt-get install libarmadillo-dev")
        message(STATUS "  CentOS/RHEL: sudo yum install armadillo-devel")
    elseif(BANDITPAM_MACOS)
        message(STATUS "  macOS: brew install armadillo")
    elseif(BANDITPAM_WINDOWS)
        message(STATUS "  Windows: vcpkg install armadillo:x64-windows")
    endif()
    if(NOT GITHUB_ACTIONS)
        message(FATAL_ERROR "Armadillo is required but not found")
    endif()
endif()

# OpenMP detection
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        banditpam_message("Found OpenMP: ${OpenMP_CXX_VERSION}")
    else()
        banditpam_message("OpenMP not found, building without parallel support")
        if(BANDITPAM_MACOS)
            message(STATUS "On macOS, install with: brew install libomp")
        endif()
    endif()
endif()

# BLAS/LAPACK (usually comes with Armadillo)
find_package(BLAS QUIET)
find_package(LAPACK QUIET)

if(BLAS_FOUND)
    banditpam_message("Found BLAS: ${BLAS_LIBRARIES}")
endif()
if(LAPACK_FOUND)
    banditpam_message("Found LAPACK: ${LAPACK_LIBRARIES}")
endif()

# Python bindings setup
if(BUILD_PYTHON_BINDINGS)
    banditpam_message("Setting up Python bindings...")

    # Find Python
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development QUIET)

    if(NOT Python_FOUND)
        # Fallback to older method
        find_package(PythonInterp 3 REQUIRED)
        find_package(PythonLibs 3 REQUIRED)
        set(Python_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS})
        set(Python_LIBRARIES ${PYTHON_LIBRARIES})
        set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
    endif()

    # Find pybind11
    find_package(pybind11 QUIET)

    if(NOT pybind11_FOUND)
        # Try to find pybind11 via Python
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE PYBIND11_CMAKE_RESULT
        )

        if(PYBIND11_CMAKE_RESULT EQUAL 0 AND EXISTS ${PYBIND11_CMAKE_DIR})
            find_package(pybind11 PATHS ${PYBIND11_CMAKE_DIR})
        endif()
    endif()

    if(pybind11_FOUND)
        banditpam_message("Found pybind11: ${pybind11_VERSION}")
    else()
        message(WARNING "pybind11 not found. Install with: pip install pybind11")
        if(NOT GITHUB_ACTIONS)
            message(FATAL_ERROR "pybind11 is required for Python bindings")
        endif()
    endif()

    # Find NumPy
    if(Python_EXECUTABLE)
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
            OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE NUMPY_RESULT
        )

        if(NUMPY_RESULT EQUAL 0)
            banditpam_message("Found NumPy include: ${NUMPY_INCLUDE_DIR}")
        else()
            message(WARNING "NumPy not found. Install with: pip install numpy")
        endif()
    endif()
endif()

# CARMA submodule check
if(BUILD_PYTHON_BINDINGS)
    set(CARMA_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/headers/carma/include")
    if(EXISTS ${CARMA_INCLUDE_DIR})
        banditpam_message("Found CARMA submodule")
    else()
        message(WARNING "CARMA submodule not found. Run: git submodule update --init --recursive")
        if(NOT GITHUB_ACTIONS)
            message(FATAL_ERROR "CARMA submodule is required")
        endif()
    endif()
endif()

# Compiler-specific flags
banditpam_message("Setting up compiler flags...")

if(MSVC)
    # Microsoft Visual C++
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /std:c++17")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi /DEBUG")

    # Disable specific warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996 /wd4244 /wd4267 /wd4305")

else()
    # GCC/Clang
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fPIC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -Wall -Wextra")

    # Platform-specific flags
    if(BANDITPAM_MACOS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
        if(NOT GITHUB_ACTIONS)
            # Only use -march=native in local builds, not CI
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
        endif()
    elseif(BANDITPAM_LINUX)
        if(NOT GITHUB_ACTIONS)
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
        endif()
    endif()
endif()

# Add OpenMP flags
if(USE_OPENMP AND OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Source files
set(BANDITPAM_SOURCES
    src/algorithms/kmedoids_algorithm.cpp
    src/algorithms/pam.cpp
    src/algorithms/banditpam.cpp
    src/algorithms/banditpam_orig.cpp
    src/algorithms/fastpam1.cpp
)

# Create core library
add_library(BanditPAM_LIB STATIC ${BANDITPAM_SOURCES})

target_include_directories(BanditPAM_LIB PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/headers/algorithms
    ${ARMADILLO_INCLUDE_DIRS}
)

target_link_libraries(BanditPAM_LIB PUBLIC
    ${ARMADILLO_LIBRARIES}
)

if(USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(BanditPAM_LIB PUBLIC OpenMP::OpenMP_CXX)
endif()

# Main executable
add_executable(BanditPAM src/main.cpp)
target_link_libraries(BanditPAM PUBLIC BanditPAM_LIB)

# Python bindings
if(BUILD_PYTHON_BINDINGS AND pybind11_FOUND)
    banditpam_message("Building Python bindings...")

    set(PYTHON_BINDING_SOURCES
        src/python_bindings/kmedoids_pywrapper.cpp
        src/python_bindings/predict_python.cpp
        src/python_bindings/sparse_support_python.cpp
    )

    pybind11_add_module(banditpam_python ${PYTHON_BINDING_SOURCES})

    target_include_directories(banditpam_python PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/headers/python_bindings
        ${CARMA_INCLUDE_DIR}
        ${NUMPY_INCLUDE_DIR}
    )

    target_link_libraries(banditpam_python PRIVATE BanditPAM_LIB)

    # Set properties for Python module
    set_target_properties(banditpam_python PROPERTIES
        OUTPUT_NAME banditpam
        PREFIX ""
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )

    if(BANDITPAM_MACOS)
        set_target_properties(banditpam_python PROPERTIES
            LINK_FLAGS "-undefined dynamic_lookup"
        )
    endif()
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()

    # Add test subdirectory if it exists
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        add_subdirectory(tests)
    endif()
endif()

# Installation
install(TARGETS BanditPAM DESTINATION bin)

if(TARGET banditpam_python)
    install(TARGETS banditpam_python
        DESTINATION ${Python_SITEARCH}
        COMPONENT python
    )
endif()

# Summary
banditpam_message("Configuration Summary:")
banditpam_message("  Version: ${PROJECT_VERSION}")
banditpam_message("  Build Type: ${CMAKE_BUILD_TYPE}")
banditpam_message("  C++ Standard: ${CMAKE_CXX_STANDARD}")
banditpam_message("  Armadillo: ${ARMADILLO_FOUND}")
banditpam_message("  OpenMP: ${OpenMP_CXX_FOUND}")
banditpam_message("  Python Bindings: ${BUILD_PYTHON_BINDINGS}")
banditpam_message("  Tests: ${BUILD_TESTS}")

if(BUILD_PYTHON_BINDINGS)
    banditpam_message("  Python: ${Python_VERSION}")
    banditpam_message("  pybind11: ${pybind11_FOUND}")
    banditpam_message("  NumPy: ${NUMPY_RESULT EQUAL 0}")
endif()