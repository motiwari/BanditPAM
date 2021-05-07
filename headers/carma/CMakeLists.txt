CMAKE_MINIMUM_REQUIRED(VERSION 3.12)

SET(CARMA_VERSION_INFO "1.0.0")

PROJECT(
    carma
    VERSION ${CARMA_VERSION_INFO}
    LANGUAGES CXX
)

SET(CMAKE_MODULE_PATH_SAVED ${CMAKE_MODULE_PATH})
LIST(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/cmake")
# ##############################################################################
#                                   SETTINGS                                   #
# ##############################################################################
SET(CMAKE_CXX_STANDARD 14)
SET(DEFAULT_BUILD_TYPE "Release")
# Set build type to Release if not specified
INCLUDE(BuildType)
INCLUDE(ReleaseSettings)

OPTION(DEV_MODE "Enable additional definitions for development" OFF)
OPTION(DEV_DEBUG_MODE "Turn on all debugging definitions" OFF)
OPTION(BUILD_TESTS "Build tests for carma" OFF)
OPTION(BUILD_EXAMPLES "Build examples for carma" OFF)

# collect user settable options
INCLUDE(${PROJECT_SOURCE_DIR}/carma_config.cmake)
INCLUDE(ParseOptions)
CONFIGURE_FILE(
    ${PROJECT_SOURCE_DIR}/include/carma_bits/config.h.in
    ${PROJECT_SOURCE_DIR}/include/carma_bits/config.h
    @ONLY
)
# ##############################################################################
#                                REQUIREMENTS                                  #
# ##############################################################################
FIND_PACKAGE(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# Needed to ensure the same Python executable is found by Pybind11
IF (NOT DEFINED PYTHON_EXECUTABLE)
    SET(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
ENDIF ()
# ##############################################################################
#                                   TARGET                                     #
# ##############################################################################
ADD_LIBRARY(carma INTERFACE)

TARGET_COMPILE_DEFINITIONS(carma INTERFACE VERSION_INFO=${CARMA_VERSION_INFO})

TARGET_LINK_LIBRARIES(carma
    INTERFACE
    armadillo
    pybind11::headers
    Python3::Module
    Python3::NumPy
)
TARGET_INCLUDE_DIRECTORIES(carma
    INTERFACE
    # path of the headers after installation
    $<INSTALL_INTERFACE:include> # <prefix>/include/carma
    # path of the headers before installation
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

INSTALL(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include" 
    DESTINATION carma
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "carma"
)
# ##############################################################################
#                              TEST TARGETS                                    #
# ##############################################################################
IF (BUILD_TESTS OR BUILD_EXAMPLES)
    # optionally build custom targets for our third_party libs
    INCLUDE(ThirdPartyTargets)
ENDIF()

IF(BUILD_TESTS)
    INCLUDE(CTest)
    ENABLE_TESTING()
    ADD_SUBDIRECTORY(tests)
    ADD_SUBDIRECTORY(third_party/Catch2)
    INCLUDE(ClangFormatTarget)
    INCLUDE(CoverageTarget)
ENDIF()

IF(BUILD_EXAMPLES)
    ADD_SUBDIRECTORY(examples)
ENDIF()
#------------------------------------------------------
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH_SAVED})
UNSET(CMAKE_MODULE_PATH_SAVED)
