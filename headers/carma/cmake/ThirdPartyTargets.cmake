# ##############################################################################
#                             THIRD PARTY LIBS                                 #
# ##############################################################################
# Use a header only version of armadillo; default CMake project is not header only
# That's why we use a custom 'armadillo' target with only include directory
IF(NOT TARGET armadillo)
    IF (ARMADILLO_ROOT_DIR)
        #add_subdirectory(${ARMADILLO_ROOT_DIR} armadillo-code)
        SET(ARMADILLO_INCLUDE_DIR ${ARMADILLO_ROOT_DIR}/include)
    ELSE()
        #add_subdirectory(third_party/armadillo-code)
        SET(ARMADILLO_INCLUDE_DIR third_party/armadillo-code/include)
    ENDIF()
    ADD_LIBRARY(armadillo INTERFACE)
    TARGET_INCLUDE_DIRECTORIES(armadillo INTERFACE ${ARMADILLO_INCLUDE_DIR})
ENDIF()

IF(NOT TARGET pybind11)
    IF (PYTHON_PREFIX_PATH)
        SET(CMAKE_PREFIX_PATH_SAVED ${CMAKE_PREFIX_PATH})
        LIST(APPEND CMAKE_PREFIX_PATH ${PYTHON_PREFIX_PATH})
    ENDIF()
    IF (PYBIND11_ROOT_DIR)
        ADD_SUBDIRECTORY(${PYBIND11_ROOT_DIR} pybind11)
    ELSE()
        ADD_SUBDIRECTORY(third_party/pybind11)
    ENDIF()
    IF (CMAKE_PREFIX_PATH_SAVED)
        SET(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH_SAVED})
        UNSET(CMAKE_PREFIX_PATH_SAVED)
    ENDIF()
ENDIF()
