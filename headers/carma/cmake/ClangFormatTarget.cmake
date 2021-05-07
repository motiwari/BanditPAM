# ##############################################################################
#                               Clang-format                                   #
# ##############################################################################
## search for clang-format and add target
IF(NOT DEFINED CARMA_DEV_TARGET OR CARMA_DEV_TARGET)
    FIND_PROGRAM(CLANG_FORMAT clang-format)
    IF (CLANG_FORMAT)
        EXEC_PROGRAM(
            ${CLANG_FORMAT} ARGS -version OUTPUT_VARIABLE CLANG_FORMAT_RAW_VERSION
        )
        STRING(REGEX MATCH "[1-9][0-9]*\\.[0-9]+\\.[0-9]+"
                CLANG_FORMAT_VERSION ${CLANG_FORMAT_RAW_VERSION})
        IF (CLANG_FORMAT_VERSION VERSION_GREATER_EQUAL "6.0.0") 
            ADD_CUSTOM_TARGET(clang-format
                COMMAND echo "running ${CLANG_FORMAT} ..."
                COMMAND ${CMAKE_COMMAND}
                -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
                -DCLANG_FORMAT="${CLANG_FORMAT}"
                -P ${PROJECT_SOURCE_DIR}/cmake/ClangFormatProcess.cmake
            )
            MESSAGE(STATUS "clang-format target for updating code format is available")
        ELSE ()
            MESSAGE(WARNING "incompatible clang-format found (<6.0.0); clang-format target is not available.")
            ADD_CUSTOM_TARGET(clang-format
                COMMAND ${CMAKE_COMMAND} -E echo ""
                COMMAND ${CMAKE_COMMAND} -E echo "*** code formatting not available since clang-format version is incompatible ***"
                COMMAND ${CMAKE_COMMAND} -E echo ""
                COMMENT "Inform about not available code format."
            )
        ENDIF()
    ELSE ()
        MESSAGE(WARNING "clang-format no found; clang-format target is not available.")
        ADD_CUSTOM_TARGET(clang-format
            COMMAND ${CMAKE_COMMAND} -E echo ""
            COMMAND ${CMAKE_COMMAND} -E echo "*** code formatting not available since clang-format has not been found ***"
            COMMAND ${CMAKE_COMMAND} -E echo ""
            COMMENT "Inform about not available code format."
        )
    ENDIF ()
ENDIF ()
