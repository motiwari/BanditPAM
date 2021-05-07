# --------------- runs clang-format in place using the style file ---------------

if(PROJECT_SOURCE_DIR AND CLANG_FORMAT)
    # get C++ sources file list (ignoring packages)
    file(GLOB_RECURSE ALL_SOURCE_FILES
            ${PROJECT_SOURCE_DIR}/include/**.h
            ${PROJECT_SOURCE_DIR}/tests/src/**.cpp
            ${PROJECT_SOURCE_DIR}/tests/src/**.h
            ${PROJECT_SOURCE_DIR}/examples/src/**.cpp
            ${PROJECT_SOURCE_DIR}/examples/src/**.h
            )

    # apply style to the file list
    foreach(SOURCE_FILE ${ALL_SOURCE_FILES})
        execute_process(COMMAND "${CLANG_FORMAT}" -style=file -verbose -i "${SOURCE_FILE}")
    endforeach()
endif()
