set(FlexiBLAS_NAMES)
set(FlexiBLAS_NAMES ${FlexiBLAS_NAMES} flexiblas)

set(FlexiBLAS_TMP_LIBRARY)
set(FlexiBLAS_TMP_LIBRARIES)


foreach (FlexiBLAS_NAME ${FlexiBLAS_NAMES})
  find_library(${FlexiBLAS_NAME}_LIBRARY
    NAMES ${FlexiBLAS_NAME}
    PATHS ${CMAKE_SYSTEM_LIBRARY_PATH} /lib64 /lib /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib /opt/local/lib64 /opt/local/lib
    )
  
  set(FlexiBLAS_TMP_LIBRARY ${${FlexiBLAS_NAME}_LIBRARY})
  
  if(FlexiBLAS_TMP_LIBRARY)
    set(FlexiBLAS_TMP_LIBRARIES ${FlexiBLAS_TMP_LIBRARIES} ${FlexiBLAS_TMP_LIBRARY})
  endif()
endforeach()


# use only one library

if(FlexiBLAS_TMP_LIBRARIES)
  list(GET FlexiBLAS_TMP_LIBRARIES 0 FlexiBLAS_LIBRARY)
endif()


if(FlexiBLAS_LIBRARY)
  set(FlexiBLAS_LIBRARIES ${FlexiBLAS_LIBRARY})
  set(FlexiBLAS_FOUND "YES")
else()
  set(FlexiBLAS_FOUND "NO")
endif()


if(FlexiBLAS_FOUND)
  if (NOT FlexiBLAS_FIND_QUIETLY)
    message(STATUS "Found FlexiBLAS: ${FlexiBLAS_LIBRARIES}")
  endif()
else()
  if(FlexiBLAS_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find FlexiBLAS")
  endif()
endif()


# mark_as_advanced(FlexiBLAS_LIBRARY)
