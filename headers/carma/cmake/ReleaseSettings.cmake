SET(PROJECT_RELEASE_DEFINITIONS ARMA_NO_DEBUG)
IF (
    CMAKE_CXX_COMPILER_ID STREQUAL "GNU" 
    OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" 
    OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
    AND NOT DISABLE_RELEASE_FLAGS
)
    SET(PROJECT_RELEASE_FLAGS "-march=native" "-mtune=native")
ELSEIF (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # workaround like https://github.com/nlohmann/json/issues/1408
    # to avoid error like: carma\third_party\armadillo-code\include\armadillo_bits/arma_str.hpp(194):
    # error C2039: '_snprintf': is not a member of 'std' (compiling source file carma\tests\src\bindings.cpp) 
    ADD_DEFINITIONS(-DHAVE_SNPRINTF)
ENDIF ()
