cmake_minimum_required(VERSION 3.17)

project(BanditPAM VERSION 1.0 LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)

# Note: ThreadSanitizer and AddressSanitizer can give spurious errors around the #pragma omp critical.
# They don't really matter because the threads always resolve with the same end state
# ThreadSanitizer also introduces a very large overhead, AddressSanitizer is more reasonable
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -g -fno-omit-frame-pointer -fsanitize=address")

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_subdirectory(src)
