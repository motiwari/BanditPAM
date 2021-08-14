cmake_minimum_required(VERSION 3.17)

project(BanditPAM VERSION 1.0 LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# set(CMAKE_CXX_FLAGS_RELEASE "-v")

# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -undefined dynamic_lookup")
# find_package(Python3 COMPONENTS Interpreter Development)
#set(CMAKE_CXX_FLAGS_RELEASE "-L/opt/intel/oneapi/intelpython/latest/lib/python3.7/config-3.7m-darwin -lpython3.7m -ldl -framework CoreFoundation")

ADD_SUBDIRECTORY(/Users/motiwari/Desktop/BanditPAM/headers/carma)
add_subdirectory(src)
