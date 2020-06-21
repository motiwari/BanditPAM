cmake_minimum_required(VERSION 3.17)
project(BanditPAM VERSION 1.0 LANGUAGES CXX)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)

#include_directories(headers)

#add_executable(cluster src/main.cpp)

#add_library(BanditPAM src/kmedoids_ucb.cpp)
#target_link_libraries(cluster PUBLIC BanditPAM)

#find_package(OpenMP REQUIRED)
#target_link_libraries(cluster PUBLIC OpenMP::OpenMP_CXX)

#find_package(Armadillo REQUIRED)
#include_directories(${ARMADILLO_INCLUDE_DIRS})
#target_link_libraries(cluster PUBLIC ${ARMADILLO_LIBRARIES})

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
