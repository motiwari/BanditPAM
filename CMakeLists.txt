cmake_minimum_required(VERSION 3.17)
project(pam VERSION 1.0 LANGUAGES CXX)

include_directories(headers)
add_library(KMedoids_UCB src/kmedoids_ucb.cpp)

add_executable(pam src/main.cpp)

# include openMP
find_package(OpenMP REQUIRED)
target_link_libraries(pam PUBLIC OpenMP::OpenMP_CXX)

# include Armadillo
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(pam PUBLIC ${ARMADILLO_LIBRARIES})

target_link_libraries(pam PUBLIC KMedoids_UCB)