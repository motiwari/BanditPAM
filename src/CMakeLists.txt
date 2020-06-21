include_directories(../headers)

add_executable(BanditPAM main.cpp)

add_library(KMedoids_UCB kmedoids_ucb.cpp)
target_link_libraries(BanditPAM PUBLIC KMedoids_UCB)

find_package(OpenMP REQUIRED)
target_link_libraries(BanditPAM PUBLIC OpenMP::OpenMP_CXX)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(BanditPAM PUBLIC ${ARMADILLO_LIBRARIES})