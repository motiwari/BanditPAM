include_directories(${PROJECT_SOURCE_DIR}/headers)

add_executable(cluster main.cpp)

add_library(BanditPAM kmedoids_ucb.cpp)
target_link_libraries(cluster PUBLIC BanditPAM)

find_package(OpenMP REQUIRED)
target_link_libraries(cluster PUBLIC OpenMP::OpenMP_CXX)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(BanditPAM PUBLIC ${ARMADILLO_LIBRARIES})