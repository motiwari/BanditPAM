include_directories(${PROJECT_SOURCE_DIR}/headers)

<<<<<<< HEAD
add_executable(cluster main2.cpp)

add_library(BanditPAM kmedoids_ucb_updated.cpp)
target_link_libraries(cluster PUBLIC BanditPAM)
=======
add_executable(BanditPAM main.cpp)

add_library(BanditPAM_LIB kmedoids_ucb.cpp)
target_link_libraries(BanditPAM PUBLIC BanditPAM_LIB)
>>>>>>> ea8a8e5cd96a0f5ccc36c16688d97cf83e97fee3

find_package(OpenMP REQUIRED)
target_link_libraries(BanditPAM PUBLIC OpenMP::OpenMP_CXX)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
<<<<<<< HEAD
target_link_libraries(BanditPAM PUBLIC ${ARMADILLO_LIBRARIES})
=======
target_link_libraries(BanditPAM_LIB PUBLIC ${ARMADILLO_LIBRARIES})
>>>>>>> ea8a8e5cd96a0f5ccc36c16688d97cf83e97fee3
