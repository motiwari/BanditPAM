include_directories(/System/Library/Frameworks/Python.framework/Versions/Current/Headers)
include_directories(${PROJECT_SOURCE_DIR}/headers)
include_directories(${PROJECT_SOURCE_DIR}/headers/carma)
include_directories(${PROJECT_SOURCE_DIR}/headers/carma/include)
include_directories(${PROJECT_SOURCE_DIR}/headers/carma/include/carma)
include_directories(${PROJECT_SOURCE_DIR}/headers/carma/include/carma/carma)

add_executable(BanditPAM main.cpp)

add_library(BanditPAM_LIB kmedoids_ucb.cpp)
target_link_libraries(BanditPAM PUBLIC BanditPAM_LIB)

find_package(OpenMP REQUIRED)
target_link_libraries(BanditPAM_LIB PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(BanditPAM PUBLIC OpenMP::OpenMP_CXX)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(BanditPAM PUBLIC ${ARMADILLO_LIBRARIES})
