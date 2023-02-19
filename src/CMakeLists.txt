include_directories(${PROJECT_SOURCE_DIR}/headers)
include_directories(${PROJECT_SOURCE_DIR}/headers/algorithms)
include_directories(${PROJECT_SOURCE_DIR}/headers/python_bindings)
include_directories(${PROJECT_SOURCE_DIR}/headers/carma/include)
include_directories(${PROJECT_SOURCE_DIR}/headers/carma/include/carma_bits)

# For Mac Github Runner to find OpenMP -- potentially unnecessary after fixing -Xpreprocessor -fopenmp issue
include_directories(/usr/local/Cellar/libomp/15.0.2/include)
include_directories(/usr/local/Cellar/libomp/15.0.7/include)
include_directories(/usr/local/Cellar/libomp)
include_directories(/usr/local/opt/libomp)
include_directories(/usr/local/opt)

add_executable(BanditPAM main.cpp)

add_library(BanditPAM_LIB algorithms/kmedoids_algorithm.cpp algorithms/pam.cpp algorithms/banditpam.cpp algorithms/banditpam_orig.cpp algorithms/fastpam1.cpp)
target_link_libraries(BanditPAM PUBLIC BanditPAM_LIB)

find_package(OpenMP REQUIRED)
target_link_libraries(BanditPAM_LIB PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(BanditPAM PUBLIC OpenMP::OpenMP_CXX)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(BanditPAM PUBLIC ${ARMADILLO_LIBRARIES})
