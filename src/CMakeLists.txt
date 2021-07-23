# include_directories(/opt/intel/oneapi/intelpython/latest/lib/python3.7/site-packages/numpy/core/include)
include_directories(/Users/motiwari/.local/lib/python3.7/site-packages/numpy/core/include)
# include_directories(/opt/intel/oneapi/intelpython/latest/include/python3.7m/)
include_directories(/opt/intel/oneapi/intelpython/latest/include/python3.7m)
include_directories(/opt/intel/oneapi/intelpython/latest/bin/)
include_directories(/opt/intel/oneapi/advisor/2021.2.0/pythonapi)
# include_directories(/System/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7)
# include_directories(/System/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7)


include_directories(${PROJECT_SOURCE_DIR}/headers)
include_directories(${PROJECT_SOURCE_DIR}/headers/carma)
include_directories(${PROJECT_SOURCE_DIR}/headers/carma/include)
include_directories(${PROJECT_SOURCE_DIR}/headers/carma/include/carma)
include_directories(${PROJECT_SOURCE_DIR}/headers/carma/include/carma/carma)


# include_directories(/opt/intel/oneapi/intelpython/latest)
# include_directories(/opt/intel/oneapi/intelpython/latest/include/python3.7m)
# include_directories(/opt/intel/oneapi/intelpython/latest/include/python3.7m)
# include_directories(/opt/intel/oneapi/intelpython/latest/lib/python3.7/site-packages)
# include_directories(/opt/intel/oneapi/intelpython/latest/lib/python3.7)
# include_directories(/opt/intel/oneapi/intelpython/latest/lib/python3.7/site-packages)
# include_directories(/opt/intel/oneapi/intelpython/latest/bin)
# include_directories(/opt/intel/oneapi/intelpython/latest/lib/python3.7)

add_executable(BanditPAM main.cpp)

add_library(BanditPAM_LIB kmedoids_ucb.cpp)
target_link_libraries(BanditPAM PUBLIC BanditPAM_LIB)

find_package(OpenMP REQUIRED)
target_link_libraries(BanditPAM_LIB PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(BanditPAM PUBLIC OpenMP::OpenMP_CXX)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(BanditPAM PUBLIC ${ARMADILLO_LIBRARIES})
