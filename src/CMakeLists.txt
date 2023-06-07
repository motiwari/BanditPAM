include_directories(${PROJECT_SOURCE_DIR}/headers)
include_directories(${PROJECT_SOURCE_DIR}/headers/algorithms)
include_directories(${PROJECT_SOURCE_DIR}/headers/python_bindings)
include_directories(${PROJECT_SOURCE_DIR}/headers/carma/include)
include_directories(${PROJECT_SOURCE_DIR}/headers/carma/include/carma_bits)
# Include directory for windows
if(WIN32)
include_directories(${PROJECT_SOURCE_DIR}/headers/armadillo/include)
include_directories(
        ${PROJECT_SOURCE_DIR}/headers/armadillo/include/armadillo_bits)
set(OPENBLAS_dir ${PROJECT_SOURCE_DIR}/headers/armadillo/examples/lib_win64)
endif()

# For Mac Github Runner to find OpenMP -- potentially unnecessary after fixing
# -Xpreprocessor -fopenmp issue
if(APPLE OR UNIX)
include_directories(/usr/local/Cellar/libomp/15.0.2/include)
include_directories(/usr/local/Cellar/libomp/15.0.7/include)
include_directories(/usr/local/Cellar/libomp)
include_directories(/usr/local/opt/libomp/include)
include_directories(/usr/local/opt/libomp/lib)
include_directories(/usr/local/opt/libomp)
include_directories(/usr/local/opt)
endif()

if(WIN32)
add_executable(BanditPAM getopt.cpp main.cpp )

elseif(APPLE OR UNIX)
add_executable(BanditPAM main.cpp )
endif()
add_library(
        BanditPAM_LIB algorithms/kmedoids_algorithm.cpp algorithms/pam.cpp
        algorithms/banditpam.cpp algorithms/banditpam_orig.cpp
        algorithms/fastpam1.cpp)

target_link_libraries(BanditPAM PUBLIC BanditPAM_LIB)

if(WIN32)
target_link_directories(BanditPAM PUBLIC ${OPENBLAS_dir})
target_link_libraries(BanditPAM_LIB libopenblas)
endif()

if(UNIX OR APPLE)
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(BanditPAM PUBLIC ${ARMADILLO_LIBRARIES})
endif()

if(WIN32)
string(REPLACE "/" "\\"  OPENBLAS_dir_win ${OPENBLAS_dir})
add_custom_command(TARGET BanditPAM_LIB POST_BUILD
            COMMAND copy ${OPENBLAS_dir_win}\\*.dll $(OutDir))
endif()
