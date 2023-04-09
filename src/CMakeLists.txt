include_directories(${PROJECT_SOURCE_DIR}/headers)
include_directories(${PROJECT_SOURCE_DIR}/headers/algorithms)
include_directories(${PROJECT_SOURCE_DIR}/headers/python_bindings)
include_directories(${PROJECT_SOURCE_DIR}/headers/carma/include)
include_directories(${PROJECT_SOURCE_DIR}/headers/carma/include/carma_bits)
include_directories(${PROJECT_SOURCE_DIR}/headers/armadillo-code/include)
include_directories(${PROJECT_SOURCE_DIR}/headers/armadillo-code/include/armadillo_bits)

# For Mac Github Runner to find OpenMP -- potentially unnecessary after fixing -Xpreprocessor -fopenmp issue
include_directories(/usr/local/Cellar/libomp/15.0.2/include)
include_directories(/usr/local/Cellar/libomp/15.0.7/include)
include_directories(/usr/local/Cellar/libomp)
include_directories(/usr/local/opt/libomp/lib)
include_directories(/usr/local/opt/libomp)
include_directories(/usr/local/opt)

set(OPENBLAS_dir ${PROJECT_SOURCE_DIR}/headers/armadillo-code/examples/lib_win64)
add_executable(BanditPAM getopt.cpp main.cpp )

add_library(BanditPAM_LIB algorithms/kmedoids_algorithm.cpp algorithms/pam.cpp algorithms/banditpam.cpp algorithms/banditpam_orig.cpp algorithms/fastpam1.cpp)
target_link_directories(BanditPAM PUBLIC ${OPENBLAS_dir})
target_link_libraries(BanditPAM_LIB libopenblas)
target_link_libraries(BanditPAM PUBLIC BanditPAM_LIB)

string(REPLACE "/" "\\"  OPENBLAS_dir_win ${OPENBLAS_dir})
message(STATUS ${OPENBLAS_dir})
message(STATUS ${OPENBLAS_dir_win}\\dll)
add_custom_command(TARGET BanditPAM_LIB POST_BUILD
    COMMAND copy ${OPENBLAS_dir_win}\\*.dll $(OutDir)
)
