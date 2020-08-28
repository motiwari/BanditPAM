find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/headers)


set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

add_executable(runTest naive_test.cpp)
add_executable(runLargeTest dataset_test.cpp)


target_link_libraries(runTest PUBLIC BanditPAM)
target_link_libraries(runTest PUBLIC Threads::Threads)

target_link_libraries(runTest PUBLIC ${GTEST_BOTH_LIBRARIES})

target_link_libraries(runLargeTest PUBLIC BanditPAM)
target_link_libraries(runLargeTest PUBLIC Threads::Threads)

target_link_libraries(runLargeTest PUBLIC ${GTEST_BOTH_LIBRARIES})

add_test(NAME anTest COMMAND runTest)
add_test(NAME twoTest COMMAND runLargeTest)
