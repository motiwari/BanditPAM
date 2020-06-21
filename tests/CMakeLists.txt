find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

add_executable(runTest naive_test.cpp)



target_link_libraries(runTest PUBLIC Threads::Threads)

target_link_libraries(runTest PUBLIC ${GTEST_BOTH_LIBRARIES})

add_test(anTest COMMAND runTest)