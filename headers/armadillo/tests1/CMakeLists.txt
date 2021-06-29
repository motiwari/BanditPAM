add_executable(smoke_test smoke_test.cpp)
target_link_libraries(smoke_test PRIVATE armadillo)
enable_testing()

add_test(NAME smoke_test COMMAND smoke_test)
