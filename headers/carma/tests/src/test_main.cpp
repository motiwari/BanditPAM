// In a Catch project with multiple files, dedicate one file to compile the
// source code of Catch itself and reuse the resulting object file for linking.

#define CATCH_CONFIG_RUNNER
#include <pybind11/embed.h>
#include <catch2/catch.hpp>

int main(int argc, char* argv[]) {
    pybind11::scoped_interpreter guard{};
    int result = Catch::Session().run(argc, argv);
    return result;
}
