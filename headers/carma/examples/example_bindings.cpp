#include <pybind11/pybind11.h>
// include numpy header for usage of array_t
#include <pybind11/numpy.h>

#include "arraystore.h"
#include "manual_conversion.h"
#include "automatic_conversion.h"

namespace py = pybind11;

PYBIND11_MODULE(example_carma, m) {
    bind_manual_example(m);
    bind_update_example(m);
    bind_automatic_example(m);
    bind_exampleclass(m);
}
