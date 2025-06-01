#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#undef STRINGIFY_IMPLEMENTATION_
#undef STRINGIFY
#define STRINGIFY_IMPLEMENTATION_(x) #x
#define STRINGIFY(x) STRINGIFY_IMPLEMENTATION_(x)

// forward declarations, see other interface cpp files for definitions
void define_creation_interface(py::module &m);
void define_runtime_interface(py::module &m);


PYBIND11_MODULE(_TeamIndex, m) {
    define_creation_interface(m);
    define_runtime_interface(m);

    // documentation
    m.doc() = R"pbdoc(
        TeamIndexBackend, a library that implements funtionality of
        TeamIndex, a disk-based and multi-dimensional secondary index.

        -----------------------
        .. currentmodule::TeamIndexBackend
        .. autosummary::
           :toctree: _generate
           add
    )pbdoc";

    #ifdef VERSION_INFO
    m.attr("__version__") = STRINGIFY(VERSION_INFO);
    #else
    m.attr("__version__") = "dev";
    #endif
}