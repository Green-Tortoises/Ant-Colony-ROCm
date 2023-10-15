#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <hip/hip_runtime.h>

// Using this namespace as recommended in the official documentation
// to reduce definition name
namespace py = pybind11;

static bool check_ants(py::array_t<double> the_colony) {
    auto r = the_colony.unchecked<2>();
    bool didAllAntsWalked = true;

    for (py::ssize_t i = 0; i < r.shape(0); i++)
        for (py::ssize_t j = 0; j < r.shape(1); j++)
            didAllAntsWalked = r(i, j) == -1 ? false : true;

    return didAllAntsWalked;
}

int run_colony_cpu(py::array_t<double> the_colony, py::array_t<double> X, py::array_t<double> Y,
                   int initial_pheromone, float evaporarion_rate, int Q) {

    auto r = the_colony.unchecked<2>();

    while(check_ants(the_colony)) {

    }


    return 12;
}

int run_colony_gpu(py::array_t<double> X, py::array_t<double> Y, int initial_pheromone, float evaporarion_rate, int Q) {
    return 0;
}

PYBIND11_MODULE(run_colony, m) {
    m.doc() = "Run colony using C++ to get some performance :)"; // optional module docstring

    m.def("run", &run_colony_cpu, "A function that runs the colony on the CPU");
    m.def("run_gpu", &run_colony_gpu, "A function that runs the colony on the GPU");
}
