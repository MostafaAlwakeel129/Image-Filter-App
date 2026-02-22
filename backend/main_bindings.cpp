#include <pybind11/pybind11.h>

namespace py = pybind11;

// Add the declerations for your functions here
int test_noise_connection(int a, int b);             

// --- BINDINGS ---
PYBIND11_MODULE(cv_backend, m) {
    m.doc() = "C++ OpenCV Backend";

    // After you add declerations you have to tell the module...
    // when should it call your function? (first argument of m.def)
    // give it a pointer to the function (second argument of m.def)
    // A description of the function     (third argument of m.def)
    m.def("test_noise_connection", &test_noise_connection, "description of function 1");
    
}