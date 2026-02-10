#include <stdio.h>

#include <pybind11/pybind11.h>

void register_pufferlib_bindings(pybind11::module_& m);

PYBIND11_MODULE(binding, m) {
  register_pufferlib_bindings(m);
}

extern "C" void test_function() {
    printf("Test function called!\n");
}