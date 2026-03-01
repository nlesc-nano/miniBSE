#pragma once
#include <pybind11/pybind11.h>
#include <string>

// Declare the parsing function so bindings.cpp can see it
pybind11::tuple parse_cp2k_mos_cpp(const std::string& filename, int n_ao_total);

