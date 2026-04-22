#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>

// Declare the parsing function so bindings.cpp can see it
pybind11::tuple parse_cp2k_mos_cpp(const std::string& filename, int n_ao_total);

pybind11::array_t<double> parse_cp2k_block_matrix_cpp(const std::string& filename, int n_ao);

pybind11::array_t<double> load_raw_binary_matrix_cpp(const std::string& bin_path, int n_ao);

