#include "cp2k_parser.hpp"
#include <pybind11/numpy.h>
#include <fstream>
#include <vector>
#include <cctype>
#include <stdexcept>
#include <thread>
#include <cmath>

namespace py = pybind11;

// 1. Ultra-fast custom float parser (Natively handles 'D' and 'E')
inline float parse_scientific(const char* p) {
    while (isspace(*p)) p++;
    bool neg = false;
    if (*p == '-') { neg = true; p++; }
    else if (*p == '+') { p++; }

    double val = 0.0;
    while (isdigit(*p)) {
        val = val * 10.0 + (*p - '0');
        p++;
    }
    if (*p == '.') {
        p++;
        double frac = 1.0;
        while (isdigit(*p)) {
            frac *= 0.1;
            val += (*p - '0') * frac;
            p++;
        }
    }
    if (*p == 'E' || *p == 'e' || *p == 'D' || *p == 'd') {
        p++;
        bool exp_neg = false;
        if (*p == '-') { exp_neg = true; p++; }
        else if (*p == '+') { p++; }
        int exp = 0;
        while (isdigit(*p)) {
            exp = exp * 10 + (*p - '0');
            p++;
        }
        val *= std::pow(10.0, exp_neg ? -exp : exp);
    }
    return neg ? -val : val;
}

// 2. Strict header check
bool is_header_line_strict(char* str, int& n_cols) {
    n_cols = 0;
    char* p = str;
    while (*p) {
        while (isspace(*p)) p++;
        if (!*p) break;
        if (*p == '+' || *p == '-') p++;
        if (!isdigit(*p)) return false;
        while (isdigit(*p)) p++;
        if (*p != '\0' && !isspace(*p)) return false;
        n_cols++;
    }
    return n_cols > 0;
}

// 3. Extract the last N floats from a line in-place
std::vector<float> extract_last_n_floats(char* line, int n) {
    std::vector<char*> toks;
    toks.reserve(n + 10);
    char* p = line;
    while (*p) {
        while (isspace(*p)) { *p = '\0'; p++; }
        if (!*p) break;
        toks.push_back(p);
        while (*p && !isspace(*p)) p++;
    }
    
    std::vector<float> vals(n);
    int start = (int)toks.size() - n;
    for (int i = 0; i < n; ++i) {
        vals[i] = parse_scientific(toks[start + i]);
    }
    return vals;
}

// Struct to hold block metadata for threading
struct BlockInfo {
    int line_idx;
    int n_cols;
    int col_offset;
};

py::tuple parse_cp2k_mos_cpp(const std::string& filename, int n_ao_total) {
    // 1. GULP THE ENTIRE FILE INTO MEMORY (~0.5 seconds)
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) throw std::runtime_error("C++ Error: Could not open file.");
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size + 1);
    if (!file.read(buffer.data(), size)) throw std::runtime_error("C++ Error: Failed to read file.");
    buffer[size] = '\0';

    // 2. FIND ALL LINES (~0.2 seconds)
    std::vector<char*> lines;
    lines.reserve(size / 50); 
    lines.push_back(buffer.data());
    for (std::streamsize i = 0; i < size; ++i) {
        if (buffer[i] == '\n') {
            buffer[i] = '\0';
            lines.push_back(buffer.data() + i + 1);
        }
    }

    // 3. FIND ALL BLOCKS (Sequential scan, ~0.05 seconds)
    std::vector<BlockInfo> blocks;
    int current_col_offset = 0;
    for (size_t i = 0; i < lines.size(); ++i) {
        int n_cols = 0;
        if (is_header_line_strict(lines[i], n_cols)) {
            blocks.push_back({(int)i, n_cols, current_col_offset});
            current_col_offset += n_cols;
            i += (3 + n_ao_total); // Jump to the end of this block!
        }
    }

    int n_mo_total = current_col_offset;
    if (n_mo_total == 0) throw std::runtime_error("C++ Error: No MO blocks detected.");

    // 4. ALLOCATE NUMPY ARRAYS
    auto eps_np = py::array_t<float>(n_mo_total);
    auto occ_np = py::array_t<float>(n_mo_total);
    auto C_np = py::array_t<float>({n_ao_total, n_mo_total});

    float* eps_ptr = eps_np.mutable_data();
    float* occ_ptr = occ_np.mutable_data();
    float* C_ptr = C_np.mutable_data();

    // 5. MULTITHREADED PARSING (~0.5 seconds)
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // fallback
    
    std::vector<std::thread> workers;
    for (int tid = 0; tid < num_threads; ++tid) {
        workers.emplace_back([&, tid]() {
            // Each thread processes a staggered subset of blocks
            for (size_t b = tid; b < blocks.size(); b += num_threads) {
                const auto& block = blocks[b];
                int start = block.line_idx;
                int n_cols = block.n_cols;
                int col_off = block.col_offset;

                // Energies
                auto eps_vals = extract_last_n_floats(lines[start + 1], n_cols);
                for (int c = 0; c < n_cols; ++c) eps_ptr[col_off + c] = eps_vals[c];

                // Occupations
                auto occ_vals = extract_last_n_floats(lines[start + 2], n_cols);
                for (int c = 0; c < n_cols; ++c) occ_ptr[col_off + c] = occ_vals[c];

                // AO Coefficients
                for (int ao = 0; ao < n_ao_total; ++ao) {
                    auto ao_vals = extract_last_n_floats(lines[start + 3 + ao], n_cols);
                    for (int c = 0; c < n_cols; ++c) {
                        C_ptr[ao * n_mo_total + (col_off + c)] = ao_vals[c];
                    }
                }
            }
        });
    }

    // Wait for all threads to finish
    for (auto& w : workers) w.join();

    return py::make_tuple(C_np, eps_np, occ_np);
}

