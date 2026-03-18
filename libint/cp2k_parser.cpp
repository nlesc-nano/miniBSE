#include "cp2k_parser.hpp"
#include <pybind11/numpy.h>
#include <fstream>
#include <vector>
#include <cctype>
#include <stdexcept>
#include <thread>
#include <cmath>
#include <mutex>

namespace py = pybind11;

// 1. Ultra-fast custom float parser (Natively handles 'D' and 'E')
inline double parse_scientific(const char* p) {
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

// 3. Extract the last N floats from a line in-place safely
std::vector<double> extract_last_n_floats(const std::string& line, int n, int line_num) {
    std::vector<double> result(n);
    int idx = n - 1;
    int end = line.size() - 1;

    while (idx >= 0 && end >= 0) {
        // Skip trailing spaces
        while (end >= 0 && isspace(line[end])) end--;
        if (end < 0) break;

        int start = end;
        while (start >= 0) {
            // Standard space delimiter
            if (isspace(line[start])) break;
            
            // Catch merged CP2K columns (e.g. 0.123-0.123)
            if (line[start] == '-' || line[start] == '+') {
                if (start > 0) {
                    char prev = line[start - 1];
                    // If the sign is NOT preceded by an exponent (E/D) or a space, it's a merged number boundary
                    if (prev != 'E' && prev != 'e' && prev != 'D' && prev != 'd' && !isspace(prev)) {
                        start--; // Step back to include the sign in the token
                        break;
                    }
                }
            }
            start--;
        }

        std::string token = line.substr(start + 1, end - start);
        result[idx] = parse_scientific(token.c_str());

        idx--;
        end = start;
    }

    if (idx >= 0) {
        throw std::runtime_error("Could not extract enough floats on line " + std::to_string(line_num) + "\nLine: " + line);
    }
    return result;
}

// Struct to hold block metadata for threading
struct BlockInfo {
    int line_idx;
    int n_cols;
    int col_offset;
};

py::tuple parse_cp2k_mos_cpp(const std::string& filename, int n_ao_total) {
    // 1. GULP THE ENTIRE FILE INTO MEMORY
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) throw std::runtime_error("C++ Error: Could not open file.");
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size + 1);
    if (!file.read(buffer.data(), size)) throw std::runtime_error("C++ Error: Failed to read file.");
    buffer[size] = '\0';

    // 2. FIND ALL LINES
    std::vector<char*> lines;
    lines.reserve(size / 50); 
    lines.push_back(buffer.data());
    for (std::streamsize i = 0; i < size; ++i) {
        if (buffer[i] == '\n') {
            buffer[i] = '\0';
            lines.push_back(buffer.data() + i + 1);
        }
    }

    // 3. FIND ALL BLOCKS 
    std::vector<BlockInfo> blocks;
    int current_col_offset = 0;
    for (size_t i = 0; i < lines.size(); ++i) {
        int n_cols = 0;
        if (is_header_line_strict(lines[i], n_cols)) {
            blocks.push_back({(int)i, n_cols, current_col_offset});
            current_col_offset += n_cols;
            i += (3 + n_ao_total); // Jump to the end of this block
        }
    }

    int n_mo_total = current_col_offset;
    if (n_mo_total == 0) throw std::runtime_error("C++ Error: No MO blocks detected.");

    // 4. ALLOCATE NUMPY ARRAYS
    auto eps_np = py::array_t<double>(n_mo_total);
    auto occ_np = py::array_t<double>(n_mo_total);
    auto C_np = py::array_t<double>({n_ao_total, n_mo_total});

    double* eps_ptr = eps_np.mutable_data();
    double* occ_ptr = occ_np.mutable_data();
    double* C_ptr = C_np.mutable_data();

    // 5. MULTITHREADED PARSING WITH ERROR CATCHING
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::vector<std::thread> workers;
    std::mutex err_mutex;
    std::string thread_error = "";

    for (int tid = 0; tid < num_threads; ++tid) {
        workers.emplace_back([&, tid]() {
            try {
                for (size_t b = tid; b < blocks.size(); b += num_threads) {
                    const auto& block = blocks[b];
                    int start = block.line_idx;
                    int n_cols = block.n_cols;
                    int col_off = block.col_offset;

                    // Bounds Check File End
                    if (start + 3 + n_ao_total >= lines.size()) {
                        throw std::runtime_error("Unexpected End of File at block starting at line " + std::to_string(start));
                    }

                    // Energies
                    auto eps_vals = extract_last_n_floats(lines[start + 1], n_cols, start + 1);
                    for (int c = 0; c < n_cols; ++c) eps_ptr[col_off + c] = eps_vals[c];

                    // Occupations
                    auto occ_vals = extract_last_n_floats(lines[start + 2], n_cols, start + 2);
                    for (int c = 0; c < n_cols; ++c) occ_ptr[col_off + c] = occ_vals[c];

                    // AO Coefficients
                    for (int ao = 0; ao < n_ao_total; ++ao) {
                        int current_line = start + 3 + ao;
                        auto ao_vals = extract_last_n_floats(lines[current_line], n_cols, current_line);
                        for (int c = 0; c < n_cols; ++c) {
                            C_ptr[ao * n_mo_total + (col_off + c)] = ao_vals[c];
                        }
                    }
                }
            } catch (const std::exception& e) {
                // Safely lock and pass the error back to the main thread
                std::lock_guard<std::mutex> lock(err_mutex);
                if (thread_error.empty()) {
                    thread_error = e.what();
                }
            }
        });
    }

    // Wait for all threads to finish
    for (auto& w : workers) w.join();

    // If any thread crashed, throw the Python exception now!
    if (!thread_error.empty()) {
        throw std::runtime_error(thread_error);
    }

    return py::make_tuple(C_np, eps_np, occ_np);
}

