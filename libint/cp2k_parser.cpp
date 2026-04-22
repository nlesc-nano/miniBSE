#include "cp2k_parser.hpp"
#include <pybind11/numpy.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>
#include <pybind11/pybind11.h>
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

// 2. Strict header check (Restored to robust version)
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

    // 3. FIND ALL BLOCKS (Robust Lookahead + Safe Jump)
    std::vector<BlockInfo> blocks;
    int current_col_offset = 0;
    
    for (size_t i = 0; i < lines.size(); ++i) {
        int n_cols = 0;
        if (is_header_line_strict(lines[i], n_cols)) {
            
            // ANTI-GHOST CHECK: Lookahead to the next line.
            // A valid header MUST be immediately followed by Energy floats.
            if (i + 1 < lines.size()) {
                int energy_tokens = 0;
                char* p = lines[i + 1];
                while (*p) {
                    while (isspace(*p)) p++;
                    if (!*p) break;
                    energy_tokens++;
                    while (!isspace(*p) && *p != '\0') p++;
                }
                
                // If the next line lacks enough tokens, it's a ghost line. Ignore it!
                if (energy_tokens < n_cols) continue;
                
                // Valid Block! Record it.
                blocks.push_back({(int)i, n_cols, current_col_offset});
                current_col_offset += n_cols;
                
                // SAFEST JUMP: i=Header, i+1=Energy, i+2=Occ, i+3=AO_1.
                // The last AO is exactly at i + 2 + n_ao_total.
                // Setting i here ensures the loop's `++i` lands exactly on the next line.
                i += (2 + n_ao_total);
            }
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

// 1. Fast State-Machine Parser for CP2K Block-Dense Text
py::array_t<double> parse_cp2k_block_matrix_cpp(const std::string& filename, int n_ao) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("C++ Error: Could not open cleaned VXC file.");

    auto result_np = py::array_t<double>({n_ao, n_ao});
    double* ptr = result_np.mutable_data();
    std::fill(ptr, ptr + (size_t)n_ao * n_ao, 0.0);

    std::string line;
    std::vector<int> current_cols;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::vector<std::string> tokens;
        std::string buf;
        while (ss >> buf) tokens.push_back(buf);

        if (tokens.empty()) continue;

        // Header lines are only numbers. Data lines have strings (e.g. "Cd", "2s")
        bool is_header = true;
        for (const auto& t : tokens) {
            if (t.find_first_not_of("0123456789") != std::string::npos) {
                is_header = false;
                break;
            }
        }

        if (is_header) {
            current_cols.clear();
            for (const auto& t : tokens) current_cols.push_back(std::stoi(t));
        } else {
            // Data Line: Row AtomIdx Sym Orb Val1 Val2 ...
            // e.g. "1 1 Cd 2s -0.367055 -0.261982 ..."
            if (tokens.size() < 4 + current_cols.size()) continue; // Safety check
            
            int row = std::stoi(tokens[0]) - 1; // 1-based to 0-based indexing
            for (size_t i = 0; i < current_cols.size(); ++i) {
                int col = current_cols[i] - 1;
                double val = std::stod(tokens[i + 4]); // Values start at token index 4
                ptr[(size_t)row * n_ao + col] = val;
            }
        }
    }
    return result_np;
}

// 2. Ultra-Fast Raw Binary Loader
py::array_t<double> load_raw_binary_matrix_cpp(const std::string& bin_path, int n_ao) {
    size_t expected_size = (size_t)n_ao * n_ao * sizeof(double);
    int fd = open(bin_path.c_str(), O_RDONLY);
    if (fd == -1) throw std::runtime_error("C++ Error: Could not open binary file.");

    auto result_np = py::array_t<double>({n_ao, n_ao});
    double* ptr = result_np.mutable_data();

    // Direct system read into pre-allocated NumPy buffer
    if (read(fd, ptr, expected_size) != (ssize_t)expected_size) {
        close(fd);
        throw std::runtime_error("C++ Error: Binary file size mismatch. Try deleting the .raw.bin file.");
    }
    close(fd);
    return result_np;
}



