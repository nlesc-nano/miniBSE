#include "integrals_core.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>          // Eigen ↔ NumPy
#include <string>                    // Required for std::string
#include <libint2.hpp>               // Required for libint2 headers
// #include <complex>  // -> uncomment when returning complex to Python


namespace py = pybind11;
using licpp::Matrix;

static std::vector<libint2::Shell>
convert_shells(const py::list& py_shells)
{
    using libint2::Shell;
    using libint2::svector;

    std::vector<Shell> shells;
    shells.reserve(py_shells.size());

    int shell_counter = 0; // Optional: for clearer numbering
    for (const auto& item : py_shells) {
        py::dict d = item.cast<py::dict>();

        int l = d["l"].cast<int>();
        auto ex_ps = d["exps"].cast<py::array_t<double>>();
        auto cf_ps = d["coefs"].cast<py::array_t<double>>();
        auto cen_ps = d["center"].cast<py::array_t<double>>();

        // --- THIS IS THE FIX ---
        // Read the 'pure' flag from the Python dictionary.
        // Default to 'false' (Cartesian) if the key is not present.
        bool pure = d.contains("pure") ? d["pure"].cast<bool>() : false;

        // ======================= YOUR PRINT STATEMENTS =======================
//        std::cout << "Shell #" << shell_counter
//                  << ", L: " << l 
//                  << ", Pure: " << (pure ? "true" : "false") << std::endl;
//
//        // Print exponents
//        std::cout << "  Exponents: ";
 //       for (ssize_t i = 0; i < ex_ps.size(); ++i) {
 //           std::cout << ex_ps.at(i) << " ";
 //       }
 //       std::cout << std::endl;
 //
 //       // Print coefficients
 //       std::cout << "  Coefficients: ";
 //       for (ssize_t i = 0; i < cf_ps.size(); ++i) {
 //           std::cout << cf_ps.at(i) << " ";
 //       }
 //       std::cout << std::endl;
 //
 //       // Print center
 //       std::cout << "  Center: " << cen_ps.at(0) << ", " << cen_ps.at(1) << ", " << cen_ps.at(2) << std::endl;
 //       std::cout << "------------------------------------" << std::endl;
 //       // =====================================================================
 //
        std::vector<double> exps(ex_ps.data(), ex_ps.data() + ex_ps.size());
        std::vector<double> coefs(cf_ps.data(), cf_ps.data() + cf_ps.size());
        std::array<double, 3> C;
        std::memcpy(C.data(), cen_ps.data(), 3 * sizeof(double));

        svector<double> ex_sv(exps.begin(), exps.end());
        svector<double> cf_sv(coefs.begin(), coefs.end());

        // Now, the 'pure' variable is used to create the shell correctly.
        Shell sh{
            ex_sv,
            { {l, pure, cf_sv} }, // Use the 'pure' variable read from Python
            {C[0], C[1], C[2]}
        };
        shells.push_back(std::move(sh));
        shell_counter++; // Optional: increment counter
    }
//    std::cout << "\n[SOC-LIBINT SHELL GROUND TRUTH]\n";
//    for(const auto& shell : shells) {
//        std::cout << shell << std::endl;
//    }
    return shells;
}

// Converts a Python list of projector parameter dicts to a C++ vector of structs
static std::vector<licpp::HghProjectorParams> // Fixed: Added licpp::
convert_projectors(const py::list& py_projectors)
{
    std::vector<licpp::HghProjectorParams> projectors; // Fixed: Added licpp::
    projectors.reserve(py_projectors.size());

    for (const auto& item : py_projectors) {
        py::dict d = item.cast<py::dict>();
        licpp::HghProjectorParams p; // Fixed: Added licpp::
        p.l = d["l"].cast<int>();
        p.i = d["i"].cast<int>();
        p.r_l = d["r_l"].cast<double>();
        
        auto cen_ps = d["center"].cast<py::array_t<double>>();
        std::memcpy(p.center.data(), cen_ps.data(), 3 * sizeof(double));
        projectors.push_back(p);
    }
    return projectors;
}

PYBIND11_MODULE(libint_cpp, m)
{
  m.doc() = "High-level libint2 wrappers (overlap, dipole, spin-orbit)";

  // Add a function to return the Libint version string
  m.def("version", []() {
      #if defined(LIBINT2_VERSION)
          return std::string(LIBINT2_VERSION);
      #elif defined(LIBINT2_VERSION_MAJOR)
          return std::to_string(LIBINT2_VERSION_MAJOR) + "." +
                 std::to_string(LIBINT2_VERSION_MINOR) + "." +
                 std::to_string(LIBINT2_VERSION_PATCH);
      #else
          return std::string("unknown");
      #endif
  }, "Returns the version of the Libint library");

  m.def("overlap",
        [](py::list py_shells, int nthreads) {
          auto shells = convert_shells(py_shells);
          libint2::initialize();
          Matrix S = licpp::overlap(shells, nthreads);
          libint2::finalize();
          return S;
        },
        py::arg("shells"), py::arg("nthreads") = 1);

  m.def("cross_overlap",
        [](py::list py_shells, size_t n_ao, size_t n_prj, int nthreads)
  {
      auto shells = convert_shells(py_shells);
      libint2::initialize();
      Matrix X = licpp::cross_overlap(shells, n_ao, n_prj, nthreads);
      libint2::finalize();
      return X;
  },
  py::arg("shells"), py::arg("n_ao"), py::arg("n_prj"),
  py::arg("nthreads") = 1);

  m.def("dipole",
        [](py::list py_shells, std::array<double,3> origin, int nthreads) {
          auto shells = convert_shells(py_shells);
          libint2::initialize();
          auto v = licpp::dipole(shells, origin, nthreads);
          libint2::finalize();

          const ssize_t nbf = v[0].rows();
          std::vector<ssize_t> shape = {3, nbf, nbf};
          py::array_t<double> out(shape);
          auto buf = out.mutable_unchecked<3>();
          for (int c=0;c<3;++c)
            for (ssize_t i=0;i<nbf;++i)
              for (ssize_t j=0;j<nbf;++j)
                buf(c,i,j) = v[c](i,j);
          return out;
        },
        py::arg("shells"), py::arg("origin"),
        py::arg("nthreads") = 1);


    // --- NEWLY ADDED BINDING (WITH CORRECTION) ---
    m.def("compute_hgh_overlaps",
        [](py::list py_ao_shells, py::list py_projectors, int nthreads) {
            auto ao_shells = convert_shells(py_ao_shells);
            auto projectors = convert_projectors(py_projectors);

            // --- FIX: Added missing initialize/finalize calls ---
            libint2::initialize();
            Matrix B = licpp::compute_hgh_projector_overlaps(ao_shells, projectors, nthreads);
            libint2::finalize();
            // ---------------------------------------------------

            return B;
        },
        py::arg("ao_shells"), py::arg("projectors"), py::arg("nthreads") = 1,
        "Computes <AO|Projector> overlaps using the derivative trick.");

    m.def("ao_ft",
          [](py::list py_shells, py::array_t<double> k_array, int nthreads){
              auto shells = convert_shells(py_shells);
  
              std::vector<licpp::KPoint> kpts;
              auto buf = k_array.unchecked<2>();
              kpts.reserve(buf.shape(0));
              for (ssize_t i=0;i<buf.shape(0);++i)
                kpts.push_back({buf(i,0),buf(i,1),buf(i,2)});
  
              libint2::initialize();
              Matrix F = licpp::ao_ft(shells, kpts, nthreads);
              libint2::finalize();
              return F;
          },
          py::arg("shells"), py::arg("kpoints"), py::arg("nthreads")=1,
          "Analytic AO Fourier transforms along a user supplied k-path");

    m.def("overlap_pbc",
          [](py::list py_shells,
             py::array_t<double> lattice_A,
             double cutoff_A,
             int nthreads) {
              auto shells = convert_shells(py_shells);
    
              if (lattice_A.ndim() != 2 || lattice_A.shape(0) != 3 || lattice_A.shape(1) != 3)
                throw std::runtime_error("lattice_A must be a (3,3) array");
              Eigen::Matrix3d L;
              std::memcpy(L.data(), lattice_A.data(), sizeof(double)*9);
    
              libint2::initialize();
              Matrix S = licpp::overlap_pbc(shells, L, cutoff_A, nthreads);
              libint2::finalize();
              return S;
          },
          py::arg("shells"),
          py::arg("lattice_A"),
          py::arg("cutoff_A") = -1.0,
          py::arg("nthreads") = 1);

    m.def("evaluate_basis_on_grid",
          [](py::list py_shells, py::array_t<double> pts_array, int nthreads) {
              auto shells = convert_shells(py_shells);
              if (pts_array.ndim() != 2 || pts_array.shape(1) != 3)
                  throw std::runtime_error("pts must be shape (N, 3)");
              
              auto buf = pts_array.unchecked<2>();
              std::vector<Eigen::Vector3d> pts;
              pts.reserve(buf.shape(0));
              for (ssize_t i = 0; i < buf.shape(0); ++i)
                  pts.emplace_back(buf(i, 0), buf(i, 1), buf(i, 2));
                  
              // The fuzzy 2 function returns (N_ao, N_pts). 
              // We transpose it so Python gets (N_pts, N_ao) for easy matrix multiplication.
              Eigen::MatrixXd result = licpp::ao_values_at_points(shells, pts, nthreads);
              return result.transpose().eval(); 
          },
          py::arg("shells"), py::arg("pts"), py::arg("nthreads") = 1,
          "Evaluate AOs on a grid. Returns shape (N_pts, N_ao).");

    m.def("evaluate_mos_on_grid",
          [](py::list py_shells, py::array_t<double> C_array, py::array_t<double> pts_array, int nthreads) {
              auto shells = convert_shells(py_shells);

              // 1. Unpack the C matrix
              auto C_buf = C_array.unchecked<2>();
              Eigen::MatrixXd C(C_buf.shape(0), C_buf.shape(1));
              for(ssize_t i=0; i<C_buf.shape(0); ++i)
                  for(ssize_t j=0; j<C_buf.shape(1); ++j)
                      C(i,j) = C_buf(i,j);

              // 2. Unpack the grid points
              auto p_buf = pts_array.unchecked<2>();
              std::vector<Eigen::Vector3d> pts;
              pts.reserve(p_buf.shape(0));
              for(ssize_t i=0; i<p_buf.shape(0); ++i)
                  pts.emplace_back(p_buf(i,0), p_buf(i,1), p_buf(i,2));

              // 3. Evaluate AOs (Multi-threaded) - shape: (N_ao, N_pts)
              Eigen::MatrixXd chi = licpp::ao_values_at_points(shells, pts, nthreads);

              // 4. Multiply in C++ to get MOs - shape: (N_pts, N_mo)
              // This completely bypasses the 1.2 GB Python transfer!
              Eigen::MatrixXd psi = chi.transpose() * C;

              return psi;
          },
          py::arg("shells"), py::arg("C"), py::arg("pts"), py::arg("nthreads") = 1,
          "Evaluates active MOs directly on a grid to avoid massive Python memory transfers.");


}
