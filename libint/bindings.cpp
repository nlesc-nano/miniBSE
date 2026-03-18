#include "integrals_core.hpp"
#include "basis_to_k.hpp"            // Required for the new k-space file
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>          
#include <pybind11/complex.h>        // Required to seamlessly handle complex matrices
#include <string>                    
#include <libint2.hpp>               
#include "cp2k_parser.hpp"

namespace py = pybind11;
using licpp::Matrix;

static std::vector<libint2::Shell>
convert_shells(const py::list& py_shells)
{
    using libint2::Shell;
    using libint2::svector;

    std::vector<Shell> shells;
    shells.reserve(py_shells.size());

    for (const auto& item : py_shells) {
        py::dict d = item.cast<py::dict>();
        int l = d["l"].cast<int>();
        auto ex_ps = d["exps"].cast<py::array_t<double>>();
        auto cf_ps = d["coefs"].cast<py::array_t<double>>();
        auto cen_ps = d["center"].cast<py::array_t<double>>();

        bool pure = d.contains("pure") ? d["pure"].cast<bool>() : false;

        std::vector<double> exps(ex_ps.data(), ex_ps.data() + ex_ps.size());
        std::vector<double> coefs(cf_ps.data(), cf_ps.data() + cf_ps.size());
        std::array<double, 3> C;
        std::memcpy(C.data(), cen_ps.data(), 3 * sizeof(double));

        svector<double> ex_sv(exps.begin(), exps.end());
        svector<double> cf_sv(coefs.begin(), coefs.end());

        Shell sh{
            ex_sv,
            { {l, pure, cf_sv} }, 
            {C[0], C[1], C[2]}
        };
        shells.push_back(std::move(sh));
    }
    return shells;
}

static std::vector<licpp::HghProjectorParams>
convert_projectors(const py::list& py_projectors)
{
    std::vector<licpp::HghProjectorParams> projectors;
    projectors.reserve(py_projectors.size());

    for (const auto& item : py_projectors) {
        py::dict d = item.cast<py::dict>();
        licpp::HghProjectorParams p;
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
  m.doc() = "High-level libint2 wrappers (overlap, dipole, spin-orbit, k-space)";

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
          //libint2::set_solid_harmonics_ordering(libint2::SHGShellOrdering_Standard);
          libint2::initialize();
          Matrix S = licpp::overlap(shells, nthreads);
          libint2::finalize();
          return S;
        },
        py::arg("shells"), py::arg("nthreads") = 1);

  m.def("cross_overlap",
        [](py::list py_shells, size_t n_ao, size_t n_prj, int nthreads) {
      auto shells = convert_shells(py_shells);
      //libint2::set_solid_harmonics_ordering(libint2::SHGShellOrdering_Standard);
      libint2::initialize();
      Matrix X = licpp::cross_overlap(shells, n_ao, n_prj, nthreads);
      libint2::finalize();
      return X;
  },
  py::arg("shells"), py::arg("n_ao"), py::arg("n_prj"), py::arg("nthreads") = 1);

  m.def("dipole",
        [](py::list py_shells, std::array<double,3> origin, int nthreads) {
          auto shells = convert_shells(py_shells);
          //libint2::set_solid_harmonics_ordering(libint2::SHGShellOrdering_Standard);
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
        py::arg("shells"), py::arg("origin"), py::arg("nthreads") = 1);

    m.def("compute_hgh_overlaps",
        [](py::list py_ao_shells, py::list py_projectors, int nthreads) {
            auto ao_shells = convert_shells(py_ao_shells);
            auto projectors = convert_projectors(py_projectors);
            //libint2::set_solid_harmonics_ordering(libint2::SHGShellOrdering_Standard);
            libint2::initialize();
            Matrix B = licpp::compute_hgh_projector_overlaps(ao_shells, projectors, nthreads);
            libint2::finalize();
            return B;
        },
        py::arg("ao_shells"), py::arg("projectors"), py::arg("nthreads") = 1);

  // --- NEW COMPLEX FOURIER TRANSFORM ---
  m.def("ao_ft_complex",
          [](py::list py_shells, py::array_t<double> k_array, int nthreads){
              auto shells = convert_shells(py_shells);
              std::vector<licpp::KPoint> kpts;
              auto buf = k_array.unchecked<2>();
              kpts.reserve(buf.shape(0));
              for (ssize_t i=0;i<buf.shape(0);++i)
                  kpts.push_back({buf(i,0),buf(i,1),buf(i,2)});
              //libint2::set_solid_harmonics_ordering(libint2::SHGShellOrdering_Standard);
              libint2::initialize();
              Eigen::MatrixXcd F = licpp::ao_ft_complex(shells, kpts, nthreads);
              libint2::finalize();
              return F;
          },
          py::arg("shells"), py::arg("kpoints"), py::arg("nthreads")=1,
          "Analytic AO Fourier transforms returning complex amplitudes");

  m.def("overlap_pbc",
          [](py::list py_shells, py::array_t<double> lattice_A, double cutoff_A, int nthreads) {
              auto shells = convert_shells(py_shells);
              if (lattice_A.ndim() != 2 || lattice_A.shape(0) != 3 || lattice_A.shape(1) != 3)
                throw std::runtime_error("lattice_A must be a (3,3) array");
              
              Eigen::Matrix3d L;
              std::memcpy(L.data(), lattice_A.data(), sizeof(double)*9);
              //libint2::set_solid_harmonics_ordering(libint2::SHGShellOrdering_Standard); 
              libint2::initialize();
              Matrix S = licpp::overlap_pbc(shells, L, cutoff_A, nthreads);
              libint2::finalize();
              return S;
          },
          py::arg("shells"), py::arg("lattice_A"), py::arg("cutoff_A") = -1.0, py::arg("nthreads") = 1);

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
                 
              Eigen::MatrixXd result = licpp::ao_values_at_points(shells, pts, nthreads);
              return result.transpose().eval(); 
          },
          py::arg("shells"), py::arg("pts"), py::arg("nthreads") = 1);

    m.def("evaluate_mos_on_grid",
          [](py::list py_shells, py::array_t<double> C_array, py::array_t<double> pts_array, int nthreads) {
              auto shells = convert_shells(py_shells);
              auto C_buf = C_array.unchecked<2>();
              Eigen::MatrixXd C(C_buf.shape(0), C_buf.shape(1));
              for(ssize_t i=0; i<C_buf.shape(0); ++i)
                  for(ssize_t j=0; j<C_buf.shape(1); ++j)
                      C(i,j) = C_buf(i,j);

              auto p_buf = pts_array.unchecked<2>();
              std::vector<Eigen::Vector3d> pts;
              pts.reserve(p_buf.shape(0));
              for(ssize_t i=0; i<p_buf.shape(0); ++i)
                  pts.emplace_back(p_buf(i,0), p_buf(i,1), p_buf(i,2));

              Eigen::MatrixXd chi = licpp::ao_values_at_points(shells, pts, nthreads);
              Eigen::MatrixXd psi = chi.transpose() * C;

              return psi;
          },
          py::arg("shells"), py::arg("C"), py::arg("pts"), py::arg("nthreads") = 1);

    m.def("parse_cp2k_mos", &parse_cp2k_mos_cpp, 
          "Ultra-fast C++ parser for CP2K MO text files",
          py::arg("filename"), py::arg("n_ao_total"));
}

