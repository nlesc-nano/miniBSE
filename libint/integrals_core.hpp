#pragma once
#include <Eigen/Dense>
#include <libint2.hpp>
#include <vector>
#include <array>
#include <thread>
#include <numeric>

namespace licpp
{
    using Matrix = Eigen::MatrixXd; // shortcut

    /// Cartesian k-point in reciprocal space.
    /// Units MUST match the units of the AO shell centers passed to Libint.
    /// (If your AO centers are in Bohr, pass k in Bohr^-1; if in Å, pass Å^-1.)
    struct KPoint {
      double kx, ky, kz;
      inline Eigen::Vector3d vec() const { return Eigen::Vector3d(kx, ky, kz); }
    };

    // A struct to hold the fundamental parameters of a single HGH projector
    // This data will be passed from Python.
    struct HghProjectorParams {
        int l;      // angular momentum
        int i;      // 1-based projector index (from paper)
        double r_l; // localization radius
        std::array<double, 3> center;
    };

    /* -------------------------------------------------- utilities -------- */
    size_t nbasis(const std::vector<libint2::Shell>&);
    size_t max_nprim(const std::vector<libint2::Shell>&);
    int max_l(const std::vector<libint2::Shell>&);
    std::vector<size_t>
    map_shell_to_basis_function(const std::vector<libint2::Shell>&);

    /* -------------------------------------------------- integrals -------- */
    Matrix overlap(const std::vector<libint2::Shell>&, int nthreads);
    std::vector<Matrix> dipole(const std::vector<libint2::Shell>&,
                               const std::array<double, 3>& origin,
                               int nthreads);
    Matrix cross_overlap(const std::vector<libint2::Shell>& shells,
                         size_t n_ao, size_t n_prj, int nthreads);

    // --- FUNCTION DECLARATION FOR GTH OVERLAPS ---
    Matrix compute_hgh_projector_overlaps(
        const std::vector<libint2::Shell>& ao_shells,
        const std::vector<HghProjectorParams>& projectors,
        int nthreads);

    // --- FUNCTION DECLARATION FOR ANALYTIC AO FOURIER TRANSFORM  ---
    Matrix ao_ft(const std::vector<libint2::Shell>& shells,
                 const std::vector<KPoint>& kpts,
                 int nthreads);
    
    // NEW: minimum-image helper + PBC overlap
    using Matrix = Eigen::MatrixXd;
    
    Matrix overlap_pbc(const std::vector<libint2::Shell>& shells,
                       const Eigen::Matrix3d& lattice_A,   // in Bohr
                       double cutoff_A = -1.0,              // <0: MIC only; >0: sum images within cutoff
                       int nthreads = 1);

    // Evaluate AO values for all shells at arbitrary points (bohr).
    // Returns shape (nao, npts).
    Eigen::MatrixXd ao_values_at_points(
        const std::vector<libint2::Shell>& shells,
        const std::vector<Eigen::Vector3d>& points,
        int nthreads);
 
} // namespace licpp

