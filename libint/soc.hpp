#pragma once
#include <vector>
#include <array>
#include <complex>
#include <Eigen/Core>
#include <libint2.hpp>

namespace licpp {

// Defines a single GTH/HGH projector for SOC
struct HghProjectorParams {
    int l;
    int i;
    double r_l;
    std::array<double, 3> center;
};

// KPoint structure for Analytic Fourier Transforms
struct KPoint {
    double kx, ky, kz;
    Eigen::Vector3d vec() const { return Eigen::Vector3d(kx, ky, kz); }
};

// 1. Computes the overlaps <AO | Projector> using derivative techniques
Eigen::MatrixXd compute_hgh_projector_overlaps(
    const std::vector<libint2::Shell>& ao_shells,
    const std::vector<HghProjectorParams>& projectors,
    int nthreads);

// 2. Analytic AO Fourier transforms along a user supplied k-path
Eigen::MatrixXd ao_ft(
    const std::vector<libint2::Shell>& shells,
    const std::vector<KPoint>& kpts,
    int nthreads);

// 3. PBC Overlap Matrix using Minimum Image Convention (MIC) and cutoff
Eigen::MatrixXd overlap_pbc(
    const std::vector<libint2::Shell>& shells,
    const Eigen::Matrix3d& lattice_A, // Bohr
    double cutoff_A, // Angstroms
    int nthreads);

// Extern declarations for helpers assumed to be in integrals_core.cpp
size_t nbasis(const std::vector<libint2::Shell>& shells);
size_t max_nprim(const std::vector<libint2::Shell>& shells);
int max_l(const std::vector<libint2::Shell>& shells);
std::vector<size_t> map_shell_to_basis_function(const std::vector<libint2::Shell>& shells);

} // namespace licpp


