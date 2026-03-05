#include "integrals_core.hpp"
#include <libint2.hpp>
#include <vector>
#include <algorithm>
#include <cmath> 
#include <complex>   

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace licpp {

/* =================== Parallel launcher ========================== */
template <typename Lambda>
inline void parallel_do(Lambda&& body, int nthreads) {
#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
  {
    body(omp_get_thread_num());
  }
#else
  std::vector<std::thread> workers;
  for (int id = 1; id < nthreads; ++id)
    workers.emplace_back(body, id);
  body(0);
  for (auto& t : workers) t.join();
#endif
}

/* ---------------- basis helpers ------------------------------------- */
size_t nbasis(const std::vector<libint2::Shell>& shells) {
  size_t n = 0;
  for (auto& s : shells) n += s.size();
  return n;
}

size_t max_nprim(const std::vector<libint2::Shell>& shells) {
  size_t m = 0;
  for (auto& s : shells) m = std::max(m, s.nprim());
  return m;
}

int max_l(const std::vector<libint2::Shell>& shells) {
  int m = 0;
  for (auto& s : shells)
    for (auto& c : s.contr) m = std::max(m, c.l);
  return m;
}

std::vector<size_t>
map_shell_to_basis_function(const std::vector<libint2::Shell>& shells) {
  std::vector<size_t> s2bf;
  s2bf.reserve(shells.size());
  size_t acc = 0;
  for (auto& s : shells) {
    s2bf.push_back(acc);
    acc += s.size();
  }
  return s2bf;
}

/* -------- generic engine farm with explicit thread count ------------ */
template <libint2::Operator OP, typename OpParams = typename
        libint2::operator_traits<OP>::oper_params_type>
std::vector<Matrix> compute_multipoles(
        const std::vector<libint2::Shell>& shells,
        int nthreads,
        OpParams params = OpParams{}) {

  constexpr unsigned int NP = libint2::operator_traits<OP>::nopers;
  const auto nbf = nbasis(shells);
  std::vector<Matrix> result(NP, Matrix::Zero(nbf, nbf));

  /* one engine per thread */
  std::vector<libint2::Engine> engines(nthreads);
  engines[0] = libint2::Engine(OP, max_nprim(shells), max_l(shells), 0);
  engines[0].set_params(params);
  for (int i = 1; i < nthreads; ++i) engines[i] = engines[0];

  const auto s2bf = map_shell_to_basis_function(shells);
  const int nS = static_cast<int>(shells.size());

  auto worker = [&](int tid) {
    const auto& buf = engines[tid].results();
    for (int s1 = 0; s1 < nS; ++s1) {
      int bf1 = s2bf[s1], n1 = shells[s1].size();
      for (int s2 = 0; s2 <= s1; ++s2) {
        if ( (s2 + s1*nS) % nthreads != tid) continue;
        int bf2 = s2bf[s2], n2 = shells[s2].size();

        engines[tid].compute(shells[s1], shells[s2]);

        for (unsigned op = 0; op < NP; ++op) {
          Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> block(buf[op], n1, n2);
          result[op].block(bf1,bf2,n1,n2) = block;
          if (s1 != s2)
            result[op].block(bf2,bf1,n2,n1) = block.transpose();
        }
      }
    }
  };
  parallel_do(worker, nthreads);
  return result;
}

/* ---------------- user-facing wrappers ------------------------------ */
Matrix overlap(const std::vector<libint2::Shell>& shells, int nthreads) {
  return compute_multipoles<libint2::Operator::overlap>(shells, nthreads)[0];
}

Matrix cross_overlap(const std::vector<libint2::Shell>& shells,
                     size_t n_ao, size_t n_prj, int nthreads) {
  Matrix S = overlap(shells, nthreads);           
  return S.block(0, n_ao, n_ao, n_prj);           
}

std::vector<Matrix> dipole(const std::vector<libint2::Shell>& shells,
                           const std::array<double,3>& origin,
                           int nthreads) {
  return compute_multipoles<libint2::Operator::emultipole1>(shells, nthreads, origin);
}

/* =======================================================================
 * HGH PROJECTOR OVERLAPS VIA DERIVATIVES
 * ======================================================================= */

double hgh_norm_prefactor(int l, int i, double r_l) {
    if (i < 1) throw std::runtime_error("Projector index 'i' must be 1-based.");
    double num = 1.0; 
    double gamma_arg = l + (4.0 * i - 1.0) * 0.5;
    double rl_pow    = std::pow(r_l, gamma_arg);
    double den       = rl_pow * std::sqrt(std::tgamma(gamma_arg));
    return num / den;
}

double libint_primitive_norm(int l, double alpha) {
    double res = std::pow(2.0 * alpha / M_PI, 0.75);
    double num = std::pow(4.0 * alpha, static_cast<double>(l) / 2.0);
    double den = 0.0;
    if (l > 0) {
        double dfact = 1.0;
        for (int i = 2*l-1; i > 0; i -= 2) {
            dfact *= i;
        }
        den = std::sqrt(dfact);
    } else {
        den = 1.0;
    }
    return res * (num / den);
}

Matrix compute_hgh_projector_overlaps(
    const std::vector<libint2::Shell>& ao_shells,
    const std::vector<HghProjectorParams>& projectors,
    int nthreads)
{
    const auto n_ao = nbasis(ao_shells);
    size_t n_proj_funcs = 0;
    for (const auto& p : projectors)
        n_proj_funcs += (2 * p.l + 1);
    Matrix B = Matrix::Zero(n_ao, n_proj_funcs);
    const auto ao_s2bf = map_shell_to_basis_function(ao_shells);

    int max_l_val = max_l(ao_shells);
    for (const auto& p : projectors)
        max_l_val = std::max(max_l_val, p.l);
    size_t max_nprim_val = max_nprim(ao_shells);

    libint2::Engine engine(libint2::Operator::overlap, max_nprim_val, max_l_val, 0);
    const auto& buf = engine.results();

    size_t proj_col_offset = 0;
    for (const auto& p : projectors) {
        const int l = p.l;
        const int i = p.i;
        const double r_l = p.r_l;
        const int n_funcs = 2 * l + 1;
        const int k = i - 1;
        const double alpha = 0.5 / (r_l * r_l);

        Matrix block = Matrix::Zero(n_ao, n_funcs);
        auto compute_block = [&](double a)->Matrix {
            libint2::Shell proj_shell{{a}, {{l, true, {1.0}}}, p.center};
            Matrix tmp(n_ao, n_funcs); tmp.setZero();
            for (size_t s1 = 0; s1 < ao_shells.size(); ++s1) {
                engine.compute(ao_shells[s1], proj_shell);
                Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                    computed_block(buf[0], ao_shells[s1].size(), n_funcs);
                tmp.block(ao_s2bf[s1], 0, ao_shells[s1].size(), n_funcs) = computed_block;
            }
            return tmp;
        };

        if      (k == 0) block = compute_block(alpha);
        else if (k == 1) {
            const double delta = 1e-6;
            block = - (compute_block(alpha + delta) - compute_block(alpha - delta)) / (2.0 * delta);
        }
        else if (k == 2) {
            const double delta = 1e-4;
            block = (compute_block(alpha + delta)
                   - 2.0 * compute_block(alpha)
                   + compute_block(alpha - delta)) / (delta * delta);
        }
        else
            throw std::runtime_error("HGH projector: k > 2 not implemented");

        double norm_fac = hgh_norm_prefactor(l, i, r_l) / libint_primitive_norm(l, alpha);
        block *= norm_fac;

        B.block(0, proj_col_offset, n_ao, n_funcs) = block;
        proj_col_offset += n_funcs;
    }
    return B;
}

/* =======================================================================
 * PBC OVERLAP MATRIX (MINIMUM IMAGE CONVENTION)
 * ======================================================================= */

inline Eigen::Vector3i nint3(const Eigen::Vector3d& u) {
  return Eigen::Vector3i{ (int)std::llround(u[0]),
                          (int)std::llround(u[1]),
                          (int)std::llround(u[2]) };
}

inline double shell_extent_bohr(const libint2::Shell& sh, double c = 4.0) {
  double amin = std::numeric_limits<double>::infinity();
  for (int p = 0; p < sh.nprim(); ++p)
    amin = std::min(amin, sh.alpha[p]);
  if (!std::isfinite(amin) || amin <= 0.0) return 0.0;
  return c / std::sqrt(amin);
}

inline void build_Tlist_screened(const Eigen::Matrix3d& A,
                                 const Eigen::Vector3i& m_mic,
                                 const Eigen::Vector3d& dR,
                                 double rcut_bohr,
                                 double extent_ij,
                                 std::vector<Eigen::Vector3d>& Tlist) {
  Tlist.clear();
  const Eigen::Vector3d T_mic = -A * m_mic.cast<double>();
  const double a0 = A.col(0).norm();
  const double a1 = A.col(1).norm();
  const double a2 = A.col(2).norm();

  auto nmax_for = [&](double a_len) -> int {
    if (!std::isfinite(rcut_bohr)) return 0;
    return std::max(0, (int)std::ceil((rcut_bohr + extent_ij) / std::max(1e-12, a_len)) + 1);
  };

  const int n0 = nmax_for(a0);
  const int n1 = nmax_for(a1);
  const int n2 = nmax_for(a2);
  for (int i = -n0; i <= n0; ++i) {
    for (int j = -n1; j <= n1; ++j) {
      for (int k = -n2; k <= n2; ++k) {
        const Eigen::Vector3d T = T_mic + A * Eigen::Vector3d(i, j, k);
        const double Rij = (dR + T).norm();
        if (!std::isfinite(rcut_bohr) || Rij <= (rcut_bohr + extent_ij))
          Tlist.push_back(T);
      }
    }
  }
}

Matrix overlap_pbc(const std::vector<libint2::Shell>& shells,
                   const Eigen::Matrix3d& lattice_A, 
                   double cutoff_A, 
                   int /*nthreads*/) {

  const size_t nsh = shells.size();
  const auto off = map_shell_to_basis_function(shells);
  const size_t nbf = off.back() + shells.back().size();

  Matrix S = Matrix::Zero(nbf, nbf);

  int max_nprim_v = 0, max_l_v = 0;
  for (const auto& sh : shells) {
    max_nprim_v = std::max<int>(max_nprim_v, sh.nprim());
    for (const auto& c : sh.contr) max_l_v = std::max<int>(max_l_v, c.l);
  }
  
  libint2::Engine eng(libint2::Operator::overlap, max_nprim_v, max_l_v, 0);
  eng.set_precision(1e-12);
  
  constexpr double BOHR_PER_ANG = 1.889726124565062;
  const bool use_MIC_only = !(cutoff_A > 0.0);
  const double rcut_bohr = use_MIC_only ?
                               std::numeric_limits<double>::infinity()
                               : cutoff_A * BOHR_PER_ANG;

  std::vector<double> ext(nsh, 0.0);
  for (size_t s = 0; s < nsh; ++s) ext[s] = shell_extent_bohr(shells[s], 4.0);
  const Eigen::Matrix3d Linv = lattice_A.inverse();
  std::vector<Eigen::Vector3d> Tlist; Tlist.reserve(64);

  for (size_t si = 0; si < nsh; ++si) {
    const auto& shi = shells[si];
    const size_t fi0 = off[si];
    const Eigen::Vector3d Ri(shi.O[0], shi.O[1], shi.O[2]);
    for (size_t sj = si; sj < nsh; ++sj) {
      const auto& shj = shells[sj];
      const size_t fj0 = off[sj];
      const Eigen::Vector3d Rj(shj.O[0], shj.O[1], shj.O[2]);
      const Eigen::Vector3d dR = Rj - Ri;
      const Eigen::Vector3d uvw = Linv * dR;
      const Eigen::Vector3i m_mic = nint3(uvw);
      const double extent_ij = ext[si] + ext[sj];
      Tlist.clear();

      if (use_MIC_only) {
        Tlist.emplace_back(-lattice_A * m_mic.cast<double>());
      } else {
        build_Tlist_screened(lattice_A, m_mic, dR, rcut_bohr, extent_ij, Tlist);
      }

      Matrix Sblock = Matrix::Zero(shi.size(), shj.size());
      for (const auto& T : Tlist) {
        libint2::Shell shj_shift = shj;
        shj_shift.O[0] += T[0];
        shj_shift.O[1] += T[1];
        shj_shift.O[2] += T[2];
        eng.compute(shi, shj_shift);
        const auto& buf = eng.results();
        if (!buf.empty() && buf[0] != nullptr) {
          const double* p = buf[0];
          for (int fi = 0; fi < (int)shi.size(); ++fi)
            for (int fj = 0; fj < (int)shj.size(); ++fj)
              Sblock(fi, fj) += p[fi * (int)shj.size() + fj];
        }
      }
      S.block(fi0, fj0, shi.size(), shj.size()) = Sblock;
      if (sj != si) S.block(fj0, fi0, shj.size(), shi.size()) = Sblock.transpose();
    }
  }
  return S;
}

} // namespace licpp

