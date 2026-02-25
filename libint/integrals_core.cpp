#include "integrals_core.hpp"
#include <libint2.hpp>
#include <vector>
#include <algorithm>
#include <cmath> // For std::tgamma and std::pow
#include <complex>   // <-- added
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

        // ======================= YOUR NEW DEBUG BLOCK =======================
//        const auto& s1_info = shells[s1];
//        const auto& s2_info = shells[s2];
//
        // Check if we have a (p,d) or (d,p) pair
//        bool is_pd_pair = (s1_info.contr[0].l == 1 && s2_info.contr[0].l == 2) || 
//                          (s1_info.contr[0].l == 2 && s2_info.contr[0].l == 1);

//        if (is_pd_pair) {
//            int buf_n1 = s1_info.size();
//            int buf_n2 = s2_info.size();
            
//            std::cout << "\n[SOC-LIBINT DEBUG]: Raw buffer for shell pair (" << s1 << ", " << s2 
//                      << ") with L=(" << s1_info.contr[0].l << "," << s2_info.contr[0].l << ")"
//                      << " size " << buf_n1 << "x" << buf_n2 << std::endl;

            // Print the raw buffer from Libint
//            for (int i = 0; i < buf_n1 * buf_n2; ++i) {
//                std::cout << buf[0][i] << " ";
//            }
//            std::cout << std::endl;
//        }
        // ====================================================================


        for (unsigned op = 0; op < NP; ++op) {
//          Eigen::Map<const Matrix> block(buf[op], n1, n2);
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
Matrix overlap(const std::vector<libint2::Shell>& shells, int nthreads)
{
  return compute_multipoles<libint2::Operator::overlap>(shells, nthreads)[0];
}

Matrix cross_overlap(const std::vector<libint2::Shell>& shells,
                     size_t n_ao, size_t n_prj, int nthreads)
{
  Matrix S = overlap(shells, nthreads);           // AO+proj × AO+proj
  return S.block(0, n_ao, n_ao, n_prj);           // top‑right block
}

std::vector<Matrix> dipole(const std::vector<libint2::Shell>& shells,
                           const std::array<double,3>& origin,
                           int nthreads)
{
  return compute_multipoles<libint2::Operator::emultipole1>(shells, nthreads, origin);
}

/* =======================================================================
 *
 * NEW FUNCTION TO COMPUTE HGH PROJECTOR OVERLAPS VIA DERIVATIVES
 *
 * ======================================================================= */

double hgh_norm_prefactor(int l, int i, double r_l) {
    if (i < 1) throw std::runtime_error("Projector index 'i' must be 1‑based.");
    /* The √2 is *not* part of the original GTH/HGH normalisation.      */
    /* (Goedecker et al., PRB 54 (1996) 1703, Eq. (6) and Table I.)     */
    double num = 1.0;                                   //  <── remove √2
    double gamma_arg = l + (4.0 * i - 1.0) * 0.5;
    double rl_pow    = std::pow(r_l, gamma_arg);
    double den       = rl_pow * std::sqrt(std::tgamma(gamma_arg));
    return num / den;
}

// Helper to compute the normalization constant for a single solid-harmonic GTO primitive
// This is needed to undo the automatic normalization applied by Libint.
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
    size_t proj_no = 0;
    for (const auto& p : projectors) {
        const int l = p.l;
        const int i = p.i;
        const double r_l = p.r_l;
        const int n_funcs = 2 * l + 1;
        const int k = i - 1;
        const double alpha = 0.5 / (r_l * r_l);

        // Build raw overlap block (finite-difference for k>0)
        Matrix block = Matrix::Zero(n_ao, n_funcs);
        auto compute_block = [&](double a)->Matrix {
            libint2::Shell proj_shell{{a}, {{l, true, {1.0}}}, p.center};

            // ======================= YOUR PRINT STATEMENTS =======================
            // Add these lines to print the information for the projector shell.
//            std::cout << "Projector Shell #" << proj_no 
//                      << ", L: " << l << std::endl;
            
            // Print the exponent 'a' for this shell
//            std::cout << "  Exponent: " << a << std::endl;
            
            // The coefficient for these uncontracted projectors is always {1.0}
//            std::cout << "  Coefficient: " << 1.0 << std::endl;
//            std::cout << "------------------------------------" << std::endl;
            // =====================================================================


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

        // Compute normalization factor for this projector
        double N_hgh = hgh_norm_prefactor(l, i, r_l);
        double N_gto = libint_primitive_norm(l, alpha);
        double norm_fac = N_hgh / N_gto;

        // Debug: print before/after normalization for each projector
        double max_before = block.cwiseAbs().maxCoeff();
        double max_after;
        // Multiply normalization
        block *= norm_fac;
        max_after = block.cwiseAbs().maxCoeff();

        //printf("[HGH DEBUG] proj#%zu (l=%d,i=%d, alpha=%.5g, r_l=%.5g): N_hgh=%.5g N_gto=%.5g norm_fac=%.5g max|block| before=%.6g after=%.6g\n",
        //    proj_no, l, i, alpha, r_l, N_hgh, N_gto, norm_fac, max_before, max_after);

        // Store result
        B.block(0, proj_col_offset, n_ao, n_funcs) = block;
        proj_col_offset += n_funcs;
        ++proj_no;
    }
    return B;
}

/* -----------------------------------------------------------------------
 *  Real spherical harmonics Y_l^m  in Libint SO order, up to l = 3.
 *  Inputs:  unit direction (x,y,z).  Output array length:
 *     l=0 → 1,  l=1 → 3,  l=2 → 5,  l=3 → 7+3 = 10
 *  Normalisations match Libint’s pure=SO convention (overall constants
 *  absorbed in radial prefactor; relative signs/√ factors are correct for
 *  interference).  Extend similarly for g (l=4) if ever needed.
 * --------------------------------------------------------------------- */
inline void rsh_array_l0(double, double, double, double* o) { o[0] = 1.0; }

inline void rsh_array_l1(double x,double y,double z,double* o) {
  o[0]=y;  o[1]=z;  o[2]=x;
}

inline void rsh_array_l2(double x,double y,double z,double* o) {
  const double r2 = 1.0;
  o[0]=x*y;             // m=-2
  o[1]=y*z;             // m=-1
  o[2]=3.0*z*z-r2;      // m= 0
  o[3]=z*x;             // m=+1
  o[4]=x*x-y*y;         // m=+2
}

inline void rsh_array_l3(double x,double y,double z,double* o) {
  const double r2 = 1.0;
  const double r2sq = r2*r2;
  /* SO order for l=3 : m = -3,-2,-1,0,+1,+2,+3  (7 funcs)
     Libint additionally stores the real tesseral family in
     alphabetical xy(y^2-3x^2) …  We mirror CP2K/Libint layout:  */
  o[0] = y*(3.0*x*x -   y*y);          // m=-3  (xyz combination)
  o[1] = x*y*z;                        // m=-2
  o[2] = y*z*z - y*r2/5.0*     3.0;    // m=-1  (~ yz(5z^2-r^2))
  o[3] = z*(5.0*z*z - 3.0*r2);         // m= 0
  o[4] = x*z*z - x*r2/5.0*     3.0;    // m=+1
  o[5] = z*(x*x - y*y);               // m=+2
  o[6] = x*(x*x - 3.0*y*y);           // m=+3

  /* Libint SO actually has 10 functions for l=3 (real cubic harmonics):
     the remaining 3 are linear‐independent but orthonormal combinations.
     For completeness we append them (indices 7‑9): */
  o[7] = x*y*(x*x - y*y);             // χ_7 (xy(x^2‑y^2))
  o[8] = (x*x - y*y)*(x*x - 3*y*y);   // χ_8 ((x^2‑y^2)(x^2‑3y^2))
  o[9] = (3*x*x - y*y)*y*z;           // χ_9 ((3x^2‑y^2)yz)
}

inline void rsh_array(int l,double x,double y,double z,double* o){
  switch(l){
    case 0: rsh_array_l0(x,y,z,o); break;
    case 1: rsh_array_l1(x,y,z,o); break;
    case 2: rsh_array_l2(x,y,z,o); break;
    case 3: rsh_array_l3(x,y,z,o); break;
    default:
      throw std::runtime_error("rsh_array: l>3 not implemented");
  }
}

/* ---- analytic FT of one Libint shell at a single k-point --------------- */
static std::vector<std::complex<double>>
shell_ft_complex(const libint2::Shell& sh,
                 const Eigen::Vector3d& k) {

  const double klen = k.norm();
  const int l = sh.contr[0].l;
  const size_t nfunc = sh.size();
  std::vector<std::complex<double>> vals(nfunc, 0.0);

  if (klen < 1e-15) {
    // limit k->0 (Gamma); rsh well-defined except for l>0 where k^l->0
    double ang[9]; // enough up to l=2
    rsh_array(l, 0.0, 0.0, 1.0, ang);  // arbitrary axis; k^l will zero l>0 anyway
    for (size_t p=0; p<sh.nprim(); ++p) {
      const double alpha = sh.alpha[p];
      const double coeff = sh.contr[0].coeff[p];
      const double pref  = coeff * std::pow(M_PI/alpha, 1.5)
                                 * std::pow(klen/(2*alpha), l)
                                 * std::exp(-klen*klen/(4*alpha));
      for (size_t f=0; f<nfunc; ++f)
        vals[f] += pref * ang[f];
    }
    // phase = 1 at k=0
    return vals;
  }

  const double x = k(0)/klen, y = k(1)/klen, z = k(2)/klen;
  double ang[9]; rsh_array(l, x,y,z, ang);

  for (size_t p = 0; p < sh.nprim(); ++p) {
    const double alpha = sh.alpha[p];
    const double coeff = sh.contr[0].coeff[p];   // <-- fixed name
    const double pref  = coeff * std::pow(M_PI/alpha, 1.5)
                               * std::pow(klen/(2*alpha), l)
                               * std::exp(-klen*klen/(4*alpha));
    for (size_t f = 0; f < nfunc; ++f)
      vals[f] += pref * ang[f];
  }

  // phase factor e^{-i k·R}
  const Eigen::Vector3d R{sh.O[0], sh.O[1], sh.O[2]};
  const std::complex<double> phase =
      std::exp(std::complex<double>(0.0, -k.dot(R)));
  for (auto& v : vals) v *= phase;
  return vals;
}

/* real-return shim (drop imag) to keep API backward compatible */
static std::vector<double>
shell_ft_real(const libint2::Shell& sh,
              const Eigen::Vector3d& k) {
  auto tmp = shell_ft_complex(sh,k);
  std::vector<double> out(tmp.size());
  for (size_t i=0;i<tmp.size();++i) out[i] = tmp[i].real(); // drop imag
  return out;
}

/* AO-FT matrix: (n_ao × n_k) real-valued (imag dropped) ------------------ */
Matrix ao_ft(const std::vector<libint2::Shell>& shells,
             const std::vector<KPoint>& kpts,
             int nthreads) {

  const size_t n_ao = nbasis(shells);
  const size_t n_k  = kpts.size();
  Matrix F(n_ao, n_k); F.setZero();
  const auto s2bf = map_shell_to_basis_function(shells);

  parallel_do([&](int tid){
    for (size_t kp = 0; kp < n_k; ++kp) {
      if (kp % nthreads != (size_t)tid) continue;
      Eigen::Vector3d kvec = kpts[kp].vec();
      size_t ao0 = 0;
      for (size_t s = 0; s < shells.size(); ++s) {
        auto vec = shell_ft_real(shells[s], kvec);
        for (size_t f = 0; f < vec.size(); ++f)
          F(ao0 + f, kp) = vec[f];
        ao0 += vec.size();
      }
    }
  }, nthreads);

  return F;
}

// --- in licpp namespace (integrals_core.cpp) ---

static std::vector<size_t> shell2bf(const std::vector<libint2::Shell>& shells) {
    std::vector<size_t> off(shells.size());
    size_t bf = 0;
    for (size_t s = 0; s < shells.size(); ++s) {
        off[s] = bf;
        bf += shells[s].size();
    }
    return off;
}

// nearest-integer for 3D vector
inline Eigen::Vector3i nint3(const Eigen::Vector3d& u) {
  return Eigen::Vector3i{ (int)std::llround(u[0]),
                          (int)std::llround(u[1]),
                          (int)std::llround(u[2]) };
}

// very simple shell extent ~ c / sqrt(alpha_min)  [Bohr]
inline double shell_extent_bohr(const libint2::Shell& sh, double c = 4.0) {
  double amin = std::numeric_limits<double>::infinity();
  for (int p = 0; p < sh.nprim(); ++p)
    amin = std::min(amin, sh.alpha[p]);
  if (!std::isfinite(amin) || amin <= 0.0) return 0.0;
  return c / std::sqrt(amin);
}

// build translations around MIC within cutoff + extent (S_PBC only)
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
                   const Eigen::Matrix3d& lattice_A, // Bohr
                   double cutoff_A, // Å (converted)
                   int /*nthreads*/) {

  const size_t nsh = shells.size();
  const auto off = shell2bf(shells);
  const size_t nbf = off.back() + shells.back().size();

  Matrix S = Matrix::Zero(nbf, nbf);

  // libint engine
  int max_nprim = 0, max_l = 0;
  for (const auto& sh : shells) {
    max_nprim = std::max<int>(max_nprim, sh.nprim());
    for (const auto& c : sh.contr) max_l = std::max<int>(max_l, c.l);
  }
  libint2::Engine eng(libint2::Operator::overlap, max_nprim, max_l, 0);
  eng.set_precision(1e-12);

  constexpr double BOHR_PER_ANG = 1.889726124565062;
  const bool use_MIC_only = !(cutoff_A > 0.0);
  const double rcut_bohr = use_MIC_only ? std::numeric_limits<double>::infinity()
                                        : cutoff_A * BOHR_PER_ANG;

  // shell extents
  std::vector<double> ext(nsh, 0.0);
  for (size_t s = 0; s < nsh; ++s) ext[s] = shell_extent_bohr(shells[s], 4.0);

  // inverse lattice for MIC
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

      // new: integer triplet for MIC
      const Eigen::Vector3d uvw = Linv * dR;
      const Eigen::Vector3i m_mic = nint3(uvw);

      // use robust translation list
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
        shj_shift.O[0] = shj.O[0] + T[0];
        shj_shift.O[1] = shj.O[1] + T[1];
        shj_shift.O[2] = shj.O[2] + T[2];
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

/* ---- Concatenate AO values for all shells on many points ---------------- */

// --- fast AO values on arbitrary points (shell-major, vectorized) ----------
Eigen::MatrixXd ao_values_at_points(
    const std::vector<libint2::Shell>& shells,
    const std::vector<Eigen::Vector3d>& points,
    int nthreads) {

  const size_t npts = points.size();
  const size_t nao  = nbasis(shells);
  Eigen::MatrixXd A(nao, npts);
  A.setZero();

  // Flatten points into three Eigen vectors for cache-friendly math
  Eigen::VectorXd Px(npts), Py(npts), Pz(npts);
  for (size_t i = 0; i < npts; ++i) {
    Px[i] = points[i][0];
    Py[i] = points[i][1];
    Pz[i] = points[i][2];
  }

  const auto s2bf = map_shell_to_basis_function(shells);

  #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
  for (ptrdiff_t s = 0; s < (ptrdiff_t)shells.size(); ++s) {
    const auto& sh = shells[(size_t)s];
    const int l = sh.contr[0].l;
    const size_t nfunc = sh.size();
    const double Cx = sh.O[0], Cy = sh.O[1], Cz = sh.O[2];

    // Coordinates relative to shell center
    Eigen::ArrayXd x = Px.array() - Cx;
    Eigen::ArrayXd y = Py.array() - Cy;
    Eigen::ArrayXd z = Pz.array() - Cz;
    Eigen::ArrayXd xx = x*x, yy = y*y, zz = z*z;
    Eigen::ArrayXd r2 = xx + yy + zz;

    // Radial contraction: sum_p coeff_p * exp(-alpha_p * r^2)
    // (Nl = 1.0 here to avoid double normalization)
    Eigen::ArrayXd radial = Eigen::ArrayXd::Zero(npts);
    for (size_t p = 0; p < sh.nprim(); ++p) {
      const double a = sh.alpha[p];
      const double c = sh.contr[0].coeff[p];
      radial += c * (-a * r2).exp();
    }

    // Solid-harmonic polynomials (SO order) multiplied by 'radial'
    Eigen::MatrixXd block(nfunc, npts);
    if (l == 0) {
      block.row(0) = radial.matrix();
    } else if (l == 1) {
      // p: [y, z, x]
      block.row(0) = (radial * y).matrix();
      block.row(1) = (radial * z).matrix();
      block.row(2) = (radial * x).matrix();
    } else if (l == 2) {
      // d (SO): [xy, yz, (3z^2 - r^2), zx, (x^2 - y^2)]
      block.row(0) = (radial * (x*y)).matrix();
      block.row(1) = (radial * (y*z)).matrix();
      block.row(2) = (radial * (3.0*zz - (xx+yy+zz))).matrix();
      block.row(3) = (radial * (z*x)).matrix();
      block.row(4) = (radial * (xx - yy)).matrix();
    } else if (l == 3) {
      // f (SO): [ y(3x^2-y^2), 2xyz, y(5z^2 - r^2), z(5z^2 - 3r^2),
      //           x(5z^2 - r^2), z(x^2 - y^2), x(x^2 - 3y^2) ]
      Eigen::ArrayXd r2 = (xx + yy + zz);
      block.row(0) = (radial * (y * (3.0*xx - yy))).matrix();
      block.row(1) = (radial * (2.0*x*y*z)).matrix();
      block.row(2) = (radial * (y * (5.0*zz - r2))).matrix();
      block.row(3) = (radial * (z * (5.0*zz - 3.0*r2))).matrix();
      block.row(4) = (radial * (x * (5.0*zz - r2))).matrix();
      block.row(5) = (radial * (z * (xx - yy))).matrix();
      block.row(6) = (radial * (x * (xx - 3.0*yy))).matrix();
    } else {
      continue;
    }
    
    const size_t bf0 = s2bf[(size_t)s];
    A.middleRows((Eigen::Index)bf0, (Eigen::Index)nfunc) = block;
  }

  return A;
}



} // namespace licpp
