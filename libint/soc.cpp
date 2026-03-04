#include "soc.hpp"
#include <cmath>
#include <stdexcept>
#include <thread>

namespace licpp {

// Simple parallel launcher for ao_ft
template <typename Lambda>
inline void parallel_do_soc(Lambda&& body, int nthreads) {
#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
  { body(omp_get_thread_num()); }
#else
  std::vector<std::thread> workers;
  for (int id = 1; id < nthreads; ++id) workers.emplace_back(body, id);
  body(0);
  for (auto& t : workers) t.join();
#endif
}

// =======================================================================
// 1. HGH PROJECTOR OVERLAPS
// =======================================================================
double hgh_norm_prefactor(int l, int i, double r_l) {
    if (i < 1) throw std::runtime_error("Projector index 'i' must be 1-based.");
    double num = 1.0; 
    double gamma_arg = l + (4.0 * i - 1.0) * 0.5;
    double rl_pow = std::pow(r_l, gamma_arg);
    double den = rl_pow * std::sqrt(std::tgamma(gamma_arg));
    return num / den;
}

double libint_primitive_norm(int l, double alpha) {
    double res = std::pow(2.0 * alpha / M_PI, 0.75);
    double num = std::pow(4.0 * alpha, static_cast<double>(l) / 2.0);
    double den = 1.0;
    if (l > 0) {
        double dfact = 1.0;
        for (int i = 2*l-1; i > 0; i -= 2) dfact *= i;
        den = std::sqrt(dfact);
    }
    return res * (num / den);
}

Eigen::MatrixXd compute_hgh_projector_overlaps(
    const std::vector<libint2::Shell>& ao_shells,
    const std::vector<HghProjectorParams>& projectors,
    int nthreads)
{
    const auto n_ao = nbasis(ao_shells);
    size_t n_proj_funcs = 0;
    for (const auto& p : projectors) n_proj_funcs += (2 * p.l + 1);
    
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n_ao, n_proj_funcs);
    const auto ao_s2bf = map_shell_to_basis_function(ao_shells);

    int max_l_val = max_l(ao_shells);
    for (const auto& p : projectors) max_l_val = std::max(max_l_val, p.l);
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

        auto compute_block = [&](double a) -> Eigen::MatrixXd {
            libint2::Shell proj_shell{{a}, {{l, true, {1.0}}}, p.center};
            Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(n_ao, n_funcs);
            for (size_t s1 = 0; s1 < ao_shells.size(); ++s1) {
                engine.compute(ao_shells[s1], proj_shell);
                Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                    computed_block(buf[0], ao_shells[s1].size(), n_funcs);
                tmp.block(ao_s2bf[s1], 0, ao_shells[s1].size(), n_funcs) = computed_block;
            }
            return tmp;
        };

        Eigen::MatrixXd block = Eigen::MatrixXd::Zero(n_ao, n_funcs);
        if (k == 0) block = compute_block(alpha);
        else if (k == 1) {
            const double delta = 1e-6;
            block = -(compute_block(alpha + delta) - compute_block(alpha - delta)) / (2.0 * delta);
        } else if (k == 2) {
            const double delta = 1e-4;
            block = (compute_block(alpha + delta) - 2.0 * compute_block(alpha) + compute_block(alpha - delta)) / (delta * delta);
        } else throw std::runtime_error("HGH projector: k > 2 not implemented");

        double norm_fac = hgh_norm_prefactor(l, i, r_l) / libint_primitive_norm(l, alpha);
        block *= norm_fac;

        B.block(0, proj_col_offset, n_ao, n_funcs) = block;
        proj_col_offset += n_funcs;
    }
    return B;
}

// =======================================================================
// 2. REAL SPHERICAL HARMONICS & ANALYTIC FT
// =======================================================================
inline void rsh_array_l0(double, double, double, double* o) { o[0] = 1.0; }
inline void rsh_array_l1(double x,double y,double z,double* o) { o[0]=y; o[1]=z; o[2]=x; }
inline void rsh_array_l2(double x,double y,double z,double* o) {
  const double r2 = 1.0;
  o[0]=x*y; o[1]=y*z; o[2]=3.0*z*z-r2; o[3]=z*x; o[4]=x*x-y*y;
}
inline void rsh_array_l3(double x,double y,double z,double* o) {
  const double r2 = 1.0;
  o[0] = y*(3.0*x*x - y*y); 
  o[1] = x*y*z; 
  o[2] = y*z*z - y*r2/5.0*3.0; 
  o[3] = z*(5.0*z*z - 3.0*r2); 
  o[4] = x*z*z - x*r2/5.0*3.0; 
  o[5] = z*(x*x - y*y); 
  o[6] = x*(x*x - 3.0*y*y); 
  o[7] = x*y*(x*x - y*y); 
  o[8] = (x*x - y*y)*(x*x - 3*y*y); 
  o[9] = (3*x*x - y*y)*y*z; 
}
inline void rsh_array(int l,double x,double y,double z,double* o){
  switch(l){
    case 0: rsh_array_l0(x,y,z,o); break;
    case 1: rsh_array_l1(x,y,z,o); break;
    case 2: rsh_array_l2(x,y,z,o); break;
    case 3: rsh_array_l3(x,y,z,o); break;
    default: throw std::runtime_error("rsh_array: l>3 not implemented");
  }
}

static std::vector<std::complex<double>> shell_ft_complex(const libint2::Shell& sh, const Eigen::Vector3d& k) {
  const double klen = k.norm();
  const int l = sh.contr[0].l;
  const size_t nfunc = sh.size();
  std::vector<std::complex<double>> vals(nfunc, 0.0);

  if (klen < 1e-15) {
    double ang[9]; rsh_array(l, 0.0, 0.0, 1.0, ang);
    for (size_t p=0; p<sh.nprim(); ++p) {
      const double alpha = sh.alpha[p];
      const double coeff = sh.contr[0].coeff[p];
      const double pref  = coeff * std::pow(M_PI/alpha, 1.5) * std::pow(klen/(2*alpha), l) * std::exp(-klen*klen/(4*alpha));
      for (size_t f=0; f<nfunc; ++f) vals[f] += pref * ang[f];
    }
    return vals;
  }

  const double x = k(0)/klen, y = k(1)/klen, z = k(2)/klen;
  double ang[9]; rsh_array(l, x,y,z, ang);

  for (size_t p = 0; p < sh.nprim(); ++p) {
    const double alpha = sh.alpha[p];
    const double coeff = sh.contr[0].coeff[p];
    const double pref  = coeff * std::pow(M_PI/alpha, 1.5) * std::pow(klen/(2*alpha), l) * std::exp(-klen*klen/(4*alpha));
    for (size_t f = 0; f < nfunc; ++f) vals[f] += pref * ang[f];
  }

  const Eigen::Vector3d R{sh.O[0], sh.O[1], sh.O[2]};
  const std::complex<double> phase = std::exp(std::complex<double>(0.0, -k.dot(R)));
  for (auto& v : vals) v *= phase;
  return vals;
}

static std::vector<double> shell_ft_real(const libint2::Shell& sh, const Eigen::Vector3d& k) {
  auto tmp = shell_ft_complex(sh,k);
  std::vector<double> out(tmp.size());
  for (size_t i=0;i<tmp.size();++i) out[i] = tmp[i].real();
  return out;
}

Eigen::MatrixXd ao_ft(const std::vector<libint2::Shell>& shells, const std::vector<KPoint>& kpts, int nthreads) {
  const size_t n_ao = nbasis(shells);
  const size_t n_k  = kpts.size();
  Eigen::MatrixXd F = Eigen::MatrixXd::Zero(n_ao, n_k);

  parallel_do_soc([&](int tid){
    for (size_t kp = 0; kp < n_k; ++kp) {
      if (kp % nthreads != (size_t)tid) continue;
      Eigen::Vector3d kvec = kpts[kp].vec();
      size_t ao0 = 0;
      for (size_t s = 0; s < shells.size(); ++s) {
        auto vec = shell_ft_real(shells[s], kvec);
        for (size_t f = 0; f < vec.size(); ++f) F(ao0 + f, kp) = vec[f];
        ao0 += vec.size();
      }
    }
  }, nthreads);
  return F;
}

// =======================================================================
// 3. PBC OVERLAPS
// =======================================================================
static std::vector<size_t> shell2bf(const std::vector<libint2::Shell>& shells) {
    std::vector<size_t> off(shells.size());
    size_t bf = 0;
    for (size_t s = 0; s < shells.size(); ++s) { off[s] = bf; bf += shells[s].size(); }
    return off;
}

inline Eigen::Vector3i nint3(const Eigen::Vector3d& u) {
  return Eigen::Vector3i{ (int)std::llround(u[0]), (int)std::llround(u[1]), (int)std::llround(u[2]) };
}

inline double shell_extent_bohr(const libint2::Shell& sh, double c = 4.0) {
  double amin = std::numeric_limits<double>::infinity();
  for (int p = 0; p < sh.nprim(); ++p) amin = std::min(amin, sh.alpha[p]);
  if (!std::isfinite(amin) || amin <= 0.0) return 0.0;
  return c / std::sqrt(amin);
}

inline void build_Tlist_screened(const Eigen::Matrix3d& A, const Eigen::Vector3i& m_mic,
                                 const Eigen::Vector3d& dR, double rcut_bohr, double extent_ij,
                                 std::vector<Eigen::Vector3d>& Tlist) {
  Tlist.clear();
  const Eigen::Vector3d T_mic = -A * m_mic.cast<double>();
  auto nmax_for = [&](double a_len) -> int {
    if (!std::isfinite(rcut_bohr)) return 0;
    return std::max(0, (int)std::ceil((rcut_bohr + extent_ij) / std::max(1e-12, a_len)) + 1);
  };
  const int n0 = nmax_for(A.col(0).norm());
  const int n1 = nmax_for(A.col(1).norm());
  const int n2 = nmax_for(A.col(2).norm());

  for (int i = -n0; i <= n0; ++i) {
    for (int j = -n1; j <= n1; ++j) {
      for (int k = -n2; k <= n2; ++k) {
        const Eigen::Vector3d T = T_mic + A * Eigen::Vector3d(i, j, k);
        if (!std::isfinite(rcut_bohr) || (dR + T).norm() <= (rcut_bohr + extent_ij)) Tlist.push_back(T);
      }
    }
  }
}

Eigen::MatrixXd overlap_pbc(const std::vector<libint2::Shell>& shells,
                            const Eigen::Matrix3d& lattice_A, double cutoff_A, int nthreads) {
  const size_t nsh = shells.size();
  const auto off = shell2bf(shells);
  const size_t nbf = off.back() + shells.back().size();

  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(nbf, nbf);
  int max_nprim_v = 0, max_l_v = 0;
  for (const auto& sh : shells) {
    max_nprim_v = std::max<int>(max_nprim_v, sh.nprim());
    for (const auto& c : sh.contr) max_l_v = std::max<int>(max_l_v, c.l);
  }
  libint2::Engine eng(libint2::Operator::overlap, max_nprim_v, max_l_v, 0);
  eng.set_precision(1e-12);

  const double BOHR_PER_ANG = 1.889726124565062;
  const bool use_MIC_only = !(cutoff_A > 0.0);
  const double rcut_bohr = use_MIC_only ? std::numeric_limits<double>::infinity() : cutoff_A * BOHR_PER_ANG;

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

      if (use_MIC_only) Tlist = {-lattice_A * m_mic.cast<double>()};
      else build_Tlist_screened(lattice_A, m_mic, dR, rcut_bohr, extent_ij, Tlist);

      Eigen::MatrixXd Sblock = Eigen::MatrixXd::Zero(shi.size(), shj.size());
      for (const auto& T : Tlist) {
        libint2::Shell shj_shift = shj;
        shj_shift.O[0] += T[0]; shj_shift.O[1] += T[1]; shj_shift.O[2] += T[2];
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

