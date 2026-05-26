import numpy as np
import time
from miniBSE.davidson import davidson
from miniBSE.exciton_hamiltonian import ExcitonHamiltonian

# --- UPDATED IMPORTS ---
from miniBSE.hardness import build_gamma, build_resta_mnok


def _assemble_truncated_exchange(q_hole, w_elec, vi, va, block_size=128, max_full_elements=12_000_000):
    """Build K[p,q] = sum_A q_hole[i_p,i_q,A]^* W_elec[a_p,a_q,A]."""
    n_p = len(vi)
    dtype = np.result_type(q_hole, w_elec)

    n_occ, _, n_atoms = q_hole.shape
    n_virt = w_elec.shape[0]
    full_dim = n_occ * n_virt

    # Dense diagonalization already needs an n_p x n_p matrix.  When the full
    # occ-virt grid is modest, assemble the cached exchange tensor with one
    # BLAS-friendly contraction and slice the truncated space from it.  This is
    # much faster than repeatedly gathering screened pairs into temporary
    # blocks, which regressed the 2500x2500 dense path.
    if full_dim * full_dim <= max_full_elements:
        K_full = np.einsum("ijA,abA->iajb", q_hole.conj(), w_elec, optimize=True)
        K_2d = K_full.reshape(full_dim, full_dim)
        idx = vi * n_virt + va
        return np.ascontiguousarray(K_2d[np.ix_(idx, idx)])

    K = np.empty((n_p, n_p), dtype=dtype)

    for p0 in range(0, n_p, block_size):
        p1 = min(p0 + block_size, n_p)
        qh = q_hole[vi[p0:p1, None], vi[None, :], :].conj()
        we = w_elec[va[p0:p1, None], va[None, :], :]
        K[p0:p1, :] = np.einsum("pqA,pqA->pq", qh, we, optimize=True)

    return K

class ExcitonSolver:
    def __init__(self, C, eps, occ, overlap, atom_symbols, atom_coords, atom_ao_ranges, 
                 homo_index, n_occ, n_virt, scissor_ev, kernel, alpha, beta=0.0, material=None, 
                 include_exchange=False, estimate_qp=False, e_thresh=None, f_thresh=0.0, 
                 mu_ia_x=None, mu_ia_y=None, mu_ia_z=None, eps_out=2.0, 
                 soc_U=None, soc_E=None, device="numpy", precomputed_sigma=None, 
                 vxc_ao_path=None, nthreads=1, spin='singlet', 
                 C_beta=None, eps_beta=None, homo_index_beta=None, charge_type='mulliken',
                 n_occ_beta=None, n_virt_beta=None):

        self.C = C
        self.overlap = overlap
        self.atom_ao_ranges = atom_ao_ranges
        self.n_occ = n_occ
        self.n_virt = n_virt
        self.homo_index = homo_index
        self.soc_flag = (soc_U is not None and soc_E is not None)
        self.spin = spin
        self.C_beta = C_beta
        self.eps_beta = eps_beta
        self.homo_index_beta = homo_index_beta

        # --- UPDATED KERNEL LOGIC in solver.py ---
        if kernel.lower() == "resta":
            print(f"  [Solver] Using Screened Resta-MNOK kernel for material: {material}")
            gamma_qp, gamma_bse = build_resta_mnok(
                atom_symbols=atom_symbols, coords=atom_coords,
                alpha=alpha, material_name=material, eps_out=eps_out
            )
        else:
            print(f"  [Solver] Using standard Grimme sTDA MNOK kernel.")
            print(f"           -> Screened W/BSE (alpha = {alpha:.3f})")
            g = build_gamma(atom_symbols=atom_symbols, coords=atom_coords, alpha=alpha, beta=0.0)
            gamma_qp, gamma_bse = g, g

        print(f"  [Solver] Building Bare Kernel V (alpha = 1.000, beta = 0.000)")
        # This MUST be beta=0.0 to preserve your baseline COH polarization!
        gamma_bare = build_gamma(atom_symbols=atom_symbols, coords=atom_coords, alpha=1.0, beta=0.0)

        print(f"  [Solver] Building Exact Exchange Penalty (beta = {beta:.3f})")
        # Build the stiffened matrix, and subtract the baseline to isolate ONLY the penalty
        gamma_beta_full = build_gamma(atom_symbols=atom_symbols, coords=atom_coords, alpha=1.0, beta=beta)
        gamma_penalty = gamma_beta_full - gamma_bare

        self.ham = ExcitonHamiltonian(
            C=C, eps=eps, overlap=overlap, atom_ao_ranges=atom_ao_ranges,
            homo_index=homo_index, n_occ=n_occ, n_virt=n_virt, scissor_ev=scissor_ev,
            gamma_qp=gamma_qp,        
            gamma_bse=gamma_bse,      
            material=material,
            gamma_bare=gamma_bare,
            gamma_penalty=gamma_penalty,
            alpha=alpha,            
            include_exchange=include_exchange, estimate_qp=estimate_qp, e_thresh=e_thresh, 
            f_thresh=f_thresh, mu_ia_x=mu_ia_x, mu_ia_y=mu_ia_y, mu_ia_z=mu_ia_z, 
            soc_U=soc_U, soc_E=soc_E, device=device, precomputed_sigma=precomputed_sigma,
            vxc_ao_path=vxc_ao_path, nthreads=nthreads, spin=spin,
            C_beta=C_beta, eps_beta=eps_beta, homo_index_beta=homo_index_beta,
            charge_type=charge_type,
            n_occ_beta=n_occ_beta, n_virt_beta=n_virt_beta
        )

    def solve(self, nroots=10, full_diag=False, tol=1e-5):
        if self.ham.dim == 0:
            print("ERROR: Active space dimension is 0! Your energy threshold is filtering out all transitions.")
            import sys; sys.exit(1)

        if full_diag:
            print(f"  Building dense Hamiltonian in truncated space ({self.ham.dim}x{self.ham.dim})...")
            
            if not self.soc_flag:
                # ==========================================================
                # SPATIAL DENSE BUILDER (Spin-Free Singlets or Triplets)
                # ==========================================================
                is_triplet = getattr(self.ham, 'spin', 'singlet') == 'triplet'
                is_uks_sp  = getattr(self.ham, 'spin', 'singlet') == 'uks_spin_preserving'

                if is_triplet:
                    print("  [Dense] Building Triplet Hamiltonian (No Coulomb J)...")
                    H = np.diag(self.ham.D).copy()
                    self.J_mat = np.zeros((self.ham.dim, self.ham.dim))
                    self.K_mat = np.zeros_like(self.J_mat)
                    if self.ham.include_exchange:
                        print("  [Dense] Building Triplet Exchange matrix (-K)...")
                        t1 = time.time()
                        vi, va = self.ham.valid_i, self.ham.valid_a
                        K_truncated = _assemble_truncated_exchange(
                            self.ham.q_occ, self.ham.W_virt, vi, va
                        )
                        H -= K_truncated
                        self.K_mat = K_truncated
                        print(f"    -> Triplet K built in {time.time()-t1:.2f}s")

                elif is_uks_sp:
                    # ----------------------------------------------------------
                    # UKS SPIN-PRESERVING (Manifold B) DENSE BUILDER
                    # ----------------------------------------------------------
                    print("  [Dense] Building UKS Spin-Preserving Hamiltonian (Manifold B)...")
                    t0 = time.time()

                    # Coulomb J: couples all transitions (alpha and beta) with factor 1.0
                    # q_flat is [dim_a + dim_b, n_atoms] — the unified charge array
                    temp = self.ham.q_flat @ self.ham.gamma
                    J_mat = temp @ self.ham.q_flat.T
                    H = np.diag(self.ham.D) + 1.0 * J_mat
                    self.J_mat = 1.0 * J_mat
                    self.K_mat = np.zeros_like(J_mat)
                    print(f"    -> Coulomb built in {time.time()-t0:.2f}s")

                    if self.ham.include_exchange:
                        print("  [Dense] Building block-diagonal Exchange matrix (-K_alpha, -K_beta)...")
                        t1 = time.time()
                        dim_a = self.ham.dim_a
                        dim_b = self.ham.dim_b
                        vi_a, va_a = self.ham.vi_a, self.ham.va_a
                        vi_b, va_b = self.ham.vi_b, self.ham.va_b
                        n_a = len(vi_a)
                        n_b = len(vi_b)

                        # Alpha block K_alpha
                        K_full = np.zeros((self.ham.dim, self.ham.dim))
                        if n_a > 0:
                            K_alpha = _assemble_truncated_exchange(
                                self.ham.q_occ_a, self.ham.W_virt_a, vi_a, va_a
                            )
                            K_full[:dim_a, :dim_a] = K_alpha

                        # Beta block K_beta
                        if n_b > 0:
                            K_beta = _assemble_truncated_exchange(
                                self.ham.q_occ_b, self.ham.W_virt_b, vi_b, va_b
                            )
                            K_full[dim_a:, dim_a:] = K_beta

                        H -= K_full
                        self.K_mat = K_full
                        print(f"    -> Exchange built in {time.time()-t1:.2f}s")

                else:
                    print("  [Dense] Building Coulomb term (2J)...")
                    t0 = time.time()
                    temp = self.ham.q_flat @ self.ham.gamma
                    J_mat = temp @ self.ham.q_flat.T
                    H = np.diag(self.ham.D) + 2.0 * J_mat
                    self.J_mat = 2.0 * J_mat
                    self.K_mat = np.zeros_like(J_mat)
                    print(f"    -> Coulomb built in {time.time()-t0:.2f}s")
                
                if self.ham.include_exchange and not is_uks_sp and not is_triplet:
                    print("  [Dense] Building Exchange matrix (-K)...")
                    t1 = time.time()
                    c_x = getattr(self.ham, 'c_x', 1.0) 
                    
                    vi, va = self.ham.valid_i, self.ham.valid_a
                    
                    print("    -> Constructing K_truncated from cached screened pairs...")
                    t_k = time.time()
                    K_truncated = _assemble_truncated_exchange(
                        self.ham.q_occ, self.ham.W_virt, vi, va
                    )
                    
                    print(f"    -> K_truncated assembled in {time.time()-t_k:.2f}s")
                    
                    H -= c_x * K_truncated
                    self.K_mat = c_x * K_truncated
                    print(f"    -> Total Exchange (-K) built in {time.time()-t1:.2f}s")
 
            else:
                # ==========================================================
                # SPINOR DENSE BUILDER (Relativistic Spin-Orbit)
                # ==========================================================
                print("  [Dense-SOC] Building Relativistic Coulomb term (J)...")
                t0 = time.time()
                # Notice: No factor of 2.0, and requires complex conjugate transpose
                temp = self.ham.q_spinor.conj() @ self.ham.gamma
                J_mat = temp @ self.ham.q_spinor.T
                H = np.diag(self.ham.D).astype(complex) + J_mat
                self.J_mat = J_mat
                self.K_mat = np.zeros_like(J_mat)
                print(f"    -> Coulomb built in {time.time()-t0:.2f}s")

                if self.ham.include_exchange:
                    print("  [Dense-SOC] Building Relativistic Exchange matrix (-K)...")
                    t1 = time.time()
                    
                    n_occ_sp = self.ham.n_occ_spinor
                    n_virt_sp = self.ham.n_virt_spinor
                    
                    if hasattr(self.ham, 'valid_spinor_idx'):
                        v_idx = self.ham.valid_spinor_idx
                        vi_sp = v_idx // n_virt_sp
                        va_sp = v_idx % n_virt_sp
                        
                        print("    -> Constructing K_truncated from cached screened pairs...")
                        t_k = time.time()
                        W_elec = self.ham.W_elec_spinor
                        K_truncated = _assemble_truncated_exchange(
                            self.ham.q_hole_spinor, W_elec, vi_sp, va_sp
                        )
                        
                        print(f"    -> K_truncated assembled in {time.time()-t_k:.2f}s")
                        
                        H -= K_truncated
                        self.K_mat = K_truncated
                    else:
                        # Fallback
                        print("    -> Contracting W = Gamma @ q_elec_spinor ...")
                        t_w = time.time()
                        W = self.ham.W_elec_spinor
                        print(f"    -> W contracted in {time.time()-t_w:.2f}s")
                        
                        print("    -> Contracting K_full in Spinor basis...")
                        t_k = time.time()
                        K_full = np.tensordot(self.ham.q_hole_spinor.conj(), W, axes=([2], [2]))
                        K_full_trans = np.transpose(K_full, (0, 2, 1, 3))
                        K_2d = K_full_trans.reshape(n_occ_sp * n_virt_sp, n_occ_sp * n_virt_sp)
                        print(f"    -> K_full completed in {time.time()-t_k:.2f}s")

                        H -= K_2d
                        self.K_mat = K_2d
 
                    print(f"    -> Total Exchange (-K) built in {time.time()-t1:.2f}s")

            # --- 3. Diagonalization ---
            print(f"  [Dense] Diagonalizing {self.ham.dim}x{self.ham.dim} matrix...")
            t_diag = time.time()
            evals, evecs = np.linalg.eigh(H)
            print(f"    -> Diagonalization complete in {time.time()-t_diag:.2f}s")
            
            return evals, evecs

        # Davidson Solver Fallback
        nroots = min(nroots, self.ham.dim - 1)
        print(f"  Using Davidson solver on {nroots} roots out of {self.ham.dim} transitions")
        return davidson(self.ham.matvec, self.ham.D, nroots, tol=tol)

    def main_transition(self, vec):
        """Extracts the dominant hole and electron indices from a state vector."""
        idx = np.argmax(np.abs(vec))
        if getattr(self.ham, 'spin', 'singlet') == 'uks_spin_preserving' and not self.soc_flag:
            # Identify which spin channel the dominant contribution belongs to
            if idx < self.ham.dim_a:
                hole = self.ham.vi_a[idx]
                elec = self.ham.va_a[idx]
            else:
                idx_b = idx - self.ham.dim_a
                hole = self.ham.vi_b[idx_b]
                elec = self.ham.va_b[idx_b]
            return hole, elec, abs(vec[idx])
        if not self.soc_flag:
            hole = self.ham.valid_i[idx]
            elec = self.ham.valid_a[idx]
        else:
            # Map the truncated index back to the full spinor grid index
            full_idx = self.ham.valid_spinor_idx[idx] if hasattr(self.ham, 'valid_spinor_idx') else idx
            hole = full_idx // self.ham.n_virt_spinor
            elec = full_idx % self.ham.n_virt_spinor
        return hole, elec, abs(vec[idx])
