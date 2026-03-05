import numpy as np
import time
from miniBSE.davidson import davidson
from miniBSE.exciton_hamiltonian import ExcitonHamiltonian

# --- UPDATED IMPORTS ---
from miniBSE.hardness import build_gamma, build_resta_mnok

class ExcitonSolver:
    def __init__(self, C, eps, occ, overlap, atom_symbols, atom_coords, atom_ao_ranges, 
                 homo_index, n_occ, n_virt, scissor_ev, kernel, alpha, material=None, 
                 include_exchange=False, e_thresh=None, f_thresh=0.0, 
                 mu_ia_x=None, mu_ia_y=None, mu_ia_z=None, eps_out=2.0, 
                 soc_U=None, soc_E=None, device="numpy"): 

        self.C = C
        self.overlap = overlap
        self.atom_ao_ranges = atom_ao_ranges
        self.n_occ = n_occ
        self.n_virt = n_virt
        self.homo_index = homo_index
        self.soc_flag = (soc_U is not None and soc_E is not None)

        # --- UPDATED KERNEL LOGIC ---
        if kernel.lower() == "resta":
            print(f"  [Solver] Using Screened Resta-MNOK kernel for material: {material}")
            gamma = build_resta_mnok(
                atom_symbols=atom_symbols, 
                coords=atom_coords, 
                alpha=alpha, 
                material_name=material,
                eps_out=eps_out  # Passed from CLI
            )
        else:
            print("  [Solver] Using standard Grimme sTDA MNOK kernel.")
            gamma = build_gamma(
                atom_symbols=atom_symbols, 
                coords=atom_coords, 
                alpha=alpha
            ) 

        self.ham = ExcitonHamiltonian(
            C=C, eps=eps, overlap=overlap, atom_ao_ranges=atom_ao_ranges,
            homo_index=homo_index, n_occ=n_occ, n_virt=n_virt, scissor_ev=scissor_ev,
            gamma=gamma, include_exchange=include_exchange, e_thresh=e_thresh, 
            f_thresh=f_thresh, mu_ia_x=mu_ia_x, mu_ia_y=mu_ia_y, mu_ia_z=mu_ia_z, 
            soc_U=soc_U, soc_E=soc_E, device=device
        )

    def solve(self, nroots=10, full_diag=False, tol=1e-5):
        if full_diag:
            print(f"  Building dense Hamiltonian in truncated space ({self.ham.dim}x{self.ham.dim})...")
            
            if not self.soc_flag:
                # ==========================================================
                # SPATIAL DENSE BUILDER (Spin-Free Singlets)
                # ==========================================================
                print("  [Dense] Building Coulomb term (2J)...")
                t0 = time.time()
                temp = self.ham.q_flat @ self.ham.gamma
                J_mat = temp @ self.ham.q_flat.T
                H = np.diag(self.ham.D) + 2.0 * J_mat
                self.J_mat = 2.0 * J_mat
                self.K_mat = np.zeros_like(J_mat)
                print(f"    -> Coulomb built in {time.time()-t0:.2f}s")
                
                if self.ham.include_exchange:
                    print("  [Dense] Building Exchange matrix (-K)...")
                    t1 = time.time()
                    c_x = getattr(self.ham, 'c_x', 1.0) 
                    
                    vi, va = self.ham.valid_i, self.ham.valid_a
                    n_p = len(vi)
                    
                    print("    -> Constructing K_truncated natively (Zero-Memory Overhead)...")
                    t_k = time.time()
                    
                    # 1. Slice natively 
                    q_occ_sub = self.ham.q_occ[vi[:, None], vi[None, :], :]
                    q_virt_sub = self.ham.q_virt[va[:, None], va[None, :], :]
                    
                    # 2. Contract Gamma only for the valid pairs
                    W_sub = (q_virt_sub.reshape(n_p * n_p, -1) @ self.ham.gamma).reshape(n_p, n_p, -1)
                    
                    # 3. Element-wise multiply and sum
                    K_truncated = np.sum(q_occ_sub * W_sub, axis=-1)
                    
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
                        n_p = len(v_idx)
                        
                        print("    -> Constructing K_truncated natively (Zero-Memory Overhead)...")
                        t_k = time.time()
                        
                        # 1. Slice natively 
                        q_hole_sub = self.ham.q_hole_spinor.conj()[vi_sp[:, None], vi_sp[None, :], :]
                        q_elec_sub = self.ham.q_elec_spinor[va_sp[:, None], va_sp[None, :], :]
                        
                        # 2. Contract Gamma only for the valid pairs
                        W_sub = (q_elec_sub.reshape(n_p * n_p, -1) @ self.ham.gamma).reshape(n_p, n_p, -1)
                        
                        # 3. Element-wise multiply and sum
                        K_truncated = np.sum(q_hole_sub * W_sub, axis=-1)
                        
                        print(f"    -> K_truncated assembled in {time.time()-t_k:.2f}s")
                        
                        H -= K_truncated
                        self.K_mat = K_truncated
                    else:
                        # Fallback
                        print("    -> Contracting W = Gamma @ q_elec_spinor ...")
                        t_w = time.time()
                        W = np.tensordot(self.ham.gamma, self.ham.q_elec_spinor, axes=([1], [2])) 
                        W = np.transpose(W, (1, 2, 0))
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
        if not self.soc_flag:
            hole = self.ham.valid_i[idx]
            elec = self.ham.valid_a[idx]
        else:
            # Map the truncated index back to the full spinor grid index
            full_idx = self.ham.valid_spinor_idx[idx] if hasattr(self.ham, 'valid_spinor_idx') else idx
            hole = full_idx // self.ham.n_virt_spinor
            elec = full_idx % self.ham.n_virt_spinor
        return hole, elec, abs(vec[idx])

