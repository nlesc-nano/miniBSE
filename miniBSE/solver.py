import numpy as np
import time
from miniBSE.davidson import davidson
from miniBSE.exciton_hamiltonian import ExcitonHamiltonian
from miniBSE.hardness import build_gamma, build_yukawa_mnok

class ExcitonSolver:
    def __init__(self, C, eps, occ, overlap, atom_symbols, atom_coords, atom_ao_ranges, 
                 homo_index, n_occ, n_virt, scissor_ev, kernel, alpha, material=None, 
                 include_exchange=False, e_thresh=None, f_thresh=0.0, 
                 mu_ia_x=None, mu_ia_y=None, mu_ia_z=None, device="numpy"):

        self.C = C
        self.overlap = overlap
        self.atom_ao_ranges = atom_ao_ranges
        self.n_occ = n_occ
        self.n_virt = n_virt
        self.homo_index = homo_index

        if kernel.lower() == "yukawa":
            print(f"  [Solver] Using Screened Yukawa-MNOK kernel for material: {material}")
            gamma = build_yukawa_mnok(
                atom_symbols=atom_symbols, 
                coords=atom_coords, 
                alpha=alpha, 
                material_name=material
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
            device=device
        )

    def solve(self, nroots=10, full_diag=False, tol=1e-5):
        if full_diag:
            print(f"  Building dense Hamiltonian in truncated space ({self.ham.dim}x{self.ham.dim})...")
            
            # --- 1. Coulomb Term (2J) ---
            print("  [Dense] Building Coulomb term (2J)...")
            t0 = time.time()
            temp = self.ham.q_flat @ self.ham.gamma
            J_mat = temp @ self.ham.q_flat.T
            H = np.diag(self.ham.D) + 2.0 * J_mat
            print(f"    -> Coulomb built in {time.time()-t0:.2f}s")
            
            # --- 2. Exchange Term (-K) ---
            if self.ham.include_exchange:
                print("  [Dense] Building Exchange matrix (-K)...")
                t1 = time.time()
                c_x = getattr(self.ham, 'c_x', 1.0) 
                
                # Fetch indices and active space dimensions
                vi, va = self.ham.valid_i, self.ham.valid_a
                n_occ_act = self.ham.q_occ.shape[0]
                n_virt_act = self.ham.q_virt.shape[0]
                
                print("    -> Contracting W = Gamma @ q_virt ...")
                t_w = time.time()
                # Tensordot is ultra-fast. W shape goes from (A, a, b) -> (a, b, A)
                W = np.tensordot(self.ham.gamma, self.ham.q_virt, axes=([1], [2])) 
                W = np.transpose(W, (1, 2, 0))
                print(f"    -> W contracted in {time.time()-t_w:.2f}s")
                
                print("    -> Contracting K_full in MO basis (Speed & Memory Optimized)...")
                t_k = time.time()
                
                # Multiply over atoms (axis 2). Output shape: (i, j, a, b)
                K_full = np.tensordot(self.ham.q_occ, W, axes=([2], [2]))
                
                # Rearrange to (i, a, j, b) and flatten to a standard 2D matrix
                K_full_trans = np.transpose(K_full, (0, 2, 1, 3))
                K_2d = K_full_trans.reshape(n_occ_act * n_virt_act, n_occ_act * n_virt_act)
                
                # Create a 1D mapping array for the valid transitions
                comp_idx = vi * n_virt_act + va
                
                # Single 2D advanced index extraction (Creates the 5070x5070 array instantly)
                K_truncated = K_2d[comp_idx[:, None], comp_idx[None, :]]
                print(f"    -> K_full and truncation completed in {time.time()-t_k:.2f}s")
                
                H -= c_x * K_truncated
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
        idx = np.argmax(np.abs(vec))
        hole = self.ham.valid_i[idx]
        elec = self.ham.valid_a[idx]
        return hole, elec, abs(vec[idx])

