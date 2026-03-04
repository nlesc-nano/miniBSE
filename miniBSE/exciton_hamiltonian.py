import numpy as np
import time
import sys

class ExcitonHamiltonian:
    def __init__(self, C, eps, overlap, atom_ao_ranges, homo_index, n_occ, n_virt, scissor_ev, gamma, 
                 include_exchange=False, e_thresh=None, f_thresh=0.0, mu_ia_x=None, mu_ia_y=None, mu_ia_z=None, 
                 charge_type='mulliken', soc_U=None, soc_E=None, device="numpy"):
        
        self.include_exchange = include_exchange
        self.gamma = gamma
        self.soc_flag = (soc_U is not None and soc_E is not None)

        occ_idx = np.arange(homo_index - n_occ + 1, homo_index + 1)
        virt_idx = np.arange(homo_index + 1, homo_index + 1 + n_virt)
        n_occ_act, n_virt_act = len(occ_idx), len(virt_idx)
        
        e_min_occ, e_homo = eps[occ_idx[0]], eps[occ_idx[-1]]
        e_lumo, e_max_virt = eps[virt_idx[0]], eps[virt_idx[-1]]

        print(f"\n--- [4] Building Exciton Hamiltonian ---")
        print(f"  Energy Window Diagnostics:")
        print(f"    HOMO-LUMO Gap (Raw):            {e_lumo - e_homo:8.4f} eV")
        print(f"    Max Possible Excitation Energy: {e_max_virt - e_min_occ + scissor_ev:8.4f} eV")

        print(f"\n  Retained Active Space Orbitals:")
        print(f"    {'Orbital':>12} | {'Index':>6} | {'Energy (eV)':>12} | {'Occ':>5}")
        print(f"    {'-'*46}")
        
        for idx in range(virt_idx[-1], homo_index, -1):
            label = "LUMO" if idx == homo_index + 1 else f"LUMO+{idx - (homo_index + 1)}"
            print(f"    {label:>12} | {idx:6d} | {eps[idx]:12.4f} | {0.0:5.1f}")
            
        print(f"    {'-- FERMI --':>12} | {'------':>6} | {'------------':>12} | {'-----':>5}")
        
        for idx in range(homo_index, occ_idx[0] - 1, -1):
            label = "HOMO" if idx == homo_index else f"HOMO-{homo_index - idx}"
            print(f"    {label:>12} | {idx:6d} | {eps[idx]:12.4f} | {2.0:5.1f}")

        # --------------------------------------------------
        # CI DUAL TRUNCATION (Energy & Intensity)
        # --------------------------------------------------
        raw_gap = eps[virt_idx].reshape(1, -1) - eps[occ_idx].reshape(-1, 1)
        
        mask_e = (raw_gap <= e_thresh) if e_thresh is not None else np.ones_like(raw_gap, dtype=bool)
        n_e_passed = np.sum(mask_e)
        
        if f_thresh > 0.0 and mu_ia_x is not None:
            gap_au = raw_gap / 27.211386
            f_ia_0 = (4.0 / 3.0) * gap_au * (mu_ia_x**2 + mu_ia_y**2 + mu_ia_z**2)
            mask_f = f_ia_0 >= f_thresh
        else:
            mask_f = np.ones_like(raw_gap, dtype=bool)

        self.valid_mask = mask_e & mask_f
        self.valid_i, self.valid_a = np.where(self.valid_mask)
        self.dim = len(self.valid_i)
        
        if self.dim == 0:
            print(f"ERROR: CI Space is empty! Energy threshold ({e_thresh}) or f_thresh ({f_thresh}) is too strict.")
            sys.exit(1)
            
        self.D_spatial = raw_gap[self.valid_mask] + scissor_ev
        self.D = self.D_spatial  # Default diagonal for solver
        
        print(f"\n  CI Space Truncation:")
        print(f"    Transitions passing Energy Threshold ({e_thresh or 'None'} eV): {n_e_passed}")
        print(f"    Final CI Space (Energy AND f0 >= {f_thresh}): {self.dim} valid transitions")

        # --------------------------------------------------
        # EXTRACT DENSE ACTIVE SPACE MATRICES
        # --------------------------------------------------
        C_occ_act = C[:, occ_idx]
        C_virt_act = C[:, virt_idx]
        if hasattr(C_occ_act, "toarray"): C_occ_act = C_occ_act.toarray()
        if hasattr(C_virt_act, "toarray"): C_virt_act = C_virt_act.toarray()
        
        self.occ_idx, self.virt_idx = occ_idx, virt_idx
        self.n_occ_act, self.n_virt_act = n_occ_act, n_virt_act
        self.C_orig_occ, self.C_orig_virt = C_occ_act, C_virt_act
        self.scissor_ev = scissor_ev

        # --------------------------------------------------
        # ULTRA-FAST Charge Construction
        # --------------------------------------------------
        self.n_atoms = len(atom_ao_ranges)
        start_q = time.time()
        
        self.q_flat = np.zeros((self.dim, self.n_atoms), dtype=np.float32)

        if charge_type == 'mulliken':
            print(f"  Building transition charges (Atom-by-Atom via {device.upper()})...")
            
            if device != "numpy" and device != "cpu":
                import torch
                dev = torch.device(device)
                S_t = torch.tensor(overlap.toarray() if hasattr(overlap, "toarray") else overlap, dtype=torch.float32, device=dev)
                C_o_t = torch.tensor(C_occ_act, dtype=torch.float32, device=dev)
                C_v_t = torch.tensor(C_virt_act, dtype=torch.float32, device=dev)

                SC_occ_t = S_t @ C_o_t
                SC_virt_t = S_t @ C_v_t

                for A, (a0, a1) in enumerate(atom_ao_ranges):
                    Ci_A = C_o_t[a0:a1, self.valid_i]
                    SCa_A = SC_virt_t[a0:a1, self.valid_a]
                    Ca_A = C_v_t[a0:a1, self.valid_a]
                    SCi_A = SC_occ_t[a0:a1, self.valid_i]
                    q_A = 0.5 * torch.sum((Ci_A * SCa_A) + (Ca_A * SCi_A), dim=0)
                    self.q_flat[:, A] = q_A.cpu().numpy()
                    
                if not include_exchange:
                    del S_t, C_o_t, C_v_t, SC_occ_t, SC_virt_t
            else:
                SC_occ = overlap @ C_occ_act
                SC_virt = overlap @ C_virt_act
                if hasattr(SC_occ, "toarray"): SC_occ = SC_occ.toarray()
                if hasattr(SC_virt, "toarray"): SC_virt = SC_virt.toarray()
                
                for A, (a0, a1) in enumerate(atom_ao_ranges):
                    Ci_A = C_occ_act[a0:a1, :][:, self.valid_i]
                    SCa_A = SC_virt[a0:a1, :][:, self.valid_a]
                    Ca_A = C_virt_act[a0:a1, :][:, self.valid_a]
                    SCi_A = SC_occ[a0:a1, :][:, self.valid_i]
                    self.q_flat[:, A] = 0.5 * (np.sum(Ci_A * SCa_A, axis=0) + np.sum(Ca_A * SCi_A, axis=0))

        if include_exchange:
            print(f"  Building hole/electron density blocks for Exchange...")
            self.q_occ = np.zeros((n_occ_act, n_occ_act, self.n_atoms))
            self.q_virt = np.zeros((n_virt_act, n_virt_act, self.n_atoms))
            for A, (a0, a1) in enumerate(atom_ao_ranges):
                if charge_type == 'mulliken':
                    Co = C_occ_act[a0:a1, :]
                    SCo = SC_occ_t[a0:a1, :].cpu().numpy() if device not in ["numpy", "cpu"] else SC_occ[a0:a1, :]
                    self.q_occ[:, :, A] = 0.5 * (Co.T @ SCo + SCo.T @ Co)
                    
                    Cv = C_virt_act[a0:a1, :]
                    SCv = SC_virt_t[a0:a1, :].cpu().numpy() if device not in ["numpy", "cpu"] else SC_virt[a0:a1, :]
                    self.q_virt[:, :, A] = 0.5 * (Cv.T @ SCv + SCv.T @ Cv)

        print(f"    Charges built in {time.time() - start_q:2.4f} s")

        # --------------------------------------------------
        # SPIN-ORBIT COUPLING (SPINOR) TRANSFORMATION
        # --------------------------------------------------
        if self.soc_flag:
            self.build_spinor_basis(soc_U, soc_E, e_thresh)


    def build_spinor_basis(self, U_mo, soc_E, e_thresh):
        print("\n--- [SOC] Transforming Exciton Hamiltonian to Spinor Basis ---")
        k = self.n_occ_act + self.n_virt_act
        self.n_occ_spinor = 2 * self.n_occ_act
        self.n_virt_spinor = 2 * self.n_virt_act
        
        # New active space dimension (Full Spinor Space, no thresholds applied here)
        self.dim_spinor = self.n_occ_spinor * self.n_virt_spinor
        self.dim = self.dim_spinor  # Update dimension for the Davidson Solver

        print(f"\n  Retained Active Space Spinors:")
        print(f"    {'Spinor':>12} | {'Index':>6} | {'Energy (eV)':>12} | {'Occ':>5}")
        print(f"    {'-'*46}")
        
        # Virtual Spinors (print top-down to LUMO)
        for idx in range(self.n_occ_spinor + self.n_virt_spinor - 1, self.n_occ_spinor - 1, -1):
            rel_idx = idx - self.n_occ_spinor
            label = "spL" if rel_idx == 0 else f"spL+{rel_idx}"
            print(f"    {label:>12} | {idx + 1:6d} | {soc_E[idx]:12.4f} | {0.0:5.1f}")
            
        print(f"    {'-- FERMI --':>12} | {'------':>6} | {'------------':>12} | {'-----':>5}")
        
        # Occupied Spinors (print HOMO down to bottom)
        for idx in range(self.n_occ_spinor - 1, -1, -1):
            rel_idx = (self.n_occ_spinor - 1) - idx
            label = "spH" if rel_idx == 0 else f"spH-{rel_idx}"
            # Spinors hold 1 electron each, so occ is 1.0
            print(f"    {label:>12} | {idx + 1:6d} | {soc_E[idx]:12.4f} | {1.0:5.1f}")
        print("\n")
        
        # 1. Extract Alpha/Beta Blocks from Single-Particle Spinor Transformation
        U_occ_a = U_mo[0 : self.n_occ_act, 0 : self.n_occ_spinor]
        U_virt_a = U_mo[self.n_occ_act : k, self.n_occ_spinor : 2*k]
        
        U_occ_b = U_mo[k : k + self.n_occ_act, 0 : self.n_occ_spinor]
        U_virt_b = U_mo[k + self.n_occ_act : 2*k, self.n_occ_spinor : 2*k]
        # 2. Build Spin-Integrated Density Overlap D_{ia}^{IA}
        # Mapping: spatial transition (i,a) -> spinor transition (I,A)
        print("  -> Computing transition density spinor mapping (Fast Broadcast)...")
        
        # Broadcasting is exponentially faster than unoptimized einsum for outer products
        D_a = U_occ_a.conj()[:, None, :, None] * U_virt_a[None, :, None, :]
        D_b = U_occ_b.conj()[:, None, :, None] * U_virt_b[None, :, None, :]
        D_tensor = D_a + D_b
        
        # Reshape to (Spatial_Transitions, Spinor_Transitions)
        self.D_matrix = D_tensor.reshape(self.n_occ_act * self.n_virt_act, self.dim_spinor)        
 
        # 3. Restore truncated spatial q_flat to full (n_occ * n_virt) size
        q_spatial_full = np.zeros((self.n_occ_act * self.n_virt_act, self.n_atoms))
        flat_valid_indices = self.valid_i * self.n_virt_act + self.valid_a
        q_spatial_full[flat_valid_indices, :] = self.q_flat
        
        # 4. Map Transition Charges: q^{IA}_C = \sum_{ia} D_{ia}^{IA} q^{ia}_C
        self.q_spinor = self.D_matrix.T @ q_spatial_full
       
        # 5. Transform Exchange Blocks
        if self.include_exchange:
            print("  -> Computing exchange density spinor mapping natively (O(N^3) Memory-Free)...")
            
            # BYPASS building the massive 4D D_hole tensor!
            # We contract the spatial (i,j) axes directly into the spinor space.
            
            # Hole overlap mapping
            qh_a = np.einsum('iI, ijC, jJ -> IJC', U_occ_a.conj(), self.q_occ, U_occ_a, optimize=True)
            qh_b = np.einsum('iI, ijC, jJ -> IJC', U_occ_b.conj(), self.q_occ, U_occ_b, optimize=True)
            self.q_hole_spinor = qh_a + qh_b
            
            # Electron overlap mapping
            qe_a = np.einsum('aA, abC, bB -> ABC', U_virt_a.conj(), self.q_virt, U_virt_a, optimize=True)
            qe_b = np.einsum('aA, abC, bB -> ABC', U_virt_b.conj(), self.q_virt, U_virt_b, optimize=True)
            self.q_elec_spinor = qe_a + qe_b
 
        # 6. Spinor Zero-Order Energies
        eps_occ_sp = soc_E[0 : self.n_occ_spinor]
        eps_virt_sp = soc_E[self.n_occ_spinor : 2*k]
        diff_sp = eps_virt_sp.reshape(1, -1) - eps_occ_sp.reshape(-1, 1)
        
        raw_gap_spinor = diff_sp.flatten()
        raw_D_spinor = raw_gap_spinor + self.scissor_ev
        
        # --- FIXED TRUNCATION LOGIC ---
        # Apply e_thresh to the RAW gap (just like the spatial code does)
        if e_thresh is not None:
            self.valid_spinor_mask = raw_gap_spinor <= e_thresh
        else:
            self.valid_spinor_mask = np.ones_like(raw_D_spinor, dtype=bool)
            
        self.valid_spinor_idx = np.where(self.valid_spinor_mask)[0]
            
        # Apply the mask globally so downstream tools don't crash
        self.D_spinor = raw_D_spinor[self.valid_spinor_mask]
        self.q_spinor = self.q_spinor[self.valid_spinor_mask, :]
        self.D_matrix = self.D_matrix[:, self.valid_spinor_mask]  # MUST slice D_matrix!
        self.D = self.D_spinor
        
        self.dim_spinor_full = len(raw_D_spinor)
        self.dim = len(self.D_spinor)
        
        print(f"  -> BSE Active Space expanded to {self.dim_spinor_full} spinor transitions.")
        print(f"  -> Truncated Spinor Space (Energy <= {e_thresh} eV): {self.dim} valid transitions")

    def matvec(self, x):
        """Matrix-vector product for Davidson Solver."""
        
        if not self.soc_flag:
            # === STANDARD SPATIAL MATVEC (Singlets) ===
            y = self.D_spatial * x
            T = self.q_flat.T @ x
            y += 2.0 * self.q_flat @ (self.gamma @ T) # 2.0 Spin-Multiplicity
            
            if self.include_exchange:
                x_mat = np.zeros((self.n_occ_act, self.n_virt_act))
                x_mat[self.valid_i, self.valid_a] = x
                W = np.einsum("AB, abB -> abA", self.gamma, self.q_virt)
                K = np.einsum("ijA, abA, jb -> ia", self.q_occ, W, x_mat)
                y -= 1.0 * K[self.valid_i, self.valid_a]
                
            return y

        # === RELATIVISTIC SPINOR MATVEC ===
        y = self.D_spinor * x
        
        # Coulomb J (No 2x factor for spinors)
        # J_action = q_spinor.conj() @ gamma @ q_spinor.T @ x
        T_C = self.q_spinor.T @ x
        V_C = self.gamma @ T_C
        y += self.q_spinor.conj() @ V_C

        if self.include_exchange:
            # Exchange K (Complex tensors)
            if hasattr(self, 'valid_spinor_idx'):
                x_full = np.zeros(self.dim_spinor_full, dtype=complex)
                x_full[self.valid_spinor_idx] = x
                x_mat = x_full.reshape(self.n_occ_spinor, self.n_virt_spinor)
            else:
                x_mat = x.reshape(self.n_occ_spinor, self.n_virt_spinor)
                
            # W[a, b, A] = \sum_B \gamma_{AB} q_elec[a, b, B]
            W = np.einsum("AB, abB -> abA", self.gamma, self.q_elec_spinor)
            # K[i, a] = \sum_{j,b,A} (q_hole[i, j, A])^* W[a, b, A] x_mat[j, b]
            K = np.einsum("ijA, abA, jb -> ia", self.q_hole_spinor.conj(), W, x_mat)
            
            if hasattr(self, 'valid_spinor_idx'):
                y -= K.flatten()[self.valid_spinor_idx]
            else:
                y -= K.flatten()
            
        return y

    def get_transition_dipoles(self, mu_ia_x, mu_ia_y, mu_ia_z):
        """Extracts transition dipoles in either the spatial or spinor basis."""
        if not self.soc_flag:
            mu_ia = np.zeros((len(self.valid_i), 3))
            mu_ia[:, 0] = mu_ia_x[self.valid_i, self.valid_a]
            mu_ia[:, 1] = mu_ia_y[self.valid_i, self.valid_a]
            mu_ia[:, 2] = mu_ia_z[self.valid_i, self.valid_a]
            return mu_ia
            
        # Map full spatial dipoles to full spinor dipoles
        # mu_{IA} = \sum_{ia} D_{ia}^{IA} \mu_{ia}
        mu_x_sp = self.D_matrix.T @ mu_ia_x.flatten()
        mu_y_sp = self.D_matrix.T @ mu_ia_y.flatten()
        mu_z_sp = self.D_matrix.T @ mu_ia_z.flatten()
        return np.column_stack((mu_x_sp, mu_y_sp, mu_z_sp))

