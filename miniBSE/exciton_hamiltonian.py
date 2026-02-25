import numpy as np
import time
import sys

class ExcitonHamiltonian:
    def __init__(self, C, eps, overlap, atom_ao_ranges, homo_index, n_occ, n_virt, scissor_ev, gamma, include_exchange=False, e_thresh=None, f_thresh=0.0, mu_ia_x=None, mu_ia_y=None, mu_ia_z=None, charge_type='mulliken', device="numpy"):
        self.include_exchange = include_exchange
        self.gamma = gamma

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
            
        self.D = raw_gap[self.valid_mask] + scissor_ev
        
        print(f"\n  CI Space Truncation:")
        print(f"    Transitions passing Energy Threshold ({e_thresh or 'None'} eV): {n_e_passed}")
        print(f"    Final CI Space (Energy AND f0 >= {f_thresh}): {self.dim} valid transitions")

        # --------------------------------------------------
        # EXTRACT DENSE ACTIVE SPACE MATRICES
        # --------------------------------------------------
        # [FIX]: Ensure the sliced MOs are strictly dense to prevent SciPy Sparse element-wise multiplication bugs
        C_occ_act = C[:, occ_idx]
        C_virt_act = C[:, virt_idx]
        if hasattr(C_occ_act, "toarray"): C_occ_act = C_occ_act.toarray()
        if hasattr(C_virt_act, "toarray"): C_virt_act = C_virt_act.toarray()
        
        self.occ_idx, self.virt_idx = occ_idx, virt_idx
        self.C_orig_occ, self.C_orig_virt = C_occ_act, C_virt_act

        # --------------------------------------------------
        # ULTRA-FAST Charge Construction
        # --------------------------------------------------
        n_atoms = len(atom_ao_ranges)
        start_q = time.time()
        
        self.q_flat = np.zeros((self.dim, n_atoms), dtype=np.float32)

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
            self.q_occ = np.zeros((n_occ_act, n_occ_act, n_atoms))
            self.q_virt = np.zeros((n_virt_act, n_virt_act, n_atoms))
            for A, (a0, a1) in enumerate(atom_ao_ranges):
                if charge_type == 'mulliken':
                    Co = C_occ_act[a0:a1, :]
                    SCo = SC_occ_t[a0:a1, :].cpu().numpy() if device not in ["numpy", "cpu"] else SC_occ[a0:a1, :]
                    self.q_occ[:, :, A] = 0.5 * (Co.T @ SCo + SCo.T @ Co)
                    
                    Cv = C_virt_act[a0:a1, :]
                    SCv = SC_virt_t[a0:a1, :].cpu().numpy() if device not in ["numpy", "cpu"] else SC_virt[a0:a1, :]
                    self.q_virt[:, :, A] = 0.5 * (Cv.T @ SCv + SCv.T @ Cv)

        print(f"    Charges built in {time.time() - start_q:2.4f} s")

    def matvec(self, x):
        y = self.D * x
        T = self.q_flat.T @ x
        y += 2.0 * self.q_flat @ (self.gamma @ T)
        
        if self.include_exchange:
            x_mat = np.zeros((len(self.occ_idx), len(self.virt_idx)))
            x_mat[self.valid_i, self.valid_a] = x
            W = np.einsum("AB, abB -> abA", self.gamma, self.q_virt)
            K = np.einsum("ijA, abA, jb -> ia", self.q_occ, W, x_mat)
            y -= 1.0 * K[self.valid_i, self.valid_a]
            
        return y

    def get_transition_dipoles(self, mu_ia_x, mu_ia_y, mu_ia_z):
        """Extracts valid transition dipoles from pre-transformed active space matrices."""
        mu_ia = np.zeros((self.dim, 3))
        mu_ia[:, 0] = mu_ia_x[self.valid_i, self.valid_a]
        mu_ia[:, 1] = mu_ia_y[self.valid_i, self.valid_a]
        mu_ia[:, 2] = mu_ia_z[self.valid_i, self.valid_a]
        return mu_ia
