import numpy as np
import time
import sys

class ExcitonHamiltonian:
    def __init__(self, C, eps, overlap, atom_ao_ranges, homo_index, n_occ, n_virt, scissor_ev, gamma_qp, gamma_bse, material=None, 
                 gamma_bare=None, gamma_penalty=None, alpha=1.0, include_exchange=False, estimate_qp=False, e_thresh=None, f_thresh=0.0, mu_ia_x=None, mu_ia_y=None, mu_ia_z=None, 
                 charge_type='mulliken', soc_U=None, soc_E=None, device="numpy", precomputed_sigma=None):
        
        self.include_exchange = include_exchange
        self.estimate_qp = estimate_qp
        self.gamma = gamma_bse
        self.material = material
        self.gamma_bare = gamma_bare
        self.gamma_penalty = gamma_penalty
        self.alpha = alpha
        self.soc_flag = (soc_U is not None and soc_E is not None)
        self.overlap = overlap                
        self.atom_ao_ranges = atom_ao_ranges  
        
        occ_idx = np.arange(homo_index - n_occ + 1, homo_index + 1)
        virt_idx = np.arange(homo_index + 1, homo_index + 1 + n_virt)
        n_occ_act, n_virt_act = len(occ_idx), len(virt_idx)
        
        e_min_occ, e_homo = eps[occ_idx[0]], eps[occ_idx[-1]]
        e_lumo, e_max_virt = eps[virt_idx[0]], eps[virt_idx[-1]]

        print(f"\n--- [4] Building Exciton Hamiltonian ---")
        print(f"  Energy Window Diagnostics:")
        print(f"    HOMO-LUMO Gap (Raw):            {e_lumo - e_homo:8.4f} eV")
        print(f"    Max Possible Excitation Energy: {e_max_virt - e_min_occ + scissor_ev:8.4f} eV")

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
        self.n_atoms = len(atom_ao_ranges)

        eps_occ_qp = eps[occ_idx].copy()
        eps_virt_qp = eps[virt_idx].copy()
        
        # --- MOVE DENSITY BUILDER UP FOR QP CORRECTIONS ---
        if self.include_exchange or (self.estimate_qp and precomputed_sigma is None):
            print(f"\n  Building hole/electron/transition density blocks for Active Space...")
            t_den = time.time()
            self.q_occ = np.zeros((n_occ_act, n_occ_act, self.n_atoms))
            self.q_virt = np.zeros((n_virt_act, n_virt_act, self.n_atoms))
            self.q_ov = np.zeros((n_occ_act, n_virt_act, self.n_atoms))

            SC_occ = overlap @ C_occ_act
            SC_virt = overlap @ C_virt_act
            if hasattr(SC_occ, "toarray"): SC_occ = SC_occ.toarray()
            if hasattr(SC_virt, "toarray"): SC_virt = SC_virt.toarray()

            for A, (a0, a1) in enumerate(atom_ao_ranges):
                Co = C_occ_act[a0:a1, :]
                SCo = SC_occ[a0:a1, :]
                self.q_occ[:, :, A] = 0.5 * (Co.T @ SCo + SCo.T @ Co)

                Cv = C_virt_act[a0:a1, :]
                SCv = SC_virt[a0:a1, :]
                self.q_virt[:, :, A] = 0.5 * (Cv.T @ SCv + SCv.T @ Cv)

                self.q_ov[:, :, A] = 0.5 * (Co.T @ SCv + SCo.T @ Cv)

            print(f"    Blocks built in {time.time() - t_den:2.4f} s")

        if self.estimate_qp:
            if precomputed_sigma is not None:
                # --- FAST PATH: Use precomputed shifts ---
                self.sigma_occ, self.sigma_virt = precomputed_sigma
                print("\n  [QP] Using precomputed spatial Quasiparticle shifts. Bypassing COHSEX recalculation.")
            else:
                print(f"\n--- [G0W0-lite] Computing COHSEX Quasiparticle Corrections ---")
                t_qp = time.time()
                
                n_all_occ = homo_index + 1
                n_valence_occ = min(100, n_all_occ)
                val_start = n_all_occ - n_valence_occ
                
                print(f"  [QP] Parameters:")
                print(f"       - Screened W Kernel (alpha) : {self.alpha:8.4f}")
                print(f"       - Active Space MOs:             {self.n_occ_act} Occ, {self.n_virt_act} Virt")
                print(f"       - Background Screening MOs:     {n_valence_occ} (Valence Occupied MOs only)")

                C_val_occ = C[:, val_start:n_all_occ]
                if hasattr(C_val_occ, "toarray"): C_val_occ = C_val_occ.toarray()
                
                SC_val_occ = overlap @ C_val_occ
                if hasattr(SC_val_occ, "toarray"): SC_val_occ = SC_val_occ.toarray()

                q_act_occ_val = np.zeros((self.n_occ_act, n_valence_occ, self.n_atoms))
                q_act_virt_val = np.zeros((self.n_virt_act, n_valence_occ, self.n_atoms))

                for A, (a0, a1) in enumerate(atom_ao_ranges):
                    Co_act = C_occ_act[a0:a1, :]
                    SCo_act = SC_occ[a0:a1, :]
                    Cv_act = C_virt_act[a0:a1, :]
                    SCv_act = SC_virt[a0:a1, :]

                    Co_val = C_val_occ[a0:a1, :]
                    SCo_val = SC_val_occ[a0:a1, :]

                    q_act_occ_val[:, :, A] = 0.5 * (Co_act.T @ SCo_val + SCo_act.T @ Co_val)
                    q_act_virt_val[:, :, A] = 0.5 * (Cv_act.T @ SCo_val + SCv_act.T @ Co_val)

                # EXACTLY your old math for the baseline
                dW = self.gamma - self.gamma_bare  
                W = self.gamma
                
                sigma_occ_raw = np.zeros(self.n_occ_act)
                sigma_virt_raw = np.zeros(self.n_virt_act)
                homo_coh, homo_sex, homo_sic = 0.0, 0.0, 0.0
                lumo_coh, lumo_sex, lumo_sic = 0.0, 0.0, 0.0

                # 1. Compute for Active Occupied Orbitals
                for i in range(self.n_occ_act):
                    q_ii = self.q_occ[i, i, :]
                    coh = 0.5 * np.dot(q_ii, dW @ q_ii)
                    sex = 0.0
                    for j in range(n_valence_occ):
                        q_ij = q_act_occ_val[i, j, :]
                        sex -= np.dot(q_ij, self.gamma_bare @ q_ij)
                    
                    # Apply the isolated Hubbard penalty (Push occupied DOWN)
                    sic = np.dot(q_ii, self.gamma_penalty @ q_ii) if self.gamma_penalty is not None else 0.0
                    sigma_occ_raw[i] = coh + sex - sic
                    if i == self.n_occ_act - 1:
                        homo_coh, homo_sex, homo_sic = coh, sex, sic

                # 2. Compute for Active Virtual Orbitals
                for a in range(self.n_virt_act):
                    q_aa = self.q_virt[a, a, :]
                    coh = 0.5 * np.dot(q_aa, dW @ q_aa)
                    sex = 0.0
                    for j in range(n_valence_occ):
                        q_aj = q_act_virt_val[a, j, :]
                        sex -= np.dot(q_aj, W @ q_aj)
                    
                    # Apply the isolated Hubbard penalty (Push virtual UP)
                    sic = np.dot(q_aa, self.gamma_penalty @ q_aa) if self.gamma_penalty is not None else 0.0
                    sigma_virt_raw[a] = coh + sex + sic

                    if a == 0:
                        lumo_coh, lumo_sex, lumo_sic = coh, sex, sic

                print(f"\n  [QP] Detailed Self-Energy Components:")
                print(f"       HOMO COH: {homo_coh:8.4f} eV  |  SEX: {homo_sex:8.4f} eV  |  SIC: -{homo_sic:8.4f} eV")
                print(f"       LUMO COH: {lumo_coh:8.4f} eV  |  SEX: {lumo_sex:8.4f} eV  |  SIC: +{lumo_sic:8.4f} eV")
                # --- PROOF OF LOCALIZATION (IPR) ---
                # Extract HOMO (last occupied) and LUMO (first virtual)
                q_homo = self.q_occ[-1, -1, :]
                q_lumo = self.q_virt[0, 0, :]
                
                ipr_homo = np.sum(q_homo ** 2)
                ipr_lumo = np.sum(q_lumo ** 2)
                
                print(f"\n  [Theory Check] Orbital Localization (IPR = Sum of q^2):")
                print(f"       HOMO IPR : {ipr_homo:8.5f}  (Higher means more localized)")
                print(f"       LUMO IPR : {ipr_lumo:8.5f}  (Higher means more localized)")
                
                # Print the raw atomic charges for the top 5 most populated atoms in each
                top_homo_atoms = np.argsort(q_homo)[-5:][::-1]
                top_lumo_atoms = np.argsort(q_lumo)[-5:][::-1]
                
                print(f"       HOMO Top 5 Atom Charges (q): {q_homo[top_homo_atoms]}")
                print(f"       LUMO Top 5 Atom Charges (q): {q_lumo[top_lumo_atoms]}")
                # -----------------------------------
    
                # 3. Vxc Cancellation (HOMO Referencing)
                homo_raw_shift = sigma_occ_raw[-1]
                self.sigma_occ = sigma_occ_raw - homo_raw_shift
                self.sigma_virt = sigma_virt_raw - homo_raw_shift 
                
                print(f"       Calculated Gap Correction: {(self.sigma_virt[0] - self.sigma_occ[-1]):+8.4f} eV")
                print(f"    Completed in {time.time() - t_qp:2.4f} s")
        else:
            # --- MANUAL SCISSOR MODE ---
            self.sigma_occ = np.zeros(self.n_occ_act)
            self.sigma_virt = np.full(self.n_virt_act, scissor_ev)
            if scissor_ev != 0.0:
                print(f"\n  [QP] Rigid Scissor applied: +{scissor_ev:.4f} eV to Virtual Orbitals.")

        # Apply shifts directly to the QP energy arrays
        eps_occ_qp += self.sigma_occ
        eps_virt_qp += self.sigma_virt

        print(f"[QP] Final QP gap: {eps_virt_qp[0] - eps_occ_qp[-1]:.3f} eV")
        
        # ==========================================================
        # UNIFIED ORBITAL PRINTOUT (Always runs!)
        # ==========================================================
        print(f"\n  Retained Active Space Orbitals (Post-Shift):")
        print(f"    {'Orbital':>12} | {'Index':>6} | {'DFT (eV)':>10} | {'Shift':>10} | {'QP Energy':>10} | {'Occ':>5}")
        print(f"    {'-'*69}")
        
        for idx_local, idx_global in reversed(list(enumerate(virt_idx))):
            label = "LUMO" if idx_global == homo_index + 1 else f"LUMO+{idx_global - (homo_index + 1)}"
            print(f"    {label:>12} | {idx_global:6d} | {eps[idx_global]:10.4f} | {self.sigma_virt[idx_local]:+10.4f} | {eps_virt_qp[idx_local]:10.4f} | {0.0:5.1f}")
            
        print(f"    {'-- FERMI --':>12} | {'------':>6} | {'----------':>10} | {'----------':>10} | {'----------':>10} | {'-----':>5}")
        
        for idx_local, idx_global in reversed(list(enumerate(occ_idx))):
            label = "HOMO" if idx_global == homo_index else f"HOMO-{homo_index - idx_global}"
            print(f"    {label:>12} | {idx_global:6d} | {eps[idx_global]:10.4f} | {self.sigma_occ[idx_local]:+10.4f} | {eps_occ_qp[idx_local]:10.4f} | {2.0:5.1f}")

        # Consume the scissor_ev so it's not double counted in the CI Diagonal D matrix later
        self.scissor_ev = 0.0
        scissor_ev = 0.0

        # --------------------------------------------------
        # CI DUAL TRUNCATION (Energy & Intensity)
        # --------------------------------------------------
        # 1. Use the RAW DFT gap for filtering to keep the active space consistent
        dft_gap_matrix = eps[virt_idx].reshape(1, -1) - eps[occ_idx].reshape(-1, 1)
        # 2. Use the QP gap for the actual Hamiltonian energies
        qp_gap_matrix = eps_virt_qp.reshape(1, -1) - eps_occ_qp.reshape(-1, 1)

        mask_e = (dft_gap_matrix <= e_thresh) if e_thresh is not None else np.ones_like(dft_gap_matrix, dtype=bool)
        n_e_passed = np.sum(mask_e)
        
        if f_thresh > 0.0 and mu_ia_x is not None:
            gap_au = dft_gap_matrix / 27.211386
            f_ia_0 = (4.0 / 3.0) * gap_au * (mu_ia_x**2 + mu_ia_y**2 + mu_ia_z**2)
            mask_f = f_ia_0 >= f_thresh
        else:
            mask_f = np.ones_like(dft_gap_matrix, dtype=bool)

        self.valid_mask = mask_e & mask_f
        self.valid_i, self.valid_a = np.where(self.valid_mask)
        self.dim = len(self.valid_i)
        
        if self.dim == 0:
            print(f"ERROR: CI Space is empty! Energy threshold ({e_thresh}) or f_thresh ({f_thresh}) is too strict.")
            sys.exit(1)
            
        # 3. Feed the corrected QP energies into the diagonal
        self.D_spatial = qp_gap_matrix[self.valid_mask] + scissor_ev
        self.D = self.D_spatial

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
        self.dim_spinor = self.n_occ_spinor * self.n_virt_spinor
        self.dim = self.dim_spinor
        self.soc_U = U_mo # Save for dipole mapping

        # 1. Extract Alpha/Beta Blocks (Truncated to Occ/Virt spaces as per original formalism)
        U_occ_a = U_mo[0 : self.n_occ_act, 0 : self.n_occ_spinor]
        U_virt_a = U_mo[self.n_occ_act : k, self.n_occ_spinor : 2*k]
        U_occ_b = U_mo[k : k + self.n_occ_act, 0 : self.n_occ_spinor]
        U_virt_b = U_mo[k + self.n_occ_act : 2*k, self.n_occ_spinor : 2*k]

        print("  -> Constructing Spinor AOs natively for fast integrals...")
        t_sp = time.time()
        
        # Convert spatial MOs to Spinor MOs natively
        C_o = self.C_orig_occ
        C_v = self.C_orig_virt
        
        C_sp_occ_a = C_o @ U_occ_a
        C_sp_occ_b = C_o @ U_occ_b
        C_sp_virt_a = C_v @ U_virt_a
        C_sp_virt_b = C_v @ U_virt_b
        
        overlap_dense = self.overlap.toarray() if hasattr(self.overlap, 'toarray') else self.overlap
        SC_sp_occ_a = overlap_dense @ C_sp_occ_a
        SC_sp_occ_b = overlap_dense @ C_sp_occ_b
        SC_sp_virt_a = overlap_dense @ C_sp_virt_a
        SC_sp_virt_b = overlap_dense @ C_sp_virt_b
        
        self.q_spinor = np.zeros((self.dim_spinor, self.n_atoms), dtype=complex)
        if self.include_exchange:
            self.q_hole_spinor = np.zeros((self.n_occ_spinor, self.n_occ_spinor, self.n_atoms), dtype=complex)
            self.q_elec_spinor = np.zeros((self.n_virt_spinor, self.n_virt_spinor, self.n_atoms), dtype=complex)

        print(f"  -> Assembling transition density blocks (Atom-by-Atom via BLAS)...")
        for A, (a0, a1) in enumerate(self.atom_ao_ranges):
            Co_a = C_sp_occ_a[a0:a1, :]
            SCo_a = SC_sp_occ_a[a0:a1, :]
            Cv_a = C_sp_virt_a[a0:a1, :]
            SCv_a = SC_sp_virt_a[a0:a1, :]
            
            Co_b = C_sp_occ_b[a0:a1, :]
            SCo_b = SC_sp_occ_b[a0:a1, :]
            Cv_b = C_sp_virt_b[a0:a1, :]
            SCv_b = SC_sp_virt_b[a0:a1, :]
            
            # Ultra-Fast Native Spinor Projection
            q_A_alpha = 0.5 * (Co_a.conj().T @ SCv_a + SCo_a.conj().T @ Cv_a)
            q_A_beta  = 0.5 * (Co_b.conj().T @ SCv_b + SCo_b.conj().T @ Cv_b)
            
            self.q_spinor[:, A] = (q_A_alpha + q_A_beta).flatten()
            
            if self.include_exchange:
                self.q_hole_spinor[:, :, A] = 0.5 * (Co_a.conj().T @ SCo_a + SCo_a.conj().T @ Co_a + Co_b.conj().T @ SCo_b + SCo_b.conj().T @ Co_b)
                self.q_elec_spinor[:, :, A] = 0.5 * (Cv_a.conj().T @ SCv_a + SCv_a.conj().T @ Cv_a + Cv_b.conj().T @ SCv_b + SCv_b.conj().T @ Cv_b)

        print(f"  -> Density mappings compiled natively in {time.time() - t_sp:.2f}s")

        # 3. Spinor Zero-Order Energies
        eps_occ_sp = soc_E[0 : self.n_occ_spinor].copy()
        eps_virt_sp = soc_E[self.n_occ_spinor : 2*k].copy()
        
        # Calculate RAW DFT gap for threshold filtering
        raw_gap_spinor_dft = (eps_virt_sp.reshape(1, -1) - eps_occ_sp.reshape(-1, 1)).flatten()

        # Carry the QP OR Scissor corrections over to the Spinor subspace automatically
        sigma_occ_sp = np.concatenate([self.sigma_occ, self.sigma_occ])
        sigma_virt_sp = np.concatenate([self.sigma_virt, self.sigma_virt])
        
        if self.estimate_qp:
            print("  [QP-SOC] Mapped spatial Quasiparticle shifts onto Spinor energies.")
        elif self.sigma_virt[0] != 0.0:
            print(f"  [QP-SOC] Mapped rigid scissor shift (+{self.sigma_virt[0]:.4f} eV) onto Spinor energies.")

        # Apply shifts
        eps_occ_sp += sigma_occ_sp
        eps_virt_sp += sigma_virt_sp

        # ==========================================================
        # NEW ALIGNED PRINTOUT
        # ==========================================================
        print(f"\n  Retained Active Space Spinors (Post-Shift Mapping):")
        print(f"    {'Spinor':>12} | {'Index':>6} | {'DFT+SOC(eV)':>12} | {'Shift':>10} | {'QP Energy':>10} | {'Occ':>5}")
        print(f"    {'-'*73}")
        for idx in range(self.n_occ_spinor + self.n_virt_spinor - 1, self.n_occ_spinor - 1, -1):
            label = "spL" if idx - self.n_occ_spinor == 0 else f"spL+{idx - self.n_occ_spinor}"
            local_virt_idx = idx - self.n_occ_spinor
            print(f"    {label:>12} | {idx + 1:6d} | {soc_E[idx]:12.4f} | {sigma_virt_sp[local_virt_idx]:+10.4f} | {eps_virt_sp[local_virt_idx]:10.4f} | {0.0:5.1f}")
            
        print(f"    {'-- FERMI --':>12} | {'------':>6} | {'------------':>12} | {'----------':>10} | {'----------':>10} | {'-----':>5}")
        
        for idx in range(self.n_occ_spinor - 1, -1, -1):
            label = "spH" if (self.n_occ_spinor - 1) - idx == 0 else f"spH-{(self.n_occ_spinor - 1) - idx}"
            local_occ_idx = idx
            print(f"    {label:>12} | {idx + 1:6d} | {soc_E[idx]:12.4f} | {sigma_occ_sp[local_occ_idx]:+10.4f} | {eps_occ_sp[local_occ_idx]:10.4f} | {1.0:5.1f}")
        print("\n")
        # ==========================================================

        # Calculate QP gap for the actual Hamiltonian diagonal
        raw_gap_spinor_qp = (eps_virt_sp.reshape(1, -1) - eps_occ_sp.reshape(-1, 1)).flatten()
        # Because we already zeroed out self.scissor_ev in the spatial block, this won't double count!
        raw_D_spinor = raw_gap_spinor_qp + self.scissor_ev


        # Apply the threshold to the RAW DFT energies
        if e_thresh is not None:
            self.valid_spinor_mask = raw_gap_spinor_dft <= e_thresh
        else:
            self.valid_spinor_mask = np.ones_like(raw_D_spinor, dtype=bool)

        self.valid_spinor_idx = np.where(self.valid_spinor_mask)[0]
        self.D_spinor = raw_D_spinor[self.valid_spinor_mask]
        self.q_spinor = self.q_spinor[self.valid_spinor_mask, :]
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
            mu_ia[:, 0], mu_ia[:, 1], mu_ia[:, 2] = mu_ia_x[self.valid_i, self.valid_a], mu_ia_y[self.valid_i, self.valid_a], mu_ia_z[self.valid_i, self.valid_a]
            return mu_ia
            
        # Fast memory-free mapping for dipoles using U directly via matrix multiplication
        k = self.n_occ_act + self.n_virt_act
        U_occ_a = self.soc_U[0 : self.n_occ_act, 0 : self.n_occ_spinor]
        U_virt_a = self.soc_U[self.n_occ_act : k, self.n_occ_spinor : 2*k]
        U_occ_b = self.soc_U[k : k + self.n_occ_act, 0 : self.n_occ_spinor]
        U_virt_b = self.soc_U[k + self.n_occ_act : 2*k, self.n_occ_spinor : 2*k]
        
        def map_dipole(mu_spatial):
            mu_sp = U_occ_a.conj().T @ mu_spatial @ U_virt_a + U_occ_b.conj().T @ mu_spatial @ U_virt_b
            return mu_sp.flatten()[self.valid_spinor_mask] if hasattr(self, 'valid_spinor_mask') else mu_sp.flatten()

        return np.column_stack((map_dipole(mu_ia_x), map_dipole(mu_ia_y), map_dipole(mu_ia_z)))

