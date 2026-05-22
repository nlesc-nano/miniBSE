import os
import numpy as np
import time
import sys
from miniBSE.io_utils import get_vxc_ao_matrix
from miniBSE.constants import HA_TO_EV

class ExcitonHamiltonian:
    def __init__(self, C, eps, overlap, atom_ao_ranges, homo_index, n_occ, n_virt, scissor_ev, gamma_qp, gamma_bse, material=None, 
                 gamma_bare=None, gamma_penalty=None, alpha=1.0, include_exchange=False, estimate_qp=False, e_thresh=None, f_thresh=0.0, mu_ia_x=None, mu_ia_y=None, mu_ia_z=None, 
                 charge_type='mulliken', soc_U=None, soc_E=None, device="numpy", precomputed_sigma=None,
                 vxc_ao_path=None, nthreads=1, spin='singlet', C_beta=None, eps_beta=None, homo_index_beta=None,
                 n_occ_beta=None, n_virt_beta=None):
        
        self.include_exchange = include_exchange
        self.estimate_qp = estimate_qp
        self.gamma_bse = gamma_bse
        self.gamma_qp = gamma_qp
        self.gamma = gamma_bse
        self.spin = spin
        self.material = material
        self.gamma_bare = gamma_bare
        self.gamma_penalty = gamma_penalty
        self.alpha = alpha
        self.soc_flag = (soc_U is not None and soc_E is not None)
        self.overlap = overlap                
        self.atom_ao_ranges = atom_ao_ranges

        # --------------------------------------------------
        # UKS SPIN-PRESERVING (Manifold B): Coupled alpha-alpha + beta-beta
        # --------------------------------------------------
        if spin == 'uks_spin_preserving':
            self.init_uks_spin_preserving(
                C_alpha=C, eps_alpha=eps, homo_index_alpha=homo_index, n_occ_alpha=n_occ, n_virt_alpha=n_virt,
                C_beta=C_beta, eps_beta=eps_beta, homo_index_beta=homo_index_beta,
                n_occ_beta=n_occ_beta if n_occ_beta is not None else n_occ,
                n_virt_beta=n_virt_beta if n_virt_beta is not None else n_virt,
                overlap=overlap, atom_ao_ranges=atom_ao_ranges, scissor_ev=scissor_ev,
                e_thresh=e_thresh, f_thresh=f_thresh, mu_ia_x=mu_ia_x, mu_ia_y=mu_ia_y, mu_ia_z=mu_ia_z,
                charge_type=charge_type, device=device, soc_U=soc_U, soc_E=soc_E
            )
            return
        
        C_beta_or_alpha = C_beta if C_beta is not None else C
        eps_beta_or_alpha = eps_beta if eps_beta is not None else eps
        homo_index_beta = homo_index_beta if homo_index_beta is not None else homo_index

        occ_idx = np.arange(homo_index - n_occ + 1, homo_index + 1)
        virt_idx = np.arange(homo_index_beta + 1, homo_index_beta + 1 + n_virt)
        n_occ_act, n_virt_act = len(occ_idx), len(virt_idx)
        
        e_min_occ, e_homo = eps[occ_idx[0]], eps[occ_idx[-1]]
        e_lumo, e_max_virt = eps_beta_or_alpha[virt_idx[0]], eps_beta_or_alpha[virt_idx[-1]]

        print(f"\n--- [4] Building Exciton Hamiltonian ---")
        print(f"  Energy Window Diagnostics:")
        print(f"    HOMO-LUMO Gap (Raw):            {e_lumo - e_homo:8.4f} eV")
        print(f"    Max Possible Excitation Energy: {e_max_virt - e_min_occ + scissor_ev:8.4f} eV")

        # --------------------------------------------------
        # EXTRACT DENSE ACTIVE SPACE MATRICES
        # --------------------------------------------------
        C_occ_act = C[:, occ_idx]
        C_virt_act = C_beta_or_alpha[:, virt_idx]
        if hasattr(C_occ_act, "toarray"): C_occ_act = C_occ_act.toarray()
        if hasattr(C_virt_act, "toarray"): C_virt_act = C_virt_act.toarray()
        
        self.occ_idx, self.virt_idx = occ_idx, virt_idx
        self.n_occ_act, self.n_virt_act = n_occ_act, n_virt_act
        self.C_orig_occ, self.C_orig_virt = C_occ_act, C_virt_act
        self.scissor_ev = scissor_ev
        self.n_atoms = len(atom_ao_ranges)
        start_q = time.time()

        eps_occ_qp = eps[occ_idx].copy()
        eps_virt_qp = eps_beta_or_alpha[virt_idx].copy()
        
        # --- MOVE DENSITY BUILDER UP FOR QP CORRECTIONS ---
        if self.include_exchange or self.soc_flag or (self.estimate_qp and precomputed_sigma is None):
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

            if self.include_exchange:
                qv = self.q_virt.reshape(n_virt_act * n_virt_act, self.n_atoms)
                self.W_virt = (qv @ self.gamma.T).reshape(n_virt_act, n_virt_act, self.n_atoms)

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

                # [EXPERIMENTAL] COHSEX self-energy — uses the QP-screened kernel (gamma_qp)
                # for SEX and the difference kernel (gamma_qp - gamma_bare) for COH.
                # This path is not used in production runs (use --qp_gap gw instead).
                dW = self.gamma_qp - self.gamma_bare  
                W = self.gamma_qp
                
                # Fully Vectorized COHSEX Computation
                q_occ_diag = np.array([self.q_occ[i, i, :] for i in range(self.n_occ_act)])
                q_virt_diag = np.array([self.q_virt[a, a, :] for a in range(self.n_virt_act)])

                coh_occ = 0.5 * np.einsum('iA,AB,iB->i', q_occ_diag, dW, q_occ_diag, optimize=True)
                Wq_occ = np.einsum('AB,ijB->ijA', W, q_act_occ_val, optimize=True)
                sex_occ = -np.einsum('ijA,ijA->i', q_act_occ_val, Wq_occ, optimize=True)
                
                if self.gamma_penalty is not None:
                    sic_occ = np.einsum('iA,AB,iB->i', q_occ_diag, self.gamma_penalty, q_occ_diag, optimize=True)
                else:
                    sic_occ = np.zeros(self.n_occ_act)
                
                sigma_occ_raw = coh_occ + sex_occ - sic_occ
                
                coh_virt = 0.5 * np.einsum('aA,AB,aB->a', q_virt_diag, dW, q_virt_diag, optimize=True)
                Wq_virt = np.einsum('AB,ajB->ajA', W, q_act_virt_val, optimize=True)
                sex_virt = -np.einsum('ajA,ajA->a', q_act_virt_val, Wq_virt, optimize=True)
                
                if self.gamma_penalty is not None:
                    sic_virt = np.einsum('aA,AB,aB->a', q_virt_diag, self.gamma_penalty, q_virt_diag, optimize=True)
                else:
                    sic_virt = np.zeros(self.n_virt_act)
                
                sigma_virt_raw = coh_virt + sex_virt + sic_virt

                # Extract for diagnostics
                homo_coh = coh_occ[-1]
                homo_sex = sex_occ[-1]
                homo_sic = sic_occ[-1]
                
                lumo_coh = coh_virt[0]
                lumo_sex = sex_virt[0]
                lumo_sic = sic_virt[0]

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
                # 3. Exact Vxc Correction from AO Matrix OR HOMO Referencing
                if vxc_ao_path is not None and os.path.exists(vxc_ao_path):
                    print("\n  [Vxc] Applying Exact State-Dependent Vxc Integrals...")
                    V_ao = get_vxc_ao_matrix(vxc_ao_path, self.overlap.shape[0])
                    
                    # Fully Vectorized Vxc Projection
                    vxc_occ = np.sum(self.C_orig_occ * (V_ao @ self.C_orig_occ), axis=0)
                    vxc_virt = np.sum(self.C_orig_virt * (V_ao @ self.C_orig_virt), axis=0)

                    vxc_occ_ev = vxc_occ * HA_TO_EV
                    vxc_virt_ev = vxc_virt * HA_TO_EV

                    print(f"       HOMO Exact Vxc: {vxc_occ_ev[-1]:8.4f} eV")
                    print(f"       LUMO Exact Vxc: {vxc_virt_ev[0]:8.4f} eV")
                    
                    # Apply Exact State-Dependent G0W0 Equation
                    self.sigma_occ = sigma_occ_raw - vxc_occ_ev
                    self.sigma_virt = sigma_virt_raw - vxc_virt_ev
                    
                    cohsex_gap_corr = self.sigma_virt[0] - self.sigma_occ[-1]
                    print(f"       Exact Vxc Gap Correction: {cohsex_gap_corr:+8.4f} eV")

                else:
                    # Fallback to standard HOMO-referencing
                    if vxc_ao_path is not None:
                        print(f"\n  [Vxc] WARNING: '{vxc_ao_path}' not found! Falling back to HOMO-referencing.")
                        
                    homo_raw_shift = sigma_occ_raw[-1]
                    self.sigma_occ = sigma_occ_raw - homo_raw_shift
                    self.sigma_virt = sigma_virt_raw - homo_raw_shift 
                    
                    cohsex_gap_corr = self.sigma_virt[0] - self.sigma_occ[-1]
                    print(f"       Pure COHSEX Gap Correction: {cohsex_gap_corr:+8.4f} eV")
 
                # 4. HYBRID GW APPROACH: Anchor to Tabulated GW Scissor
                if scissor_ev != 0.0:
                    print(f"       Tabulated GW Target Shift : {scissor_ev:+8.4f} eV")
                    residual_shift = scissor_ev - cohsex_gap_corr
                    self.sigma_virt += residual_shift
                    print(f"       -> Applied residual shift of {residual_shift:+8.4f} eV to virtuals to match Tabulated GW Gap.")
                    
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
        # ULTRA-FAST Charge Construction
        # --------------------------------------------------
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

        elif charge_type == 'lowdin':
            print(f"  Building transition charges (Atom-by-Atom via Lowdin symmetric orthogonalization)...")
            from miniBSE.lowdin import build_lowdin_transition_charges_flat
            S_dense = overlap.toarray() if hasattr(overlap, "toarray") else overlap
            self.q_flat = build_lowdin_transition_charges_flat(
                C_occ_act, C_virt_act, S_dense, atom_ao_ranges, self.valid_i, self.valid_a
            )

        print(f"    Charges built in {time.time() - start_q:2.4f} s")

        # --------------------------------------------------
        # SPIN-ORBIT COUPLING (SPINOR) TRANSFORMATION
        # --------------------------------------------------
        if self.soc_flag:
            self.build_spinor_basis(soc_U, soc_E, e_thresh)

    def init_uks_spin_preserving(self, C_alpha, eps_alpha, homo_index_alpha, n_occ_alpha, n_virt_alpha,
                                   C_beta, eps_beta, homo_index_beta, n_occ_beta, n_virt_beta,
                                   overlap, atom_ao_ranges, scissor_ev, e_thresh, f_thresh,
                                   mu_ia_x, mu_ia_y, mu_ia_z, charge_type, device, soc_U=None, soc_E=None):
        """
        Manifold B: Coupled spin-preserving excitations for an open-shell UKS reference.
        Alpha transitions (alpha_occ -> alpha_virt) and beta transitions (beta_occ -> beta_virt)
        are coupled by the spin-independent Coulomb interaction J.
        Exchange K is block-diagonal: K_alpha acts only within alpha transitions, K_beta only within beta.
        """
        print(f"\n--- [4] Building Exciton Hamiltonian (UKS Spin-Preserving, Manifold B) ---")
        n_atoms = len(atom_ao_ranges)
        self.n_atoms = n_atoms
        self.scissor_ev = 0.0  # will be consumed into D below

        if C_beta is None:
            raise ValueError("UKS spin-preserving mode requires C_beta (beta MOs).")
        if hasattr(C_alpha, 'toarray'): C_alpha = C_alpha.toarray()
        if hasattr(C_beta, 'toarray'): C_beta = C_beta.toarray()

        # ---- Alpha channel active space ----
        occ_idx_a  = np.arange(homo_index_alpha - n_occ_alpha + 1, homo_index_alpha + 1)
        virt_idx_a = np.arange(homo_index_alpha + 1, homo_index_alpha + 1 + n_virt_alpha)
        C_occ_a  = C_alpha[:, occ_idx_a]
        C_virt_a = C_alpha[:, virt_idx_a]
        eps_occ_a  = eps_alpha[occ_idx_a].copy()
        eps_virt_a = eps_alpha[virt_idx_a].copy()
        # Apply scissor to virtual alpha energies
        eps_virt_a += scissor_ev

        # ---- Beta channel active space ----
        occ_idx_b  = np.arange(homo_index_beta - n_occ_beta + 1, homo_index_beta + 1)
        virt_idx_b = np.arange(homo_index_beta + 1, homo_index_beta + 1 + n_virt_beta)
        C_occ_b  = C_beta[:, occ_idx_b]
        C_virt_b = C_beta[:, virt_idx_b]
        eps_occ_b  = eps_beta[occ_idx_b].copy()
        eps_virt_b = eps_beta[virt_idx_b].copy()
        # Apply scissor to virtual beta energies
        eps_virt_b += scissor_ev

        n_occ_a, n_virt_a = len(occ_idx_a), len(virt_idx_a)
        n_occ_b, n_virt_b = len(occ_idx_b), len(virt_idx_b)

        print(f"  Alpha channel: {n_occ_a} occ x {n_virt_a} virt = {n_occ_a * n_virt_a} transitions")
        print(f"  Beta  channel: {n_occ_b} occ x {n_virt_b} virt = {n_occ_b * n_virt_b} transitions")

        # Store for later use in solver / analysis
        self.n_occ_act   = n_occ_a   # primary (alpha) sizes for compatibility
        self.n_virt_act  = n_virt_a
        self.n_occ_act_b = n_occ_b
        self.n_virt_act_b = n_virt_b
        self.homo_index_alpha = homo_index_alpha
        self.homo_index_beta  = homo_index_beta
        self.occ_idx_a, self.virt_idx_a = occ_idx_a, virt_idx_a
        self.occ_idx_b, self.virt_idx_b = occ_idx_b, virt_idx_b
        self.C_orig_occ  = C_occ_a
        self.C_orig_virt = C_virt_a
        self.C_orig_occ_b = C_occ_b
        self.C_orig_virt_b = C_virt_b
        self.sigma_occ_a = np.zeros(n_occ_a)
        self.sigma_virt_a = np.full(n_virt_a, scissor_ev)
        self.sigma_occ_b = np.zeros(n_occ_b)
        self.sigma_virt_b = np.full(n_virt_b, scissor_ev)
        self.sigma_occ = self.sigma_occ_a
        self.sigma_virt = self.sigma_virt_a

        # ---- Compute raw orbital energy gaps (DFT, for threshold filtering) ----
        dft_gap_a = eps_alpha[virt_idx_a].reshape(1, -1) - eps_alpha[occ_idx_a].reshape(-1, 1)
        dft_gap_b = eps_beta[virt_idx_b].reshape(1, -1)  - eps_beta[occ_idx_b].reshape(-1, 1)

        # ---- Apply e_thresh mask ----
        mask_e_a = (dft_gap_a <= e_thresh) if e_thresh is not None else np.ones_like(dft_gap_a, dtype=bool)
        mask_e_b = (dft_gap_b <= e_thresh) if e_thresh is not None else np.ones_like(dft_gap_b, dtype=bool)

        # ---- Oscillator strength pre-filter (alpha block only, beta assigned zero) ----
        if f_thresh > 0.0 and mu_ia_x is not None:
            mu_x_a, mu_x_b = mu_ia_x
            mu_y_a, mu_y_b = mu_ia_y
            mu_z_a, mu_z_b = mu_ia_z
            gap_au_a = dft_gap_a / 27.211386
            f_ia_0_a = (4.0 / 3.0) * gap_au_a * (mu_x_a**2 + mu_y_a**2 + mu_z_a**2)
            mask_f_a = f_ia_0_a >= f_thresh
            gap_au_b = dft_gap_b / 27.211386
            f_ia_0_b = (4.0 / 3.0) * gap_au_b * (mu_x_b**2 + mu_y_b**2 + mu_z_b**2)
            mask_f_b = f_ia_0_b >= f_thresh
        else:
            mask_f_a = np.ones_like(dft_gap_a, dtype=bool)
            mask_f_b = np.ones_like(dft_gap_b, dtype=bool)

        valid_mask_a = mask_e_a & mask_f_a
        valid_mask_b = mask_e_b & mask_f_b
        vi_a, va_a = np.where(valid_mask_a)
        vi_b, va_b = np.where(valid_mask_b)
        dim_a, dim_b = len(vi_a), len(vi_b)

        if (dim_a + dim_b) == 0:
            print(f"ERROR: UKS spin-preserving CI space is empty! e_thresh ({e_thresh}) is too strict.")
            sys.exit(1)

        print(f"  Alpha transitions after threshold: {dim_a}")
        print(f"  Beta  transitions after threshold: {dim_b}")
        print(f"  Total CI dimension: {dim_a + dim_b}")

        # Store valid indices for both channels
        self.valid_i,   self.valid_a   = vi_a, va_a   # alpha (legacy compat)
        self.valid_mask                 = valid_mask_a
        self.vi_a, self.va_a           = vi_a, va_a
        self.vi_b, self.va_b           = vi_b, va_b
        self.dim_a, self.dim_b         = dim_a, dim_b
        self.dim = dim_a + dim_b

        # ---- QP diagonal (use QP-corrected energies with scissor applied) ----
        qp_gap_a = eps_virt_a.reshape(1, -1) - eps_occ_a.reshape(-1, 1)
        qp_gap_b = eps_virt_b.reshape(1, -1) - eps_occ_b.reshape(-1, 1)
        D_a = qp_gap_a[valid_mask_a]
        D_b = qp_gap_b[valid_mask_b]
        self.D_spatial = np.concatenate([D_a, D_b])
        self.D = self.D_spatial

        print(f"  Alpha energy range: {D_a.min():.3f} – {D_a.max():.3f} eV" if dim_a else "  Alpha: no transitions")
        print(f"  Beta  energy range: {D_b.min():.3f} – {D_b.max():.3f} eV" if dim_b else "  Beta:  no transitions")

        # ---- Build transition charges (concatenated, flat) ----
        start_q = time.time()
        S = overlap.toarray() if hasattr(overlap, 'toarray') else overlap

        if charge_type == 'mulliken':
            print(f"  Building Mulliken transition charges (alpha + beta channels)...")
            SC_occ_a  = S @ C_occ_a
            SC_virt_a = S @ C_virt_a
            SC_occ_b  = S @ C_occ_b
            SC_virt_b = S @ C_virt_b
            sc_built = True

            q_flat_a = np.zeros((dim_a, n_atoms), dtype=np.float32)
            q_flat_b = np.zeros((dim_b, n_atoms), dtype=np.float32)

            for A, (a0, a1) in enumerate(atom_ao_ranges):
                if dim_a:
                    Ci_A  = C_occ_a[a0:a1, :][:, vi_a]
                    SCa_A = SC_virt_a[a0:a1, :][:, va_a]
                    Ca_A  = C_virt_a[a0:a1, :][:, va_a]
                    SCi_A = SC_occ_a[a0:a1, :][:, vi_a]
                    q_flat_a[:, A] = 0.5 * (np.sum(Ci_A * SCa_A, axis=0) + np.sum(Ca_A * SCi_A, axis=0))
                if dim_b:
                    Ci_B  = C_occ_b[a0:a1, :][:, vi_b]
                    SCa_B = SC_virt_b[a0:a1, :][:, va_b]
                    Ca_B  = C_virt_b[a0:a1, :][:, va_b]
                    SCi_B = SC_occ_b[a0:a1, :][:, vi_b]
                    q_flat_b[:, A] = 0.5 * (np.sum(Ci_B * SCa_B, axis=0) + np.sum(Ca_B * SCi_B, axis=0))

        elif charge_type == 'lowdin':
            print(f"  Building Löwdin transition charges (alpha + beta channels)...")
            from miniBSE.lowdin import build_lowdin_transition_charges_flat
            q_flat_a = build_lowdin_transition_charges_flat(C_occ_a, C_virt_a, S, atom_ao_ranges, vi_a, va_a) if dim_a else np.zeros((0, n_atoms), dtype=np.float32)
            q_flat_b = build_lowdin_transition_charges_flat(C_occ_b, C_virt_b, S, atom_ao_ranges, vi_b, va_b) if dim_b else np.zeros((0, n_atoms), dtype=np.float32)
            sc_built = False
        else:
            raise ValueError(f"Unknown charge_type '{charge_type}'. Use 'mulliken' or 'lowdin'.")

        # Concatenate into unified flat charge array [dim_a + dim_b, n_atoms]
        self.q_flat   = np.concatenate([q_flat_a, q_flat_b], axis=0).astype(np.float32)
        self.q_flat_a = q_flat_a
        self.q_flat_b = q_flat_b
        print(f"    Charges built in {time.time() - start_q:.4f} s")

        # ---- Build full channel density blocks for exchange and/or SOC rotation ----
        if self.include_exchange or self.soc_flag:
            print(f"  Building same-spin density blocks...")
            t_ex = time.time()
            # Compute S@C products if not already done (e.g. lowdin path skipped them)
            if not sc_built:
                SC_occ_a  = S @ C_occ_a
                SC_virt_a = S @ C_virt_a
                SC_occ_b  = S @ C_occ_b
                SC_virt_b = S @ C_virt_b
            # Alpha block: q_occ_a[i,j,A] and q_virt_a[a,b,A]
            self.q_occ_a  = np.zeros((n_occ_a, n_occ_a, n_atoms))
            self.q_virt_a = np.zeros((n_virt_a, n_virt_a, n_atoms))
            self.q_ov_a = np.zeros((n_occ_a, n_virt_a, n_atoms))
            for A, (a0, a1) in enumerate(atom_ao_ranges):
                Co  = C_occ_a[a0:a1, :]
                SCo = SC_occ_a[a0:a1, :]
                self.q_occ_a[:, :, A] = 0.5 * (Co.T @ SCo + SCo.T @ Co)
                Cv  = C_virt_a[a0:a1, :]
                SCv = SC_virt_a[a0:a1, :]
                self.q_virt_a[:, :, A] = 0.5 * (Cv.T @ SCv + SCv.T @ Cv)
                self.q_ov_a[:, :, A] = 0.5 * (Co.T @ SCv + SCo.T @ Cv)
            # Beta block
            self.q_occ_b  = np.zeros((n_occ_b, n_occ_b, n_atoms))
            self.q_virt_b = np.zeros((n_virt_b, n_virt_b, n_atoms))
            self.q_ov_b = np.zeros((n_occ_b, n_virt_b, n_atoms))
            for A, (a0, a1) in enumerate(atom_ao_ranges):
                Co  = C_occ_b[a0:a1, :]
                SCo = SC_occ_b[a0:a1, :]
                self.q_occ_b[:, :, A] = 0.5 * (Co.T @ SCo + SCo.T @ Co)
                Cv  = C_virt_b[a0:a1, :]
                SCv = SC_virt_b[a0:a1, :]
                self.q_virt_b[:, :, A] = 0.5 * (Cv.T @ SCv + SCv.T @ Cv)
                self.q_ov_b[:, :, A] = 0.5 * (Co.T @ SCv + SCo.T @ Cv)

            # Pre-contract screened virtual blocks  W = gamma @ q_virt
            if self.include_exchange:
                qva = self.q_virt_a.reshape(n_virt_a * n_virt_a, n_atoms)
                self.W_virt_a = (qva @ self.gamma.T).reshape(n_virt_a, n_virt_a, n_atoms)
                qvb = self.q_virt_b.reshape(n_virt_b * n_virt_b, n_atoms)
                self.W_virt_b = (qvb @ self.gamma.T).reshape(n_virt_b, n_virt_b, n_atoms)
            print(f"    Same-spin density blocks built in {time.time() - t_ex:.4f} s")

        if self.soc_flag:
            self.build_spinor_basis_uks(soc_U, soc_E, e_thresh)

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

        print("  -> Rotating spatial charge tensors into the spinor basis...")
        t_sp = time.time()

        q_trans_a = np.einsum("ip,iaA,aq->pqA", U_occ_a.conj(), self.q_ov, U_virt_a, optimize=True)
        q_trans_b = np.einsum("ip,iaA,aq->pqA", U_occ_b.conj(), self.q_ov, U_virt_b, optimize=True)
        self.q_spinor = (q_trans_a + q_trans_b).reshape(self.dim_spinor, self.n_atoms)
        del q_trans_a, q_trans_b

        if self.include_exchange:
            self.q_hole_spinor = (
                np.einsum("ip,ijA,jq->pqA", U_occ_a.conj(), self.q_occ, U_occ_a, optimize=True)
                + np.einsum("ip,ijA,jq->pqA", U_occ_b.conj(), self.q_occ, U_occ_b, optimize=True)
            )
            self.q_elec_spinor = (
                np.einsum("ap,abA,bq->pqA", U_virt_a.conj(), self.q_virt, U_virt_a, optimize=True)
                + np.einsum("ap,abA,bq->pqA", U_virt_b.conj(), self.q_virt, U_virt_b, optimize=True)
            )

        if self.include_exchange:
            qe = self.q_elec_spinor.reshape(self.n_virt_spinor * self.n_virt_spinor, self.n_atoms)
            self.W_elec_spinor = (qe @ self.gamma.T).reshape(self.n_virt_spinor, self.n_virt_spinor, self.n_atoms)

        print(f"  -> Density mappings compiled in {time.time() - t_sp:.2f}s")

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

    def build_spinor_basis_uks(self, U_mo, soc_E, e_thresh):
        print("\n--- [SOC-UKS] Transforming Exciton Hamiltonian to Spinor Basis ---")
        n_alpha = self.n_occ_act + self.n_virt_act
        n_beta = self.n_occ_act_b + self.n_virt_act_b
        self.n_occ_spinor = self.n_occ_act + self.n_occ_act_b
        self.n_virt_spinor = self.n_virt_act + self.n_virt_act_b
        self.dim_spinor = self.n_occ_spinor * self.n_virt_spinor
        self.dim = self.dim_spinor
        self.soc_U = U_mo

        occ_cols = slice(0, self.n_occ_spinor)
        virt_cols = slice(self.n_occ_spinor, self.n_occ_spinor + self.n_virt_spinor)

        U_occ_a = U_mo[0:self.n_occ_act, occ_cols]
        U_virt_a = U_mo[self.n_occ_act:n_alpha, virt_cols]
        U_occ_b = U_mo[n_alpha:n_alpha + self.n_occ_act_b, occ_cols]
        U_virt_b = U_mo[n_alpha + self.n_occ_act_b:n_alpha + n_beta, virt_cols]

        print("  -> Rotating UKS alpha/beta charge tensors into the spinor basis...")
        t_sp = time.time()
        q_trans_a = np.einsum("ip,iaA,aq->pqA", U_occ_a.conj(), self.q_ov_a, U_virt_a, optimize=True)
        q_trans_b = np.einsum("ip,iaA,aq->pqA", U_occ_b.conj(), self.q_ov_b, U_virt_b, optimize=True)
        self.q_spinor = (q_trans_a + q_trans_b).reshape(self.dim_spinor, self.n_atoms)

        if self.include_exchange:
            self.q_hole_spinor = (
                np.einsum("ip,ijA,jq->pqA", U_occ_a.conj(), self.q_occ_a, U_occ_a, optimize=True)
                + np.einsum("ip,ijA,jq->pqA", U_occ_b.conj(), self.q_occ_b, U_occ_b, optimize=True)
            )
            self.q_elec_spinor = (
                np.einsum("ap,abA,bq->pqA", U_virt_a.conj(), self.q_virt_a, U_virt_a, optimize=True)
                + np.einsum("ap,abA,bq->pqA", U_virt_b.conj(), self.q_virt_b, U_virt_b, optimize=True)
            )
            qe = self.q_elec_spinor.reshape(self.n_virt_spinor * self.n_virt_spinor, self.n_atoms)
            self.W_elec_spinor = (qe @ self.gamma.T).reshape(self.n_virt_spinor, self.n_virt_spinor, self.n_atoms)

        print(f"  -> UKS density mappings compiled in {time.time() - t_sp:.2f}s")

        eps_occ_sp = soc_E[:self.n_occ_spinor].copy()
        eps_virt_sp = soc_E[self.n_occ_spinor:self.n_occ_spinor + self.n_virt_spinor].copy()
        raw_gap_spinor_dft = (eps_virt_sp.reshape(1, -1) - eps_occ_sp.reshape(-1, 1)).flatten()

        sigma_occ_sp = np.concatenate([self.sigma_occ_a, self.sigma_occ_b])
        sigma_virt_sp = np.concatenate([self.sigma_virt_a, self.sigma_virt_b])

        eps_occ_sp += sigma_occ_sp
        eps_virt_sp += sigma_virt_sp

        raw_gap_spinor_qp = (eps_virt_sp.reshape(1, -1) - eps_occ_sp.reshape(-1, 1)).flatten()
        raw_D_spinor = raw_gap_spinor_qp + self.scissor_ev

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
        print(f"  -> UKS BSE Active Space expanded to {self.dim_spinor_full} spinor transitions.")
        print(f"  -> Truncated UKS Spinor Space (Energy <= {e_thresh} eV): {self.dim} valid transitions")

    def matvec(self, x):
        """Matrix-vector product for Davidson Solver."""
        
        if not self.soc_flag:
            if getattr(self, 'spin', 'singlet') == 'triplet':
                # === TRIPLET SPATIAL MATVEC ===
                y = self.D_spatial * x
                # No J term
                if self.include_exchange:
                    x_mat = np.zeros((self.n_occ_act, self.n_virt_act))
                    x_mat[self.valid_i, self.valid_a] = x
                    K = np.einsum("ijA, abA, jb -> ia", self.q_occ, self.W_virt, x_mat, optimize=True)
                    y -= 1.0 * K[self.valid_i, self.valid_a]
                return y

            if getattr(self, 'spin', 'singlet') == 'uks_spin_preserving':
                # === UKS SPIN-PRESERVING MATVEC (Manifold B) ===
                # Diagonal term
                y = self.D_spatial * x

                # Full Coulomb J (spin-independent, couples all alpha and beta transitions)
                # Factor is 1.0 (we are explicit in spin-channel space, no 2x factor)
                T = self.q_flat.T @ x          # [n_atoms]
                y += 1.0 * self.q_flat @ (self.gamma @ T)

                if self.include_exchange:
                    # Alpha block: K_alpha applied to x_alpha part
                    x_a = x[:self.dim_a]
                    x_mat_a = np.zeros((self.n_occ_act, self.n_virt_act))
                    x_mat_a[self.vi_a, self.va_a] = x_a
                    K_a = np.einsum("ijA, abA, jb -> ia", self.q_occ_a, self.W_virt_a, x_mat_a, optimize=True)
                    y[:self.dim_a] -= K_a[self.vi_a, self.va_a]

                    # Beta block: K_beta applied to x_beta part
                    x_b = x[self.dim_a:]
                    x_mat_b = np.zeros((self.n_occ_act_b, self.n_virt_act_b))
                    x_mat_b[self.vi_b, self.va_b] = x_b
                    K_b = np.einsum("ijA, abA, jb -> ia", self.q_occ_b, self.W_virt_b, x_mat_b, optimize=True)
                    y[self.dim_a:] -= K_b[self.vi_b, self.va_b]

                return y

            # === STANDARD SPATIAL MATVEC (Singlets) ===
            y = self.D_spatial * x
            T = self.q_flat.T @ x
            y += 2.0 * self.q_flat @ (self.gamma @ T) # 2.0 Spin-Multiplicity
            
            if self.include_exchange:
                x_mat = np.zeros((self.n_occ_act, self.n_virt_act))
                x_mat[self.valid_i, self.valid_a] = x
                K = np.einsum("ijA, abA, jb -> ia", self.q_occ, self.W_virt, x_mat, optimize=True)
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
                
            # K[i, a] = \sum_{j,b,A} (q_hole[i, j, A])^* W[a, b, A] x_mat[j, b]
            K = np.einsum("ijA, abA, jb -> ia", self.q_hole_spinor.conj(), self.W_elec_spinor, x_mat, optimize=True)
            
            if hasattr(self, 'valid_spinor_idx'):
                y -= K.flatten()[self.valid_spinor_idx]
            else:
                y -= K.flatten()
            
        return y

    def get_transition_dipoles(self, mu_ia_x, mu_ia_y, mu_ia_z):
        """Extracts transition dipoles in either the spatial or spinor basis."""
        if getattr(self, 'spin', 'singlet') == 'triplet':
            return np.zeros((self.dim, 3))

        if getattr(self, 'spin', 'singlet') == 'uks_spin_preserving' and not self.soc_flag:
            # mu_ia_x/y/z are tuples: (mu_alpha, mu_beta)
            mu_x_a, mu_x_b = mu_ia_x
            mu_y_a, mu_y_b = mu_ia_y
            mu_z_a, mu_z_b = mu_ia_z
            mu_a = np.zeros((self.dim_a, 3))
            mu_b = np.zeros((self.dim_b, 3))
            if self.dim_a:
                mu_a[:, 0] = mu_x_a[self.vi_a, self.va_a]
                mu_a[:, 1] = mu_y_a[self.vi_a, self.va_a]
                mu_a[:, 2] = mu_z_a[self.vi_a, self.va_a]
            if self.dim_b:
                mu_b[:, 0] = mu_x_b[self.vi_b, self.va_b]
                mu_b[:, 1] = mu_y_b[self.vi_b, self.va_b]
                mu_b[:, 2] = mu_z_b[self.vi_b, self.va_b]
            return np.concatenate([mu_a, mu_b], axis=0)

        if not self.soc_flag:
            mu_ia = np.zeros((len(self.valid_i), 3))
            mu_ia[:, 0], mu_ia[:, 1], mu_ia[:, 2] = mu_ia_x[self.valid_i, self.valid_a], mu_ia_y[self.valid_i, self.valid_a], mu_ia_z[self.valid_i, self.valid_a]
            return mu_ia

        if getattr(self, 'spin', 'singlet') == 'uks_spin_preserving':
            n_alpha = self.n_occ_act + self.n_virt_act
            occ_cols = slice(0, self.n_occ_spinor)
            virt_cols = slice(self.n_occ_spinor, self.n_occ_spinor + self.n_virt_spinor)
            U_occ_a = self.soc_U[0:self.n_occ_act, occ_cols]
            U_virt_a = self.soc_U[self.n_occ_act:n_alpha, virt_cols]
            U_occ_b = self.soc_U[n_alpha:n_alpha + self.n_occ_act_b, occ_cols]
            U_virt_b = self.soc_U[n_alpha + self.n_occ_act_b:n_alpha + self.n_occ_act_b + self.n_virt_act_b, virt_cols]

            mu_x_a, mu_x_b = mu_ia_x
            mu_y_a, mu_y_b = mu_ia_y
            mu_z_a, mu_z_b = mu_ia_z

            def map_dipole_uks(mu_a, mu_b):
                mu_sp = U_occ_a.conj().T @ mu_a @ U_virt_a + U_occ_b.conj().T @ mu_b @ U_virt_b
                return mu_sp.flatten()[self.valid_spinor_mask] if hasattr(self, 'valid_spinor_mask') else mu_sp.flatten()

            return np.column_stack((
                map_dipole_uks(mu_x_a, mu_x_b),
                map_dipole_uks(mu_y_a, mu_y_b),
                map_dipole_uks(mu_z_a, mu_z_b),
            ))
            
        # Fast memory-free mapping for dipoles using U directly via matrix multiplication
        k = self.n_occ_act + self.n_virt_act
        U_occ_a = self.soc_U[0 : self.n_occ_act, 0 : self.n_occ_spinor]
        U_virt_a = self.soc_U[self.n_occ_act : k, self.n_occ_spinor : 2*k]
        U_occ_b = self.soc_U[k : k + self.n_occ_act, 0 : self.n_occ_spinor]
        U_virt_b = self.soc_U[k + self.n_occ_act : 2*k, self.n_occ_spinor : 2*k]
        
        def map_dipole(mu_spatial):
            mu_sp = U_occ_a.conj().T @ mu_spatial @ U_virt_a + U_occ_b.conj().T @ mu_spatial @ U_virt_b
            return mu_sp.flatten()[self.valid_spinor_mask] if hasattr(self, 'valid_spinor_mask') else mu_sp.flatten()

        return np.column_stack((map_dipole(mu_ia_x), map_dipole(mu_ia_y), map_dipole(map_dipole(mu_ia_z) if False else mu_ia_z)))
