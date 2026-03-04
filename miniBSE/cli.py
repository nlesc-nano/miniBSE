import argparse
import numpy as np
import sys
import os
import time 
import gc
import platform
import yaml 

import libint_cpp

from miniBSE.io_utils import (
    read_xyz,
    parse_basis,
    build_shell_dicts,
    count_ao_from_shells,
    build_atom_ao_ranges,
    read_mos_auto
)

from miniBSE.solver import ExcitonSolver
from miniBSE.constants import HA_TO_EV
from miniBSE.exciton_analysis import ExcitonAnalyzer, plot_analysis_summary
from miniBSE.exciton_cube import generate_exciton_cubes
from miniBSE.integrals import compute_dipole_ao
from miniBSE.oscillator import compute_oscillator_strengths
from miniBSE.hardness import MATERIAL_DB

def compute_spin_character(vec, soc_U, n_occ_sp, n_virt_sp, valid_mask=None):
    """
    Projects the Spinor BSE eigenvector back onto the spatial basis 
    to calculate the total Singlet and Triplet weights.
    """
    # Reconstruct the full (n_occ_sp x n_virt_sp) matrix
    X_IA = np.zeros((n_occ_sp, n_virt_sp), dtype=vec.dtype)
    if valid_mask is not None:
        X_IA[valid_mask] = vec
    else:
        X_IA = vec.reshape((n_occ_sp, n_virt_sp))
    
    # soc_U has shape (2*n_mo, 2*n_mo). Top half is alpha, bottom is beta.
    n_mo = soc_U.shape[0] // 2
    
    # Extract alpha and beta components of the occupied and virtual spinors
    U_occ_a = soc_U[:n_mo, :n_occ_sp]
    U_occ_b = soc_U[n_mo:, :n_occ_sp]
    
    U_virt_a = soc_U[:n_mo, n_occ_sp:]
    U_virt_b = soc_U[n_mo:, n_occ_sp:]
    
    # [Rest of the function remains exactly the same]
    rho_aa = U_occ_a.conj() @ X_IA @ U_virt_a.T
    rho_bb = U_occ_b.conj() @ X_IA @ U_virt_b.T
    
    S_mat = (rho_aa + rho_bb) / np.sqrt(2.0)
    singlet_weight = np.sum(np.abs(S_mat)**2)
    
    singlet_weight = min(1.0, max(0.0, singlet_weight))
    triplet_weight = 1.0 - singlet_weight
    
    return singlet_weight * 100, triplet_weight * 100

def print_orbital_summary(energies_eV, occ, homo_idx, C, S, syms, shells, is_soc=False, offset=0, print_range=15):
    """
    Fast Mulliken population analysis broken down by Element and Angular Momentum (s, p, d).
    """
    print("\n" + "="*115)
    print(f"{'Orbital':>14} | {'Index':>6} | {'Energy (eV)':>12} | {'Occ':>5} | {'Main Contributions':>45}")
    print("-" * 115)
    
    n_ao = S.shape[0] if hasattr(S, 'shape') else S.shape[0]
    n_states = C.shape[1]
    
    # 1. Build AO mapping array (e.g. ['Pb(s)', 'Pb(p)', 'Pb(p)', 'Pb(p)', ...])
    l_char = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}
    ao_labels = []
    for sh in shells:
        sym = sh['sym']
        l_int = int(sh['l'])
        l_str = l_char.get(l_int, str(l_int))
        label = f"{sym}({l_str})"
        
        # Spherical harmonics have 2*l + 1 basis functions per shell
        nbf = 2 * l_int + 1
        ao_labels.extend([label] * nbf)
        
    unique_labels = list(dict.fromkeys(ao_labels))
    label_pops = np.zeros((len(unique_labels), n_states))

    # 2. Fast Mulliken: C_mu * (S @ C)_mu
    if is_soc:
        # SOC spinors have 2*n_ao rows (alpha on top, beta on bottom)
        C_alpha, C_beta = C[:n_ao, :], C[n_ao:, :]
        SC_alpha, SC_beta = S @ C_alpha, S @ C_beta
        pops = np.real(C_alpha.conj() * SC_alpha) + np.real(C_beta.conj() * SC_beta)
    else:
        SC = S @ C
        pops = np.real(C.conj() * SC)
        
    # 3. Sum contributions by unique Label (Element + L)
    for i_ao, label in enumerate(ao_labels):
        lbl_idx = unique_labels.index(label)
        label_pops[lbl_idx, :] += pops[i_ao, :]
        
    # Normalize to exactly 100% per state
    col_sums = np.sum(label_pops, axis=0)
    col_sums[col_sums == 0] = 1.0
    label_pops = label_pops / col_sums[np.newaxis, :]
    
    start_idx = max(0, homo_idx - print_range + 1)
    end_idx = min(n_states, homo_idx + 1 + print_range)
    
    for idx in range(end_idx - 1, start_idx - 1, -1):
        if not is_soc:
            rel = idx - homo_idx
            label = ("LUMO" if rel == 1 else f"LUMO+{rel-1}") if rel > 0 else ("HOMO" if rel == 0 else f"HOMO{rel}")
        else:
            rel = idx - homo_idx
            label = ("spL" if rel == 1 else f"spL+{rel-1}") if rel > 0 else ("spH" if rel == 0 else f"spH{rel}")
                
        # Format top elemental+L contributions
        state_pops = label_pops[:, idx]
        top_indices = np.argsort(-state_pops)[:5]  # Keep up to top 5 contributors
        contrib_str = ", ".join([f"{unique_labels[i]} ({state_pops[i]*100:.0f}%)" for i in top_indices if state_pops[i] > 0.05])
        
        print(f"{label:>14} | {idx + offset:6d} | {energies_eV[idx]:12.4f} | {occ[idx]:5.1f} | {contrib_str}")
        
        if idx == homo_idx + 1:
            print(f"   {'-- FERMI --':>11} | {'------':>6} | {'------------':>12} | {'-----':>5} | {'-'*45}")
    print("=" * 115 + "\n")

def run_solver_and_analysis(solver, coords_ang, syms, shells, mu_ia_x, mu_ia_y, mu_ia_z, dft_gap, scissor, confinement_energy, args, suffix="", soc_gap=None, soc_U=None, soc_E=None):
    """Encapsulates the solving, printing, analysis, and exporting."""
 
    label = "SOC" if solver.soc_flag else "SPIN-FREE"
    print(f"\n===================================================")
    print(f" [ {label} ] EXCITON CALCULATION ")
    print(f"===================================================")

    start_solve = time.time()
    energies_ev, vectors = solver.solve(nroots=args.nroots, full_diag=args.full_diag, tol=args.tol)
    
    if args.soc != 0.0:
        energies_ev = energies_ev - args.soc
        print(f"  [SOC Shift] Applied empirical energy shift: -{args.soc:.3f} eV")

    print(f"  {label} Solver converged in {time.time() - start_solve:2.2f} s")


    mu_ia = solver.ham.get_transition_dipoles(mu_ia_x, mu_ia_y, mu_ia_z)
    f_strengths = compute_oscillator_strengths(energies_ev, vectors, mu_ia, is_spinor=solver.soc_flag)

    print("\n" + "-"*60)
    print(f" SYSTEM ENERGY SUMMARY ({label})")
    print("-" * 60)
    print(f"  Raw DFT Gap           : {dft_gap:8.4f} eV")
    if solver.soc_flag and soc_gap is not None:
        print(f"  SOC Gap               : {soc_gap:8.4f} eV")
    print(f"  QP Correction (Shift) : {scissor:8.4f} eV")
    print(f"  Confinement Energy    : {confinement_energy:8.4f} eV")
    print(f"  Dielectric Tuning (α) : {args.alpha:8.4f}")
    print("-" * 60)

    print("\n" + "="*125)
    print(f"{'State':>5} {'Energy':>10} {'Main Trans':>12} {'Weight':>8} {'f_osc':>10} | {'PR':>5} | {'dE(eV)':>8} {'J(eV)':>8} {'K(eV)':>8} | {'Spin Character':>18}")
    print("-" * 125)

    n_print = min(100, len(energies_ev))

    for n in range(n_print):
        vec = vectors[:, n]
        hole_idx, elec_idx, weight = solver.main_transition(vec)
        
        if solver.soc_flag:
            # Spinors have 2x the orbitals. We will 1-index them.
            n_occ_sp = solver.ham.n_occ_spinor
            abs_h = hole_idx + 1                   # Hole is in the occupied spinor block
            abs_e = elec_idx + 1 + n_occ_sp        # Elec is in the virtual spinor block
            
            # Label them relative to the Spinor HOMO
            h_lbl = f"spH" if abs_h == n_occ_sp else f"spH-{n_occ_sp - abs_h}"
            e_lbl = f"spL" if abs_e == n_occ_sp + 1 else f"spL+{abs_e - (n_occ_sp + 1)}"
            trans_str = f"{h_lbl}->{e_lbl}"
        else:
            # Spatial MO logic
            abs_h = (solver.homo_index - solver.ham.n_occ_act + 1) + hole_idx
            abs_e = (solver.homo_index + 1) + elec_idx
            trans_str = f"{abs_h:3d}->{abs_e:3d}"

        # Calculate Expectation Values: X^T * Matrix * X
        vec_conj = vec.conj() if np.iscomplexobj(vec) else vec
        
        dE_val = np.sum((np.abs(vec)**2) * solver.ham.D)
        
        if hasattr(solver, 'J_mat'):
            J_val = np.real(vec_conj.T @ solver.J_mat @ vec)
            K_val = -np.real(vec_conj.T @ solver.K_mat @ vec) if solver.ham.include_exchange else 0.0
        else:
            J_val, K_val = 0.0, 0.0  # Fallback if using Davidson

        pr = 1.0 / np.sum(np.abs(vec)**4)

        if solver.soc_flag:
            soc_mask = None
            if args.e_thresh is not None and soc_E is not None:
                n_occ_sp = solver.ham.n_occ_spinor
                n_virt_sp = solver.ham.n_virt_spinor
                # The first n_occ_sp elements are occupied, the rest are virtual
                soc_occ_E = soc_E[:n_occ_sp]
                soc_virt_E = soc_E[n_occ_sp:n_occ_sp + n_virt_sp]
                # Recreate the exact mask used by the relativistic Hamiltonian
                soc_mask = (soc_virt_E.reshape(1, -1) - soc_occ_E.reshape(-1, 1)) <= args.e_thresh
                
            s_pct, t_pct = compute_spin_character(vec, soc_U, solver.ham.n_occ_spinor, solver.ham.n_virt_spinor, valid_mask=soc_mask)
            spin_str = f"{s_pct:5.1f}% S / {t_pct:5.1f}% T"
        else:
            # Spin-free states are purely singlets (if solving the singlet BSE block)
            spin_str = "100.0% S /   0.0% T"

        print(f"{n+1:5d} {energies_ev[n]:10.4f}  {trans_str:>12}  {weight**2:8.3f}  {f_strengths[n]:10.5f} | {pr:5.1f} | {dE_val:8.4f} {J_val:8.4f} {K_val:8.4f} | {spin_str:>18}")

    if len(energies_ev) > 100:
        print(f" ... {len(energies_ev) - 100} additional states computed (output truncated) ...")
    print("="*125)

    print(f"\n--- Performing Dreuw/Plasser Analysis ({label}) ---")
    analyzer = ExcitonAnalyzer(solver, np.array(coords_ang), syms)
    analysis_results = []

    print(f"{'State':>5} {'Energy':>8} {'f_osc':>8} | {'PR':>5} {'d_eh(A)':>7} {'d_CT(A)':>7} {'sig_h':>6} {'sig_e':>6} | {'Type':>8}")
    print("-" * 95)

    for n in range(n_print):
        vec = vectors[:, n]
        
        # Project spinor back to spatial for Dreuw-Plasser
        if solver.soc_flag:
            vec_spatial_complex = solver.ham.D_matrix @ vec
            
            # 1. Squeeze it down to match valid_i and valid_a
            if hasattr(solver.ham, 'valid_mask'):
                vec_spatial_complex = vec_spatial_complex[solver.ham.valid_mask.flatten()]
                
            # 2. Take the absolute magnitude to completely destroy the ComplexWarning
            vec_spatial = np.abs(vec_spatial_complex)
        else:
            vec_spatial = vec

        # Use vec_spatial for spatial properties
        res = analyzer.analyze_state(vec_spatial, energies_ev[n], f_strengths[n])
        analysis_results.append(res)
        
        ct_ratio = res['CT_Character']
        if ct_ratio > 0.6: ex_type = "CT"
        elif res['d_eh'] < 3.0 and ct_ratio < 0.2: ex_type = "Frenkel"
        else: ex_type = "Wannier"

        print(f"{n+1:5d} {res['energy']:8.3f} {res['f_osc']:8.4f} | "
              f"{res['PR']:5.1f} {res['d_eh']:7.2f} {res['d_CT']:7.2f} "
              f"{res['sigma_h']:6.1f} {res['sigma_e']:6.1f} | {ex_type:>8}")


    if args.write_csv:
        import csv
        csv_file = f"exciton_results{suffix}.csv"
        n_to_write = min(args.csv_roots, len(analysis_results))
 
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "State", "Energy_eV", "f_osc", "mu_x", "mu_y", "mu_z", "PR", "d_eh_A", "d_CT_A", "sigma_h_A", "sigma_e_A", "Type"])

            for n in range(n_to_write):
                res = analysis_results[n]
                mu_state = np.sum(vectors[:, n][:, None] * mu_ia, axis=0)
                
                ct_ratio = res['CT_Character']
                ex_type = "CT" if ct_ratio > 0.6 else "Frenkel" if res['d_eh'] < 3.0 and ct_ratio < 0.2 else "Wannier"

                writer.writerow([
                    args.time, n + 1, f"{energies_ev[n]:.6f}", f"{f_strengths[n]:.6e}", 
                    f"{np.real(mu_state[0]):.6f}", f"{np.real(mu_state[1]):.6f}", f"{np.real(mu_state[2]):.6f}",
                    f"{res['PR']:.3f}", f"{res['d_eh']:.4f}", f"{res['d_CT']:.4f}", 
                    f"{res['sigma_h']:.4f}", f"{res['sigma_e']:.4f}", ex_type
                ])

    if args.save_xia:
        npz_filename = f"xia{suffix}_{args.time:08.2f}fs.npz"
        n_to_write = min(args.csv_roots, len(energies_ev))
        np.savez_compressed(
            npz_filename,
            time=args.time,
            energies=energies_ev[:n_to_write],
            X_ia=vectors[:, :n_to_write], 
            valid_i=solver.ham.valid_i,
            valid_a=solver.ham.valid_a,
            homo_index=solver.homo_index,
            n_occ=solver.n_occ
        )
        print(f"  Saved X_ia coefficients to {npz_filename}")

    if args.broadening != "none":
        from miniBSE.spectrum import generate_spectrum, plot_spectrum
        e_min = max(0.0, np.min(energies_ev) - 2.5)
        e_max = np.max(energies_ev) + 2.5
        x_grid, y_grid = generate_spectrum(energies_ev, f_strengths, e_min=e_min, e_max=e_max, sigma=args.sigma, profile=args.broadening)
        
        spec_file = f"spectrum{suffix}.dat"
        np.savetxt(spec_file, np.column_stack((x_grid, y_grid)), header=f"Energy(eV) Intensity(arb.u.) | {args.broadening}, sigma={args.sigma}")
        
        if args.plot or args.show:
            plot_file = f"spectrum{suffix}.png" if args.plot else None
            plot_spectrum(x_grid, y_grid, energies_ev, f_strengths, filename=plot_file, show=args.show)


    if args.plot or args.show:
        # Calculate binding energy dynamically based on the current regime gap
        ref_gap = soc_gap + scissor if (solver.soc_flag and soc_gap is not None) else args.qp_gap
        
        metrics = {
            "dft_gap": dft_gap, 
            "qp_correction": scissor, 
            "confinement_energy": confinement_energy, 
            "binding_energy": ref_gap - energies_ev[0] if len(energies_ev) > 0 else 0.0, 
            "first_exc_energy": energies_ev[0] if len(energies_ev) > 0 else 0.0,
            "is_soc": solver.soc_flag
        }
        if solver.soc_flag and soc_gap is not None:
            metrics["soc_gap"] = soc_gap
            
        plot_file = f"exciton_analysis{suffix}.html" if args.plot else None
        plot_analysis_summary(
            analysis_results, 
            physics_metrics=metrics, 
            filename=plot_file, 
            show=args.show,
            broadening=args.broadening,
            sigma=args.sigma
        )

def main():
    parser = argparse.ArgumentParser(description="miniBSE - Lightweight post-DFT exciton solver")

    parser.add_argument("--config", type=str, help="Path to a YAML configuration file.")

    parser.add_argument("--mo_file")
    parser.add_argument("--xyz") 
    parser.add_argument("--basis_txt")
    parser.add_argument("--basis_name")

    parser.add_argument("--n-occ", type=int, default=50)
    parser.add_argument("--n-virt", type=int, default=50)
    parser.add_argument("--e_thresh", type=float, default=None)
    parser.add_argument("--f_thresh", type=float, default=1e-4)

    parser.add_argument("--qp_gap", type=float)
    parser.add_argument("--soc", type=float, default=0.0, help="Empirical global energy shift.")
    
    parser.add_argument("--soc_flag", action="store_true", help="Enable Relativistic Spin-Orbit Coupling.")
    parser.add_argument("--gth_file", type=str, default=None, help="Path to GTH potentials file for SOC.")

    parser.add_argument("--kernel", choices=["bse", "resta"], default="bse")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--exchange", action="store_true")
    parser.add_argument("--material", type=str, default="DEFAULT")
    parser.add_argument("--eps-out", type=float, default=2.0)

    parser.add_argument("--broadening", choices=["gaussian", "lorentzian", "none"], default="gaussian")
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--cube", action="store_true")
    parser.add_argument("--cube-spacing", type=float, default=0.5)
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--csv-roots", type=int, default=10)
    parser.add_argument("--time", type=float, default=0.0)
    parser.add_argument("--save-xia", action="store_true")
 
    parser.add_argument("--nroots", type=int, default=10)
    parser.add_argument("--full-diag", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps", "numpy"], default="auto")

    args = parser.parse_args()

    if args.config:
        print(f"Loading configuration from {args.config}...")
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
            
        # Flatten the nested dictionary and apply to args
        for section, parameters in config_data.items():
            if isinstance(parameters, dict):
                for key, value in parameters.items():
                    setattr(args, key, value)
            else:
                # Fallback if someone puts a flat variable in the YAML
                setattr(args, section, parameters)

    required_args = ['mo_file', 'xyz', 'basis_txt', 'basis_name', 'qp_gap']
    missing = [arg for arg in required_args if getattr(args, arg) is None]
    if missing:
        parser.error(f"Missing required arguments (must be provided via CLI or YAML): {', '.join(missing)}")

    print("\n===================================================")
    print(" miniBSE - Post-DFT Exciton Solver")
    print("===================================================")

    syms, coords_ang = read_xyz(args.xyz)
    basis_dict = parse_basis(args.basis_txt, args.basis_name)
    shells = build_shell_dicts(syms, coords_ang, basis_dict)
    n_ao = count_ao_from_shells(shells)
    atom_ao_ranges = build_atom_ao_ranges(shells)

    print("\n--- Computing AO overlap ---")
    S = libint_cpp.overlap(shells, args.nthreads)

    C, eps, occ = read_mos_auto(args.mo_file, n_ao, verbose=True)
    homo_index = np.where(occ > 0.0)[0].max()
    eps = eps * HA_TO_EV
    
    dft_gap = eps[homo_index + 1] - eps[homo_index]
    scissor = args.qp_gap - dft_gap

    # NEW: Print Spin-Free MO populations
    print("\n--- Spin-Free MO Population Analysis ---")
    C_dense = C.toarray() if hasattr(C, 'toarray') else C
    S_dense = S.toarray() if hasattr(S, 'toarray') else S
    print_orbital_summary(eps, occ, homo_index, C_dense, S_dense, syms, shells, is_soc=False)

    if args.e_thresh is not None:
        raw_gap = eps[homo_index + 1:].reshape(1, -1) - eps[:homo_index + 1].reshape(-1, 1)
        valid_pairs = raw_gap <= args.e_thresh
        if not np.any(valid_pairs): sys.exit("ERROR: Energy threshold too tight.")
        
        n_occ = homo_index - np.where(np.any(valid_pairs, axis=1))[0][0] + 1
        n_virt = np.where(np.any(valid_pairs, axis=0))[0][-1] + 1
    else:
        n_occ, n_virt = args.n_occ, args.n_virt

    print("\n--- Computing Transition Dipoles ---")
    mu_ao_x, mu_ao_y, mu_ao_z = compute_dipole_ao(shells, nthreads=args.nthreads)

    compute_device = "numpy"
    if args.device != "numpy":
        try:
            import torch
            if args.device == "auto":
                if torch.cuda.is_available(): compute_device = "cuda"
                elif hasattr(torch.backends, "mps") and platform.machine() == 'arm64': compute_device = "mps"
                else: compute_device = "cpu"
            else: compute_device = args.device
            torch.set_num_threads(args.nthreads)
        except ImportError: pass

    C_occ = C[:, homo_index - n_occ + 1 : homo_index + 1].astype(np.float32)
    C_virt = C[:, homo_index + 1 : homo_index + 1 + n_virt].astype(np.float32)

    def transform_dipole(mu_ao):
        if compute_device in ["cuda", "mps"]:
            dev = torch.device(compute_device)
            return (torch.tensor(C_occ, device=dev).T @ (torch.tensor(mu_ao.astype(np.float32), device=dev) @ torch.tensor(C_virt, device=dev))).cpu().numpy().astype(np.float64)
        return (C_occ.T @ (mu_ao.astype(np.float32) @ C_virt)).astype(np.float64)

    mu_ia_x, mu_ia_y, mu_ia_z = transform_dipole(mu_ao_x), transform_dipole(mu_ao_y), transform_dipole(mu_ao_z)
    del mu_ao_x, mu_ao_y, mu_ao_z; gc.collect()

    db_gap = MATERIAL_DB.get(args.material.upper(), MATERIAL_DB["DEFAULT"])[3]
    confinement_energy = args.qp_gap - db_gap

    # =========================================================================
    # EXECUTION BLOCK 1: SPIN-FREE SOLVER (Always runs)
    # =========================================================================
    solver_sf = ExcitonSolver(
        C=C, eps=eps, occ=occ, overlap=S, atom_symbols=syms, atom_coords=np.array(coords_ang),
        atom_ao_ranges=atom_ao_ranges, homo_index=homo_index, n_occ=n_occ, n_virt=n_virt, 
        scissor_ev=scissor, kernel=args.kernel, alpha=args.alpha, include_exchange=args.exchange, 
        material=args.material, e_thresh=args.e_thresh, f_thresh=args.f_thresh, 
        mu_ia_x=mu_ia_x, mu_ia_y=mu_ia_y, mu_ia_z=mu_ia_z, eps_out=args.eps_out,
        soc_U=None, soc_E=None, device=compute_device
    )

    suffix_sf = "_sf" if args.soc_flag else ""
    run_solver_and_analysis(solver_sf, np.array(coords_ang), syms, shells, mu_ia_x, mu_ia_y, mu_ia_z, 
                            dft_gap, scissor, confinement_energy, args, suffix=suffix_sf)

    # =========================================================================
    # EXECUTION BLOCK 2: SPIN-ORBIT COUPLING SOLVER
    # =========================================================================
    if args.soc_flag:
        if args.gth_file is None:
            sys.exit("ERROR: --gth_file is required when --soc_flag is enabled.")
            
        from miniBSE.soc_utils import compute_spinor_subspace
        
        active_indices = np.arange(homo_index - n_occ + 1, homo_index + 1 + n_virt)
        
        soc_E, soc_U = compute_spinor_subspace(
            atom_symbols=syms, coords_ang=coords_ang, shells=shells, 
            C_AO=C, eps_Ha=eps / HA_TO_EV, S_AO=S, 
            active_indices=active_indices, gth_file=args.gth_file, nthreads=args.nthreads
        )

        soc_E *= HA_TO_EV 

        # NEW: Print SOC Spinor populations
        print("\n--- SOC Spinor Population Analysis ---")
        n_mo_act = len(active_indices)
        C_act = C[:, active_indices].toarray() if hasattr(C, 'toarray') else C[:, active_indices]
        
        # Build spinor AO coefficients: [alpha_AO; beta_AO] vertically stacked
        C_spinor_ao = np.vstack([
            C_act @ soc_U[:n_mo_act, :],
            C_act @ soc_U[n_mo_act:, :]
        ])
        
        # The active space has `n_occ` spatial orbitals, which means 2*n_occ spinors are fully occupied
        soc_homo_idx = (2 * n_occ) - 1
        soc_occ = np.zeros_like(soc_E)
        soc_occ[:soc_homo_idx + 1] = 1.0
        
        # Optional offset so the printed indices match the "absolute" size of the system
        soc_offset = (homo_index - n_occ + 1) * 2 
        print_orbital_summary(soc_E, soc_occ, soc_homo_idx, C_spinor_ao, S_dense, 
                              syms, shells, is_soc=True, offset=soc_offset)
 
        # Calculate SOC Gap: virtual spinor 0 minus occupied spinor -1
        soc_homo = soc_E[2 * n_occ - 1]
        soc_lumo = soc_E[2 * n_occ]
        calculated_soc_gap = soc_lumo - soc_homo
        
        solver_soc = ExcitonSolver(
            C=C, eps=eps, occ=occ, overlap=S, atom_symbols=syms, atom_coords=np.array(coords_ang),
            atom_ao_ranges=atom_ao_ranges, homo_index=homo_index, n_occ=n_occ, n_virt=n_virt, 
            scissor_ev=scissor, kernel=args.kernel, alpha=args.alpha, include_exchange=args.exchange, 
            material=args.material, e_thresh=args.e_thresh, f_thresh=args.f_thresh, 
            mu_ia_x=mu_ia_x, mu_ia_y=mu_ia_y, mu_ia_z=mu_ia_z, eps_out=args.eps_out,
            soc_U=soc_U, soc_E=soc_E, device=compute_device
        )

        run_solver_and_analysis(solver_soc, np.array(coords_ang), syms, shells, mu_ia_x, mu_ia_y, mu_ia_z, 
                                dft_gap, scissor, confinement_energy, args, suffix="_soc", soc_gap=calculated_soc_gap, soc_U=soc_U, soc_E=soc_E)

    print("\nAll calculations finished successfully.")

if __name__ == "__main__":
    main()

