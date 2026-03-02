import argparse
import numpy as np
import sys
import time 
import gc
import platform

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
from miniBSE.exciton_analysis import ExcitonAnalyzer, plot_analysis_summary, plot_exciton_3d_plotly
from miniBSE.exciton_cube import generate_exciton_cubes
from miniBSE.hardness import compute_sos_screening
from miniBSE.integrals import compute_dipole_ao
from miniBSE.oscillator import compute_oscillator_strengths

def main():
    parser = argparse.ArgumentParser(
        description="miniBSE - Lightweight post-DFT exciton solver"
    )

    # Required inputs
    parser.add_argument("--mo_file", required=True)
    parser.add_argument("--xyz", required=True)
    parser.add_argument("--basis_txt", required=True)
    parser.add_argument("--basis_name", required=True)

    # Subspace & CI Truncation
    parser.add_argument("--n-occ", type=int, default=50)
    parser.add_argument("--n-virt", type=int, default=50)
    parser.add_argument("--e_thresh", type=float, default=None, 
                        help="Energy threshold (eV) for truncating the CI space")
    parser.add_argument("--f_thresh", type=float, default=1e-4, 
                        help="Zero-order oscillator strength threshold for truncating CI space (default: 1e-4)")

    # Physics
    parser.add_argument("--qp_gap", type=float, required=True, 
                        help="Quasi-particle gap in eV (replaces scissor)")
    parser.add_argument("--soc", type=float, default=0.0, 
                        help="Spin-orbit coupling correction in eV. Shifts excitation energies down.")
    parser.add_argument("--kernel", choices=["bse", "stda", "resta", "yukawa"], default="bse")
    parser.add_argument("--alpha", type=float, default=None, help="Manual screening factor. Overrides automatic screening computation.")
    parser.add_argument("--exchange", action="store_true", help="Activate STDA-like exchange term")
    parser.add_argument("--screening", choices=["auto", "geometric", "polariz", "hybrid", "dielectric_conf"],
                        default="auto",
                        help="Screening method for alpha (default: auto = polariz if active space large else geometric)")
 
    # Material / Auto-Alpha overrides
    parser.add_argument("--material", type=str, default=None, help="Material name for auto-alpha")
    parser.add_argument("--eps-bulk", type=float, default=None, help="Override bulk dielectric constant")
    parser.add_argument("--L-scale", type=float, default=None, help="Override screening length")
    parser.add_argument("--eps-out", type=float, default=2.0, help="Dielectric constant of the surrounding medium (ligands/solvent) for Resta kernel.")

    # Spectrum Broadening
    parser.add_argument("--broadening", choices=["gaussian", "lorentzian", "none"], default="gaussian")
    parser.add_argument("--sigma", type=float, default=0.1, help="Broadening width in eV (default: 0.1)")

    # Plotting
    parser.add_argument("--plot", action="store_true", help="Generate a PNG plot of the spectrum")
    parser.add_argument("--show", action="store_true", help="Pop up the plot window on screen")
    parser.add_argument("--cube", action="store_true", help="Generate volumetric .cube files for the brightest exciton")
    parser.add_argument("--cube-spacing", type=float, default=0.5, help="Grid spacing in Angstroms for cube files")
 
    # Solver
    parser.add_argument("--nroots", type=int, default=10)
    parser.add_argument("--full-diag", action="store_true", help="Force full diagonalization")
    parser.add_argument("--tol", type=float, default=1e-5, help="Davidson convergence tolerance")
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps", "numpy"], default="auto", 
                        help="Compute device for heavy tensor math")

    # CSV Output Controls
    parser.add_argument("--write-csv", action="store_true", help="Enable exporting results to a CSV file")
    parser.add_argument("--csv-roots", type=int, default=10, help="Number of roots to include in the CSV")
    parser.add_argument("--time", type=float, default=0.0, help="Current time in the MD trajectory (fs or ps)")
    parser.add_argument("--save-xia", action="store_true", help="Save BSE eigenvectors (X_ia) to a compressed .npz file for NAMD overlaps")

    args = parser.parse_args()

    print("\n===================================================")
    print(" miniBSE - Post-DFT Exciton Solver")
    print("===================================================")

    # ------------------------------------------------------------
    # 1. Structure + Basis
    # ------------------------------------------------------------
    print("\n--- [1] Reading structure and basis ---")
    syms, coords_ang = read_xyz(args.xyz)
    basis_dict = parse_basis(args.basis_txt, args.basis_name)
    shells = build_shell_dicts(syms, coords_ang, basis_dict)
    n_ao = count_ao_from_shells(shells)

    print(f"  Atoms: {len(syms)}")
    print(f"  Total AOs: {n_ao}")

    # ------------------------------------------------------------
    # 2. Overlap
    # ------------------------------------------------------------
    print("\n--- [2] Computing AO overlap ---")
    S = libint_cpp.overlap(shells, args.nthreads)
    if S.shape[0] != n_ao:
        print("ERROR: AO dimension mismatch")
        sys.exit(1)
    print(f"  Overlap shape: {S.shape}")

    # ------------------------------------------------------------
    # 3. Read MOs & Dynamic Active Space
    # ------------------------------------------------------------
    print("\n--- [3] Reading molecular orbitals ---")
    C, eps, occ = read_mos_auto(args.mo_file, n_ao, verbose=True)
    homo_index = np.where(occ > 0.0)[0].max()
    print(f"  HOMO automatically detected at index {homo_index}")
    eps = eps * HA_TO_EV
    
    dft_gap = eps[homo_index + 1] - eps[homo_index]
    scissor = args.qp_gap - dft_gap
    print(f"  DFT Gap: {dft_gap:.3f} eV")
    print(f"  Target QP Gap: {args.qp_gap:.3f} eV")
    print(f"  Calculated Scissor shift: {scissor:.3f} eV")

    # --- DYNAMIC ACTIVE SPACE OVERRIDE ---
    if args.e_thresh is not None:
        occ_energies = eps[:homo_index + 1]
        virt_energies = eps[homo_index + 1:]
        
        gaps = virt_energies.reshape(1, -1) - occ_energies.reshape(-1, 1)
        valid_pairs = gaps <= args.e_thresh
        
        if not np.any(valid_pairs):
            print(f"ERROR: Energy threshold {args.e_thresh} eV is too tight. No transitions found.")
            sys.exit(1)
            
        active_occ_bool = np.any(valid_pairs, axis=1)
        active_virt_bool = np.any(valid_pairs, axis=0)
        
        min_occ_idx = np.where(active_occ_bool)[0][0]
        max_virt_idx = np.where(active_virt_bool)[0][-1]
        
        n_occ = homo_index - min_occ_idx + 1
        n_virt = max_virt_idx + 1
        
        print(f"  [Auto-Space] Threshold {args.e_thresh} eV active.")
        print(f"  [Auto-Space] Dynamically allocated n_occ = {n_occ}, n_virt = {n_virt}")
    else:
        n_occ = args.n_occ
        n_virt = args.n_virt

    # ------------------------------------------------------------
    # 4. Compute Pre-Solver Dipoles & Auto-Alpha (SOS)
    # ------------------------------------------------------------
    print("\n--- [4] Computing Transition Dipoles & Screening ---")
    print(f"    Computing AO Dipole Integrals for {n_ao} AOs (Parallel)...")
    start_int = time.time()
    mu_ao_x, mu_ao_y, mu_ao_z = compute_dipole_ao(shells, nthreads=args.nthreads)
    print(f"    Integrals completed in {time.time() - start_int:.2f}s")

    # --- PyTorch Device Detection ---
    compute_device = "numpy"
    if args.device != "numpy":
        try:
            import torch
            if args.device == "auto":
                if torch.cuda.is_available(): compute_device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and platform.machine() == 'arm64': 
                    compute_device = "mps"
                else: compute_device = "cpu"
            else:
                compute_device = args.device
            print(f"  [System] Tensor backend: Torch {compute_device.upper()}")
            torch.set_num_threads(args.nthreads)
        except ImportError:
            print("  [System] PyTorch not found. Using NumPy.")

    print(f"    Transforming Dipoles to MO basis (Active Space: {n_occ}x{n_virt})...")
    t0 = time.time()
    C_occ = C[:, homo_index - n_occ + 1 : homo_index + 1].astype(np.float32)
    C_virt = C[:, homo_index + 1 : homo_index + 1 + n_virt].astype(np.float32)

    def transform_dipole_global(mu_ao):
        if compute_device in ["cuda", "mps"]:
            dev = torch.device(compute_device)
            C_o_t = torch.tensor(C_occ, device=dev)
            C_v_t = torch.tensor(C_virt, device=dev)
            mu_t = torch.tensor(mu_ao.astype(np.float32), device=dev)
            return (C_o_t.T @ (mu_t @ C_v_t)).cpu().numpy().astype(np.float64)
        else:
            return (C_occ.T @ (mu_ao.astype(np.float32) @ C_virt)).astype(np.float64)

    mu_ia_x = transform_dipole_global(mu_ao_x)
    mu_ia_y = transform_dipole_global(mu_ao_y)
    mu_ia_z = transform_dipole_global(mu_ao_z)
    print(f"    Transformation completed in {time.time()-t0:.2f}s")

    del mu_ao_x, mu_ao_y, mu_ao_z
    gc.collect()

    alpha, d_eff, eps_eff, alpha_iso, bulk_gap, eps_bulk = compute_sos_screening(
        eps, homo_index, n_occ, n_virt,
        mu_ia_x, mu_ia_y, mu_ia_z, np.array(coords_ang),
        atom_symbols=syms,
        material_name=args.material, eps_bulk=args.eps_bulk,
        screening_mode=args.screening,
        manual_alpha=args.alpha
    )

    confinement_energy = args.qp_gap - bulk_gap

    print(f"  [SOS-Alpha] Effective Diameter: {d_eff:.2f} Å")
    print(f"  [SOS-Alpha] ε_bulk = {eps_bulk:.2f} | ε_eff = {eps_eff:.3f} | α = {alpha:.4f} | Mode: {args.screening}")
    print(f"  [SOS-Alpha] Bulk Gap: {bulk_gap:.3f} eV")
    print(f"  [Physics]   Confinement Energy (dE): {confinement_energy:.3f} eV")

    atom_ao_ranges = build_atom_ao_ranges(shells)

    # ------------------------------------------------------------
    # 5. Build solver
    # ------------------------------------------------------------
    solver = ExcitonSolver(
        C=C, eps=eps, occ=occ, overlap=S, atom_symbols=syms, atom_coords=np.array(coords_ang),
        atom_ao_ranges=atom_ao_ranges, homo_index=homo_index,
        n_occ=n_occ, n_virt=n_virt, scissor_ev=scissor, kernel=args.kernel,
        alpha=alpha, include_exchange=args.exchange, 
        material=args.material,
        e_thresh=args.e_thresh, f_thresh=args.f_thresh, 
        mu_ia_x=mu_ia_x, mu_ia_y=mu_ia_y, mu_ia_z=mu_ia_z,
        eps_out=args.eps_out,
        device=compute_device
    )

    # ------------------------------------------------------------
    # 6. Solve
    # ------------------------------------------------------------
    print("\n--- [5] Solving excited states ---")
    start_solve = time.time()
    energies_ev, vectors = solver.solve(nroots=args.nroots, full_diag=args.full_diag, tol=args.tol)
    
    if args.soc != 0.0:
        energies_ev = energies_ev - args.soc
        print(f"  [SOC] Applied Spin-Orbit Coupling: shifted excitation energies by -{args.soc:.3f} eV")

    print(f"  Solver converged in {time.time() - start_solve:2.2f} s")

    # ------------------------------------------------------------
    # 7. Transition Dipoles
    # ------------------------------------------------------------
    print("\n--- [6] Computing oscillator strengths ---")
    mu_ia = solver.ham.get_transition_dipoles(mu_ia_x, mu_ia_y, mu_ia_z)
    f_strengths = compute_oscillator_strengths(energies_ev, vectors, mu_ia)

    # ------------------------------------------------------------
    # 8. Print Loop (Max 100 outputs)
    # ------------------------------------------------------------
    print("\n" + "="*125)
    print(f" SYSTEM PARAMETERS: QP Gap = {args.qp_gap:.3f} eV | SOC = {args.soc:.3f} eV | Screening Alpha = {alpha:.4f} | Confinement = {confinement_energy:.3f} eV")
    print("="*125)
    print(f"{'State':>5} {'Energy':>10} {'Main Trans':>12} {'Weight':>8} {'f':>10} | {'Gap_avg':>10} {'2*J_coll':>10} {'-K_coll':>10} | {'PR':>5}")
    print("-" * 125)

    n_transitions = len(solver.ham.valid_i)
    basis_gaps = []
    for idx in range(n_transitions):
        i_rel = solver.ham.valid_i[idx]
        a_rel = solver.ham.valid_a[idx]
        abs_h = (homo_index - n_occ + 1) + i_rel
        abs_e = (homo_index + 1) + a_rel
        basis_gaps.append((eps[abs_e] - eps[abs_h]) + scissor)
    basis_gaps = np.array(basis_gaps)

    n_print = min(100, len(energies_ev))
    for n in range(n_print):
        vec = vectors[:, n]
        gap_avg_coll = np.sum(vec**2 * basis_gaps)

        q_exc = vec @ solver.ham.q_flat
        e_exchange_coll = 2.0 * np.dot(q_exc, solver.ham.gamma @ q_exc)
        
        # Un-shift SOC before calculating Coulomb binding diagnostic so it doesn't skew -K_coll
        e_kernel_coll_neg = (energies_ev[n] + args.soc) - gap_avg_coll - e_exchange_coll
        
        max_idx = np.argmax(np.abs(vec))
        max_weight = vec[max_idx]**2
        abs_h_max = (homo_index - n_occ + 1) + solver.ham.valid_i[max_idx]
        abs_e_max = (homo_index + 1) + solver.ham.valid_a[max_idx]
        trans_str = f"{abs_h_max:3d}->{abs_e_max:3d}"
        pr = 1.0 / np.sum(vec**4)

        print(f"{n+1:5d} {energies_ev[n]:10.4f}  {trans_str:>12}  {max_weight:8.3f}  {f_strengths[n]:10.5f} | "
              f"{gap_avg_coll:10.3f} {e_exchange_coll:10.3f} {e_kernel_coll_neg:10.3f} | {pr:5.1f}")

    if len(energies_ev) > 100:
        print(f" ... {len(energies_ev) - 100} additional states computed (output truncated to 100) ...")
    print("="*125)

    # ------------------------------------------------------------
    # 9. Dreuw/Plasser Analysis
    # ------------------------------------------------------------
    print(f"\n--- [7] Performing Dreuw/Plasser Exciton Analysis (First {n_print} states) ---")
    analyzer = ExcitonAnalyzer(solver, np.array(coords_ang), syms)
    analysis_results = []

    print(f"{'State':>5} {'Energy':>8} {'f_osc':>8} | {'PR':>5} {'d_eh(A)':>7} {'d_CT(A)':>7} {'sig_h':>6} {'sig_e':>6} | {'Cov':>6} {'Corr':>6} | {'Type':>8}")
    print("-" * 110)

    # [FIXED]: Loop truncated strictly to n_print for spatial analysis
    for n in range(n_print):
        results = analyzer.analyze_state(vectors[:, n], energies_ev[n], f_strengths[n])
        analysis_results.append(results)
        
        ct_ratio = results['CT_Character']
        if ct_ratio > 0.6: ex_type = "CT"
        elif results['d_eh'] < 3.0 and ct_ratio < 0.2: ex_type = "Frenkel"
        else: ex_type = "Wannier"

        print(f"{n+1:5d} {results['energy']:8.3f} {results['f_osc']:8.4f} | "
              f"{results['PR']:5.1f} {results['d_eh']:7.2f} {results['d_CT']:7.2f} "
              f"{results['sigma_h']:6.1f} {results['sigma_e']:6.1f} | "
              f"{results['cov_eh']:6.1f} {results['corr_eh']:6.2f} | {ex_type:>8}")

    if len(energies_ev) > n_print:
        print(f" ... remaining {len(energies_ev) - n_print} states skipped for spatial analysis ...")

    if args.cube:
        brightest_idx = np.argmax(f_strengths)
        print(f"\n--- [8] Generating 3D Densities for State {brightest_idx + 1} ---")
        generate_exciton_cubes(
            solver=solver, bse_vec=vectors[:, brightest_idx], shells=shells, symbols=syms, coords=np.array(coords_ang),
            prefix=f"state_{brightest_idx + 1}", spacing_ang=args.cube_spacing, margin_ang=3.5, nthreads=args.nthreads
        )

    if args.plot or args.show:
        print("\nGenerate interactive analysis plots...")
        
        # Package the physics metrics for the HTML dashboard
        physics_metrics = {
            "dft_gap": dft_gap,
            "qp_correction": scissor,
            "confinement_energy": confinement_energy,
            # Binding energy is negative (stabilization from Coulomb attraction)
            "binding_energy": energies_ev[0] - args.qp_gap if len(energies_ev) > 0 else 0.0,
            "first_exc_energy": energies_ev[0] if len(energies_ev) > 0 else 0.0
        }
        
        analysis_plot_file = "exciton_analysis.html" if args.plot else None
        plot_analysis_summary(
            analysis_results, 
            physics_metrics=physics_metrics, 
            filename=analysis_plot_file, 
            show=args.show
        )

    # ------------------------------------------------------------
    # [NEW] Export Comprehensive Results to CSV
    # ------------------------------------------------------------

    if args.write_csv:
        import csv
        csv_file = "exciton_results.csv"
        n_to_write = min(args.csv_roots, len(energies_ev))
        
        # We need the active space transition dipoles to get the state vector
        mu_ia_active = solver.ham.get_transition_dipoles(mu_ia_x, mu_ia_y, mu_ia_z)

        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            # ADDED mu_x, mu_y, mu_z to the header
            header = [
                "Time", "State", "Energy_eV", "f_osc", "mu_x", "mu_y", "mu_z", "PR", 
                "Main_Transition", "d_eh_A", "d_CT_A", "sigma_h_A", "sigma_e_A", "Type"
            ]
            writer.writerow(header)

            for n in range(n_to_write):
                res = analyzer.analyze_state(vectors[:, n], energies_ev[n], f_strengths[n])
                
                # Calculate the 3D transition dipole vector for this specific state
                mu_state = np.sum(vectors[:, n][:, None] * mu_ia_active, axis=0)
                
                ct_ratio = res['CT_Character']
                if ct_ratio > 0.6: ex_type = "CT"
                elif res['d_eh'] < 3.0 and ct_ratio < 0.2: ex_type = "Frenkel"
                else: ex_type = "Wannier"

                writer.writerow([
                    args.time, n + 1, f"{energies_ev[n]:.6f}", f"{f_strengths[n]:.6e}", 
                    f"{mu_state[0]:.6f}", f"{mu_state[1]:.6f}", f"{mu_state[2]:.6f}", # Added vectors
                    f"{res['PR']:.3f}", "...", f"{res['d_eh']:.4f}", f"{res['d_CT']:.4f}", 
                    f"{res['sigma_h']:.4f}", f"{res['sigma_e']:.4f}", ex_type
                ])

    # ------------------------------------------------------------
    # [NEW] Export BSE Eigenvectors (X_ia) for NAMD Diabatization
    # ------------------------------------------------------------
    if args.save_xia:
        # Create a unique filename based on the trajectory time
        # E.g., "xia_0008.00fs.npz"
        npz_filename = f"xia_{args.time:08.2f}fs.npz"
        
        # Determine how many roots to save (usually match the CSV, or all computed roots)
        n_to_write = min(args.csv_roots, len(energies_ev))
        
        print(f"\n--- [9] Exporting BSE Eigenvectors to {npz_filename} ---")
        
        # Save to a compressed binary format to save disk space and I/O time
        np.savez_compressed(
            npz_filename,
            time=args.time,
            energies=energies_ev[:n_to_write],
            # vectors shape is (n_transitions, n_roots). We slice it to save only the roots we care about.
            X_ia=vectors[:, :n_to_write], 
            # Save the active space transition indices so we can align them between steps if the active space changes!
            valid_i=solver.ham.valid_i,
            valid_a=solver.ham.valid_a,
            homo_index=homo_index,
            n_occ=n_occ
        )
        print(f"  Successfully saved X_ia coefficients for {n_to_write} states.")

    # ------------------------------------------------------------
    # 10. Convolute Spectrum
    # ------------------------------------------------------------
    if args.broadening != "none":
        print("\n--- [8] Generating UV-Vis Spectrum ---")
        from miniBSE.spectrum import generate_spectrum, plot_spectrum
        
        # WIDER PADDING: +/- 2.5 eV ensures tails flatten fully
        e_min = max(0.0, np.min(energies_ev) - 2.5)
        e_max = np.max(energies_ev) + 2.5
        
        # [NOTE]: All 5000+ energies and f_strengths are passed here naturally
        x_grid, y_grid = generate_spectrum(energies_ev, f_strengths, e_min=e_min, e_max=e_max, sigma=args.sigma, profile=args.broadening)
        spec_file = "spectrum.dat"
        np.savetxt(spec_file, np.column_stack((x_grid, y_grid)), header=f"Energy(eV) Intensity(arb.u.) | {args.broadening}, sigma={args.sigma}", fmt="%8.4f %12.6e")
        print(f"  Data saved to '{spec_file}'")
        
        if args.plot or args.show:
            plot_file = "spectrum.png" if args.plot else None
            plot_spectrum(x_grid, y_grid, energies_ev, f_strengths, filename=plot_file, show=args.show)

    print("\nDone.")

if __name__ == "__main__":
    main()

