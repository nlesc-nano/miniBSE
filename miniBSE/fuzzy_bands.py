import numpy as np
import time
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from scipy.spatial.distance import pdist

def generate_automated_kpath(cif_path, coords_ang, line_density=50, return_reciprocal=False):
    print(f"  [Fuzzy] Loading CIF: {cif_path}")
    struct = Structure.from_file(cif_path)
    sga = SpacegroupAnalyzer(struct)
    prim_struct = sga.get_primitive_standard_structure()
    kpath = HighSymmKpath(prim_struct)
    kpts_frac, labels = kpath.get_kpoints(line_density=line_density, coords_are_cartesian=False)
    
    # Standard reciprocal mapping
    reciprocal_matrix = prim_struct.lattice.reciprocal_lattice.matrix
    kpts_cart = np.dot(kpts_frac, reciprocal_matrix)
    
    # ====================================================================
    # SCIENTIFIC FIX: LATTICE SCALING & PCA ROTATIONAL ALIGNMENT
    # ====================================================================
    center_of_mass = np.mean(coords_ang, axis=0)
    xyz_centered = coords_ang - center_of_mass
    
    # 1. SCALING: Compare CIF nearest-neighbor to XYZ core nearest-neighbor
    cif_dists = np.unique(np.round(struct.distance_matrix.flatten(), 3))
    cif_bond = cif_dists[cif_dists > 0.5][0] if len(cif_dists[cif_dists > 0.5]) > 0 else 1.0
    
    dists_to_center = np.linalg.norm(xyz_centered, axis=1)
    core_indices = np.argsort(dists_to_center)[:min(40, len(coords_ang))]
    xyz_dists = pdist(xyz_centered[core_indices])
    xyz_dists = xyz_dists[(xyz_dists > 0.5) & (xyz_dists < cif_bond * 1.5)]
    
    if len(xyz_dists) > 0:
        xyz_bond = np.percentile(xyz_dists, 5) 
        scale_factor = xyz_bond / cif_bond
        print(f"  [Fuzzy] Phase correction: Scaling k-points by 1/({scale_factor:.4f}) to match XYZ bonds.")
        kpts_cart = kpts_cart / scale_factor
        reciprocal_matrix = reciprocal_matrix / scale_factor

    # 2. ROTATION: Align Principal Axes (Inertia Tensors)
    try:
        # Build a sphere from the CIF to match the XYZ cluster size
        max_radius = np.max(dists_to_center)
        sphere_sites = struct.get_sites_in_sphere(prim_struct[0].coords, max_radius)
        cif_coords = np.array([site.coords for site in sphere_sites])
        cif_centered = cif_coords - np.mean(cif_coords, axis=0)
        
        # Compute Covariance (Inertia) Matrices
        cov_xyz = np.cov(xyz_centered.T)
        cov_cif = np.cov(cif_centered.T)
        
        # Get Principal Axes
        _, vecs_xyz = np.linalg.eigh(cov_xyz)
        _, vecs_cif = np.linalg.eigh(cov_cif)
        
        # Ensure right-handed coordinate systems
        if np.linalg.det(vecs_xyz) < 0: vecs_xyz[:, 2] *= -1
        if np.linalg.det(vecs_cif) < 0: vecs_cif[:, 2] *= -1
        
        # Compute Rotation Matrix connecting CIF orientation to XYZ orientation
        R = vecs_xyz @ vecs_cif.T
        
        print(f"  [Fuzzy] Applying PCA rotation to k-path to correct optimizer drift.")
        kpts_cart = (R @ kpts_cart.T).T
        reciprocal_matrix = (R @ reciprocal_matrix.T).T
    except Exception as e:
        print(f"  [Fuzzy] Warning: PCA Rotational alignment failed: {e}")

    print(f"  [Fuzzy] Generated {len(kpts_cart)} k-points for spacegroup {sga.get_space_group_symbol()}.")
    if return_reciprocal:
        return kpts_cart, labels, reciprocal_matrix
    return kpts_cart, labels


def make_reciprocal_replicas(reciprocal_matrix, g_shell):
    coeffs = np.arange(-g_shell, g_shell + 1, dtype=int)
    hkl = np.array(np.meshgrid(coeffs, coeffs, coeffs, indexing="ij")).reshape(3, -1).T
    return hkl @ reciprocal_matrix


def fuzzy_energy_mask(eps_dft, ewin, sigma_ev, qp_energies=None):
    margin = 4.0 * sigma_ev
    mask = (eps_dft >= ewin[0] - margin) & (eps_dft <= ewin[1] + margin)
    if qp_energies is not None:
        mask |= (qp_energies >= ewin[0] - margin) & (qp_energies <= ewin[1] + margin)
    return mask


def fuzzy_energy_indices(eps_dft, ewin, sigma_ev, homo_index=None, qp_energies=None):
    mask = fuzzy_energy_mask(eps_dft, ewin, sigma_ev, qp_energies=qp_energies)
    if homo_index is not None:
        if 0 <= homo_index < len(mask):
            mask[homo_index] = True
        if 0 <= homo_index + 1 < len(mask):
            mask[homo_index + 1] = True
    return np.where(mask)[0].astype(int)


def compute_fuzzy_intensity(C_dense, shells, kpts_cart, nthreads, fold_to_bz=False, g_shell=0, reciprocal_matrix=None, mo_indices=None):
    import libint_cpp

    if g_shell < 0:
        raise ValueError("g_shell must be >= 0")

    C_project = C_dense if mo_indices is None else C_dense[:, mo_indices]

    if not fold_to_bz:
        qpts_cart = kpts_cart
        qpts_bohr = qpts_cart / 1.8897259886
        F_ao = libint_cpp.ao_ft_complex(shells, qpts_bohr, nthreads)
        F_mo = C_project.T.conj() @ F_ao
        return np.abs(F_mo)**2

    if reciprocal_matrix is None:
        raise ValueError("reciprocal_matrix is required when fold_to_bz=True")

    # BZ-folded spectral projection of finite QD molecular orbitals.
    # This sums reciprocal replicas of the same finite-MO Fourier amplitude;
    # it is not a true Bloch band structure.
    G_vecs = make_reciprocal_replicas(reciprocal_matrix, g_shell)
    print(f"  [Fuzzy] BZ folding enabled: g_shell={g_shell} ({len(G_vecs)} reciprocal replicas).")
    intensity = np.zeros((C_project.shape[1], len(kpts_cart)), dtype=float)
    for G in G_vecs:
        qpts_bohr = (kpts_cart + G) / 1.8897259886
        F_ao = libint_cpp.ao_ft_complex(shells, qpts_bohr, nthreads)
        F_mo = C_project.T.conj() @ F_ao
        intensity += np.abs(F_mo)**2
    return intensity


def build_qp_energies(eps_dft, homo_index, scissor_ev=None, sigma_occ=None, sigma_virt=None):
    if sigma_occ is None and sigma_virt is None and scissor_ev is None:
        return None

    eps_qp = np.array(eps_dft, dtype=float, copy=True)
    occ_slice = slice(0, homo_index + 1)
    virt_slice = slice(homo_index + 1, len(eps_qp))

    if sigma_occ is not None:
        eps_qp[occ_slice] += np.asarray(sigma_occ, dtype=float)
    if sigma_virt is not None:
        eps_qp[virt_slice] += np.asarray(sigma_virt, dtype=float)
    elif scissor_ev is not None:
        eps_qp[virt_slice] += float(scissor_ev)

    return eps_qp


def build_qp_energies_vacuum(eps_abs, homo_index, qp_homo=None, qp_lumo=None, occ_shift=None, virt_shift=None):
    eps_qp = np.array(eps_abs, dtype=float, copy=True)
    if occ_shift is None:
        occ_shift = float(qp_homo) - float(eps_abs[homo_index])
    if virt_shift is None:
        virt_shift = float(qp_lumo) - float(eps_abs[homo_index + 1])
    eps_qp[:homo_index + 1] += occ_shift
    eps_qp[homo_index + 1:] += virt_shift
    return eps_qp


def build_soc_qp_energies_vacuum(soc_E_abs, soc_U, eps_abs_spin, eps_qp_abs_spin):
    delta_qp = np.asarray(eps_qp_abs_spin, dtype=float) - np.asarray(eps_abs_spin, dtype=float)
    return np.asarray(soc_E_abs, dtype=float) + (np.abs(soc_U) ** 2).T @ delta_qp


def auto_energy_window(*energy_arrays, sigma_ev=0.03):
    vals = []
    for energies in energy_arrays:
        if energies is None:
            continue
        arr = np.asarray(energies, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(arr)
    if not vals:
        return np.array([-5.0, 5.0], dtype=float)

    all_e = np.concatenate(vals)
    e_min = float(np.min(all_e))
    e_max = float(np.max(all_e))
    if np.isclose(e_min, e_max):
        pad = max(10.0 * float(sigma_ev), 0.1)
    else:
        pad = max(4.0 * float(sigma_ev), 0.02 * (e_max - e_min), 0.05)
    return np.array([e_min - pad, e_max + pad], dtype=float)


def build_smeared_fuzzy(intensity, eps_plot, ewin, sigma_ev):
    window_mask = (eps_plot >= ewin[0] - 4 * sigma_ev) & (eps_plot <= ewin[1] + 4 * sigma_ev)
    E_w = eps_plot[window_mask]
    I_w = intensity[window_mask, :]

    dE = max(0.5 * sigma_ev, 0.01)
    edges = np.arange(ewin[0], ewin[1] + dE, dE)
    centres = 0.5 * (edges[:-1] + edges[1:])

    Z = np.zeros((centres.size, I_w.shape[1]), dtype=float)
    for En, Ik in zip(E_w, I_w):
        w = np.exp(-0.5 * ((centres - En) / sigma_ev) ** 2)
        Z += np.outer(w, Ik)
    return centres, Z


def smear_and_export_fuzzy(intensity, eps_plot, labels, ewin, sigma_ev, prefix="sf"):
    t0 = time.time()
    centres, Z = build_smeared_fuzzy(intensity, eps_plot, ewin, sigma_ev)

    # ====================================================================
    # SCIENTIFIC FIX: CLEAN K-PATH LABELS & MERGE PATH BREAKS
    # ====================================================================
    valid_idx = []
    valid_labels = []
    for i, lbl in enumerate(labels):
        if lbl:
            # 1. Clean up LaTeX and Pymatgen formatting (convert to Unicode)
            clean_lbl = lbl.replace("\\Gamma", "Γ").replace("GAMMA", "Γ").replace("$", "")
            
            # 2. Handle Path Breaks (Adjacent duplicate or different labels)
            if valid_idx and (i - valid_idx[-1] <= 1):
                # Combine labels with a vertical bar if they are different
                if clean_lbl not in valid_labels[-1].split(" | "):
                    valid_labels[-1] = f"{valid_labels[-1]} | {clean_lbl}"
                valid_idx[-1] = i  # Snap to the exact current index
            else:
                valid_idx.append(i)
                valid_labels.append(clean_lbl)

    out_name = f"fuzzy_data_{prefix}.npz"
    np.savez_compressed(
        out_name,
        centres=centres.astype(np.float32),
        intensity=Z.astype(np.float32),
        tick_positions=np.array(valid_idx, dtype=np.float32),
        tick_labels=np.array(valid_labels, dtype=object),
        ewin=np.array(ewin, dtype=np.float32),
        extent=np.array([0.0, float(Z.shape[1] - 1), float(ewin[0]), float(ewin[1])])
    )
    print(f"  [Fuzzy] Exported {out_name} in {time.time()-t0:.2f} s")


def smear_and_export_spin_fuzzy(intensity_alpha, eps_alpha, intensity_beta, eps_beta, labels, ewin, sigma_ev, prefix="uks"):
    t0 = time.time()
    centres, Z_a = build_smeared_fuzzy(intensity_alpha, eps_alpha, ewin, sigma_ev)
    _, Z_b = build_smeared_fuzzy(intensity_beta, eps_beta, ewin, sigma_ev)
    Z_total = Z_a + Z_b
    spinpol = np.divide(Z_a - Z_b, Z_total, out=np.zeros_like(Z_total), where=Z_total > 1e-14)

    valid_idx = []
    valid_labels = []
    for i, lbl in enumerate(labels):
        if lbl:
            clean_lbl = lbl.replace("\\Gamma", "Γ").replace("GAMMA", "Γ").replace("$", "")
            if valid_idx and (i - valid_idx[-1] <= 1):
                if clean_lbl not in valid_labels[-1].split(" | "):
                    valid_labels[-1] = f"{valid_labels[-1]} | {clean_lbl}"
                valid_idx[-1] = i
            else:
                valid_idx.append(i)
                valid_labels.append(clean_lbl)

    common = dict(
        centres=centres.astype(np.float32),
        tick_positions=np.array(valid_idx, dtype=np.float32),
        tick_labels=np.array(valid_labels, dtype=object),
        ewin=np.array(ewin, dtype=np.float32),
        extent=np.array([0.0, float(Z_total.shape[1] - 1), float(ewin[0]), float(ewin[1])])
    )
    np.savez_compressed(f"fuzzy_data_{prefix}.npz", intensity=Z_total.astype(np.float32), spinpol=spinpol.astype(np.float32), **common)
    np.savez_compressed(f"fuzzy_data_{prefix}_alpha.npz", intensity=Z_a.astype(np.float32), **common)
    np.savez_compressed(f"fuzzy_data_{prefix}_beta.npz", intensity=Z_b.astype(np.float32), **common)
    print(f"  [Fuzzy-UKS] Exported fuzzy_data_{prefix}.npz with spin polarization overlay in {time.time()-t0:.2f} s")

def run_fuzzy_bands_and_pdos(args, C_dense, S_dense, eps_shifted, occ, homo_index, e_homo, e_lumo, e_fermi_raw, syms, coords_ang, shells, pops_sf, soc_active_indices=None, soc_E_act=None, soc_U_act=None, spinor_homo_idx=None, qp_energies=None, eps_abs=None, qp_energies_abs=None, soc_E_abs_act=None, C_beta_dense=None, eps_beta_shifted=None, eps_beta_abs=None, homo_index_beta=None, qp_energies_beta=None, qp_energies_beta_abs=None, soc_active_indices_beta=None):
    import time
    import numpy as np
    from miniBSE.pdos_coop import compute_pdos_and_coop, export_pdos_coop_data
    
    print("\n===================================================")
    print(" [ FUZZY BANDS & PDOS ]")
    print("===================================================")
    
    fold_to_bz = bool(getattr(args, 'fold_to_bz', False))
    g_shell = int(getattr(args, 'g_shell', 0))
    kpath_result = generate_automated_kpath(args.cif, np.array(coords_ang), line_density=50, return_reciprocal=fold_to_bz)
    if fold_to_bz:
        kpts_cart, labels, reciprocal_matrix = kpath_result
    else:
        kpts_cart, labels = kpath_result
        reciprocal_matrix = None
    
    # --- 1. SPIN-FREE CALCULATION ---
    n_occ = homo_index + 1
    n_virt = len(eps_shifted) - n_occ
    print(f"\n  [Fuzzy] --- Spin-Free MO Statistics ---")
    print(f"  [Fuzzy] Total MOs: {len(eps_shifted)} ({n_occ} Occupied, {n_virt} Virtual)")
    print(f"  [Fuzzy] MO HOMO (Idx {homo_index}): {e_homo:8.4f} eV")
    print(f"  [Fuzzy] MO LUMO (Idx {homo_index + 1}): {e_lumo:8.4f} eV")
    print(f"  [Fuzzy] Fermi Level (raw shifted to 0.0): {e_fermi_raw:8.4f} eV")
    print(f"  [Fuzzy] -------------------------------")

    sigma_use = getattr(args, 'fuzzy_sigma', 0.03)
    pdos_sigma_use = getattr(args, 'pdos_sigma', 0.10)
    dashboard_energy_mode = str(getattr(args, 'dashboard_energy_mode', 'dft')).lower()
    if dashboard_energy_mode not in ("dft", "qp", "both"):
        raise ValueError("dashboard_energy_mode must be one of: dft, qp, both")
    qp_energy_reference = str(getattr(args, "qp_energy_reference", "vacuum")).lower()
    if qp_energy_reference not in ("vacuum", "fermi"):
        raise ValueError("qp_energy_reference must be one of: vacuum, fermi")
    qp_plot_energies = qp_energies_abs if qp_energy_reference == "vacuum" else qp_energies
    qp_plot_energies_beta = qp_energies_beta_abs if qp_energy_reference == "vacuum" else qp_energies_beta
    dft_ewin = list(getattr(args, "ewin", [-5.0, 5.0]))
    qp_ewin = dft_ewin if qp_energy_reference == "fermi" else (auto_energy_window(qp_plot_energies, sigma_ev=sigma_use) if qp_plot_energies is not None else None)
    qp_mask_energies = qp_plot_energies if qp_energy_reference == "fermi" else None
    fuzzy_indices = fuzzy_energy_indices(
        eps_shifted, dft_ewin, sigma_use, homo_index=homo_index, qp_energies=qp_mask_energies
    )
    eps_fuzzy = eps_shifted[fuzzy_indices]
    qp_fuzzy = qp_plot_energies[fuzzy_indices] if qp_plot_energies is not None else None

    print(f"  [Fuzzy] Projecting {len(fuzzy_indices)} / {len(eps_shifted)} MOs in ewin [{dft_ewin[0]:.3f}, {dft_ewin[1]:.3f}] eV relative to mid-gap.")
    if qp_ewin is not None:
        print(f"  [Fuzzy] QP plot window: [{qp_ewin[0]:.3f}, {qp_ewin[1]:.3f}] eV")

    print("  [Fuzzy] Computing Analytic AO-FT via C++ ...")
    intensity_sf = compute_fuzzy_intensity(
        C_dense, shells, kpts_cart, args.nthreads,
        fold_to_bz=fold_to_bz, g_shell=g_shell, reciprocal_matrix=reciprocal_matrix,
        mo_indices=fuzzy_indices
    )

    if fold_to_bz and g_shell == 0:
        intensity_ref = compute_fuzzy_intensity(C_dense, shells, kpts_cart, args.nthreads, fold_to_bz=False, mo_indices=fuzzy_indices)
        _, Z_ref = build_smeared_fuzzy(intensity_ref, eps_fuzzy, dft_ewin, sigma_use)
        _, Z_fold = build_smeared_fuzzy(intensity_sf, eps_fuzzy, dft_ewin, sigma_use)
        print(
            "  [Fuzzy] g_shell=0 folding diagnostic: "
            f"max |dW|={np.max(np.abs(intensity_sf - intensity_ref)):.3e}, "
            f"max |dA|={np.max(np.abs(Z_fold - Z_ref)):.3e}"
        )
    
    smear_and_export_fuzzy(intensity_sf, eps_fuzzy, labels, dft_ewin, sigma_use, prefix="sf")
    
    pdos_analysis_sf = None
    if getattr(args, 'pdos_atoms', None) and getattr(args, 'coop_pairs', None):
        print("  [PDOS/COOP] Computing Spin-Free population analysis...")
        pdos_analysis_sf = compute_pdos_and_coop(C_dense, S_dense, eps_shifted, shells, args.pdos_atoms, args.coop_pairs, dft_ewin, sigma=pdos_sigma_use, is_soc=False, prefix="sf", pops=pops_sf)

    if dashboard_energy_mode in ("qp", "both"):
        if qp_plot_energies is None:
            msg = "QP dashboard requested, but QP-corrected orbital energies were not found."
            if dashboard_energy_mode == "qp":
                raise ValueError(msg)
            print(f"  [Warning] {msg} Skipping QP dashboard.")
        else:
            smear_and_export_fuzzy(intensity_sf, qp_fuzzy, labels, qp_ewin, sigma_use, prefix="sf_qp")
            if pdos_analysis_sf is not None:
                export_pdos_coop_data(pdos_analysis_sf, qp_plot_energies, args.pdos_atoms, args.coop_pairs, qp_ewin, sigma=pdos_sigma_use, is_soc=False, prefix="sf_qp")

    is_uks = C_beta_dense is not None and eps_beta_shifted is not None and homo_index_beta is not None
    if is_uks:
        print(f"\n  [Fuzzy-UKS] Computing alpha/beta fuzzy channels with total intensity and spin polarization...")
        dft_uks_ewin = dft_ewin
        qp_uks_ewin = auto_energy_window(qp_plot_energies, qp_plot_energies_beta, sigma_ev=sigma_use) if qp_plot_energies is not None and qp_plot_energies_beta is not None else None
        qp_mask_energies_beta = qp_plot_energies_beta if qp_energy_reference == "fermi" else None
        fuzzy_indices_b = fuzzy_energy_indices(
            eps_beta_shifted, dft_uks_ewin, sigma_use, homo_index=homo_index_beta, qp_energies=qp_mask_energies_beta
        )
        eps_fuzzy_b = eps_beta_shifted[fuzzy_indices_b]
        intensity_a = intensity_sf
        intensity_b = compute_fuzzy_intensity(
            C_beta_dense, shells, kpts_cart, args.nthreads,
            fold_to_bz=fold_to_bz, g_shell=g_shell, reciprocal_matrix=reciprocal_matrix,
            mo_indices=fuzzy_indices_b
        )
        smear_and_export_spin_fuzzy(intensity_a, eps_fuzzy, intensity_b, eps_fuzzy_b, labels, dft_uks_ewin, sigma_use, prefix="uks")

        if dashboard_energy_mode in ("qp", "both") and qp_plot_energies is not None and qp_plot_energies_beta is not None:
            smear_and_export_spin_fuzzy(
                intensity_a, qp_plot_energies[fuzzy_indices],
                intensity_b, qp_plot_energies_beta[fuzzy_indices_b],
                labels, qp_uks_ewin, sigma_use, prefix="uks_qp"
            )

    # --- 2. SOC CALCULATION ---
    if args.soc_flag and soc_active_indices is not None:
        n_act_mo = len(soc_active_indices)
        n_act_occ = np.sum(soc_active_indices <= homo_index)
        n_act_virt = n_act_mo - n_act_occ

        print(f"\n  [Fuzzy-SOC] Applying Precomputed Unified SOC Projection...")
        print(f"  [Fuzzy-SOC] --- Dual-Window SOC Statistics ---")
        print(f"  [Fuzzy-SOC] Active Space Spatial MOs: {n_act_mo} ({n_act_occ} Occ, {n_act_virt} Virt)")
        print(f"  [Fuzzy-SOC] Full spinor basis available: {len(eps_shifted) * 2} states")

        alpha_plot_indices = fuzzy_energy_indices(eps_shifted, dft_ewin, sigma_use, homo_index=homo_index)
        core_idx = alpha_plot_indices[alpha_plot_indices < soc_active_indices[0]]
        virt_idx = alpha_plot_indices[alpha_plot_indices > soc_active_indices[-1]]
        
        if fold_to_bz:
            G_vecs = make_reciprocal_replicas(reciprocal_matrix, g_shell)
            qpts_cart = (kpts_cart[:, None, :] + G_vecs[None, :, :]).reshape(-1, 3)
        else:
            G_vecs = None
            qpts_cart = kpts_cart

        qpts_bohr = qpts_cart / 1.8897259886
        import libint_cpp
        F_ao = libint_cpp.ao_ft_complex(shells, qpts_bohr, args.nthreads)

        alpha_needed = np.unique(np.concatenate([core_idx, soc_active_indices, virt_idx])).astype(int)
        F_mo_sf_needed = C_dense[:, alpha_needed].T.conj() @ F_ao
        alpha_pos = {int(idx): pos for pos, idx in enumerate(alpha_needed)}

        def take_alpha(indices):
            if len(indices) == 0:
                return np.zeros((0, F_ao.shape[1]), dtype=complex)
            return F_mo_sf_needed[[alpha_pos[int(idx)] for idx in indices], :]

        if is_uks and soc_active_indices_beta is not None:
            beta_plot_indices = fuzzy_energy_indices(eps_beta_shifted, dft_ewin, sigma_use, homo_index=homo_index_beta)
            core_idx_b = beta_plot_indices[beta_plot_indices < soc_active_indices_beta[0]]
            virt_idx_b = beta_plot_indices[beta_plot_indices > soc_active_indices_beta[-1]]
            beta_needed = np.unique(np.concatenate([core_idx_b, soc_active_indices_beta, virt_idx_b])).astype(int)
            F_mo_beta_needed = C_beta_dense[:, beta_needed].T.conj() @ F_ao
            beta_pos = {int(idx): pos for pos, idx in enumerate(beta_needed)}

            def take_beta(indices):
                if len(indices) == 0:
                    return np.zeros((0, F_ao.shape[1]), dtype=complex)
                return F_mo_beta_needed[[beta_pos[int(idx)] for idx in indices], :]

            F_mo_act = take_alpha(soc_active_indices)
            F_mo_act_b = take_beta(soc_active_indices_beta)
            F_spinor_act = soc_U_act.T.conj() @ np.vstack([F_mo_act, F_mo_act_b])
            F_spinor_core = np.vstack([take_alpha(core_idx), take_beta(core_idx_b)])
            F_spinor_virt = np.vstack([take_alpha(virt_idx), take_beta(virt_idx_b)])
            E_core = np.concatenate([eps_shifted[core_idx], eps_beta_shifted[core_idx_b]])
            E_virt = np.concatenate([eps_shifted[virt_idx], eps_beta_shifted[virt_idx_b]])
            if eps_abs is not None and eps_beta_abs is not None and qp_energies_abs is not None and qp_energies_beta_abs is not None and soc_E_abs_act is not None:
                eps_abs_spin_act = np.concatenate([eps_abs[soc_active_indices], eps_beta_abs[soc_active_indices_beta]])
                eps_qp_abs_spin_act = np.concatenate([qp_energies_abs[soc_active_indices], qp_energies_beta_abs[soc_active_indices_beta]])
                soc_E_qp_act = build_soc_qp_energies_vacuum(soc_E_abs_act, soc_U_act, eps_abs_spin_act, eps_qp_abs_spin_act)
                E_core_qp = np.concatenate([qp_energies_abs[core_idx], qp_energies_beta_abs[core_idx_b]])
                E_virt_qp = np.concatenate([qp_energies_abs[virt_idx], qp_energies_beta_abs[virt_idx_b]])
            else:
                soc_E_qp_act = E_core_qp = E_virt_qp = None
        else:
            F_mo_act = take_alpha(soc_active_indices)
            F_spinor_act = soc_U_act.T.conj() @ np.vstack([F_mo_act, F_mo_act])
            F_spinor_core = np.vstack([take_alpha(core_idx), take_alpha(core_idx)])
            F_spinor_virt = np.vstack([take_alpha(virt_idx), take_alpha(virt_idx)])
            E_core = np.concatenate([eps_shifted[core_idx], eps_shifted[core_idx]])
            E_virt = np.concatenate([eps_shifted[virt_idx], eps_shifted[virt_idx]])
            if eps_abs is not None and qp_energies_abs is not None and soc_E_abs_act is not None:
                eps_abs_spin_act = np.concatenate([eps_abs[soc_active_indices], eps_abs[soc_active_indices]])
                eps_qp_abs_spin_act = np.concatenate([qp_energies_abs[soc_active_indices], qp_energies_abs[soc_active_indices]])
                soc_E_qp_act = build_soc_qp_energies_vacuum(soc_E_abs_act, soc_U_act, eps_abs_spin_act, eps_qp_abs_spin_act)
                E_core_qp = np.concatenate([qp_energies_abs[core_idx], qp_energies_abs[core_idx]])
                E_virt_qp = np.concatenate([qp_energies_abs[virt_idx], qp_energies_abs[virt_idx]])
            else:
                soc_E_qp_act = E_core_qp = E_virt_qp = None

        F_spinor_unsorted = np.vstack([F_spinor_core, F_spinor_act, F_spinor_virt])
        eps_soc_unsorted = np.concatenate([E_core, soc_E_act, E_virt])

        plot_keep = fuzzy_energy_mask(eps_soc_unsorted, dft_ewin, sigma_use)
        below_zero = np.where(eps_soc_unsorted <= 0.0)[0]
        above_zero = np.where(eps_soc_unsorted > 0.0)[0]
        if below_zero.size:
            plot_keep[below_zero[np.argmax(eps_soc_unsorted[below_zero])]] = True
        if above_zero.size:
            plot_keep[above_zero[np.argmin(eps_soc_unsorted[above_zero])]] = True

        F_spinor_plot_unsorted = F_spinor_unsorted[plot_keep, :]
        eps_soc_plot_unsorted = eps_soc_unsorted[plot_keep]
        sort_idx = np.argsort(eps_soc_plot_unsorted)
        eps_soc = eps_soc_plot_unsorted[sort_idx]
        F_spinor = F_spinor_plot_unsorted[sort_idx, :]
        if fold_to_bz:
            intensity_soc = np.sum(np.abs(F_spinor.reshape(F_spinor.shape[0], len(kpts_cart), len(G_vecs)))**2, axis=2)
        else:
            intensity_soc = np.abs(F_spinor)**2
        soc_ewin = dft_ewin
        occupied_plot = np.where(eps_soc <= 0.0)[0]
        global_spinor_homo_idx = int(occupied_plot[-1]) if occupied_plot.size else 0
 
        print(f"  [Fuzzy-SOC] Plotting {len(eps_soc)} spinors in ewin [{soc_ewin[0]:.3f}, {soc_ewin[1]:.3f}] eV relative to mid-gap.")
        print(f"  [Fuzzy-SOC] Spinor HOMO (Idx {global_spinor_homo_idx}): {eps_soc[global_spinor_homo_idx]:8.4f} eV")
        print(f"  [Fuzzy-SOC] Spinor LUMO (Idx {global_spinor_homo_idx + 1}): {eps_soc[global_spinor_homo_idx + 1]:8.4f} eV")
        print(f"  [Fuzzy-SOC] ----------------------------------") 
        
        smear_and_export_fuzzy(intensity_soc, eps_soc, labels, soc_ewin, sigma_use, prefix="soc")

        eps_soc_qp = None
        if dashboard_energy_mode in ("qp", "both") and soc_E_qp_act is not None:
            eps_soc_qp_unsorted = np.concatenate([E_core_qp, soc_E_qp_act, E_virt_qp])[plot_keep]
            soc_qp_ewin = auto_energy_window(eps_soc_qp_unsorted, sigma_ev=sigma_use)
            sort_idx_qp = np.argsort(eps_soc_qp_unsorted)
            eps_soc_qp = eps_soc_qp_unsorted[sort_idx_qp]
            F_spinor_qp = F_spinor_plot_unsorted[sort_idx_qp, :]
            if fold_to_bz:
                intensity_soc_qp = np.sum(np.abs(F_spinor_qp.reshape(F_spinor_qp.shape[0], len(kpts_cart), len(G_vecs)))**2, axis=2)
            else:
                intensity_soc_qp = np.abs(F_spinor_qp)**2
            smear_and_export_fuzzy(intensity_soc_qp, eps_soc_qp, labels, soc_qp_ewin, sigma_use, prefix="soc_qp")
        
        if getattr(args, 'pdos_atoms', None) and getattr(args, 'coop_pairs', None):
            print("  [PDOS/COOP] Computing SOC Spinor population analysis...")
            t_pop = time.time()
            n_ao = S_dense.shape[0]

            C_spinor_ao = np.zeros((2 * n_ao, len(eps_soc_unsorted)), dtype=complex)
            SC_dense = S_dense @ C_dense
            SC_spinor_ao = np.zeros((2 * n_ao, len(eps_soc_unsorted)), dtype=complex)

            if is_uks and soc_active_indices_beta is not None:
                SC_dense_beta = S_dense @ C_beta_dense
                n_alpha_act = len(soc_active_indices)
                n_beta_act = len(soc_active_indices_beta)

                cursor = 0
                C_spinor_ao[:n_ao, cursor:cursor + len(core_idx)] = C_dense[:, core_idx]
                SC_spinor_ao[:n_ao, cursor:cursor + len(core_idx)] = SC_dense[:, core_idx]
                cursor += len(core_idx)

                C_spinor_ao[n_ao:, cursor:cursor + len(core_idx_b)] = C_beta_dense[:, core_idx_b]
                SC_spinor_ao[n_ao:, cursor:cursor + len(core_idx_b)] = SC_dense_beta[:, core_idx_b]
                cursor += len(core_idx_b)

                C_act_a = C_dense[:, soc_active_indices]
                C_act_b = C_beta_dense[:, soc_active_indices_beta]
                SC_act_a = SC_dense[:, soc_active_indices]
                SC_act_b = SC_dense_beta[:, soc_active_indices_beta]
                n_spinor_act = soc_U_act.shape[1]
                C_spinor_ao[:, cursor:cursor + n_spinor_act] = np.vstack([
                    C_act_a @ soc_U_act[:n_alpha_act, :],
                    C_act_b @ soc_U_act[n_alpha_act:n_alpha_act + n_beta_act, :]
                ])
                SC_spinor_ao[:, cursor:cursor + n_spinor_act] = np.vstack([
                    SC_act_a @ soc_U_act[:n_alpha_act, :],
                    SC_act_b @ soc_U_act[n_alpha_act:n_alpha_act + n_beta_act, :]
                ])
                cursor += n_spinor_act

                C_spinor_ao[:n_ao, cursor:cursor + len(virt_idx)] = C_dense[:, virt_idx]
                SC_spinor_ao[:n_ao, cursor:cursor + len(virt_idx)] = SC_dense[:, virt_idx]
                cursor += len(virt_idx)

                C_spinor_ao[n_ao:, cursor:cursor + len(virt_idx_b)] = C_beta_dense[:, virt_idx_b]
                SC_spinor_ao[n_ao:, cursor:cursor + len(virt_idx_b)] = SC_dense_beta[:, virt_idx_b]
            else:
                C_spinor_ao[:n_ao, :len(core_idx)] = C_dense[:, core_idx]
                C_spinor_ao[n_ao:, len(core_idx):2*len(core_idx)] = C_dense[:, core_idx]

                C_act = C_dense[:, soc_active_indices]
                C_spinor_act_a = C_act @ soc_U_act[:len(soc_active_indices), :]
                C_spinor_act_b = C_act @ soc_U_act[len(soc_active_indices):, :]
                C_spinor_ao[:, 2*len(core_idx) : 2*len(core_idx) + 2*len(soc_active_indices)] = np.vstack([C_spinor_act_a, C_spinor_act_b])

                virt_start = 2*len(core_idx) + 2*len(soc_active_indices)
                C_spinor_ao[:n_ao, virt_start : virt_start+len(virt_idx)] = C_dense[:, virt_idx]
                C_spinor_ao[n_ao:, virt_start+len(virt_idx) :] = C_dense[:, virt_idx]

                SC_spinor_ao[:n_ao, :len(core_idx)] = SC_dense[:, core_idx]
                SC_spinor_ao[n_ao:, len(core_idx):2*len(core_idx)] = SC_dense[:, core_idx]

                SC_act = SC_dense[:, soc_active_indices]
                SC_spinor_ao[:, 2*len(core_idx) : 2*len(core_idx) + 2*len(soc_active_indices)] = np.vstack([SC_act @ soc_U_act[:len(soc_active_indices), :], SC_act @ soc_U_act[len(soc_active_indices):, :]])

                SC_spinor_ao[:n_ao, virt_start : virt_start+len(virt_idx)] = SC_dense[:, virt_idx]
                SC_spinor_ao[n_ao:, virt_start+len(virt_idx) :] = SC_dense[:, virt_idx]

            C_spinor_ao = C_spinor_ao[:, plot_keep][:, sort_idx]
            SC_spinor_ao = SC_spinor_ao[:, plot_keep][:, sort_idx]

            pops_soc_full = np.real(C_spinor_ao[:n_ao, :].conj() * SC_spinor_ao[:n_ao, :]) + \
                            np.real(C_spinor_ao[n_ao:, :].conj() * SC_spinor_ao[n_ao:, :])
            print(f"  [PDOS/COOP] Pre-computed populations in {time.time() - t_pop:.2f}s")
            
            compute_pdos_and_coop(C_spinor_ao, S_dense, eps_soc, shells, args.pdos_atoms, args.coop_pairs, soc_ewin, sigma=pdos_sigma_use, is_soc=True, prefix="soc", pops=pops_soc_full)

    # --- 3. Generate Multi-Row Interactive Plotly HTML ---
    if getattr(args, 'plot', True) or getattr(args, 'plot_fuzzy', True):
        from miniBSE.plot_fuzzy import generate_interactive_plot
        ef_dict = {"sf": 0.0}; homo_dict = {"sf": e_homo}; lumo_dict = {"sf": e_lumo}
        
        if args.soc_flag:
            ef_dict["soc"] = 0.0
            homo_dict["soc"] = eps_soc[global_spinor_homo_idx]
            lumo_dict["soc"] = eps_soc[global_spinor_homo_idx + 1]

        if dashboard_energy_mode in ("dft", "both"):
            generate_interactive_plot(
                prefix="sf",
                material=args.material,
                ef=ef_dict.get("sf", 0.0),
                e_homo=homo_dict.get("sf"),
                e_lumo=lumo_dict.get("sf"),
                normalize_coop=False,
                energy_label="DFT MO energy (eV)",
                output_html="fuzzy_dashboard_sf.html"
            )

        if dashboard_energy_mode in ("qp", "both") and qp_plot_energies is not None:
            generate_interactive_plot(
                prefix="sf_qp",
                material=args.material,
                ef=0.0 if qp_energy_reference == "fermi" else None,
                e_homo=qp_plot_energies[homo_index],
                e_lumo=qp_plot_energies[homo_index + 1],
                normalize_coop=False,
                energy_label="QP energy vs vacuum (eV)" if qp_energy_reference == "vacuum" else "QP-corrected energy relative to Fermi (eV)",
                output_html="fuzzy_dashboard_sf_qp.html"
            )

        if is_uks and dashboard_energy_mode in ("dft", "both"):
            generate_interactive_plot(
                prefix="uks",
                material=args.material,
                ef=0.0,
                e_homo=max(eps_shifted[homo_index], eps_beta_shifted[homo_index_beta]),
                e_lumo=min(eps_shifted[homo_index + 1], eps_beta_shifted[homo_index_beta + 1]),
                normalize_coop=False,
                energy_label="DFT MO energy (eV)",
                output_html="fuzzy_dashboard_uks.html"
            )

        if is_uks and dashboard_energy_mode in ("qp", "both") and qp_plot_energies is not None and qp_plot_energies_beta is not None:
            generate_interactive_plot(
                prefix="uks_qp",
                material=args.material,
                ef=0.0 if qp_energy_reference == "fermi" else None,
                e_homo=max(qp_plot_energies[homo_index], qp_plot_energies_beta[homo_index_beta]),
                e_lumo=min(qp_plot_energies[homo_index + 1], qp_plot_energies_beta[homo_index_beta + 1]),
                normalize_coop=False,
                energy_label="QP energy vs vacuum (eV)" if qp_energy_reference == "vacuum" else "QP-corrected energy relative to Fermi (eV)",
                output_html="fuzzy_dashboard_uks_qp.html"
            )
    
        # Generate SOC Dashboard (if requested)
        if args.soc_flag and dashboard_energy_mode in ("dft", "both"):
            generate_interactive_plot(
                prefix="soc", 
                material=args.material, 
                ef=ef_dict.get("soc", 0.0), 
                e_homo=homo_dict.get("soc"), 
                e_lumo=lumo_dict.get("soc"), 
                normalize_coop=False,
                energy_label="DFT MO energy (eV)",
                output_html="fuzzy_dashboard_soc.html"
            )

        if args.soc_flag and dashboard_energy_mode in ("qp", "both") and eps_soc_qp is not None:
            generate_interactive_plot(
                prefix="soc_qp",
                material=args.material,
                ef=None,
                e_homo=eps_soc_qp[global_spinor_homo_idx],
                e_lumo=eps_soc_qp[global_spinor_homo_idx + 1],
                normalize_coop=False,
                energy_label="QP+SOC energy vs vacuum (eV)",
                output_html="fuzzy_dashboard_soc_qp.html"
            )
