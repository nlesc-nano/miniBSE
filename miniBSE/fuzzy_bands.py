import numpy as np
import time
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from scipy.spatial.distance import pdist

def generate_automated_kpath(cif_path, coords_ang, line_density=50):
    print(f"  [Fuzzy] Loading CIF: {cif_path}")
    struct = Structure.from_file(cif_path)
    sga = SpacegroupAnalyzer(struct)
    prim_struct = sga.get_primitive_standard_structure()
    kpath = HighSymmKpath(prim_struct)
    kpts_frac, labels = kpath.get_kpoints(line_density=line_density, coords_are_cartesian=False)
    
    # Standard reciprocal mapping
    kpts_cart = np.dot(kpts_frac, prim_struct.lattice.reciprocal_lattice.matrix)
    
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
    except Exception as e:
        print(f"  [Fuzzy] Warning: PCA Rotational alignment failed: {e}")

    print(f"  [Fuzzy] Generated {len(kpts_cart)} k-points for spacegroup {sga.get_space_group_symbol()}.")
    return kpts_cart, labels

def smear_and_export_fuzzy(intensity, eps_plot, labels, ewin, sigma_ev, prefix="sf"):
    t0 = time.time()
    
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

def run_fuzzy_bands_and_pdos(args, C_dense, S_dense, eps_shifted, occ, homo_index, e_homo, e_lumo, e_fermi_raw, syms, coords_ang, shells, pops_sf, soc_active_indices=None, soc_E_act=None, soc_U_act=None, spinor_homo_idx=None):
    import time
    import numpy as np
    import libint_cpp
    from miniBSE.pdos_coop import compute_pdos_and_coop
    
    print("\n===================================================")
    print(" [ FUZZY BANDS & PDOS ]")
    print("===================================================")
    
    kpts_cart, labels = generate_automated_kpath(args.cif, np.array(coords_ang), line_density=50)
    kpts_bohr = kpts_cart / 1.8897259886 
    
    print("  [Fuzzy] Computing Analytic AO-FT via C++ ...")
    F_ao = libint_cpp.ao_ft_complex(shells, kpts_bohr, args.nthreads)
    
    # --- 1. SPIN-FREE CALCULATION ---
    n_occ = homo_index + 1
    n_virt = len(eps_shifted) - n_occ
    print(f"\n  [Fuzzy] --- Spin-Free MO Statistics ---")
    print(f"  [Fuzzy] Total MOs: {len(eps_shifted)} ({n_occ} Occupied, {n_virt} Virtual)")
    print(f"  [Fuzzy] MO HOMO (Idx {homo_index}): {e_homo:8.4f} eV")
    print(f"  [Fuzzy] MO LUMO (Idx {homo_index + 1}): {e_lumo:8.4f} eV")
    print(f"  [Fuzzy] Fermi Level (raw shifted to 0.0): {e_fermi_raw:8.4f} eV")
    print(f"  [Fuzzy] -------------------------------")

    F_mo_sf = C_dense.T.conj() @ F_ao
    intensity_sf = np.abs(F_mo_sf)**2
    
    sigma_use = getattr(args, 'fuzzy_sigma', 0.03)
    pdos_sigma_use = getattr(args, 'pdos_sigma', 0.10)
    
    smear_and_export_fuzzy(intensity_sf, eps_shifted, labels, args.ewin, sigma_use, prefix="sf")
    
    if hasattr(args, 'pdos_atoms') and hasattr(args, 'coop_pairs'):
        print("  [PDOS/COOP] Computing Spin-Free population analysis...")
        compute_pdos_and_coop(C_dense, S_dense, eps_shifted, shells, args.pdos_atoms, args.coop_pairs, args.ewin, sigma=pdos_sigma_use, is_soc=False, prefix="sf", pops=pops_sf)

    # --- 2. SOC CALCULATION ---
    if args.soc_flag and soc_active_indices is not None:
        n_act_mo = len(soc_active_indices)
        n_act_occ = np.sum(soc_active_indices <= homo_index)
        n_act_virt = n_act_mo - n_act_occ

        print(f"\n  [Fuzzy-SOC] Applying Precomputed Unified SOC Projection...")
        print(f"  [Fuzzy-SOC] --- Dual-Window SOC Statistics ---")
        print(f"  [Fuzzy-SOC] Active Space Spatial MOs: {n_act_mo} ({n_act_occ} Occ, {n_act_virt} Virt)")
        print(f"  [Fuzzy-SOC] Spinor Expansion: {len(eps_shifted) * 2} Total Spinors Generated")
        
        core_idx = np.arange(0, soc_active_indices[0])
        virt_idx = np.arange(soc_active_indices[-1] + 1, len(eps_shifted))
        
        F_mo_act = F_mo_sf[soc_active_indices, :]
        F_spinor_act = soc_U_act.T.conj() @ np.vstack([F_mo_act, F_mo_act])
        F_spinor_core = np.vstack([F_mo_sf[core_idx, :], F_mo_sf[core_idx, :]])
        F_spinor_virt = np.vstack([F_mo_sf[virt_idx, :], F_mo_sf[virt_idx, :]])
        
        F_spinor = np.vstack([F_spinor_core, F_spinor_act, F_spinor_virt])
        E_core = np.concatenate([eps_shifted[core_idx], eps_shifted[core_idx]])
        E_virt = np.concatenate([eps_shifted[virt_idx], eps_shifted[virt_idx]])
        eps_soc = np.concatenate([E_core, soc_E_act, E_virt])
        
        sort_idx = np.argsort(eps_soc)
        eps_soc = eps_soc[sort_idx]
        intensity_soc = np.abs(F_spinor[sort_idx, :])**2
        # FIX: The true global spinor HOMO is exactly (Total Spatial Occ * 2) - 1
        global_spinor_homo_idx = ((homo_index + 1) * 2) - 1
 
        print(f"  [Fuzzy-SOC] Spinor HOMO (Idx {global_spinor_homo_idx}): {eps_soc[global_spinor_homo_idx]:8.4f} eV")
        print(f"  [Fuzzy-SOC] Spinor LUMO (Idx {global_spinor_homo_idx + 1}): {eps_soc[global_spinor_homo_idx + 1]:8.4f} eV")
        print(f"  [Fuzzy-SOC] ----------------------------------") 
        
        smear_and_export_fuzzy(intensity_soc, eps_soc, labels, args.ewin, sigma_use, prefix="soc")
        
        if hasattr(args, 'pdos_atoms') and hasattr(args, 'coop_pairs'):
            print("  [PDOS/COOP] Computing SOC Spinor population analysis...")
            t_pop = time.time()
            n_ao = S_dense.shape[0]
            
            C_spinor_ao = np.zeros((2 * n_ao, len(eps_soc)), dtype=complex)
            C_spinor_ao[:n_ao, :len(core_idx)] = C_dense[:, core_idx]
            C_spinor_ao[n_ao:, len(core_idx):2*len(core_idx)] = C_dense[:, core_idx]
            
            C_act = C_dense[:, soc_active_indices]
            C_spinor_act_a = C_act @ soc_U_act[:len(soc_active_indices), :]
            C_spinor_act_b = C_act @ soc_U_act[len(soc_active_indices):, :]
            C_spinor_ao[:, 2*len(core_idx) : 2*len(core_idx) + 2*len(soc_active_indices)] = np.vstack([C_spinor_act_a, C_spinor_act_b])
            
            virt_start = 2*len(core_idx) + 2*len(soc_active_indices)
            C_spinor_ao[:n_ao, virt_start : virt_start+len(virt_idx)] = C_dense[:, virt_idx]
            C_spinor_ao[n_ao:, virt_start+len(virt_idx) :] = C_dense[:, virt_idx]
            C_spinor_ao = C_spinor_ao[:, sort_idx]
            
            # ULTRA-FAST POPS (Avoids building the full SC matrix)
            SC_dense = S_dense @ C_dense
            SC_spinor_ao = np.zeros((2 * n_ao, len(eps_soc)), dtype=complex)
            SC_spinor_ao[:n_ao, :len(core_idx)] = SC_dense[:, core_idx]
            SC_spinor_ao[n_ao:, len(core_idx):2*len(core_idx)] = SC_dense[:, core_idx]
            
            SC_act = SC_dense[:, soc_active_indices]
            SC_spinor_ao[:, 2*len(core_idx) : 2*len(core_idx) + 2*len(soc_active_indices)] = np.vstack([SC_act @ soc_U_act[:len(soc_active_indices), :], SC_act @ soc_U_act[len(soc_active_indices):, :]])
            
            SC_spinor_ao[:n_ao, virt_start : virt_start+len(virt_idx)] = SC_dense[:, virt_idx]
            SC_spinor_ao[n_ao:, virt_start+len(virt_idx) :] = SC_dense[:, virt_idx]
            SC_spinor_ao = SC_spinor_ao[:, sort_idx]
            
            pops_soc_full = np.real(C_spinor_ao[:n_ao, :].conj() * SC_spinor_ao[:n_ao, :]) + \
                            np.real(C_spinor_ao[n_ao:, :].conj() * SC_spinor_ao[n_ao:, :])
            print(f"  [PDOS/COOP] Pre-computed populations in {time.time() - t_pop:.2f}s")
            
            compute_pdos_and_coop(C_spinor_ao, S_dense, eps_soc, shells, args.pdos_atoms, args.coop_pairs, args.ewin, sigma=pdos_sigma_use, is_soc=True, prefix="soc", pops=pops_soc_full)

    # --- 3. Generate Multi-Row Interactive Plotly HTML ---
    if getattr(args, 'plot', True) or getattr(args, 'plot_fuzzy', True):
        from miniBSE.plot_fuzzy import generate_interactive_plot
        ef_dict = {"sf": 0.0}; homo_dict = {"sf": e_homo}; lumo_dict = {"sf": e_lumo}
        
        if args.soc_flag:
            ef_dict["soc"] = 0.0
            homo_dict["soc"] = eps_soc[global_spinor_homo_idx]
            lumo_dict["soc"] = eps_soc[global_spinor_homo_idx + 1]

        # Generate Spin-Free Dashboard
    generate_interactive_plot(
        prefix="sf", 
        material=args.material, 
        ef=ef_dict.get("sf", 0.0), 
        e_homo=homo_dict.get("sf"), 
        e_lumo=lumo_dict.get("sf"), 
        normalize_coop=False
    )
    
    # Generate SOC Dashboard (if requested)
    if args.soc_flag:
        generate_interactive_plot(
            prefix="soc", 
            material=args.material, 
            ef=ef_dict.get("soc", 0.0), 
            e_homo=homo_dict.get("soc"), 
            e_lumo=lumo_dict.get("soc"), 
            normalize_coop=False
        )


