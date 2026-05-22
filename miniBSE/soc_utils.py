import time
import numpy as np
from scipy.linalg import eigh
import libint_cpp
from miniBSE.io_utils import parse_gth_soc_potentials
from miniBSE.constants import HA_TO_EV, valence_electrons, BOHR_PER_ANG


def get_angular_momentum_matrices(l):
    """Returns Lx, Ly, Lz matrices in the complex spherical harmonic basis."""
    m = np.arange(-l, l + 1)
    Lz = np.diag(m).astype(complex)
    if l == 0:
        return np.zeros((1, 1), dtype=complex), np.zeros((1, 1), dtype=complex), np.zeros((1, 1), dtype=complex)
    
    Lp = np.diag(np.sqrt(l * (l + 1) - m[:-1] * (m[:-1] + 1)), 1).astype(complex)
    Lm = np.diag(np.sqrt(l * (l + 1) - m[1:] * (m[1:] - 1)), -1).astype(complex)
    
    Lx = 0.5 * (Lp + Lm)
    Ly = -0.5j * (Lp - Lm)
    return Lx, Ly, Lz

def compute_spinor_subspace(atom_symbols, coords_ang, shells, C_AO, eps_Ha, S_AO, active_indices, gth_file, nthreads=1):
    print("\n" + "="*60)
    print(" [SOC] Spin-Orbit Coupling Module Initialized")
    print("="*60)

    # 1. Parse GTH Potentials
    print(f"  -> Reading GTH Potentials from: {gth_file}")
    t0 = time.time()
    elements = {sym: valence_electrons.get(sym) for sym in set(atom_symbols)}
    soc_tbl = parse_gth_soc_potentials(gth_file, elements)
    print(f"  -> Parsed potentials in {time.time()-t0:.2f}s")

    # 2. Build Projectors
    projectors = []
    proj_groups = {}
    for atom_idx, sym in enumerate(atom_symbols):
        if sym not in soc_tbl or not soc_tbl[sym]['so']: continue
        
        # FIX: Convert Angstroms to Bohr for the C++ libint engine!
        center_bohr = np.array(coords_ang[atom_idx]) * BOHR_PER_ANG
        
        for block in soc_tbl[sym]['so']:
            if not block.get('k_coeffs'): continue # Skip if no SOC (e.g. l=0)
            
            l = block['l']
            key = (atom_idx, l)
            if key not in proj_groups: 
                proj_groups[key] = {'nprj': block['nprj'], 'sym': sym, 'k_coeffs': block['k_coeffs']}
                
            for i in range(1, block['nprj'] + 1):
                p = {'sym': sym, 'atom_idx': atom_idx, 'l': l, 'i': i, 'r_l': block['r'], 'center': center_bohr}
                projectors.append(p)
    
    print(f"  -> Generated {len(projectors)} HGH projectors across {len(proj_groups)} angular blocks.")

    # 3. Compute Overlaps
    print("  -> Computing <AO|Projector> overlaps via Libint C++...")
    t0 = time.time()
    B_raw = libint_cpp.compute_hgh_overlaps(shells, projectors, nthreads)
    print(f"  -> Overlaps computed in {time.time()-t0:.2f}s. Matrix shape: {B_raw.shape}")

    # 4. Assemble Matrices directly in the MO Subspace (Blazing Fast)
    print("  -> Projecting overlaps to Active Subspace to accelerate assembly...")
    t0 = time.time()
    
    C_act = C_AO[:, active_indices]
    S_sub = C_act.T @ S_AO @ C_act
    C_ortho = C_act @ np.linalg.inv(np.linalg.cholesky(S_sub))
    
    # Project full B_raw matrix: (126, n_ao) @ (n_ao, n_proj) -> (126, n_proj)
    B_mo_raw = C_ortho.T @ B_raw
    
    n_mo = len(active_indices)
    Hx_mo = np.zeros((n_mo, n_mo), dtype=complex)
    Hy_mo = np.zeros((n_mo, n_mo), dtype=complex)
    Hz_mo = np.zeros((n_mo, n_mo), dtype=complex)
    
    col_offset = 0
    for key in sorted(proj_groups.keys()):
        atom_idx, l = key
        grp = proj_groups[key]
        nprj = grp['nprj']
        num_cols = nprj * (2 * l + 1)
        
        B_mo_block = B_mo_raw[:, col_offset : col_offset + num_cols]
        
        h_soc = np.zeros((nprj, nprj))
        k_idx = 0
        for i in range(nprj):
            for j in range(i, nprj):
                h_soc[i, j] = h_soc[j, i] = grp['k_coeffs'][k_idx]
                k_idx += 1
                
        Lx, Ly, Lz = get_angular_momentum_matrices(l)
        Kx, Ky, Kz = np.kron(h_soc, Lx), np.kron(h_soc, Ly), np.kron(h_soc, Lz)
        
        # Multiply and project using conjugate transpose
        Hx_mo += (B_mo_block @ Kx @ B_mo_block.conj().T) * 0.5
        Hy_mo += (B_mo_block @ Ky @ B_mo_block.conj().T) * 0.5
        Hz_mo += (B_mo_block @ Kz @ B_mo_block.conj().T) * 0.5
        
        col_offset += num_cols

    # Symmetrize to clean numerical noise
    Hx_mo = 0.5 * (Hx_mo + Hx_mo.conj().T)
    Hy_mo = 0.5 * (Hy_mo + Hy_mo.conj().T)
    Hz_mo = 0.5 * (Hz_mo + Hz_mo.conj().T)

    print(f"  -> Hamiltonian assembly completed in {time.time()-t0:.2f}s")

    # 5. Solve in Active Subspace
    print(f"  -> Diagonalizing Single-Particle Spinor Hamiltonian (Active Space = {n_mo} MOs)...")
    t0 = time.time()
    
    H0 = np.kron(np.eye(2), np.diag(eps_Ha[active_indices]))
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    H_SO = 0.5 * (np.kron(sigma_x, Hx_mo) + np.kron(sigma_y, Hy_mo) + np.kron(sigma_z, Hz_mo))
    H_total = H0 - H_SO 
    
    soc_E, soc_U = eigh(H_total)
    
    print(f"  -> Spinor diagonalization completed in {time.time()-t0:.2f}s")
    
    H0_diag = np.sort(np.diag(H0).real)
    max_shift = np.max(np.abs(soc_E - H0_diag)) * HA_TO_EV
    print(f"  -> Max SOC-induced energy shift: {max_shift:.3f} eV")
    print("="*60 + "\n")
    
    return soc_E, soc_U


def compute_spinor_subspace_uks(atom_symbols, coords_ang, shells, C_alpha_AO, eps_alpha_Ha, active_alpha_indices,
                                C_beta_AO, eps_beta_Ha, active_beta_indices, S_AO, gth_file, nthreads=1):
    print("\n" + "="*60)
    print(" [SOC-UKS] Spin-Orbit Coupling Module Initialized")
    print("="*60)

    print(f"  -> Reading GTH Potentials from: {gth_file}")
    t0 = time.time()
    elements = {sym: valence_electrons.get(sym) for sym in set(atom_symbols)}
    soc_tbl = parse_gth_soc_potentials(gth_file, elements)
    print(f"  -> Parsed potentials in {time.time()-t0:.2f}s")

    projectors = []
    proj_groups = {}
    for atom_idx, sym in enumerate(atom_symbols):
        if sym not in soc_tbl or not soc_tbl[sym]['so']:
            continue

        center_bohr = np.array(coords_ang[atom_idx]) * BOHR_PER_ANG

        for block in soc_tbl[sym]['so']:
            if not block.get('k_coeffs'):
                continue

            l = block['l']
            key = (atom_idx, l)
            if key not in proj_groups:
                proj_groups[key] = {'nprj': block['nprj'], 'sym': sym, 'k_coeffs': block['k_coeffs']}

            for i in range(1, block['nprj'] + 1):
                projectors.append({
                    'sym': sym, 'atom_idx': atom_idx, 'l': l, 'i': i,
                    'r_l': block['r'], 'center': center_bohr
                })

    print(f"  -> Generated {len(projectors)} HGH projectors across {len(proj_groups)} angular blocks.")

    print("  -> Computing <AO|Projector> overlaps via Libint C++...")
    t0 = time.time()
    B_raw = libint_cpp.compute_hgh_overlaps(shells, projectors, nthreads)
    print(f"  -> Overlaps computed in {time.time()-t0:.2f}s. Matrix shape: {B_raw.shape}")

    print("  -> Projecting overlaps to UKS alpha/beta active subspaces...")
    t0 = time.time()
    C_alpha_act = C_alpha_AO[:, active_alpha_indices]
    C_beta_act = C_beta_AO[:, active_beta_indices]

    S_alpha = C_alpha_act.T @ S_AO @ C_alpha_act
    S_beta = C_beta_act.T @ S_AO @ C_beta_act
    C_alpha_ortho = C_alpha_act @ np.linalg.inv(np.linalg.cholesky(S_alpha))
    C_beta_ortho = C_beta_act @ np.linalg.inv(np.linalg.cholesky(S_beta))

    B_alpha = C_alpha_ortho.T @ B_raw
    B_beta = C_beta_ortho.T @ B_raw

    n_alpha = len(active_alpha_indices)
    n_beta = len(active_beta_indices)
    Hx_aa = np.zeros((n_alpha, n_alpha), dtype=complex)
    Hy_aa = np.zeros_like(Hx_aa)
    Hz_aa = np.zeros_like(Hx_aa)
    Hx_ab = np.zeros((n_alpha, n_beta), dtype=complex)
    Hy_ab = np.zeros_like(Hx_ab)
    Hz_ab = np.zeros_like(Hx_ab)
    Hx_ba = np.zeros((n_beta, n_alpha), dtype=complex)
    Hy_ba = np.zeros_like(Hx_ba)
    Hz_ba = np.zeros_like(Hx_ba)
    Hx_bb = np.zeros((n_beta, n_beta), dtype=complex)
    Hy_bb = np.zeros_like(Hx_bb)
    Hz_bb = np.zeros_like(Hx_bb)

    col_offset = 0
    for key in sorted(proj_groups.keys()):
        atom_idx, l = key
        grp = proj_groups[key]
        nprj = grp['nprj']
        num_cols = nprj * (2 * l + 1)

        B_a = B_alpha[:, col_offset:col_offset + num_cols]
        B_b = B_beta[:, col_offset:col_offset + num_cols]

        h_soc = np.zeros((nprj, nprj))
        k_idx = 0
        for i in range(nprj):
            for j in range(i, nprj):
                h_soc[i, j] = h_soc[j, i] = grp['k_coeffs'][k_idx]
                k_idx += 1

        Lx, Ly, Lz = get_angular_momentum_matrices(l)
        Kx, Ky, Kz = np.kron(h_soc, Lx), np.kron(h_soc, Ly), np.kron(h_soc, Lz)

        for K, Haa, Hab, Hba, Hbb in (
            (Kx, Hx_aa, Hx_ab, Hx_ba, Hx_bb),
            (Ky, Hy_aa, Hy_ab, Hy_ba, Hy_bb),
            (Kz, Hz_aa, Hz_ab, Hz_ba, Hz_bb),
        ):
            Haa += (B_a @ K @ B_a.conj().T) * 0.5
            Hab += (B_a @ K @ B_b.conj().T) * 0.5
            Hba += (B_b @ K @ B_a.conj().T) * 0.5
            Hbb += (B_b @ K @ B_b.conj().T) * 0.5

        col_offset += num_cols

    Hx_aa = 0.5 * (Hx_aa + Hx_aa.conj().T)
    Hy_aa = 0.5 * (Hy_aa + Hy_aa.conj().T)
    Hz_aa = 0.5 * (Hz_aa + Hz_aa.conj().T)
    Hx_bb = 0.5 * (Hx_bb + Hx_bb.conj().T)
    Hy_bb = 0.5 * (Hy_bb + Hy_bb.conj().T)
    Hz_bb = 0.5 * (Hz_bb + Hz_bb.conj().T)
    Hx_ba = Hx_ab.conj().T
    Hy_ba = Hy_ab.conj().T
    Hz_ba = Hz_ab.conj().T

    print(f"  -> UKS Hamiltonian assembly completed in {time.time()-t0:.2f}s")

    print(f"  -> Diagonalizing UKS Single-Particle Spinor Hamiltonian (Alpha={n_alpha}, Beta={n_beta})...")
    t0 = time.time()

    H0 = np.block([
        [np.diag(eps_alpha_Ha[active_alpha_indices]), np.zeros((n_alpha, n_beta))],
        [np.zeros((n_beta, n_alpha)), np.diag(eps_beta_Ha[active_beta_indices])],
    ]).astype(complex)

    H_SO = np.block([
        [0.5 * Hz_aa, 0.5 * (Hx_ab - 1j * Hy_ab)],
        [0.5 * (Hx_ba + 1j * Hy_ba), -0.5 * Hz_bb],
    ])
    H_total = H0 - H_SO

    soc_E, soc_U = eigh(H_total)

    print(f"  -> UKS spinor diagonalization completed in {time.time()-t0:.2f}s")
    H0_diag = np.sort(np.diag(H0).real)
    max_shift = np.max(np.abs(soc_E - H0_diag)) * HA_TO_EV
    print(f"  -> Max SOC-induced energy shift: {max_shift:.3f} eV")
    print("="*60 + "\n")

    return soc_E, soc_U
