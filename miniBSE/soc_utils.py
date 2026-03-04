import time
import numpy as np
from scipy.linalg import eigh
import libint_cpp
from miniBSE.io_utils import parse_gth_soc_potentials
from miniBSE.constants import HA_TO_EV, valence_electrons

BOHR_PER_ANG = 1.8897259886

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

