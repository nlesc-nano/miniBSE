import time

import numpy as np
from scipy.linalg import eigh

import libint_cpp
from miniBSE.constants import BOHR_PER_ANG, HA_TO_EV, valence_electrons
from miniBSE.io_utils import parse_gth_soc_potentials

_L_CACHE = {}


def get_angular_momentum_matrices(l):
    """Returns Lx, Ly, Lz matrices in the complex spherical harmonic basis."""
    if l in _L_CACHE:
        return _L_CACHE[l]

    m = np.arange(-l, l + 1)
    Lz = np.diag(m).astype(complex)
    if l == 0:
        mats = (
            np.zeros((1, 1), dtype=complex),
            np.zeros((1, 1), dtype=complex),
            np.zeros((1, 1), dtype=complex),
        )
    else:
        Lp = np.diag(np.sqrt(l * (l + 1) - m[:-1] * (m[:-1] + 1)), 1).astype(complex)
        Lm = np.diag(np.sqrt(l * (l + 1) - m[1:] * (m[1:] - 1)), -1).astype(complex)
        Lx = 0.5 * (Lp + Lm)
        Ly = -0.5j * (Lp - Lm)
        mats = (Lx, Ly, Lz)

    _L_CACHE[l] = mats
    return mats


def _build_h_soc(k_coeffs, nprj):
    h_soc = np.zeros((nprj, nprj))
    k_idx = 0
    for i in range(nprj):
        for j in range(i, nprj):
            h_soc[i, j] = h_soc[j, i] = k_coeffs[k_idx]
            k_idx += 1
    return h_soc


def _build_soc_projectors(atom_symbols, coords_ang, soc_tbl):
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
                proj_groups[key] = {
                    'nprj': block['nprj'],
                    'sym': sym,
                    'k_coeffs': block['k_coeffs'],
                    'l': l,
                }

            for i in range(1, block['nprj'] + 1):
                projectors.append({
                    'sym': sym,
                    'atom_idx': atom_idx,
                    'l': l,
                    'i': i,
                    'r_l': block['r'],
                    'center': center_bohr,
                })

    return projectors, proj_groups


def _ortho_active_coeffs(C_act, S_AO):
    S_sub = C_act.T @ S_AO @ C_act
    return C_act @ np.linalg.inv(np.linalg.cholesky(S_sub))


def _accumulate_soc_component(B_left, B_right, h_soc, L, out):
    """out += 0.5 * B_left @ kron(h_soc, L) @ B_right.conj().T."""
    K = np.kron(h_soc, L)
    BK = B_left @ K
    out += 0.5 * (BK @ B_right.conj().T)



def _assemble_rks_soc_blocks(B_mo, proj_groups):
    n_mo = B_mo.shape[0]
    Hx = np.zeros((n_mo, n_mo), dtype=complex)
    Hy = np.zeros_like(Hx)
    Hz = np.zeros_like(Hx)

    col_offset = 0
    for key in sorted(proj_groups.keys()):
        grp = proj_groups[key]
        l = grp['l']
        nprj = grp['nprj']
        num_cols = nprj * (2 * l + 1)
        B_block = B_mo[:, col_offset:col_offset + num_cols]
        h_soc = _build_h_soc(grp['k_coeffs'], nprj)
        Lx, Ly, Lz = get_angular_momentum_matrices(l)
        _accumulate_soc_component(B_block, B_block, h_soc, Lx, Hx)
        _accumulate_soc_component(B_block, B_block, h_soc, Ly, Hy)
        _accumulate_soc_component(B_block, B_block, h_soc, Lz, Hz)
        col_offset += num_cols

    Hx = 0.5 * (Hx + Hx.conj().T)
    Hy = 0.5 * (Hy + Hy.conj().T)
    Hz = 0.5 * (Hz + Hz.conj().T)
    return Hx, Hy, Hz


def _assemble_uks_soc_chunk(keys, B_alpha, B_beta, proj_groups, col_offsets):
    n_alpha = B_alpha.shape[0]
    n_beta = B_beta.shape[0]
    Hx_aa = np.zeros((n_alpha, n_alpha), dtype=complex)
    Hy_aa = np.zeros_like(Hx_aa)
    Hz_aa = np.zeros_like(Hx_aa)
    Hx_ab = np.zeros((n_alpha, n_beta), dtype=complex)
    Hy_ab = np.zeros_like(Hx_ab)
    Hz_ab = np.zeros_like(Hx_ab)
    Hx_bb = np.zeros((n_beta, n_beta), dtype=complex)
    Hy_bb = np.zeros_like(Hx_bb)
    Hz_bb = np.zeros_like(Hx_bb)

    for key in keys:
        grp = proj_groups[key]
        l = grp['l']
        nprj = grp['nprj']
        num_cols = nprj * (2 * l + 1)
        col_offset = col_offsets[key]
        B_a = B_alpha[:, col_offset:col_offset + num_cols]
        B_b = B_beta[:, col_offset:col_offset + num_cols]
        h_soc = _build_h_soc(grp['k_coeffs'], nprj)
        Lx, Ly, Lz = get_angular_momentum_matrices(l)

        for L, Haa, Hab, Hbb in (
            (Lx, Hx_aa, Hx_ab, Hx_bb),
            (Ly, Hy_aa, Hy_ab, Hy_bb),
            (Lz, Hz_aa, Hz_ab, Hz_bb),
        ):
            K = np.kron(h_soc, L)
            Ka = B_a @ K
            Haa += 0.5 * (Ka @ B_a.conj().T)
            Hab += 0.5 * (Ka @ B_b.conj().T)
            Hbb += 0.5 * ((B_b @ K) @ B_b.conj().T)

    return Hx_aa, Hy_aa, Hz_aa, Hx_ab, Hy_ab, Hz_ab, Hx_bb, Hy_bb, Hz_bb


def _assemble_uks_soc_blocks(B_alpha, B_beta, proj_groups):
    keys = sorted(proj_groups.keys())
    col_offsets = {}
    col_offset = 0
    for key in keys:
        grp = proj_groups[key]
        col_offsets[key] = col_offset
        col_offset += grp['nprj'] * (2 * grp['l'] + 1)

    chunk = _assemble_uks_soc_chunk(keys, B_alpha, B_beta, proj_groups, col_offsets)
    Hx_aa, Hy_aa, Hz_aa, Hx_ab, Hy_ab, Hz_ab, Hx_bb, Hy_bb, Hz_bb = chunk

    Hx_aa = 0.5 * (Hx_aa + Hx_aa.conj().T)
    Hy_aa = 0.5 * (Hy_aa + Hy_aa.conj().T)
    Hz_aa = 0.5 * (Hz_aa + Hz_aa.conj().T)
    Hx_bb = 0.5 * (Hx_bb + Hx_bb.conj().T)
    Hy_bb = 0.5 * (Hy_bb + Hy_bb.conj().T)
    Hz_bb = 0.5 * (Hz_bb + Hz_bb.conj().T)
    Hx_ba = Hx_ab.conj().T
    Hy_ba = Hy_ab.conj().T
    Hz_ba = Hz_ab.conj().T
    return Hx_aa, Hy_aa, Hz_aa, Hx_ab, Hy_ab, Hz_ab, Hx_ba, Hy_ba, Hz_ba, Hx_bb, Hy_bb, Hz_bb


def prepare_soc_overlap_cache(atom_symbols, coords_ang, shells, gth_file, nthreads=1):
    """Compute and cache AO-projector overlaps shared across SOC active windows."""
    elements = {sym: valence_electrons.get(sym) for sym in set(atom_symbols)}
    soc_tbl = parse_gth_soc_potentials(gth_file, elements)
    projectors, proj_groups = _build_soc_projectors(atom_symbols, coords_ang, soc_tbl)
    B_raw = libint_cpp.compute_hgh_overlaps(shells, projectors, nthreads)
    return {
        'soc_tbl': soc_tbl,
        'projectors': projectors,
        'proj_groups': proj_groups,
        'B_raw': B_raw,
    }


def compute_spinor_subspace(
    atom_symbols, coords_ang, shells, C_AO, eps_Ha, S_AO, active_indices, gth_file,
    nthreads=1, soc_cache=None,
):
    print("\n" + "=" * 60)
    print(" [SOC] Spin-Orbit Coupling Module Initialized")
    print("=" * 60)

    if soc_cache is None:
        print(f"  -> Reading GTH Potentials from: {gth_file}")
        t0 = time.time()
        soc_cache = prepare_soc_overlap_cache(
            atom_symbols, coords_ang, shells, gth_file, nthreads=nthreads
        )
        print(f"  -> Parsed potentials and overlaps in {time.time() - t0:.2f}s")
    else:
        print("  -> Reusing cached AO-projector overlaps")

    proj_groups = soc_cache['proj_groups']
    B_raw = soc_cache['B_raw']
    print(
        f"  -> Using {len(soc_cache['projectors'])} HGH projectors "
        f"across {len(proj_groups)} angular blocks."
    )
    print(f"  -> Cached overlap matrix shape: {B_raw.shape}")

    print("  -> Projecting overlaps to Active Subspace to accelerate assembly...")
    t0 = time.time()

    C_act = C_AO[:, active_indices]
    C_ortho = _ortho_active_coeffs(C_act, S_AO)
    B_mo_raw = C_ortho.T @ B_raw
    Hx_mo, Hy_mo, Hz_mo = _assemble_rks_soc_blocks(B_mo_raw, proj_groups)

    print(f"  -> Hamiltonian assembly completed in {time.time() - t0:.2f}s")

    n_mo = len(active_indices)
    print(f"  -> Diagonalizing Single-Particle Spinor Hamiltonian (Active Space = {n_mo} MOs)...")
    t0 = time.time()

    H0 = np.kron(np.eye(2), np.diag(eps_Ha[active_indices]))
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    H_SO = 0.5 * (
        np.kron(sigma_x, Hx_mo) + np.kron(sigma_y, Hy_mo) + np.kron(sigma_z, Hz_mo)
    )
    H_total = H0 - H_SO
    soc_E, soc_U = eigh(H_total)

    print(f"  -> Spinor diagonalization completed in {time.time() - t0:.2f}s")
    H0_diag = np.sort(np.diag(H0).real)
    max_shift = np.max(np.abs(soc_E - H0_diag)) * HA_TO_EV
    print(f"  -> Max SOC-induced energy shift: {max_shift:.3f} eV")
    print("=" * 60 + "\n")

    return soc_E, soc_U, soc_cache


def compute_spinor_subspace_uks(
    atom_symbols, coords_ang, shells, C_alpha_AO, eps_alpha_Ha, active_alpha_indices,
    C_beta_AO, eps_beta_Ha, active_beta_indices, S_AO, gth_file, nthreads=1, soc_cache=None,
):
    print("\n" + "=" * 60)
    print(" [SOC-UKS] Spin-Orbit Coupling Module Initialized")
    print("=" * 60)

    if soc_cache is None:
        print(f"  -> Reading GTH Potentials from: {gth_file}")
        t0 = time.time()
        soc_cache = prepare_soc_overlap_cache(
            atom_symbols, coords_ang, shells, gth_file, nthreads=nthreads
        )
        print(f"  -> Parsed potentials and overlaps in {time.time() - t0:.2f}s")
    else:
        print("  -> Reusing cached AO-projector overlaps")

    proj_groups = soc_cache['proj_groups']
    B_raw = soc_cache['B_raw']
    print(
        f"  -> Using {len(soc_cache['projectors'])} HGH projectors "
        f"across {len(proj_groups)} angular blocks."
    )
    print(f"  -> Cached overlap matrix shape: {B_raw.shape}")

    print("  -> Projecting overlaps to UKS alpha/beta active subspaces...")
    t0 = time.time()

    C_alpha_act = C_alpha_AO[:, active_alpha_indices]
    C_beta_act = C_beta_AO[:, active_beta_indices]
    C_alpha_ortho = _ortho_active_coeffs(C_alpha_act, S_AO)
    C_beta_ortho = _ortho_active_coeffs(C_beta_act, S_AO)
    B_alpha = C_alpha_ortho.T @ B_raw
    B_beta = C_beta_ortho.T @ B_raw
    t_proj = time.time()
    blocks = _assemble_uks_soc_blocks(B_alpha, B_beta, proj_groups)
    t_asm = time.time()
    Hx_aa, Hy_aa, Hz_aa, Hx_ab, Hy_ab, Hz_ab, Hx_ba, Hy_ba, Hz_ba, Hx_bb, Hy_bb, Hz_bb = blocks

    print(f"  -> MO projection completed in {t_proj - t0:.2f}s")
    print(f"  -> UKS SOC block accumulation completed in {t_asm - t_proj:.2f}s")
    print(f"  -> UKS Hamiltonian assembly completed in {t_asm - t0:.2f}s")

    n_alpha = len(active_alpha_indices)
    n_beta = len(active_beta_indices)
    print(
        f"  -> Diagonalizing UKS Single-Particle Spinor Hamiltonian "
        f"(Alpha={n_alpha}, Beta={n_beta})..."
    )
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

    print(f"  -> UKS spinor diagonalization completed in {time.time() - t0:.2f}s")
    H0_diag = np.sort(np.diag(H0).real)
    max_shift = np.max(np.abs(soc_E - H0_diag)) * HA_TO_EV
    print(f"  -> Max SOC-induced energy shift: {max_shift:.3f} eV")
    print("=" * 60 + "\n")

    return soc_E, soc_U, soc_cache
