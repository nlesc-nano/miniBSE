import os
import re
import time
import numpy as np
import collections
from scipy.sparse import issparse, csr_matrix
import libint_cpp


BOHR_PER_ANGSTROM = 1.889726124565062


# ============================================================
# XYZ READER
# ============================================================

def read_xyz(path):
    with open(path) as f:
        lines = f.readlines()

    nat = int(lines[0].strip())
    syms = []
    coords = []

    for line in lines[2:2 + nat]:
        p = line.split()
        syms.append(p[0])
        coords.append([float(x) for x in p[1:4]])

    return syms, np.asarray(coords)


# ============================================================
# BASIS PARSER (CP2K MOLOPT)
# ============================================================

def parse_basis(fname, wanted):

    basis = collections.defaultdict(list)

    with open(fname) as f:
        lines = f.readlines()

    it = iter(lines)

    for line in it:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        elem, bname = parts[0], parts[1]

        if bname != wanted:
            continue

        nset = int(next(it).split()[0])

        for _ in range(nset):
            hdr = next(it).split()
            lmin = int(hdr[1])
            nexp = int(hdr[3])
            counts = list(map(int, hdr[4:]))

            exps = []
            coef_rows = []

            for _ in range(nexp):
                row = next(it).split()
                exps.append(float(row[0]))
                coef_rows.append([float(c) for c in row[1:]])

            exps = np.array(exps)
            coef_cols = np.array(coef_rows).T

            idx = 0
            for j, n_shells in enumerate(counts):
                l = lmin + j
                for _ in range(n_shells):
                    basis[elem].append(
                        (l, exps.copy(), coef_cols[idx].copy())
                    )
                    idx += 1

    return basis


# ============================================================
# BUILD SHELL DICTS FOR LIBINT
# ============================================================

def build_shell_dicts(syms, coords_ang, basis_dict):

    shells = []

    for atom_idx, (sym, xyz_ang) in enumerate(zip(syms, coords_ang)):

        if sym not in basis_dict:
            raise KeyError(f"No basis for element {sym}")

        xyz_bohr = np.asarray(xyz_ang) * BOHR_PER_ANGSTROM

        for l, exps, coefs in basis_dict[sym]:
            shells.append(dict(
                sym=sym,
                atom_idx=atom_idx,
                l=int(l),
                exps=np.asarray(exps, dtype=float),
                coefs=np.asarray(coefs, dtype=float),
                center=xyz_bohr,
                pure=True
            ))

    return shells


# ============================================================
# AO COUNT
# ============================================================

def count_ao_from_shells(shells):
    return sum(2 * int(sh["l"]) + 1 for sh in shells)


# ============================================================
# AO RANGES PER ATOM
# ============================================================

def build_atom_ao_ranges(shells):

    atom_ranges = {}
    ao_counter = 0

    for sh in shells:
        atom_idx = sh["atom_idx"]
        nbf = 2 * int(sh["l"]) + 1

        if atom_idx not in atom_ranges:
            atom_ranges[atom_idx] = [ao_counter, ao_counter + nbf]
        else:
            atom_ranges[atom_idx][1] += nbf

        ao_counter += nbf

    return [tuple(atom_ranges[i]) for i in sorted(atom_ranges.keys())]


# ============================================================
# MO UTILITIES
# ============================================================

_NUM_RE = re.compile(r"""
    [\+\-]?
    (?:
        \d+\.\d* |
        \.\d+ |
        \d+
    )
    (?:[EeDd][\+\-]?\d+)?
""", re.VERBOSE)


def _extract_numbers(s):
    toks = _NUM_RE.findall(s)
    return [float(t.replace("D", "E").replace("d", "E")) for t in toks]


def read_mos_auto(path, n_ao_total, verbose=False):

    ext = os.path.splitext(path)[-1].lower()

    if ext == ".npz":
        d = np.load(path)
        C = csr_matrix((d["data"], d["indices"], d["indptr"]),
                       shape=d["shape"])
        eps = d["eps"]
        occ = d["occ"]

        if verbose:
            print(f"[MOs] Loaded NPZ: {C.shape}")

        return C, eps, occ

    return read_mos_txt_cc(path, n_ao_total, verbose=verbose)


def read_mos_txt(path, n_ao_total, verbose=False):

    t0 = time.perf_counter()

    blocks = []
    n_mo_total = 0

    def is_int_line(s):
        toks = s.split()
        return toks and all(tok.lstrip("+").isdigit() for tok in toks)

    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break

            s = line.strip()
            if not is_int_line(s):
                continue

            n_cols = len(s.split())

            f.readline()
            f.readline()

            for _ in range(n_ao_total):
                f.readline()

            blocks.append((n_mo_total, n_cols))
            n_mo_total += n_cols

    if n_mo_total == 0:
        raise RuntimeError("No MO blocks detected")

    C = np.zeros((n_ao_total, n_mo_total), dtype=np.float32)
    eps = np.zeros(n_mo_total)
    occ = np.zeros(n_mo_total)

    with open(path) as f:
        blk_idx = 0

        while True:
            line = f.readline()
            if not line:
                break

            s = line.strip()
            if not is_int_line(s):
                continue

            offset, n_cols = blocks[blk_idx]
            blk_idx += 1

            col_slice = slice(offset, offset + n_cols)

            # energies
            vals = _extract_numbers(f.readline())
            eps[col_slice] = vals[:n_cols]

            # occupations
            vals = _extract_numbers(f.readline())
            occ[col_slice] = vals[:n_cols]

            # coefficients
            for ao_row in range(n_ao_total):
                line_str = f.readline()
                toks = line_str.split()
                # The actual coefficients are always the last n_cols items on the line
                coef_toks = toks[-n_cols:]
                
                # Convert 'D' to 'E' for scientific notation
                vals = [float(x.replace("D", "E").replace("d", "E")) for x in coef_toks]
                C[ao_row, col_slice] = vals

    if verbose:
        dt = time.perf_counter() - t0
        print(f"[MOs] Parsed in {dt:.2f} s | C shape {C.shape}")

    return C, eps, occ

def read_mos_txt_fast(path, n_ao_total, verbose=False):
    t0 = time.perf_counter()

    # 1. Global Replace in C-speed: Read entire file into memory and fix 'D' to 'E'
    with open(path, 'r') as f:
        raw_text = f.read().replace('D', 'E').replace('d', 'E')

    # Split into lines once
    lines = raw_text.splitlines()

    C_blocks = []
    eps_list = []
    occ_list = []

    i = 0
    n_lines = len(lines)

    while i < n_lines:
        s = lines[i].strip()

        # 2. Fast check for the header line (avoids splitting every single line)
        if s and s[0].lstrip('+').isdigit():
            toks = s.split()
            
            if all(t.lstrip("+").isdigit() for t in toks):
                n_cols = len(toks)

                # Energies (Assuming numbers are the last n_cols items on the line)
                eps_list.extend(float(x) for x in lines[i+1].split()[-n_cols:])
                
                # Occupations
                occ_list.extend(float(x) for x in lines[i+2].split()[-n_cols:])

                # Coefficients block
                block_lines = lines[i+3 : i+3+n_ao_total]
                
                # 3. Fast list comprehension to flatten the block
                block_vals = [val for row in block_lines for val in row.split()[-n_cols:]]
                
                # Convert straight to a localized 2D NumPy array
                C_blocks.append(np.array(block_vals, dtype=np.float32).reshape(n_ao_total, n_cols))

                # Jump the index completely past this block
                i += 3 + n_ao_total
                continue

        i += 1

    if not C_blocks:
        raise RuntimeError("No MO blocks detected")

    # 4. Horizontally stack all blocks simultaneously 
    C = np.hstack(C_blocks)
    eps = np.array(eps_list, dtype=np.float32)
    occ = np.array(occ_list, dtype=np.float32)

    if verbose:
        dt = time.perf_counter() - t0
        print(f"[MOs] Parsed in {dt:.4f} s | C shape {C.shape}")

    return C, eps, occ

def read_mos_txt_cc(path, n_ao_total, verbose=False):
    t0 = time.perf_counter()
    
    # 1 line of Python. C++ handles everything else.
    C, eps, occ = libint_cpp.parse_cp2k_mos(path, n_ao_total)
    
    if verbose:
        dt = time.perf_counter() - t0
        print(f"[MOs] Parsed in {dt:.4f} s | C shape {C.shape}")
        
    return C, eps, occ

import collections

def parse_gth_soc_potentials(path, elements_to_parse):
    """Parses GTH potentials for SOC parameters."""
    ecp_dict = collections.defaultdict(lambda: {'so': []})
    needed_elements = set(elements_to_parse.keys())

    def _collect_coeffs(line_iter, n, init):
        coeffs = list(init)
        while len(coeffs) < n:
            line = next(line_iter).strip()
            if line and not line.startswith('#'):
                coeffs.extend([float(x) for x in line.split()])
        return coeffs

    try:
        with open(path, "r") as f:
            line_iter = iter(f.readlines())
    except FileNotFoundError:
        print(f"Warning: Potential file not found at {path}. SOC will be zero.")
        return ecp_dict

    for line in line_iter:
        if not needed_elements: break
        parts = line.strip().split()
        if not parts or parts[0] not in needed_elements: continue
        
        sym, q = parts[0], elements_to_parse.get(parts[0])
        if q is None or not any(f"q{q}" in p for p in parts): continue
        
        try:
            next(line_iter); next(line_iter) # Skip header
            n_soc_sets = int(next(line_iter).strip().split()[0])
            for l in range(n_soc_sets):
                proj_line = next(line_iter)
                while not proj_line.strip() or proj_line.strip().startswith('#'):
                    proj_line = next(line_iter)
                proj_parts = proj_line.split()
                r, nprj = float(proj_parts[0]), int(proj_parts[1])
                n_coeffs = nprj * (nprj + 1) // 2
                h = _collect_coeffs(line_iter, n_coeffs, proj_parts[2:])
                k = _collect_coeffs(line_iter, n_coeffs, []) if l > 0 else []
                ecp_dict[sym]['so'].append({'l': l, 'r': r, 'nprj': nprj, 'h_coeffs': h, 'k_coeffs': k})
            needed_elements.remove(sym)
        except Exception as e:
            continue
            
    return ecp_dict


