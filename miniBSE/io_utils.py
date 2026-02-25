import os
import re
import time
import numpy as np
import collections
from scipy.sparse import issparse, csr_matrix


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

    return read_mos_txt(path, n_ao_total, verbose=verbose)


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


