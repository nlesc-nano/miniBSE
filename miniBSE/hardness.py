import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull, distance_matrix

# Atomic Hardness values (eta) in eV. 
# Used for the Ohno-Klopman damping in the Coulomb kernel.
HARDNESS_DICT = {
    'h': 6.4299, 'he': 12.5449, 'li': 2.3746, 'be': 3.4968, 'b': 4.619, 'c': 5.7410,
    'n': 6.8624, 'o': 7.9854, 'f': 9.1065, 'ne': 10.2303, 'na': 2.4441, 'mg': 3.0146,
    'al': 3.5849, 'si': 4.1551, 'p': 4.7258, 's': 5.2960, 'cl': 5.8662, 'ar': 6.4366,
    'k': 2.3273, 'ca': 2.7587, 'sc': 2.8582, 'ti': 2.9578, 'v': 3.0573, 'cr': 3.1567,
    'mn': 3.2564, 'fe': 3.3559, 'co': 3.4556, 'ni': 3.555, 'cu': 3.6544, 'zn': 3.7542,
    'ga': 4.1855, 'ge': 4.6166, 'as': 5.0662, 'se': 5.4795, 'br': 5.9111, 'kr': 6.3418,
    'rb': 2.1204, 'sr': 2.5374, 'y': 2.6335, 'zr': 2.7297, 'nb': 2.8260, 'mo': 2.9221,
    'tc': 3.0184, 'ru': 3.1146, 'rh': 3.2107, 'pd': 3.3069, 'ag': 3.4032, 'cd': 3.4994,
    'in': 3.9164, 'sn': 4.3332, 'sb': 4.7501, 'te': 5.167, 'i': 5.5839, 'xe': 6.0009,
    'cs': 0.6829, 'ba': 0.9201, 'la': 1.1571, 'ce': 1.3943, 'pr': 1.6315, 'nd': 1.8686,
    'pm': 2.1056, 'sm': 2.3427, 'eu': 2.5798, 'gd': 2.8170, 'tb': 3.0540, 'dy': 3.2912,
    'ho': 3.5283, 'er': 3.7655, 'tm': 4.0026, 'yb': 4.2395, 'lu': 4.4766, 'hf': 4.7065,
    'ta': 4.9508, 'w': 5.1879, 're': 5.4256, 'os': 5.6619, 'ir': 5.900, 'pt': 6.1367,
    'au': 6.3741, 'hg': 6.6103, 'tl': 1.7043, 'pb': 1.9435, 'bi': 2.1785, 'po': 2.4158,
    'at': 2.6528, 'rn': 2.8899, 'fr': 0.9882, 'ra': 1.2819, 'ac': 1.3497, 'th': 1.4175,
    'pa': 1.9368, 'u': 2.2305, 'np': 2.5241, 'pu': 3.0436, 'am': 3.4169, 'cm': 3.4050,
    'bk': 3.9244, 'cf': 4.2181, 'es': 4.5116, 'fm': 4.8051, 'md': 5.0100, 'no': 5.3926,
    'lr': 5.4607
}

MATERIAL_DB = {
# (eps_inf, Bohr_diam_A, lattice_A, gap_exp_eV, eps_static, m_eff, E_LO_eV, gap_pbe_scalar_eV, gap_gw_scalar_eV)

# Halide Perovskites
"CSPBCL3": (4.0, 25.0, 5.60, 3.00, 20.0, 0.15, 0.022, 2.35, 3.50),
"CSPBBR3": (4.8, 35.0, 5.83, 2.30, 25.0, 0.12, 0.018, 1.65, 2.65),
"CSPBI3":  (5.1, 60.0, 6.20, 1.73, 30.0, 0.10, 0.015, 1.35, 2.15),
"MAPBI3":  (6.5, 45.0, 6.27, 1.55, 30.0, 0.10, 0.015, 1.55, 2.40),
"FAPBI3":  (6.2, 50.0, 6.36, 1.48, 28.0, 0.10, 0.015, 1.45, 2.30),

# II–VI Semiconductors
"ZNS":  (5.1, 25.0, 5.41, 3.60, 8.9, 0.28, 0.043, 2.10, 3.70),
"ZNSE": (5.9, 38.0, 5.67, 2.70, 8.6, 0.17, 0.031, 1.20, 2.70),
"ZNTE": (6.7, 52.0, 6.10, 2.26, 9.8, 0.12, 0.026, 1.05, 2.35),
"CDS":  (5.4, 30.0, 5.82, 2.42, 8.9, 0.16, 0.037, 1.15, 2.50),
"CDSE": (6.2, 56.0, 6.05, 1.74, 9.5, 0.13, 0.026, 0.60, 1.80),
"CDTE": (7.1, 73.0, 6.48, 1.44, 10.2, 0.10, 0.021, 0.55, 1.60),
"HGS":  (11.3, 50.0, 5.85, 0.50, 13.0, 0.20, 0.030, 0.00, 0.50),
"HGSE": (14.0, 460.0, 6.08, 0.00, 18.0, 0.04, 0.017, -0.40, 0.10),
"HGTE": (15.0, 400.0, 6.46, 0.00, 20.0, 0.03, 0.015, -0.60, 0.10),

# III–V Semiconductors
"ALP":  (7.5, 15.0, 5.46, 2.45, 10.0, 0.20, 0.050, 1.55, 2.50),
"ALAS": (8.2, 30.0, 5.66, 2.16, 10.1, 0.15, 0.049, 1.35, 2.20),
"ALSB": (10.2, 60.0, 6.14, 1.62, 12.0, 0.14, 0.036, 1.00, 1.65),
"GAP":  (9.1, 15.0, 5.45, 2.26, 11.1, 0.15, 0.049, 1.55, 2.40),
"GAAS": (10.9, 100.0, 5.65, 1.42, 13.1, 0.067, 0.036, 0.55, 1.40),
"GASB": (14.4, 200.0, 6.10, 0.73, 15.7, 0.04, 0.028, 0.10, 0.80),
"INP":  (9.6, 150.0, 5.87, 1.34, 12.4, 0.08, 0.042, 0.55, 1.40),
"INAS": (11.8, 340.0, 6.06, 0.35, 15.0, 0.023, 0.029, -0.15, 0.65),
"INSB": (15.7, 650.0, 6.48, 0.17, 17.9, 0.014, 0.023, -0.30, 0.35),

# IV–VI Semiconductors
"PBS":  (17.2, 200.0, 5.94, 0.41, 23.0, 0.09, 0.027, 0.15, 0.60),
"PBSE": (22.9, 460.0, 6.12, 0.27, 30.0, 0.07, 0.017, 0.05, 0.45),

"DEFAULT": (1.0, 1.0, 5.0, 0.0, 1.0, 1.0, 0.02, 0.00, 0.00)
}

# Core inorganic elements for each material (ignores organic ligands like MA/FA)
MATERIAL_ELEMENTS = {
    "CSPBCL3": ["Cs", "Pb", "Cl"], "CSPBBR3": ["Cs", "Pb", "Br"], "CSPBI3":  ["Cs", "Pb", "I"],
    "MAPBI3":  ["Pb", "I"], "FAPBI3":  ["Pb", "I"], 
    "ZNS":     ["Zn", "S"], "ZNSE":    ["Zn", "Se"], "ZNTE":    ["Zn", "Te"],
    "CDS":     ["Cd", "S"], "CDSE":    ["Cd", "Se"], "CDTE":    ["Cd", "Te"],
    "HGS":     ["Hg", "S"], "HGSE":    ["Hg", "Se"], "HGTE":    ["Hg", "Te"],
    "ALP":     ["Al", "P"], "ALAS":    ["Al", "As"], "ALSB":    ["Al", "Sb"],
    "GAP":     ["Ga", "P"], "GAAS":    ["Ga", "As"], "GASB":    ["Ga", "Sb"],
    "INP":     ["In", "P"], "INAS":    ["In", "As"], "INSB":    ["In", "Sb"],
    "PBS":     ["Pb", "S"], "PBSE":    ["Pb", "Se"]
}

# --- Add this function to the bottom of hardness.py ---
def estimate_gw_qp_gap(coords, atom_symbols, material_name, eps_out):
    """
    Computes the parameter-free Quasiparticle Scissor using tabulated bulk GW data
    and scaling it dynamically for dielectric confinement via surface polarization.
    """
    if material_name is None:
        print("  [Warning] Material not specified. Cannot compute GW scaling.")
        return None
        
    m_name = material_name.upper()
    if m_name not in MATERIAL_DB:
        print(f"  [Warning] Material {m_name} not found in MATERIAL_DB. GW estimation failed.")
        return None
        
    entry = MATERIAL_DB[m_name]
    if len(entry) < 9:
        print(f"  [Warning] MATERIAL_DB entry for {m_name} is outdated. GW estimation failed.")
        return None
        
    eps_inf = entry[0]
    gap_pbe_scalar = entry[7]
    gap_gw_scalar = entry[8]
    
    if gap_gw_scalar == 0.0:
        print(f"  [Warning] Missing GW bulk gap for {m_name}. GW estimation failed.")
        return None
        
    metrics = get_cluster_size_metrics(coords, atom_symbols, material_name)
    R_QD_ang = metrics['R_eff_hull']
    
    # 1. Bulk GW Correction (Many-Body Shift fixing the DFT Error)
    delta_bulk_qp = gap_gw_scalar - gap_pbe_scalar
    
    # 2. Size-dependent Surface Polarization (Dielectric Confinement)
    # Formula: Sigma_pol = (14.4 * 0.8 / R_QD) * (1/eps_out - 1/eps_inf)
    sigma_pol = (11.52 / R_QD_ang) * ((1.0 / eps_out) - (1.0 / eps_inf))
    
    # Total Scissor to apply to the raw DFT virtual states
    total_scissor = delta_bulk_qp + sigma_pol
    
    print(f"\n  [Scaled GW Model] Quasiparticle Correction for {m_name}:")
    print(f"    Cluster Radius (R_QD) : {R_QD_ang:.2f} Å")
    print(f"    Bulk PBE Gap          : {gap_pbe_scalar:.3f} eV")
    print(f"    Bulk GW Gap           : {gap_gw_scalar:.3f} eV")
    print(f"    -> Bulk Shift         : {delta_bulk_qp:+.3f} eV")
    print(f"    -> Polarization Shift : {sigma_pol:+.3f} eV  (eps_out={eps_out}, eps_inf={eps_inf})")
    print(f"    ==> Total GW Scissor  : {total_scissor:+.3f} eV")
    
    return total_scissor

def compute_delta_xc(material):

    entry = MATERIAL_DB.get(material.upper(), None)

    if entry is None:
        return 0.0

    # Experimental bulk gap
    gap_exp = entry[3]

    # If PBE gap is not present return zero correction
    if len(entry) < 8:
        print(f"[Δxc] Warning: no PBE gap for {material}. Using Δxc = 0.")
        return 0.0

    gap_pbe = entry[7]

    delta_xc = gap_exp - gap_pbe

    print(f"[Δxc] Bulk correction for {material}: {delta_xc:.3f} eV")

    return max(delta_xc, 0.0)

def get_cluster_size_metrics(coords_ang, atom_symbols=None, material_name=None):
    """Robust, rotation-invariant size metrics for nanoclusters/QDs (Angstrom)."""
    coords = np.asarray(coords_ang, dtype=float)

    if atom_symbols is not None and material_name is not None:
        m_name = material_name.upper()
        if m_name in MATERIAL_ELEMENTS:
            core_elements = [el.lower() for el in MATERIAL_ELEMENTS[m_name]]
            core_coords = [coords[i] for i, sym in enumerate(atom_symbols) if sym.lower() in core_elements]
            if len(core_coords) > 0:
                coords = np.array(core_coords)

    if len(coords) < 2:
        return {'R_eff_hull': 1.0, 'diameter_hull': 2.0}

    com = np.mean(coords, axis=0)
    r_vec = coords - com
    Rg = np.sqrt(np.mean(np.sum(r_vec**2, axis=1)))
    R_eff_gyr = Rg * np.sqrt(5.0 / 3.0)

    try:
        hull = ConvexHull(coords, qhull_options='QJ')
        R_eff_hull = (3.0 * hull.volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    except Exception:
        R_eff_hull = R_eff_gyr

    return {'R_eff_hull': R_eff_hull, 'diameter_hull': 2.0 * R_eff_hull}

# =====================================================================
# Model 1: Classic MNOK Kernel (sTDA style)
# =====================================================================
def build_gamma(atom_symbols, coords, alpha, beta=0.0, eta_dict=HARDNESS_DICT):
    """
    Standard Ohno-Klopman kernel. 
    `alpha` scales the entire matrix (macroscopic screening, e.g., 1/eps_inf).
    `beta` [0.0 to 1.0] applies exact-exchange Hubbard stiffening to the diagonal.
    """
    BOHR_TO_ANG = 0.52917721
    HA_TO_EV = 27.211386
    coords_au = coords / BOHR_TO_ANG
    r_mat_au = squareform(pdist(coords_au))
    
    etas_ev = np.array([eta_dict.get(s.lower(), 5.0) for s in atom_symbols])
    etas_au = etas_ev / HA_TO_EV
    a_au = 1.0 / etas_au
    damp_mat_au = 0.5 * (a_au[:, np.newaxis] + a_au[np.newaxis, :])
    
    gamma_au = 1.0 / np.sqrt(r_mat_au**2 + damp_mat_au**2)
    gamma_ev = gamma_au * HA_TO_EV
    
    # Apply macroscopic screening
    gamma_mat = alpha * gamma_ev
    
    # Apply short-range beta stiffening to the diagonal
    if beta > 0.0:
        for i, sym in enumerate(atom_symbols):
            sym_lower = sym.lower()
            eta_A = etas_ev[i]
            # Fetch Mann's F0, fallback to 2.1 * eta_A if exotic element
            u_bare_A = U_BARE_DICT.get(sym_lower, 2.1 * eta_A)
            
            # The diagonal perfectly interpolates between relaxed hardness and bare Hubbard U
            gamma_mat[i, i] = eta_A + beta * (u_bare_A - eta_A)
            
    return gamma_mat

# =====================================================================
# Model 2: Universal Tunable Resta-MNOK Kernel (Dual-Screening)
# =====================================================================
# =====================================================================
# Model 2: Universal Tunable Resta-MNOK Kernel (Dual-Screening)
# =====================================================================
def build_resta_mnok(atom_symbols, coords, alpha, material_name, eps_out=2.0, eta_dict=HARDNESS_DICT):

    BOHR_TO_ANG = 0.52917721
    HA_TO_EV = 27.211386

    # --------------------------------------------------
    # 1. Geometry and Size Metrics
    # --------------------------------------------------
    metrics = get_cluster_size_metrics(coords, atom_symbols, material_name)
    R_QD_ang = metrics['R_eff_hull']

    m_name = material_name.upper() if material_name else "DEFAULT"
    entry = MATERIAL_DB.get(m_name, MATERIAL_DB["DEFAULT"])

    eps_inf_bulk = entry[0]   # electronic dielectric
    eps_0_bulk   = entry[4]   # static dielectric

    # --------------------------------------------------
    # 2. Core atoms for geometry
    # --------------------------------------------------
    core_coords = coords
    if m_name in MATERIAL_ELEMENTS:
        core_elements = [el.lower() for el in MATERIAL_ELEMENTS[m_name]]
        core_coords = np.array(
            [coords[i] for i, sym in enumerate(atom_symbols) if sym.lower() in core_elements]
        )

    if len(core_coords) > 1:
        r_core_ang = squareform(pdist(core_coords))
        np.fill_diagonal(r_core_ang, np.inf)
        d_NN_ang = np.median(np.min(r_core_ang, axis=1))
    else:
        d_NN_ang = 2.5

    # --------------------------------------------------
    # 3. Screening length (Thomas-Fermi / Resta Model)
    # --------------------------------------------------
    d_NN_au = d_NN_ang / BOHR_TO_ANG

    # Tie screening length to bulk dielectric response (Microscopic)
    k_s_au = np.sqrt(max(0.0, eps_inf_bulk - 1.0)) / d_NN_au
    
    # Calculate the physical screening length in Angstroms
    lambda_s_ang = (1.0 / k_s_au) * BOHR_TO_ANG if k_s_au > 0 else 0.0

    # --------------------------------------------------
    # 4. Dielectric confinement (Self-Consistent Scaling)
    # --------------------------------------------------
    # Replace Exciton Bohr radius (a_B) with the Plasma Screening Length (lambda_s)
    confinement_ratio = R_QD_ang / lambda_s_ang if lambda_s_ang > 0 else np.inf

    # Electronic screening (used for QP correction)
    eps_eff_inf = eps_out + (eps_inf_bulk - eps_out) / (1.0 + lambda_s_ang / R_QD_ang)

    # Static screening (used for BSE electron-hole interaction)
    eps_eff_0 = eps_out + (eps_0_bulk - eps_out) / (1.0 + lambda_s_ang / R_QD_ang)

    print("\n    [Kernel: Dual-Screening Resta-MNOK]")
    print(f"    Material            = {m_name}")
    print(f"    R_QD (hull_eff)     = {R_QD_ang:.3f} Å")
    print(f"    R / lambda_s        = {confinement_ratio:.3f}")
    print(f"    ε_eff (Electronic)  = {eps_eff_inf:.3f} (bulk limit {eps_inf_bulk})")
    print(f"    ε_eff (Static)      = {eps_eff_0:.3f} (bulk limit {eps_0_bulk})")
    print(f"    Screening length    = {lambda_s_ang/BOHR_TO_ANG:.3f} a.u. ({lambda_s_ang:.3f} Å)")
    print()

    # --------------------------------------------------
    # 5. Build MNOK matrices
    # --------------------------------------------------
    coords_au = coords / BOHR_TO_ANG
    r_mat_au = squareform(pdist(coords_au))

    etas_au = np.array([eta_dict[s.lower()] for s in atom_symbols]) / HA_TO_EV
    a_au = 1.0 / etas_au

    damp_mat_au = 0.5 * (a_au[:, np.newaxis] + a_au[np.newaxis, :])
    mnok_denom_au = np.sqrt(r_mat_au**2 + damp_mat_au**2)

    # --------------------------------------------------
    # 6. Dual screening kernels
    # --------------------------------------------------
    c_inf = 1.0 / eps_eff_inf
    gamma_qp_au = (c_inf + (1.0 - c_inf) * np.exp(-k_s_au * r_mat_au)) / mnok_denom_au

    c_static = 1.0 / eps_eff_0
    gamma_bse_au = (c_static + (1.0 - c_static) * np.exp(-k_s_au * r_mat_au)) / mnok_denom_au

    return gamma_qp_au * HA_TO_EV, gamma_bse_au * HA_TO_EV


def estimate_brus_qp_gap(material_name, coords, atom_symbols):
    """Analytically estimates the Quantum Confined QP Gap using the Brus equation with Non-Parabolic corrections."""
    m_name = material_name.upper() if material_name else "DEFAULT"
    entry = MATERIAL_DB.get(m_name, MATERIAL_DB["DEFAULT"])
    
    E_bulk = entry[3]
    m_eff = entry[5]
    
    if E_bulk == 0.0 or m_eff == 0.0:
        print(f"  [Warning] Missing bulk gap or effective mass for {m_name}. Brus estimation failed.")
        return None
        
    metrics = get_cluster_size_metrics(coords, atom_symbols, material_name)
    R_QD_ang = metrics['R_eff_hull']
    
    # 1. Standard Parabolic Kinetic Confinement Energy
    E_conf_parabolic = (3.80998 * np.pi**2) / (m_eff * (R_QD_ang ** 2))
    
    # 2. Non-Parabolicity Correction (Hyperbolic Band Model)
    # Highly necessary for narrow gap materials like InAs, PbS, PbSe (E_bulk < 1.0 eV)
    if E_bulk < 2.0:
        predicted_gap = np.sqrt(E_bulk**2 + 2 * E_bulk * E_conf_parabolic)
        is_non_parabolic = True
    else:
        predicted_gap = E_bulk + E_conf_parabolic
        is_non_parabolic = False
        
    print(f"\n  [Brus Model] Estimating Confinement for {m_name}:")
    print(f"    Radius (R_QD)    : {R_QD_ang:.2f} Å")
    print(f"    Bulk Gap         : {E_bulk:.3f} eV")
    print(f"    Effective Mass   : {m_eff:.3f} m_e")
    
    if is_non_parabolic:
        print(f"    Model Used       : Hyperbolic (Non-Parabolic)")
        print(f"    Raw Parabolic dE : +{E_conf_parabolic:.3f} eV (Unphysical, applying correction...)")
    else:
        print(f"    Model Used       : Standard Parabolic")
        print(f"    Confinement (dE) : +{E_conf_parabolic:.3f} eV")
        
    print(f"    Predicted QP Gap : {predicted_gap:.3f} eV")
    
    return predicted_gap

