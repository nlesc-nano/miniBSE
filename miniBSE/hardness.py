import numpy as np
import time
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

# Material database:
# (eps_inf, Bohr_diameter_A, lattice_A, bulk_gap_eV,
#  eps_static, m_eff (reduced), E_LO_eV)
MATERIAL_DB = {
    # Halide Perovskites
    "CSPBCL3":  (4.0, 25.0, 5.60, 3.00, 20.0, 0.15, 0.022),
    "CSPBBR3":  (4.8, 35.0, 5.83, 2.30, 25.0, 0.12, 0.018),
    "CSPBI3":   (5.1, 60.0, 6.20, 1.73, 30.0, 0.10, 0.015),
    "MAPBI3":   (6.5, 45.0, 6.27, 1.55, 30.0, 0.10, 0.015),
    "FAPBI3":   (6.2, 50.0, 6.36, 1.48, 28.0, 0.10, 0.015),

    # II–VI Semiconductors
    "ZNS":      (5.1, 25.0, 5.41, 3.60, 8.9, 0.28, 0.043),
    "ZNSE":     (5.9, 38.0, 5.67, 2.70, 8.6, 0.17, 0.031),
    "ZNTE":     (6.7, 52.0, 6.10, 2.26, 9.8, 0.12, 0.026),
    "CDS":      (5.4, 30.0, 5.82, 2.42, 8.9, 0.16, 0.037),
    "CDSE":     (6.2, 56.0, 6.05, 1.74, 9.5, 0.13, 0.026),
    "CDTE":     (7.1, 73.0, 6.48, 1.44, 10.2, 0.10, 0.021),
    "HGS":      (11.3, 50.0, 5.85, 0.50, 13.0, 0.20, 0.030),
    "HGSE":     (14.0, 460.0, 6.08, 0.00, 18.0, 0.04, 0.017),
    "HGTE":     (15.0, 400.0, 6.46, 0.00, 20.0, 0.03, 0.015),

    # III–V Semiconductors
    "ALP":      (7.5, 15.0, 5.46, 2.45, 10.0, 0.20, 0.050),
    "ALAS":     (8.2, 30.0, 5.66, 2.16, 10.1, 0.15, 0.049),
    "ALSB":     (10.2, 60.0, 6.14, 1.62, 12.0, 0.14, 0.036),
    "GAP":      (9.1, 15.0, 5.45, 2.26, 11.1, 0.15, 0.049),
    "GAAS":     (10.9, 100.0, 5.65, 1.42, 13.1, 0.067, 0.036),
    "GASB":     (14.4, 200.0, 6.10, 0.73, 15.7, 0.04, 0.028),
    "INP":      (9.6, 150.0, 5.87, 1.34, 12.4, 0.08, 0.042),
    "INAS":     (11.8, 340.0, 6.06, 0.35, 15.0, 0.023, 0.029),
    "INSB":     (15.7, 650.0, 6.48, 0.17, 17.9, 0.014, 0.023),

    # IV–VI Semiconductors
    "PBS":      (17.2, 200.0, 5.94, 0.41, 23.0, 0.09, 0.027),
    "PBSE":     (22.9, 460.0, 6.12, 0.27, 30.0, 0.07, 0.017),

    "DEFAULT":  (1.0, 1.0, 5.0, 0.0, 1.0, 1.0, 0.02),

    # Solvents (keep 4-tuple)
    "VACUUM":          (1.0,  1.0, 1.0, 0.0),
    "TOLUENE":         (2.38, 1.0, 1.0, 0.0),
    "HEXANE":          (1.89, 1.0, 1.0, 0.0),
    "HEXDECANE":       (2.05, 1.0, 1.0, 0.0),
    "OCTADECENE":      (2.25, 1.0, 1.0, 0.0),
    "CHLOROFORM":      (4.81, 1.0, 1.0, 0.0),
    "DICHLOROMETHANE": (8.93, 1.0, 1.0, 0.0),
}

# Core inorganic elements for each material (ignores organic ligands like MA/FA)
MATERIAL_ELEMENTS = {
    "CSPBCL3": ["Cs", "Pb", "Cl"],
    "CSPBBR3": ["Cs", "Pb", "Br"],
    "CSPBI3":  ["Cs", "Pb", "I"],
    "MAPBI3":  ["Pb", "I"], 
    "FAPBI3":  ["Pb", "I"], 
    "ZNS":     ["Zn", "S"],
    "ZNSE":    ["Zn", "Se"],
    "ZNTE":    ["Zn", "Te"],
    "CDS":     ["Cd", "S"],
    "CDSE":    ["Cd", "Se"],
    "CDTE":    ["Cd", "Te"],
    "HGS":     ["Hg", "S"],
    "HGSE":    ["Hg", "Se"],
    "HGTE":    ["Hg", "Te"],
    "ALP":     ["Al", "P"],
    "ALAS":    ["Al", "As"],
    "ALSB":    ["Al", "Sb"],
    "GAP":     ["Ga", "P"],
    "GAAS":    ["Ga", "As"],
    "GASB":    ["Ga", "Sb"],
    "INP":     ["In", "P"],
    "INAS":    ["In", "As"],
    "INSB":    ["In", "Sb"],
    "PBS":     ["Pb", "S"],
    "PBSE":    ["Pb", "Se"]
}

def compute_polaron_dielectric(eps_inf, eps_static):
    if eps_static <= eps_inf:
        return eps_inf
    return (eps_inf * eps_static / (eps_static - eps_inf)) * np.log(eps_static / eps_inf)

def get_cluster_size_metrics(coords_ang, atom_symbols=None, material_name=None):
    """Robust, rotation-invariant size metrics for nanoclusters/QDs (Angstrom).
       Filters for inorganic core atoms if atom_symbols and material_name are provided.
    """
    coords = np.asarray(coords_ang, dtype=float)

    # Filter out ligands if symbols and material are provided
    if atom_symbols is not None and material_name is not None:
        m_name = material_name.upper()
        if m_name in MATERIAL_ELEMENTS:
            core_elements = [el.lower() for el in MATERIAL_ELEMENTS[m_name]]
            core_coords = [
                coords[i] for i, sym in enumerate(atom_symbols) 
                if sym.lower() in core_elements
            ]
            if len(core_coords) > 0:
                coords = np.array(core_coords)
                print(f"    [Geometry] Filtered core atoms: {len(coords)} / {len(coords_ang)} total atoms.")
            else:
                print(f"    [Geometry] WARNING: No core atoms matched for {m_name}. Using all atoms.")

    if len(coords) < 2:
        return {'R_eff_hull': 1.0, 'diameter_hull': 2.0, 'volume_hull_ang3': 4.1888,
                'Rg': 0.5, 'diameter_max': 0.0}

    com = np.mean(coords, axis=0)
    r_vec = coords - com
    Rg = np.sqrt(np.mean(np.sum(r_vec**2, axis=1)))
    R_eff_gyr = Rg * np.sqrt(5.0 / 3.0)

    try:
        hull = ConvexHull(coords, qhull_options='QJ')
        vol_ang3 = hull.volume
        R_eff_hull = (3.0 * vol_ang3 / (4.0 * np.pi)) ** (1.0 / 3.0)
        diameter_hull = 2.0 * R_eff_hull
    except Exception:
        vol_ang3 = 0.0
        R_eff_hull = R_eff_gyr
        diameter_hull = 2.0 * R_eff_gyr

    diameter_max = np.max(distance_matrix(coords, coords)) if len(coords) > 1 else 0.0

    return {
        'Rg': Rg,
        'R_eff_gyr': R_eff_gyr,
        'R_eff_hull': R_eff_hull,
        'volume_hull_ang3': vol_ang3,
        'diameter_hull': diameter_hull,
        'diameter_max': diameter_max
    }

def get_auto_alpha(coords, material_name=None, eps_bulk=None, L_scale=None):
    dmin = np.min(coords, axis=0)
    dmax = np.max(coords, axis=0)
    diameter = np.linalg.norm(dmax - dmin)
    R_eff = diameter / 2.0

    m_name = material_name.upper() if material_name else "DEFAULT"
    db_eps, db_bohr, db_alat, db_gap = MATERIAL_DB.get(m_name, MATERIAL_DB["DEFAULT"])
    final_eps = eps_bulk if eps_bulk is not None else db_eps

    if final_eps > 1.0:
        L_diel = db_alat / (2.0 * np.sqrt(final_eps - 1.0))
    else:
        L_diel = 5.0

    alpha_inf = 1.0 / final_eps
    alpha = alpha_inf + (1.0 - alpha_inf) * np.exp(-R_eff / L_diel)
    return alpha, diameter, final_eps, L_diel

def build_gamma(atom_symbols, coords, alpha, eta_dict=HARDNESS_DICT):
    BOHR_TO_ANG = 0.52917721
    HA_TO_EV = 27.211386
    coords_au = coords / BOHR_TO_ANG
    r_mat_au = squareform(pdist(coords_au))
    etas_au = np.array([eta_dict[s.lower()] for s in atom_symbols]) / HA_TO_EV
    a_au = 1.0 / etas_au
    damp_mat_au = 0.5 * (a_au[:, np.newaxis] + a_au[np.newaxis, :])
    gamma_au = 1.0 / np.sqrt(r_mat_au**2 + damp_mat_au**2)
    return alpha * gamma_au * HA_TO_EV

def compute_polaron_radius_nm(m_eff, E_LO_eV):
    if m_eff <= 0 or E_LO_eV <= 0:
        return 0.0
    return 0.53 / np.sqrt(m_eff) * np.sqrt(1.0 / E_LO_eV)

# =====================================================================
# NEW: Universal Beta-Tuned Yukawa-MNOK Kernel
# =====================================================================
def build_yukawa_mnok(atom_symbols, coords, alpha, material_name, eta_dict=HARDNESS_DICT):
    """
    Computes the Universal beta-tuned Yukawa-MNOK potential.
    
    - alpha: Used here as the dimensionless tuning parameter (beta) for the 
             dielectric confinement turn-on rate.
    - material_name: Used to fetch the bulk dielectric and Bohr radius.
    """
    BOHR_TO_ANG = 0.52917721
    HA_TO_EV = 27.211386
    
    # 1. Geometry: Get the Quantum Dot Radius (R_QD) from Convex Hull volume
    metrics = get_cluster_size_metrics(coords, atom_symbols, material_name)
    R_QD_ang = metrics['R_eff_hull']
    
    # 2. Material Data: Fetch eps_bulk and Bohr Exciton Radius
    m_name = material_name.upper() if material_name else "DEFAULT"
    entry = MATERIAL_DB.get(m_name, MATERIAL_DB["DEFAULT"])
    
    eps_bulk = entry[0]
    db_bohr_diam_ang = entry[1]
    
    # The Exciton Bohr Radius (a_B) is half the database diameter
    a_B_ang = db_bohr_diam_ang / 2.0 if db_bohr_diam_ang > 0 else 10.0
    
    # 3. Size-Dependent Effective Dielectric Constant (Resta/Geometric Model)
    # Here, your CLI argument "alpha" is mapped to the physics parameter "beta"
    beta_tune = alpha 
    confinement_ratio = R_QD_ang / a_B_ang
    
    # Exponential turn-on function
    interp_factor = 1.0 - np.exp(-beta_tune * confinement_ratio)
    eps_eff = 1.0 + (eps_bulk - 1.0) * interp_factor
    
    # 4. Setup the Final Screened Kernel Parameters
    c_screen = 1.0 / eps_eff
    
    # Inverse screening length for the exponential tail (in atomic units)
    a_B_au = a_B_ang / BOHR_TO_ANG
    k_s = 1.0 / a_B_au if a_B_au > 0 else 0.0
    
    # --- Informative Print Statements ---
    print("\n    [Kernel: Universal β-Yukawa-MNOK]")
    print(f"    Material            = {m_name}")
    print(f"    R_QD (hull_eff)     = {R_QD_ang:.3f} Å")
    print(f"    a_B  (exciton)      = {a_B_ang:.3f} Å")
    print(f"    Confinement Ratio   = {confinement_ratio:.3f} (R_QD / a_B)")
    print(f"    β (tuning param)    = {beta_tune:.3f}")
    print(f"    ε_bulk (database)   = {eps_bulk:.3f}")
    print(f"    ε_eff  (computed)   = {eps_eff:.3f}")
    print(f"    Prefactor (1/ε_eff) = {c_screen:.4f}")
    print(f"    Yukawa Tail (k_s)   = {k_s:.4f} a.u.^-1\n")
    
    # 5. Build Distance and Hardness Matrices
    coords_au = coords / BOHR_TO_ANG
    r_mat_au = squareform(pdist(coords_au))
    
    etas_au = np.array([eta_dict[s.lower()] for s in atom_symbols]) / HA_TO_EV
    a_au = 1.0 / etas_au
    damp_mat_au = 0.5 * (a_au[:, np.newaxis] + a_au[np.newaxis, :])
    
    # 6. The Screened Yukawa-MNOK Math
    mnok_denom_au = np.sqrt(r_mat_au**2 + damp_mat_au**2)
    gamma_au = c_screen * np.exp(-k_s * r_mat_au) / mnok_denom_au
    
    return gamma_au * HA_TO_EV


def compute_sos_polarizability(mu_ia_x, mu_ia_y, mu_ia_z, delta_e_ia_ev):
    delta_e_au = delta_e_ia_ev / 27.211386
    ax = 2.0 * np.sum((mu_ia_x ** 2) / delta_e_au)
    ay = 2.0 * np.sum((mu_ia_y ** 2) / delta_e_au)
    az = 2.0 * np.sum((mu_ia_z ** 2) / delta_e_au)
    return (ax + ay + az) / 3.0

def eps_from_clausius_mossotti(alpha_au3, vol_ang3):
    if vol_ang3 <= 0:
        return 1.01
    bohr3_to_ang3 = (0.52917721) ** 3
    alpha_vol_ang3 = alpha_au3 * bohr3_to_ang3
    f = (4.0 * np.pi / 3.0) * (alpha_vol_ang3 / vol_ang3)
    if f >= 1.0:
        f = 0.999
    eps = (1.0 + 2.0 * f) / (1.0 - f)
    return max(eps, 1.01)

def compute_alpha_polariz_screening(coords_ang, mu_ia_x, mu_ia_y, mu_ia_z,
                                    delta_e_ia_ev, atom_symbols=None, material_name=None,
                                    eps_bulk=None, n_transitions=None):
    metrics = get_cluster_size_metrics(coords_ang, atom_symbols, material_name)
    vol = metrics['volume_hull_ang3']
    if vol <= 0:
        vol = (4.0 * np.pi / 3.0) * metrics['R_eff_hull'] ** 3

    alpha_iso = compute_sos_polarizability(mu_ia_x, mu_ia_y, mu_ia_z, delta_e_ia_ev)
    eps_target = eps_from_clausius_mossotti(alpha_iso, vol)

    if eps_bulk is not None and eps_bulk > eps_target:
        eps_target = 0.7 * eps_target + 0.3 * eps_bulk   

    alpha = 1.0 / eps_target

    print(f"    [Screening-Polariz] α_iso = {alpha_iso:.2f} bohr³ | V_hull = {vol:.1f} Å³")
    print(f"    [Screening-Polariz] R_eff = {metrics['R_eff_hull']:.2f} Å | ε_eff = {eps_target:.3f} | α = {alpha:.4f}")

    if n_transitions is not None and n_transitions < 2000:
        print("    [WARNING] Active-space truncated SOS polarizability detected (n_occ×n_virt < 2000).")
        print("              Results are approximate; consider larger --n-occ/--n-virt or --e_thresh.")
    return alpha, metrics['diameter_hull'], eps_target, alpha_iso, 0.0

def compute_sos_screening(eps, homo_index, n_occ, n_virt, mu_ia_x, mu_ia_y, mu_ia_z,
                          coords, atom_symbols=None, material_name=None, eps_bulk=None,
                          screening_mode="auto", manual_alpha=None):
    
    if manual_alpha is not None:
        metrics = get_cluster_size_metrics(coords, atom_symbols, material_name)
        m_name = material_name.upper() if material_name else "DEFAULT"
        entry = MATERIAL_DB.get(m_name, MATERIAL_DB["DEFAULT"])
        
        db_eps = entry[0]
        db_gap = entry[3]
        final_eps_bulk = eps_bulk if eps_bulk is not None else db_eps
        eps_eff = 1.0 / manual_alpha if manual_alpha != 0 else float('inf')
        
        print("    [Screening] MANUAL OVERRIDE active. Computation shut off.")
        print(f"    [Screening-Manual] α = {manual_alpha:.4f} | ε_eff = {eps_eff:.3f}")
        
        return manual_alpha, metrics['diameter_hull'], eps_eff, 0.0, db_gap, final_eps_bulk

    n_trans = n_occ * n_virt
    occ_e = eps[homo_index - n_occ + 1 : homo_index + 1]
    virt_e = eps[homo_index + 1 : homo_index + 1 + n_virt]
    delta_e_ia_ev = virt_e[None, :] - occ_e[:, None]

    print(f"    [Screening] Mode = {screening_mode.upper()} | Active transitions = {n_trans}")

    if screening_mode.upper() == "GEOMETRIC" or (screening_mode.upper() == "AUTO" and n_trans < 2000):
        metrics = get_cluster_size_metrics(coords, atom_symbols, material_name)
        R_eff = metrics['R_eff_hull']
        diameter = metrics['diameter_hull']

        m_name = material_name.upper() if material_name else "DEFAULT"
        entry = MATERIAL_DB.get(m_name, MATERIAL_DB["DEFAULT"])
        db_eps, _, db_alat, db_gap = entry[:4]          

        final_eps_bulk = eps_bulk if eps_bulk is not None else db_eps
        L_scale = db_alat * 1.5 if db_alat > 0 else 10.0
        ratio = R_eff / L_scale if L_scale > 0 else 0.0

        interp_factor = 1.0 - np.exp(-R_eff / L_scale)
        eps_eff = 1.0 + (final_eps_bulk - 1.0) * interp_factor
        alpha = 1.0 / eps_eff

        print(f"    [Screening-Geometric] R_eff          = {R_eff:6.2f} Å")
        print(f"    [Screening-Geometric] L_scale        = {L_scale:6.2f} Å")
        print(f"    [Screening-Geometric] R_eff / L_scale = {ratio:6.2f}")
        print(f"    [Screening-Geometric] ε_bulk         = {final_eps_bulk:6.2f}")
        print(f"    [Screening-Geometric] ε_eff          = {eps_eff:6.3f} | α = {alpha:.4f}")

        return alpha, diameter, eps_eff, 0.0, db_gap, final_eps_bulk

    elif screening_mode in ("polariz", "hybrid"):
        alpha_p, diam, eps_p, alpha_iso, _ = compute_alpha_polariz_screening(
            coords, mu_ia_x, mu_ia_y, mu_ia_z, delta_e_ia_ev,
            atom_symbols, material_name, eps_bulk, n_trans)

        if screening_mode == "hybrid":
            alpha_g, _, eps_g, _, _ = compute_sos_screening(eps, homo_index, n_occ, n_virt,
                                                            mu_ia_x, mu_ia_y, mu_ia_z, coords,
                                                            atom_symbols, material_name, eps_bulk, "geometric")
            alpha = 0.5 * (alpha_p + alpha_g)
            eps_target = 1.0 / alpha
            print(f"    [Screening-Hybrid] Final α = {alpha:.4f} (blended)")
        else:
            alpha = alpha_p
            eps_target = eps_p

        return alpha, diam, eps_target, alpha_iso, 0.0

    elif screening_mode.lower() in ["dielectric_conf", "dielectric_confinement"]:
        m_name = material_name.upper() if material_name else "DEFAULT"
        entry = MATERIAL_DB.get(m_name, MATERIAL_DB["DEFAULT"])
        db_eps, _, _, db_gap = entry[:4]
    
        final_eps_bulk = eps_bulk if eps_bulk is not None else db_eps
    
        alpha, diam, eps_eff, _ = compute_alpha_dielectric_confinement(
            coords,
            atom_symbols=atom_symbols,
            material_name=material_name,
            eps_bulk=eps_bulk,
            eps_out="VACUUM"
        )
    
        return alpha, diam, eps_eff, 0.0, db_gap, final_eps_bulk

    else:
        raise ValueError(f"Unknown screening_mode '{screening_mode}'")

def compute_alpha_dielectric_confinement(coords_ang,
                                         atom_symbols=None,
                                         material_name=None,
                                         eps_bulk=None,
                                         eps_out=1.0):
    metrics = get_cluster_size_metrics(coords_ang, atom_symbols, material_name)
    R_eff = metrics['R_eff_hull']       
    R_eff_nm = R_eff / 10.0            

    m_name = material_name.upper() if material_name else "DEFAULT"
    entry = MATERIAL_DB.get(m_name, MATERIAL_DB["DEFAULT"])

    if len(entry) >= 7:
        eps_inf, db_bohr, db_alat, db_gap, eps_static, m_eff, E_LO = entry
    else:
        eps_inf, db_bohr, db_alat, db_gap = entry[:4]
        eps_static = eps_inf
        m_eff = 1.0
        E_LO = 0.02

    eps_inf_final = eps_bulk if eps_bulk is not None else eps_inf

    if isinstance(eps_out, str):
        eps_out_name = eps_out.upper()
        eps_out = MATERIAL_DB.get(eps_out_name, MATERIAL_DB["VACUUM"])[0]

    if eps_static > eps_inf_final:
        eps_pol = (eps_inf_final * eps_static / (eps_static - eps_inf_final)) * \
                  np.log(eps_static / eps_inf_final)
    else:
        eps_pol = eps_inf_final

    if m_eff > 0 and E_LO > 0:
        Rp_nm = 0.53 / np.sqrt(m_eff) * np.sqrt(1.0 / E_LO)
    else:
        Rp_nm = 0.0

    aB_nm = db_bohr / 20.0 if db_bohr > 0 else 1.0
    nonlocal_factor = (R_eff_nm**2) / (R_eff_nm**2 + aB_nm**2)
    eps_in = eps_inf_final + (eps_pol - eps_inf_final) * nonlocal_factor
    eps_eff = 0.5 * (eps_in + eps_out)
    alpha = 1.0 / eps_eff if eps_eff > 0 else 1.0

    print("\n    [Screening-Dielectric-Confinement-Polaron]")
    print(f"    R_eff (hull)        = {R_eff_nm:6.3f} nm")
    print(f"    a_B (exciton)       = {aB_nm:6.3f} nm")
    print(f"    nonlocal factor     = {nonlocal_factor:6.3f}")
    print(f"    ε_inf               = {eps_inf_final:6.3f}")
    print(f"    ε_static            = {eps_static:6.3f}")
    print(f"    ε_pol               = {eps_pol:6.3f}")
    print(f"    m_eff               = {m_eff:6.3f}")
    print(f"    E_LO (eV)           = {E_LO:6.4f}")
    print(f"    Rp (bulk)           = {Rp_nm:6.3f} nm")
    print(f"    ε_out               = {eps_out:6.3f}")
    print(f"    ε_in(R)             = {eps_in:6.3f}")
    print(f"    ε_eff               = {eps_eff:6.3f}")
    print(f"    α (1/ε_eff)         = {alpha:6.4f}\n")

    return alpha, metrics['diameter_hull'], eps_eff, 0.0

