import numpy as np
from .constants import HA_TO_EV

def compute_oscillator_strengths(eigvals_ev, eigvecs, mu_ia):
    f = []
    # Convert eV to Hartree for the dimensionless formula
    eigvals_ha = eigvals_ev / HA_TO_EV

    for s in range(len(eigvals_ha)):
        # Spatial transition dipole for state s: mu_s = sum_{ia} X_{ia,s} * mu_{ia}
        mu_s = np.sum(eigvecs[:, s][:, None] * mu_ia, axis=0)
        
        # f_s = (2/3) * omega_s * |mu_s_total|^2
        # For closed-shell singlets, |mu_s_total|^2 = 2 * |mu_s_spatial|^2
        # Therefore: f_s = (4/3) * omega_s * |mu_s_spatial|^2
        f_s = (4.0/3.0) * eigvals_ha[s] * np.sum(mu_s**2)
        f.append(f_s)

    return np.array(f)
