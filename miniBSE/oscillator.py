import numpy as np
from .constants import HA_TO_EV

def compute_oscillator_strengths(eigvals_ev, eigvecs, mu_ia, is_spinor=False):
    """
    Computes dimensionless oscillator strengths from excitation energies and transition dipoles.
    
    - is_spinor=False: Assumes spatial MOs (closed-shell singlets). Uses 4/3 prefactor.
    - is_spinor=True: Assumes Spinor basis. Spin degeneracy is already broken/explicit. Uses 2/3 prefactor.
    """
    f = []
    # Convert eV to Hartree for the dimensionless formula
    eigvals_ha = eigvals_ev / HA_TO_EV
    
    # Set the appropriate spin-multiplicity prefactor
    prefactor = 2.0 / 3.0 if is_spinor else 4.0 / 3.0

    for s in range(len(eigvals_ha)):
        # Transition dipole for state s: mu_s = sum_{ia} X_{ia,s} * mu_{ia}
        mu_s = np.sum(eigvecs[:, s][:, None] * mu_ia, axis=0)

        # f_s = prefactor * omega_s * |mu_s|^2
        # We use np.abs()**2 to correctly handle complex arithmetic when SOC is active
        f_s = prefactor * eigvals_ha[s] * np.sum(np.abs(mu_s)**2)
        f.append(f_s)

    return np.array(f)

