import numpy as np

def lowdin_sqrt(S):
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.clip(eigvals, a_min=1e-15, a_max=None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

def transform_mos(C, S):
    S_half = lowdin_sqrt(S)
    return S_half @ C

def build_lowdin_transition_charges_flat(C_occ_act, C_virt_act, S, atom_ao_ranges, valid_i, valid_a):
    """
    Computes transition charges using Löwdin symmetric orthogonalization:
    C^L = S^{1/2} * C.
    q^L_{ia, A} = sum_{mu in A} C^L_{mu i} * C^L_{mu a}.
    """
    S_half = lowdin_sqrt(S)
    C_occ_lowdin = S_half @ C_occ_act
    C_virt_lowdin = S_half @ C_virt_act
    
    dim = len(valid_i)
    n_atoms = len(atom_ao_ranges)
    q_flat = np.zeros((dim, n_atoms))
    
    for A, (a0, a1) in enumerate(atom_ao_ranges):
        Ci_A = C_occ_lowdin[a0:a1, :][:, valid_i]
        Ca_A = C_virt_lowdin[a0:a1, :][:, valid_a]
        q_flat[:, A] = np.sum(Ci_A * Ca_A, axis=0)
        
    return q_flat
