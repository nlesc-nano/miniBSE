import numpy as np

def atomic_populations(C_tilde, atom_ao_ranges):
    nmo = C_tilde.shape[1]
    nat = len(atom_ao_ranges)
    Q = np.zeros((nmo, nat))

    for p in range(nmo):
        for A, (i0, i1) in enumerate(atom_ao_ranges):
            Q[p, A] = np.sum(C_tilde[i0:i1, p]**2)

    return Q

def transition_charges(C_tilde, occ_idx, virt_idx, atom_ao_ranges):
    n_occ = len(occ_idx)
    n_virt = len(virt_idx)
    nat = len(atom_ao_ranges)

    q = np.zeros((n_occ, n_virt, nat))

    for i_i, i in enumerate(occ_idx):
        for a_i, a in enumerate(virt_idx):
            for A, (i0, i1) in enumerate(atom_ao_ranges):
                q[i_i, a_i, A] = np.sum(
                    C_tilde[i0:i1, i] *
                    C_tilde[i0:i1, a]
                )

    return q


