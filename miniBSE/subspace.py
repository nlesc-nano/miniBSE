import numpy as np

def build_subspace(homo_index, n_occ, n_virt, n_mo):
    occ_start = max(0, homo_index - n_occ + 1)
    occ_end = homo_index + 1
    virt_start = homo_index + 1
    virt_end = min(n_mo, homo_index + 1 + n_virt)

    occ_idx = np.arange(occ_start, occ_end)
    virt_idx = np.arange(virt_start, virt_end)

    return occ_idx, virt_idx

