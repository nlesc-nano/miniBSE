import numpy as np

def compute_spin_character(vec, soc_U, n_occ_sp, n_virt_sp, valid_mask=None):
    """
    Projects the Spinor BSE eigenvector back onto the spatial basis 
    to calculate the total Singlet and Triplet weights.
    """
    X_IA = np.zeros((n_occ_sp, n_virt_sp), dtype=vec.dtype)
    if valid_mask is not None:
        X_IA[valid_mask] = vec
    else:
        X_IA = vec.reshape((n_occ_sp, n_virt_sp))
    
    n_mo = soc_U.shape[0] // 2
    U_occ_a = soc_U[:n_mo, :n_occ_sp]
    U_occ_b = soc_U[n_mo:, :n_occ_sp]
    
    U_virt_a = soc_U[:n_mo, n_occ_sp:]
    U_virt_b = soc_U[n_mo:, n_occ_sp:]
    
    rho_aa = U_occ_a.conj() @ X_IA @ U_virt_a.T
    rho_bb = U_occ_b.conj() @ X_IA @ U_virt_b.T
    
    S_mat = (rho_aa + rho_bb) / np.sqrt(2.0)
    singlet_weight = np.sum(np.abs(S_mat)**2)
    
    singlet_weight = min(1.0, max(0.0, singlet_weight))
    triplet_weight = 1.0 - singlet_weight
    
    return singlet_weight * 100, triplet_weight * 100

def print_orbital_summary(energies_eV, occ, homo_idx, pops, syms, shells, is_soc=False, offset=0, print_range=15):
    """
    Fast Mulliken population analysis broken down by Element and Angular Momentum (s, p, d).
    Expects precomputed 'pops' matrix to avoid duplicating S @ C multiplications.
    """
    print("\n" + "="*115)
    print(f"{'Orbital':>14} | {'Index':>6} | {'Energy (eV)':>12} | {'Occ':>5} | {'Main Contributions':>45}")
    print("-" * 115)
    
    n_states = pops.shape[1]
    l_char = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}
    ao_labels = []
    
    for sh in shells:
        sym = sh['sym']
        l_int = int(sh['l'])
        l_str = l_char.get(l_int, str(l_int))
        label = f"{sym}({l_str})"
        nbf = 2 * l_int + 1
        ao_labels.extend([label] * nbf)
        
    unique_labels = list(dict.fromkeys(ao_labels))
    label_pops = np.zeros((len(unique_labels), n_states))
        
    for i_ao, label in enumerate(ao_labels):
        lbl_idx = unique_labels.index(label)
        label_pops[lbl_idx, :] += pops[i_ao, :]
        
    col_sums = np.sum(label_pops, axis=0)
    col_sums[col_sums == 0] = 1.0
    label_pops = label_pops / col_sums[np.newaxis, :]
    
    start_idx = max(0, homo_idx - print_range + 1)
    end_idx = min(n_states, homo_idx + 1 + print_range)
    
    for idx in range(end_idx - 1, start_idx - 1, -1):
        if not is_soc:
            rel = idx - homo_idx
            label = ("LUMO" if rel == 1 else f"LUMO+{rel-1}") if rel > 0 else ("HOMO" if rel == 0 else f"HOMO{rel}")
        else:
            rel = idx - homo_idx
            label = ("spL" if rel == 1 else f"spL+{rel-1}") if rel > 0 else ("spH" if rel == 0 else f"spH{rel}")
                
        state_pops = label_pops[:, idx]
        top_indices = np.argsort(-state_pops)[:5]
        contrib_str = ", ".join([f"{unique_labels[i]} ({state_pops[i]*100:.0f}%)" for i in top_indices if state_pops[i] > 0.05])
        
        print(f"{label:>14} | {idx + offset:6d} | {energies_eV[idx]:12.4f} | {occ[idx]:5.1f} | {contrib_str}")
        
        if idx == homo_idx + 1:
            print(f"   {'-- FERMI --':>11} | {'------':>6} | {'------------':>12} | {'-----':>5} | {'-'*45}")
    print("=" * 115 + "\n")

