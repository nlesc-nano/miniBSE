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


def spin_label_from_S(S):
    labels = {
        0: "S",
        1: "T",
        2: "Q",
        3: "7",
        4: "9",
    }
    S_int = int(round(S))
    return labels.get(S_int, f"{2 * S_int + 1}")


def infer_reference_spin(n_alpha, n_beta):
    return abs(float(n_alpha) - float(n_beta)) / 2.0


def compute_uks_soc_spin_character(vec, ham, soc_U, n_alpha_ref, n_beta_ref):
    """
    Approximate UKS-SOC spin-sector character.

    For a closed-shell reference this reduces to the usual singlet/triplet
    alpha/beta transition decomposition. For high-spin UKS references, the
    scalar excitation component is assigned to S_ref and the spin-vector
    component is distributed over S_ref-1, S_ref, S_ref+1 by product-space
    dimensions. This is a labeling diagnostic, not an exact <S^2> projection.
    """
    if not hasattr(ham, "n_occ_act_b"):
        s_pct, t_pct = compute_spin_character(vec, soc_U, ham.n_occ_spinor, ham.n_virt_spinor)
        return {"S": s_pct, "T": t_pct}, 0.0

    X_spinor = np.zeros(ham.dim_spinor_full, dtype=complex)
    if hasattr(ham, "valid_spinor_idx"):
        X_spinor[ham.valid_spinor_idx] = vec
    else:
        X_spinor[:len(vec)] = vec
    X_spinor = X_spinor.reshape(ham.n_occ_spinor, ham.n_virt_spinor)

    n_alpha = ham.n_occ_act + ham.n_virt_act
    occ_cols = slice(0, ham.n_occ_spinor)
    virt_cols = slice(ham.n_occ_spinor, ham.n_occ_spinor + ham.n_virt_spinor)
    U_occ_a = soc_U[0:ham.n_occ_act, occ_cols]
    U_virt_a = soc_U[ham.n_occ_act:n_alpha, virt_cols]
    U_occ_b = soc_U[n_alpha:n_alpha + ham.n_occ_act_b, occ_cols]
    U_virt_b = soc_U[n_alpha + ham.n_occ_act_b:n_alpha + ham.n_occ_act_b + ham.n_virt_act_b, virt_cols]

    X_a = U_occ_a.conj() @ X_spinor @ U_virt_a.T
    X_b = U_occ_b.conj() @ X_spinor @ U_virt_b.T

    n_i = min(X_a.shape[0], X_b.shape[0])
    n_a = min(X_a.shape[1], X_b.shape[1])
    paired_a = X_a[:n_i, :n_a]
    paired_b = X_b[:n_i, :n_a]

    scalar = np.sum(np.abs((paired_a + paired_b) / np.sqrt(2.0)) ** 2)
    vector = np.sum(np.abs((paired_a - paired_b) / np.sqrt(2.0)) ** 2)
    vector += np.sum(np.abs(X_a[n_i:, :]) ** 2) + np.sum(np.abs(X_b[n_i:, :]) ** 2)
    vector += np.sum(np.abs(X_a[:n_i, n_a:]) ** 2) + np.sum(np.abs(X_b[:n_i, n_a:]) ** 2)

    S_ref = infer_reference_spin(n_alpha_ref, n_beta_ref)
    sectors = {}
    if S_ref < 1e-8:
        sectors["S"] = scalar
        sectors["T"] = vector
    else:
        sectors[spin_label_from_S(S_ref)] = sectors.get(spin_label_from_S(S_ref), 0.0) + scalar
        allowed = [S for S in (S_ref - 1.0, S_ref, S_ref + 1.0) if S >= 0.0]
        dim_sum = sum(2.0 * S + 1.0 for S in allowed)
        for S in allowed:
            label = spin_label_from_S(S)
            sectors[label] = sectors.get(label, 0.0) + vector * ((2.0 * S + 1.0) / dim_sum)

    total = sum(sectors.values())
    if total <= 1e-14:
        return {k: 0.0 for k in sectors}, S_ref
    return {k: 100.0 * v / total for k, v in sectors.items()}, S_ref


def format_spin_character(sectors):
    order = ["S", "T", "Q", "7", "9"]
    keys = [k for k in order if k in sectors] + [k for k in sectors if k not in order]
    return " / ".join(f"{sectors[k]:4.1f}% {k}" for k in keys)

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
