import numpy as np
import csv
import time

def compute_pdos_and_coop(C, S, eps_eV, shells, pdos_atoms, coop_pairs, ewin, sigma=0.03, is_soc=False, prefix="sf", pops=None):
    t0 = time.time()
    print(f"  [PDOS/COOP] Analyzing {len(pdos_atoms)} elements and {len(coop_pairs)} bonds...")
    
    C_dense = C.toarray() if hasattr(C, 'toarray') else C
    S_dense = S.toarray() if hasattr(S, 'toarray') else S
    
    if pops is not None:
        P_weights = pops
    else:
        if is_soc:
            n_ao = S_dense.shape[0]
            C_a, C_b = C_dense[:n_ao, :], C_dense[n_ao:, :]
            SC_a, SC_b = S_dense @ C_a, S_dense @ C_b
            P_weights = np.real(C_a.conj() * SC_a) + np.real(C_b.conj() * SC_b)
        else:
            SC = S_dense @ C_dense
            P_weights = np.real(C_dense.conj() * SC)
    
    atom_symbols = np.array([sh["sym"] for sh in shells])
    energy_grid = np.linspace(ewin[0], ewin[1], 1000)
    
    pdos_curves, labels_p = [], []
    for sym in pdos_atoms:
        indices = np.where(atom_symbols == sym)[0]
        if len(indices) == 0: continue
        sym_weight = np.sum(P_weights[indices, :], axis=0)

        X = (energy_grid[:, None] - eps_eV[None, :]) / sigma
        G = np.exp(-0.5 * X * X) / (sigma * np.sqrt(2 * np.pi))
        
        # Apply Spin Degeneracy (x2) for Spatial MOs!
        pdos_val = np.sum(sym_weight * G, axis=1)
        if not is_soc:
            pdos_val *= 2.0
            
        pdos_curves.append(pdos_val)
        labels_p.append(sym)
        
    if pdos_curves:
        Ycum = np.cumsum(np.column_stack(pdos_curves), axis=1)
        with open(f"pdos_data_{prefix}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Energy_eV"] + labels_p)
            for iE, E in enumerate(energy_grid):
                w.writerow([E] + list(Ycum[iE, :]))
                
    coop_results = {}
    if is_soc: C_a, C_b = C_dense[:S_dense.shape[0], :], C_dense[S_dense.shape[0]:, :]
        
    for pair in coop_pairs:
        a_sym, b_sym = pair.split("-")
        idx_A = np.where(atom_symbols == a_sym)[0]
        idx_B = np.where(atom_symbols == b_sym)[0]
        if len(idx_A) == 0 or len(idx_B) == 0: continue
        
        # ULTRA-FAST COOP SLICING: Slice S to (N_A x N_B) before multiplying
        S_AB = S_dense[np.ix_(idx_A, idx_B)]
        
        if is_soc:
            XB_a, XB_b = S_AB @ C_a[idx_B, :], S_AB @ C_b[idx_B, :]
            coop_n = 2.0 * (np.sum(C_a[idx_A, :].conj() * XB_a, axis=0).real + 
                            np.sum(C_b[idx_A, :].conj() * XB_b, axis=0).real)
        else:
            XB = S_AB @ C_dense[idx_B, :]
            coop_n = 2.0 * np.sum(C_dense[idx_A, :].conj() * XB, axis=0).real
            
        coop_results[pair] = coop_n

    if coop_results:
        mask = (eps_eV >= ewin[0]) & (eps_eV <= ewin[1])
        E_sticks = eps_eV[mask]
        with open(f"coop_data_{prefix}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["MO_Energy_eV"] + list(coop_results.keys()))
            for i, en in enumerate(E_sticks):
                idx = np.where(mask)[0][i]
                w.writerow([en] + [coop_results[p][idx] for p in coop_results.keys()])
                
    print(f"  [PDOS/COOP] Exported {prefix} data in {time.time() - t0:.2f} s")

