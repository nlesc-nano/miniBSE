import numpy as np
import csv
import time
from miniBSE.io_utils import count_ao_from_shells

PDOS_PALETTE = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692"]

def _ao_metadata(shells):
    ao_to_sym = []
    ao_to_coord = []

    # Use the robust internal counter so we never misalign AOs
    for sh in shells:
        n_funcs = count_ao_from_shells([sh])
        sym = sh.get("sym", "X")
        coord = sh.get("O", sh.get("center", [0.0, 0.0, 0.0]))
        for _ in range(n_funcs):
            ao_to_sym.append(sym)
            ao_to_coord.append(coord)
            
    ao_to_sym = np.array(ao_to_sym)
    ao_to_coord = np.array(ao_to_coord)

    # Surface vs Core Logic (Outer 25% of the radius is considered Surface)
    unique_coords = np.unique(ao_to_coord, axis=0)
    COM = np.mean(unique_coords, axis=0)
    ao_dists = np.linalg.norm(ao_to_coord - COM, axis=1)
    R_max = np.max(ao_dists)
    surface_ao_mask = np.ones(len(ao_dists), dtype=bool) if R_max < 1e-3 else (ao_dists >= 0.75 * R_max)

    return ao_to_sym, surface_ao_mask


def export_pdos_coop_data(analysis, eps_eV, pdos_atoms, coop_pairs, ewin, sigma=0.03, is_soc=False, prefix="sf"):
    """Export PDOS/COOP/IPR using precomputed weights and a supplied energy axis."""
    P_weights = analysis["P_weights"]
    ao_to_sym = analysis["ao_to_sym"]
    surface_ao_mask = analysis["surface_ao_mask"]
    coop_results = analysis["coop_results"]

    mask = (eps_eV >= ewin[0]) & (eps_eV <= ewin[1])
    E_sticks = eps_eV[mask]

    # --- 1. Compute Inverse Participation Ratio (IPR) ---
    IPR = analysis["IPR"]
    
    with open(f"ipr_data_{prefix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["MO_Energy_eV", "IPR"])
        for en, ipr in zip(E_sticks, IPR[mask]):
            w.writerow([en, ipr])
            
    # --- 2. Compute Surface vs Core Character ---
    total_pop = np.sum(P_weights, axis=0)
    total_pop[total_pop == 0] = 1.0 # avoid div by zero
    
    surf_char = np.sum(P_weights[surface_ao_mask, :], axis=0) / total_pop
    core_char = np.sum(P_weights[~surface_ao_mask, :], axis=0) / total_pop
    
    with open(f"surf_core_data_{prefix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["MO_Energy_eV", "Surface", "Core"])
        for en, s, c in zip(E_sticks, surf_char[mask], core_char[mask]):
            w.writerow([en, s, c])
    
    # --- 3. Compute PDOS ---
    energy_grid = np.linspace(ewin[0], ewin[1], 1000)
    pdos_curves, labels_p = [], []
    for sym in pdos_atoms:
        indices = np.where(ao_to_sym == sym)[0]
        if len(indices) == 0: continue
        sym_weight = np.sum(P_weights[indices, :], axis=0)

        X = (energy_grid[:, None] - eps_eV[None, :]) / sigma
        G = np.exp(-0.5 * X * X) / (sigma * np.sqrt(2 * np.pi))
        
        pdos_val = np.sum(sym_weight * G, axis=1)
        if not is_soc: pdos_val *= 2.0
            
        pdos_curves.append(pdos_val)
        labels_p.append(sym)
        
    if pdos_curves:
        Ycum = np.cumsum(np.column_stack(pdos_curves), axis=1)
        with open(f"pdos_data_{prefix}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Energy_eV"] + labels_p)
            for iE, E in enumerate(energy_grid):
                w.writerow([E] + list(Ycum[iE, :]))
                
    if coop_results:
        with open(f"coop_data_{prefix}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["MO_Energy_eV"] + list(coop_results.keys()))
            for i, en in enumerate(E_sticks):
                idx = np.where(mask)[0][i]
                w.writerow([en] + [coop_results[p][idx] for p in coop_results.keys()])

    export_population_bar_plot(analysis, eps_eV, pdos_atoms, ewin, prefix=prefix)


def export_population_bar_plot(analysis, eps_eV, pdos_atoms, ewin, prefix="sf"):
    """Export a non-broadened stacked horizontal population bar plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  [PDOS/COOP] Warning: matplotlib unavailable; skipping population bar plot ({exc}).")
        return

    P_weights = analysis["P_weights"]
    ao_to_sym = analysis["ao_to_sym"]
    mask = (eps_eV >= ewin[0]) & (eps_eV <= ewin[1])
    idx = np.where(mask)[0]
    if idx.size == 0:
        return

    labels, weights = [], []
    for sym in pdos_atoms:
        ao_idx = np.where(ao_to_sym == sym)[0]
        if ao_idx.size == 0:
            continue
        labels.append(sym)
        weights.append(np.sum(P_weights[ao_idx[:, None], idx], axis=0))

    if not weights:
        return

    W = np.vstack(weights).T
    W = np.clip(np.real(W), 0.0, None)
    total = np.sum(W, axis=1)
    total[total <= 1e-14] = 1.0
    frac = W / total[:, None]
    energies = eps_eV[idx]

    order = np.argsort(energies)
    energies = energies[order]
    frac = frac[order, :]

    if len(energies) > 1:
        span = max(float(ewin[1] - ewin[0]), 1e-6)
        height = min(0.035, max(0.003, 0.75 * span / len(energies)))
    else:
        height = 0.035

    fig_h = max(5.0, min(16.0, 0.018 * len(energies) + 4.0))
    fig, ax = plt.subplots(figsize=(4.2, fig_h))
    left = np.zeros(len(energies))
    for j, lab in enumerate(labels):
        ax.barh(
            energies, frac[:, j], left=left, height=height,
            color=PDOS_PALETTE[j % len(PDOS_PALETTE)], edgecolor="none", label=lab
        )
        left += frac[:, j]

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(float(ewin[0]), float(ewin[1]))
    ax.set_xlabel("Population fraction")
    ax.set_ylabel("Energy (eV)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=max(1, min(len(labels), 4)))
    ax.tick_params(direction="out", width=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    fig.tight_layout()
    fig.savefig(f"population_bars_{prefix}.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"population_bars_{prefix}.svg", bbox_inches="tight")
    plt.close(fig)

    with open(f"population_bars_{prefix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["MO_Energy_eV"] + labels)
        for en, row in zip(energies, frac):
            w.writerow([en] + list(row))


def compute_pdos_and_coop(C, S, eps_eV, shells, pdos_atoms, coop_pairs, ewin, sigma=0.03, is_soc=False, prefix="sf", pops=None):
    t0 = time.time()
    print(f"  [PDOS/COOP] Analyzing {len(pdos_atoms)} elements, {len(coop_pairs)} bonds, IPR, and Surface/Core...")
    
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
            
    # --- 0. Fix AO Mapping & Identify Surface AOs ---
    ao_to_sym, surface_ao_mask = _ao_metadata(shells)

    # --- 4. Compute COOP weights once. QP dashboards reuse these unchanged. ---
    coop_results = {}
    if is_soc:
        C_a, C_b = C_dense[:S_dense.shape[0], :], C_dense[S_dense.shape[0]:, :]

    for pair in coop_pairs:
        a_sym, b_sym = pair.split("-")
        idx_A = np.where(ao_to_sym == a_sym)[0]
        idx_B = np.where(ao_to_sym == b_sym)[0]
        if len(idx_A) == 0 or len(idx_B) == 0:
            continue

        S_AB = S_dense[np.ix_(idx_A, idx_B)]

        if is_soc:
            XB_a, XB_b = S_AB @ C_a[idx_B, :], S_AB @ C_b[idx_B, :]
            coop_n = 2.0 * (np.sum(C_a[idx_A, :].conj() * XB_a, axis=0).real +
                            np.sum(C_b[idx_A, :].conj() * XB_b, axis=0).real)
        else:
            XB = S_AB @ C_dense[idx_B, :]
            coop_n = 2.0 * np.sum(C_dense[idx_A, :].conj() * XB, axis=0).real

        coop_results[pair] = coop_n

    analysis = {
        "P_weights": P_weights,
        "ao_to_sym": ao_to_sym,
        "surface_ao_mask": surface_ao_mask,
        "IPR": np.sum(P_weights**2, axis=0),
        "coop_results": coop_results,
    }

    export_pdos_coop_data(analysis, eps_eV, pdos_atoms, coop_pairs, ewin, sigma=sigma, is_soc=is_soc, prefix=prefix)
                
    print(f"  [PDOS/COOP] Exported {prefix} data in {time.time() - t0:.2f} s")
    return analysis
