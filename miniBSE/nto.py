import csv
import time
from collections import defaultdict

import numpy as np


def transition_matrix_spatial(solver, vec, soc_U=None):
    """
    Return the transition-amplitude matrix X_ia in the spatial active MO basis.

    For SOC calculations the spinor excitation vector is projected back onto the
    alpha+beta spatial transition channel, matching the existing analysis path.
    """
    n_occ = solver.ham.n_occ_act
    n_virt = solver.ham.n_virt_act

    if not solver.soc_flag:
        x_mat = np.zeros((n_occ, n_virt), dtype=np.result_type(vec, np.complex128))
        x_mat[solver.ham.valid_i, solver.ham.valid_a] = vec
        return x_mat

    if soc_U is None:
        raise ValueError("soc_U is required for SOC NTO analysis")

    x_spinor = np.zeros(solver.ham.dim_spinor_full, dtype=complex)
    if hasattr(solver.ham, "valid_spinor_idx"):
        x_spinor[solver.ham.valid_spinor_idx] = vec
    else:
        x_spinor = vec
    x_spinor = x_spinor.reshape(solver.ham.n_occ_spinor, solver.ham.n_virt_spinor)

    k = solver.ham.n_occ_act + solver.ham.n_virt_act
    n_occ_sp = solver.ham.n_occ_spinor
    u_occ_a = soc_U[0:solver.ham.n_occ_act, 0:n_occ_sp]
    u_virt_a = soc_U[solver.ham.n_occ_act:k, n_occ_sp:2 * k]
    u_occ_b = soc_U[k:k + solver.ham.n_occ_act, 0:n_occ_sp]
    u_virt_b = soc_U[k + solver.ham.n_occ_act:2 * k, n_occ_sp:2 * k]

    return u_occ_a.conj() @ x_spinor @ u_virt_a.T + u_occ_b.conj() @ x_spinor @ u_virt_b.T


def compute_nto_pairs(x_mat):
    """Compute NTO pairs from X_ia = U diag(s) Vh."""
    norm = np.linalg.norm(x_mat)
    if norm > 0.0:
        x_work = x_mat / norm
    else:
        x_work = x_mat.copy()

    u_hole, singular_values, vh_elec = np.linalg.svd(x_work, full_matrices=False)
    weights = singular_values**2
    total = np.sum(weights)
    if total > 0.0:
        weights = weights / total
    v_elec = vh_elec.conj().T
    return u_hole, v_elec, singular_values, weights


def nto_compactness(weights):
    weights = np.asarray(weights, dtype=float)
    weights = weights[weights > 1e-14]
    if weights.size == 0:
        return {"lead_weight": 0.0, "pr": 0.0, "entropy": 0.0, "n90": 0, "n99": 0}

    cumulative = np.cumsum(weights)
    return {
        "lead_weight": float(weights[0]),
        "pr": float(1.0 / np.sum(weights**2)),
        "entropy": float(-np.sum(weights * np.log(weights))),
        "n90": int(np.searchsorted(cumulative, 0.90) + 1),
        "n99": int(np.searchsorted(cumulative, 0.99) + 1),
    }


def _atom_population_from_coeffs(coeff_ao, sc_ao, atom_ao_ranges):
    pop_ao = np.real(np.conj(coeff_ao) * sc_ao)
    pop_atom = np.empty(len(atom_ao_ranges), dtype=float)
    for atom_idx, (start, end) in enumerate(atom_ao_ranges):
        pop_atom[atom_idx] = np.sum(pop_ao[start:end])
    return _population_weights(pop_atom)


def _population_weights(pop_atom):
    pop = np.real(np.asarray(pop_atom, dtype=float))
    if np.any(pop < -1e-7):
        pop = np.maximum(pop, 0.0)
    total = np.sum(pop)
    if abs(total) < 1e-14:
        return np.zeros_like(pop)
    return pop / total


def _spatial_descriptors(q_atom, coords):
    if np.sum(q_atom) <= 0.0:
        center = np.zeros(3)
        return center, 0.0, 0.0

    center = q_atom @ coords
    variance = np.sum(q_atom * np.sum((coords - center) ** 2, axis=1))
    participation = 1.0 / (np.sum(q_atom**2) + 1e-14)
    return center, float(np.sqrt(max(variance, 0.0))), float(participation)


def _element_summary(q_atom, symbols, max_items=3):
    by_element = defaultdict(float)
    for sym, q in zip(symbols, q_atom):
        by_element[str(sym)] += float(q)
    ranked = sorted(by_element.items(), key=lambda item: item[1], reverse=True)
    return ";".join(f"{sym}:{100.0 * val:.1f}%" for sym, val in ranked[:max_items] if val > 1e-4)


def _dipole_pair_contribution(mu_ia, u_pair, v_pair, singular_value):
    if mu_ia is None:
        return np.zeros(3, dtype=complex)
    return singular_value * np.einsum("i,iax,a->x", u_pair, mu_ia, v_pair, optimize=True)


def spatial_transition_dipoles(solver, mu_ia):
    if mu_ia is None or solver.soc_flag:
        return None

    n_occ = solver.ham.n_occ_act
    n_virt = solver.ham.n_virt_act
    if mu_ia.shape == (n_occ, n_virt, 3):
        return mu_ia

    mu_full = np.zeros((n_occ, n_virt, 3), dtype=mu_ia.dtype)
    mu_full[solver.ham.valid_i, solver.ham.valid_a, :] = mu_ia
    return mu_full


def build_nto_context(solver):
    c_occ = solver.ham.C_orig_occ
    c_virt = solver.ham.C_orig_virt
    sc_occ = solver.overlap @ c_occ
    sc_virt = solver.overlap @ c_virt
    if hasattr(sc_occ, "toarray"):
        sc_occ = sc_occ.toarray()
    if hasattr(sc_virt, "toarray"):
        sc_virt = sc_virt.toarray()
    return {"c_occ": c_occ, "c_virt": c_virt, "sc_occ": sc_occ, "sc_virt": sc_virt}


def analyze_nto_state(solver, vec, energy_ev, f_osc, state_index, coords, symbols, mu_ia=None, soc_U=None, top_n=3, context=None):
    x_mat = transition_matrix_spatial(solver, vec, soc_U=soc_U)
    u_hole, v_elec, singular_values, weights = compute_nto_pairs(x_mat)
    compact = nto_compactness(weights)

    if context is None:
        context = build_nto_context(solver)
    c_occ = context["c_occ"]
    c_virt = context["c_virt"]
    sc_occ = context["sc_occ"]
    sc_virt = context["sc_virt"]
    cumulative = np.cumsum(weights)

    rows = []
    for pair_idx in range(min(top_n, len(weights))):
        hole_orb = c_occ @ u_hole[:, pair_idx]
        elec_orb = c_virt @ v_elec[:, pair_idx]
        hole_sc = sc_occ @ u_hole[:, pair_idx]
        elec_sc = sc_virt @ v_elec[:, pair_idx]

        q_h = _atom_population_from_coeffs(hole_orb, hole_sc, solver.atom_ao_ranges)
        q_e = _atom_population_from_coeffs(elec_orb, elec_sc, solver.atom_ao_ranges)

        r_h, sigma_h, pr_h_atom = _spatial_descriptors(q_h, coords)
        r_e, sigma_e, pr_e_atom = _spatial_descriptors(q_e, coords)
        d_ct = float(np.linalg.norm(r_e - r_h))
        mu_pair = _dipole_pair_contribution(mu_ia, u_hole[:, pair_idx], v_elec[:, pair_idx], singular_values[pair_idx])

        rows.append({
            "state": state_index,
            "energy_ev": float(energy_ev),
            "f_osc": float(f_osc),
            "pair": pair_idx + 1,
            "weight": float(weights[pair_idx]),
            "cum_weight": float(cumulative[pair_idx]),
            "s_value": float(singular_values[pair_idx]),
            "d_ct_ang": d_ct,
            "sigma_h_ang": sigma_h,
            "sigma_e_ang": sigma_e,
            "atom_pr_h": pr_h_atom,
            "atom_pr_e": pr_e_atom,
            "hole_elements": _element_summary(q_h, symbols),
            "electron_elements": _element_summary(q_e, symbols),
            "mu_pair_norm": float(np.linalg.norm(mu_pair)),
        })

    state = {
        "state": state_index,
        "energy_ev": float(energy_ev),
        "f_osc": float(f_osc),
        **compact,
        "rows": rows,
    }
    return state


def selected_state_indices(args, n_states):
    requested = getattr(args, "nto_states", None)
    if requested:
        return [idx - 1 for idx in requested if 1 <= idx <= n_states]
    n_default = min(getattr(args, "csv_roots", 10), n_states)
    return list(range(n_default))


def print_nto_analysis(states, label):
    print(f"\n--- NTO Analysis ({label}) ---")
    print(f"{'State':>5} {'Energy':>8} {'f_osc':>8} | {'w1':>7} {'NTO_PR':>7} {'S_nto':>7} {'N90':>4} {'N99':>4}")
    print("-" * 75)
    for state in states:
        print(
            f"{state['state']:5d} {state['energy_ev']:8.3f} {state['f_osc']:8.4f} | "
            f"{state['lead_weight']:7.3f} {state['pr']:7.2f} {state['entropy']:7.3f} "
            f"{state['n90']:4d} {state['n99']:4d}"
        )

    print("\n  Dominant NTO pairs")
    print(
        f"{'State':>5} {'Pair':>4} {'Weight':>8} {'Cum':>8} | "
        f"{'d_CT':>7} {'sig_h':>7} {'sig_e':>7} {'PR_h':>6} {'PR_e':>6} | "
        f"{'Hole elem':>22} {'Electron elem':>22}"
    )
    print("-" * 125)
    for state in states:
        for row in state["rows"]:
            print(
                f"{row['state']:5d} {row['pair']:4d} {row['weight']:8.3f} {row['cum_weight']:8.3f} | "
                f"{row['d_ct_ang']:7.2f} {row['sigma_h_ang']:7.2f} {row['sigma_e_ang']:7.2f} "
                f"{row['atom_pr_h']:6.1f} {row['atom_pr_e']:6.1f} | "
                f"{row['hole_elements'][:22]:>22} {row['electron_elements'][:22]:>22}"
            )


def write_nto_csv(states, filename):
    fieldnames = [
        "state", "energy_ev", "f_osc", "lead_weight", "nto_pr", "nto_entropy", "n90", "n99",
        "pair", "weight", "cum_weight", "s_value", "d_ct_ang", "sigma_h_ang", "sigma_e_ang",
        "atom_pr_h", "atom_pr_e", "hole_elements", "electron_elements", "mu_pair_norm",
    ]
    with open(filename, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for state in states:
            for row in state["rows"]:
                writer.writerow({
                    "state": state["state"],
                    "energy_ev": f"{state['energy_ev']:.8f}",
                    "f_osc": f"{state['f_osc']:.8e}",
                    "lead_weight": f"{state['lead_weight']:.8f}",
                    "nto_pr": f"{state['pr']:.8f}",
                    "nto_entropy": f"{state['entropy']:.8f}",
                    "n90": state["n90"],
                    "n99": state["n99"],
                    **row,
                })


def run_nto_analysis(solver, vectors, energies_ev, f_strengths, coords, symbols, mu_ia=None, args=None, suffix="", soc_U=None):
    top_n = getattr(args, "nto_top", 3) if args is not None else 3
    label = "SOC" if solver.soc_flag else "SPIN-FREE"
    indices = selected_state_indices(args, len(energies_ev)) if args is not None else list(range(min(10, len(energies_ev))))

    t0 = time.time()
    states = []
    mu_spatial = spatial_transition_dipoles(solver, mu_ia)
    context = build_nto_context(solver)
    for idx in indices:
        states.append(
            analyze_nto_state(
                solver=solver,
                vec=vectors[:, idx],
                energy_ev=energies_ev[idx],
                f_osc=f_strengths[idx],
                state_index=idx + 1,
                coords=coords,
                symbols=symbols,
                mu_ia=mu_spatial,
                soc_U=soc_U,
                top_n=top_n,
                context=context,
            )
        )

    print_nto_analysis(states, label)
    print(f"  [NTO] Completed in {time.time() - t0:.2f}s")

    if args is not None and getattr(args, "nto_csv", False):
        filename = f"nto_results{suffix}.csv"
        write_nto_csv(states, filename)
        print(f"  [NTO] Wrote {filename}")

    return states
