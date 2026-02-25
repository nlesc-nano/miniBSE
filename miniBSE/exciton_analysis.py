import numpy as np
import time

class ExcitonAnalyzer:
    """
    Highly optimized statistical analysis of excitons, including 
    electron-hole spatial correlation (Covariance) and True d_eh.
    """
    def __init__(self, solver, atom_coords, atom_symbols):
        self.solver = solver
        self.coords = atom_coords
        self.symbols = atom_symbols
        self.n_atoms = len(atom_symbols)
        
        print(f"  [Analyzer] Initializing Exciton Analyzer...")
        t0 = time.time()
        
        # --- BULLETPROOF ACTIVE SPACE MAPPING ---
        self.C_occ = self.solver.ham.C_orig_occ
        self.C_virt = self.solver.ham.C_orig_virt
        self.n_occ_act = self.C_occ.shape[1]
        self.n_virt_act = self.C_virt.shape[1]
        
        print(f"  [Analyzer] Extracted active space: {self.n_occ_act} occupied, {self.n_virt_act} virtual orbitals.")

        self.SC_occ = self.solver.overlap @ self.C_occ
        self.SC_virt = self.solver.overlap @ self.C_virt

        # --- PRE-COMPUTE MO SPATIAL CENTERS ---
        print(f"  [Analyzer] Precomputing spatial centers for orbitals...")
        pop_occ_ao = self.C_occ * self.SC_occ
        pop_virt_ao = self.C_virt * self.SC_virt
        
        q_occ = np.zeros((self.n_atoms, self.n_occ_act))
        q_virt = np.zeros((self.n_atoms, self.n_virt_act))
        
        for i_atom in range(self.n_atoms):
            start, end = self.solver.atom_ao_ranges[i_atom]
            q_occ[i_atom, :] = np.sum(pop_occ_ao[start:end, :], axis=0)
            q_virt[i_atom, :] = np.sum(pop_virt_ao[start:end, :], axis=0)
            
        # STRICT NORMALIZATION: Fixes the Covariance scaling bug
        q_occ /= (np.sum(q_occ, axis=0, keepdims=True) + 1e-12)
        q_virt /= (np.sum(q_virt, axis=0, keepdims=True) + 1e-12)
        
        # Calculate strict center of mass for each MO
        self.r_occ = q_occ.T @ self.coords  # Shape: (n_occ_act, 3)
        self.r_virt = q_virt.T @ self.coords # Shape: (n_virt_act, 3)
        
        print(f"  [Analyzer] Initialization completed in {time.time()-t0:.2f}s.")

    def get_particle_densities(self, bse_vec):
        weights = bse_vec**2
        
        hole_weights_mo = np.zeros(self.n_occ_act)
        elec_weights_mo = np.zeros(self.n_virt_act)

        for idx, w in enumerate(weights):
            i_rel = self.solver.ham.valid_i[idx]
            a_rel = self.solver.ham.valid_a[idx]
            hole_weights_mo[i_rel] += w
            elec_weights_mo[a_rel] += w

        pop_hole_ao = np.sum((self.C_occ * hole_weights_mo) * self.SC_occ, axis=1)
        pop_elec_ao = np.sum((self.C_virt * elec_weights_mo) * self.SC_virt, axis=1)

        pop_h_atom = np.zeros(self.n_atoms)
        pop_e_atom = np.zeros(self.n_atoms)

        for i_atom in range(self.n_atoms):
            start, end = self.solver.atom_ao_ranges[i_atom]
            pop_h_atom[i_atom] = np.sum(pop_hole_ao[start:end])
            pop_e_atom[i_atom] = np.sum(pop_elec_ao[start:end])

        return pop_h_atom / (np.sum(pop_h_atom) + 1e-12), pop_e_atom / (np.sum(pop_e_atom) + 1e-12)

    def analyze_state(self, bse_vec, energy, f_osc):
        results = {'energy': energy, 'f_osc': f_osc}

        weights = bse_vec**2
        weights /= (np.sum(weights) + 1e-12)

        results['PR'] = 1.0 / np.sum(weights**2)

        q_h, q_e = self.get_particle_densities(bse_vec)
        results['q_h'] = q_h
        results['q_e'] = q_e

        r_h_vec = q_h @ self.coords
        r_e_vec = q_e @ self.coords
        results['d_CT'] = np.linalg.norm(r_e_vec - r_h_vec)

        var_h = np.sum(q_h * np.sum((self.coords - r_h_vec)**2, axis=1))
        var_e = np.sum(q_e * np.sum((self.coords - r_e_vec)**2, axis=1))
        results['sigma_h'] = np.sqrt(var_h)
        results['sigma_e'] = np.sqrt(var_e)

        vi = self.solver.ham.valid_i
        va = self.solver.ham.valid_a
        
        r_h_active = self.r_occ[vi] 
        r_e_active = self.r_virt[va] 
        
        dr_h = r_h_active - r_h_vec
        dr_e = r_e_active - r_e_vec
        
        cov_eh = np.sum(np.sum(dr_h * dr_e, axis=1) * weights)
        results['cov_eh'] = cov_eh
        results['corr_eh'] = cov_eh / (results['sigma_h'] * results['sigma_e'] + 1e-8)

        d_eh_sq_true = var_h + var_e + results['d_CT']**2 - 2 * cov_eh
        results['d_eh'] = np.sqrt(max(0.0, d_eh_sq_true))
        results['CT_Character'] = results['d_CT'] / (results['d_eh'] + 1e-6)

        return results

# ================= PLOTTING FUNCTIONS =================

def plot_analysis_summary(analysis_results, physics_metrics=None, filename="exciton_analysis.html", show=True):
    """
    Generates a beautifully styled, interactive HTML dashboard using Plotly.
    Includes MathJax for formulas and a detailed glossary of descriptors.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  [Warning] plotly is not installed. Skipping interactive plot generation.")
        return

    energies = [r['energy'] for r in analysis_results]
    f_osc = np.array([r['f_osc'] for r in analysis_results])
    
    # Size mapping for the bubbles
    sizes = 8 + 25 * (f_osc / (np.max(f_osc) + 1e-6))

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Exciton Delocalization (PR)", 
            "Exciton Size (True d_eh)", 
            "Particle Spread (\u03C3)", 
            "Charge Transfer Distance (d_CT)",
            "Spatial Correlation (Pearson R)"
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.08
    )

    # 1. PR
    fig.add_trace(go.Scatter(
        x=energies, y=[r['PR'] for r in analysis_results],
        mode='markers', hovertemplate="<b>Energy:</b> %{x:.3f} eV<br><b>PR:</b> %{y:.2f}<br><b>f:</b> %{text}<extra></extra>",
        text=[f"{f:.4f}" for f in f_osc],
        marker=dict(size=sizes, color=energies, colorscale='Viridis', line=dict(width=1, color='DarkSlateGrey')),
        name='PR'), row=1, col=1)

    # 2. d_eh
    fig.add_trace(go.Scatter(
        x=energies, y=[r['d_eh'] for r in analysis_results],
        mode='markers', hovertemplate="<b>Energy:</b> %{x:.3f} eV<br><b>d_eh:</b> %{y:.2f} Å<extra></extra>",
        marker=dict(size=sizes, color='crimson', line=dict(width=1, color='DarkSlateGrey')),
        name='d_eh'), row=1, col=2)

    # 3. sigmas
    fig.add_trace(go.Scatter(
        x=energies, y=[r['sigma_h'] for r in analysis_results],
        mode='markers', name='Hole (\u03C3_h)',
        marker=dict(size=sizes, color='royalblue', symbol='circle', opacity=0.7)), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=energies, y=[r['sigma_e'] for r in analysis_results],
        mode='markers', name='Electron (\u03C3_e)',
        marker=dict(size=sizes, color='firebrick', symbol='square', opacity=0.7)), row=2, col=1)

    # 4. d_CT
    ct_char = [r['CT_Character'] for r in analysis_results]
    fig.add_trace(go.Scatter(
        x=energies, y=[r['d_CT'] for r in analysis_results],
        mode='markers', hovertemplate="<b>d_CT:</b> %{y:.2f} Å<br><b>CT Ratio:</b> %{marker.color:.2f}<extra></extra>",
        marker=dict(size=sizes, color=ct_char, colorscale='RdBu_r', cmin=0, cmax=1, 
                    colorbar=dict(title="CT Ratio", x=1.02, y=0.5, len=0.3)),
        name='d_CT'), row=2, col=2)

    # 5. Spatial Correlation
    corr_vals = [r['corr_eh'] for r in analysis_results]
    fig.add_trace(go.Scatter(
        x=energies, y=corr_vals,
        mode='markers', marker=dict(size=sizes, color=corr_vals, colorscale='PiYG', cmin=-1, cmax=1,
                    colorbar=dict(title="Pearson R", x=1.02, y=0.15, len=0.3)),
        name='Correlation'), row=3, col=1)

    fig.update_layout(height=950, width=1100, template="plotly_white", showlegend=True, margin=dict(t=60))

    if filename:
        html_plot = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Physics Header
        header_html = ""
        if physics_metrics:
            header_html = f"""
            <div style="font-family: sans-serif; padding: 20px; background: #f8f9fa; border-radius: 10px; margin: 20px auto; max-width: 1100px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                <h2 style="text-align: center; color: #2c3e50;">System Photophysics Summary</h2>
                <div style="background: #eef2f5; padding: 10px; border-radius: 8px; text-align: center; margin-bottom: 20px;">
                    \\( E_{{\\text{{opt}}}} = E_{{\\text{{gap}}}}^{{\\text{{DFT}}}} + \\Delta^{{\\text{{QP}}}} + \\Delta_{{\\text{{conf}}}} + E_{{\\text{{bind}}}} \\)
                </div>
                <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 10px;">
                    <div style="text-align: center; border-left: 4px solid #3498db; padding: 10px; background: white; flex: 1;">
                        <small>DFT GAP</small><br><b>{physics_metrics['dft_gap']:.3f} eV</b>
                    </div>
                    <div style="text-align: center; border-left: 4px solid #f39c12; padding: 10px; background: white; flex: 1;">
                        <small>QP (SCISSOR)</small><br><b>+{physics_metrics['qp_correction']:.3f} eV</b>
                    </div>
                    <div style="text-align: center; border-left: 4px solid #2ecc71; padding: 10px; background: white; flex: 1;">
                        <small>CONFINEMENT</small><br><b>+{physics_metrics['confinement_energy']:.3f} eV</b>
                    </div>
                    <div style="text-align: center; border-left: 4px solid #9b59b6; padding: 10px; background: white; flex: 1;">
                        <small>BINDING ENERGY</small><br><b>{physics_metrics['binding_energy']:.3f} eV</b>
                    </div>
                    <div style="text-align: center; border-left: 4px solid #e74c3c; padding: 10px; background: white; flex: 1;">
                        <small>S1 ENERGY</small><br><b>{physics_metrics['first_exc_energy']:.3f} eV</b>
                    </div>
                </div>
            </div>
            """

        # Documentation Section
        footer_html = """
        <div style="font-family: sans-serif; margin: 20px auto; max-width: 1100px; background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
            <h3 style="color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px;">Glossary of Descriptors</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; font-size: 14px; line-height: 1.6;">
                <div>
                    <p><b>Circle Size (\(f\)):</b> Represents the Oscillator Strength. Larger bubbles indicate transitions that dominate the absorption spectrum.</p>
                    <p><b>Participation Ratio (PR):</b> Quantifies delocalization. A higher PR indicates the exciton is spread over many orbital transitions.</p>
                    <p><b>RMS Spread (\(\sigma_h, \sigma_e\)):</b> The spatial standard deviation of the hole and electron densities. High values imply diffuse charges.</p>
                </div>
                <div>
                    <p><b>CT Distance (\(d_{CT}\)):</b> The distance between the center of the hole and the center of the electron. High \(d_{CT}\) indicates Charge Transfer character.</p>
                    <p><b>Spatial Correlation (\(R\)):</b> Measures if the electron and hole move together. Positive \(R\) means they are "bound" in space; negative means they avoid each other.</p>
                    <p><b>True Separation (\(d_{eh}\)):</b> The effective many-body distance between the pair, accounting for both \(d_{CT}\) and spatial fluctuations (\(\sigma\)).</p>
                </div>
            </div>
        </div>
        """

        full_html = f"""
        <html>
        <head>
            <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
            <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        </head>
        <body style="background: #ecf0f1; margin: 0; padding: 20px;">
            {header_html}
            <div style="margin: 0 auto; max-width: 1150px; background: white; padding: 20px; border-radius: 10px;">
                {html_plot}
            </div>
            {footer_html}
        </body>
        </html>
        """
        with open(filename, 'w', encoding='utf-8') as f: f.write(full_html)
        print(f"  Interactive HTML dashboard saved to '{filename}'")

    if show: fig.show()


def plot_exciton_3d_plotly(coords, symbols, q_h, q_e, state_idx, energy, filename="exciton_3d.html"):
    """
    Plots the 3D real-space density of the hole and electron on the molecular framework.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  [Warning] plotly is not installed. Skipping 3D HTML plot generation.")
        return

    color_map = {'PB': 'darkgrey', 'CS': 'purple', 'BR': 'saddlebrown', 'I': 'darkviolet', 'CL': 'green', 'IN': 'silver', 'AS': 'orange'}
    base_colors = [color_map.get(s.upper(), 'lightgrey') for s in symbols]
    fig = go.Figure()

    # Base molecular structure
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode='markers', marker=dict(size=3, color=base_colors, opacity=0.3),
        name='Atomic Framework', hoverinfo='text',
        text=[f"{s} {i}" for i, s in enumerate(symbols)]
    ))

    # Hole density
    threshold = 0.002
    h_idx = np.where(q_h > threshold)[0]
    fig.add_trace(go.Scatter3d(
        x=coords[h_idx, 0], y=coords[h_idx, 1], z=coords[h_idx, 2],
        mode='markers', marker=dict(size=q_h[h_idx] * 400, color='blue', opacity=0.5, sizemode='diameter'),
        name='Hole Density (+)', hoverinfo='text',
        text=[f"Atom: {symbols[i]} | Hole Pop: {q_h[i]:.3f}" for i in h_idx]
    ))

    # Electron density
    e_idx = np.where(q_e > threshold)[0]
    fig.add_trace(go.Scatter3d(
        x=coords[e_idx, 0], y=coords[e_idx, 1], z=coords[e_idx, 2],
        mode='markers', marker=dict(size=q_e[e_idx] * 400, color='red', opacity=0.5, sizemode='diameter'),
        name='Electron Density (-)', hoverinfo='text',
        text=[f"Atom: {symbols[i]} | Elec Pop: {q_e[i]:.3f}" for i in e_idx]
    ))

    fig.update_layout(
        title=f"Exciton Real-Space Density | State {state_idx} | E = {energy:.3f} eV",
        scene=dict(xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)', aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fig.write_html(filename)

