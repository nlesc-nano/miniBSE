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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_analysis_summary(analysis_results, physics_metrics=None, filename=None, show=False, broadening="gaussian", sigma=0.1):
    energies = [res['energy'] for res in analysis_results]
    f_osc = [res['f_osc'] for res in analysis_results]
    d_eh = [res['d_eh'] for res in analysis_results]
    d_CT = [res['d_CT'] for res in analysis_results]
    PR = [res['PR'] for res in analysis_results]
    sigma_h = [res['sigma_h'] for res in analysis_results]
    sigma_e = [res['sigma_e'] for res in analysis_results]
    corr_eh = [res.get('corr_eh', 0) for res in analysis_results]
    ct_ratio = [res.get('CT_Character', 0) for res in analysis_results]

    max_f = max(f_osc) if max(f_osc) > 0 else 1.0
    marker_sizes = [8 + (f / max_f) * 20 for f in f_osc] 

    # 3x2 Grid with a secondary Y-axis for the 6th plot
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Exciton Delocalization (PR)",
            "Exciton Size (True d_eh)",
            "Particle Spread (σ)",
            "Charge Transfer Distance (d_CT)",
            "Spatial Correlation (Pearson R)",
            "Simulated UV-Vis Spectrum"
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
        specs=[[{}, {}], [{}, {}], [{}, {"secondary_y": True}]] 
    )

    # --- ROW 1 ---
    fig.add_trace(go.Scatter(
        x=energies, y=PR, mode='markers',
        marker=dict(size=marker_sizes, color=energies, colorscale='Viridis', opacity=0.8, line=dict(width=1, color='black')),
        name='PR', hovertemplate="Energy: %{x:.3f} eV<br>PR: %{y:.1f}<br>f_osc: %{customdata[0]:.4f}<extra></extra>",
        customdata=np.column_stack([f_osc])
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=energies, y=d_eh, mode='markers',
        marker=dict(size=marker_sizes, color='#d62728', opacity=0.7, line=dict(width=1, color='black')),
        name='d_eh', hovertemplate="Energy: %{x:.3f} eV<br>d_eh: %{y:.2f} Å<br>f_osc: %{customdata[0]:.4f}<extra></extra>",
        customdata=np.column_stack([f_osc])
    ), row=1, col=2)

    # --- ROW 2 ---
    fig.add_trace(go.Scatter(
        x=energies, y=sigma_h, mode='markers',
        marker=dict(size=marker_sizes, color='royalblue', symbol='circle', opacity=0.7, line=dict(width=1, color='white')),
        name='Hole (σ_h)', hovertemplate="Energy: %{x:.3f} eV<br>σ_h: %{y:.2f} Å<extra></extra>"
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=energies, y=sigma_e, mode='markers',
        marker=dict(size=marker_sizes, color='indianred', symbol='square', opacity=0.7, line=dict(width=1, color='white')),
        name='Electron (σ_e)', hovertemplate="Energy: %{x:.3f} eV<br>σ_e: %{y:.2f} Å<extra></extra>"
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=energies, y=d_CT, mode='markers',
        marker=dict(
            size=marker_sizes, color=ct_ratio, colorscale='RdBu_r', showscale=True, 
            cmin=0, cmax=1, opacity=0.8, line=dict(width=1, color='black'),
            colorbar=dict(title="CT Ratio", x=1.02, y=0.5, len=0.3)
        ),
        name='d_CT', hovertemplate="Energy: %{x:.3f} eV<br>d_CT: %{y:.2f} Å<br>CT Ratio: %{marker.color:.2f}<extra></extra>"
    ), row=2, col=2)

    # --- ROW 3 ---
    fig.add_trace(go.Scatter(
        x=energies, y=corr_eh, mode='markers',
        marker=dict(
            size=marker_sizes, color=corr_eh, colorscale='PiYG', showscale=True, 
            cmin=-1, cmax=1, opacity=0.8, line=dict(width=1, color='black'),
            colorbar=dict(title="Pearson R", x=0.45, y=0.15, len=0.3)
        ),
        name='Correlation', hovertemplate="Energy: %{x:.3f} eV<br>Pearson R: %{y:.3f}<extra></extra>"
    ), row=3, col=1)

    # --- 6. Simulated Spectrum Overlay ---
    if broadening != "none" and len(energies) > 0:
        # Import dynamically to avoid circular dependencies if any exist
        from miniBSE.spectrum import generate_spectrum
        
        e_min = max(0.0, np.min(energies) - 2.5)
        e_max = np.max(energies) + 2.5
        x_grid, y_grid = generate_spectrum(energies, f_osc, e_min=e_min, e_max=e_max, sigma=sigma, profile=broadening)
        
        # Convoluted Envelope
        fig.add_trace(go.Scatter(
            x=x_grid, y=y_grid, mode='lines',
            line=dict(color='black', width=1.5),
            fill='tozeroy', fillcolor='rgba(65, 105, 225, 0.3)', 
            name='Absorption Envelope', hoverinfo='skip', showlegend=False
        ), row=3, col=2, secondary_y=False)

    # Oscillator Strength Stems
    fig.add_trace(go.Bar(
        x=energies, y=f_osc, width=0.015, marker_color='crimson', opacity=0.8,
        showlegend=False, hoverinfo='skip'
    ), row=3, col=2, secondary_y=True)
    
    fig.add_trace(go.Scatter(
        x=energies, y=f_osc, mode='markers',
        marker=dict(color='crimson', size=6, line=dict(width=1, color='black')),
        name='f_osc', showlegend=False,
        hovertemplate="Energy: %{x:.3f} eV<br>f_osc: %{y:.4f}<extra></extra>"
    ), row=3, col=2, secondary_y=True)

    # --- AXES FORMATTING ---
    for i in range(1, 4):
        fig.update_xaxes(title_text="Energy (eV)", row=i, col=1)
        if i < 4: fig.update_xaxes(title_text="Energy (eV)", row=i, col=2)

    fig.update_yaxes(title_text="PR (States)", row=1, col=1)
    fig.update_yaxes(title_text="Distance (Å)", row=1, col=2)
    fig.update_yaxes(title_text="Spread (Å)", row=2, col=1)
    fig.update_yaxes(title_text="Distance (Å)", row=2, col=2)
    fig.update_yaxes(title_text="Pearson R", row=3, col=1)
    
    # Spectrum Axes
    fig.update_yaxes(title_text="Intensity (a.u.)", row=3, col=2, secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="f_osc", row=3, col=2, secondary_y=True, rangemode="tozero", color="crimson")

    fig.update_layout(
        height=1000, width=1400,
        template="plotly_white",
        margin=dict(t=40, r=120, b=60, l=80),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05)
    )

    # --- HTML EXPORT WITH CUSTOM HEADER & FOOTER ---
    if filename:
        m = physics_metrics or {}
        
        is_soc = m.get('is_soc', False)
        if is_soc and 'soc_gap' in m:
            gap_val = m['soc_gap']
            gap_label = "SOC GAP"
        else:
            gap_val = m.get('dft_gap', 0)
            gap_label = "DFT GAP"

        gap_text = f"{gap_val:.3f} eV"
        qp = f"{m.get('qp_correction', 0):+.3f} eV"
        conf = f"{m.get('confinement_energy', 0):+.3f} eV"
        bind = f"{m.get('binding_energy', 0):+.3f} eV"
        s1 = f"{m.get('first_exc_energy', 0):.3f} eV"

        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f4f6f9; margin: 0; padding: 20px; }}
                .dashboard-container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
                .header-title {{ text-align: center; color: #2c3e50; font-size: 28px; font-weight: 700; margin-bottom: 20px; }}
                .equation-box {{ background-color: #f8f9fa; padding: 15px; text-align: center; border-radius: 8px; margin-bottom: 30px; font-size: 20px; color: #343a40; }}
                .metrics-row {{ display: flex; justify-content: space-between; gap: 15px; margin-bottom: 40px; }}
                .metric-card {{ flex: 1; background: white; padding: 15px 20px; text-align: center; border-radius: 6px; box-shadow: 0 2px 5px rgba(0,0,0,0.04); display: flex; flex-direction: column; justify-content: center; }}
                .metric-label {{ font-size: 13px; font-weight: 600; color: #6c757d; letter-spacing: 0.5px; margin-bottom: 5px; text-transform: uppercase; }}
                .metric-value {{ font-size: 20px; font-weight: 700; color: #212529; }}
                .b-blue {{ border-left: 5px solid #4a90e2; }}
                .b-orange {{ border-left: 5px solid #f39c12; }}
                .b-green {{ border-left: 5px solid #2ecc71; }}
                .b-purple {{ border-left: 5px solid #9b59b6; }}
                .b-red {{ border-left: 5px solid #e74c3c; }}
                .glossary-box {{ margin-top: 40px; padding: 25px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #e9ecef; }}
                .glossary-title {{ font-size: 18px; font-weight: 700; color: #2c3e50; margin-bottom: 15px; }}
                .glossary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .glossary-item strong {{ color: #343a40; font-size: 15px; display: block; margin-bottom: 4px; }}
                .glossary-item p {{ margin: 0; font-size: 14px; color: #6c757d; line-height: 1.5; }}
            </style>
        </head>
        <body>
            <div class="dashboard-container">
                <div class="header-title">System Photophysics Summary { '(SOC)' if is_soc else '(Spin-Free)' }</div>
                
                <div class="equation-box">
                    <i>E</i><sub>opt</sub> = <i>E</i><sub>gap</sub><sup>{ 'SOC' if is_soc else 'DFT' }</sup> + Δ<sup>QP</sup> + Δ<sub>conf</sub> + <i>E</i><sub>bind</sub>
                </div>

                <div class="metrics-row">
                    <div class="metric-card b-blue">
                        <div class="metric-label">{gap_label}</div>
                        <div class="metric-value">{gap_text}</div>
                    </div>
                    <div class="metric-card b-orange">
                        <div class="metric-label">QP (Scissor)</div>
                        <div class="metric-value">{qp}</div>
                    </div>
                    <div class="metric-card b-green">
                        <div class="metric-label">Confinement</div>
                        <div class="metric-value">{conf}</div>
                    </div>
                    <div class="metric-card b-purple">
                        <div class="metric-label">Binding Energy</div>
                        <div class="metric-value">{bind}</div>
                    </div>
                    <div class="metric-card b-red">
                        <div class="metric-label">S1 Energy</div>
                        <div class="metric-value">{s1}</div>
                    </div>
                </div>

                {fig.to_html(full_html=False, include_plotlyjs='cdn')}

                <div class="glossary-box">
                    <div class="glossary-title">Descriptor Glossary</div>
                    <div class="glossary-grid">
                        <div class="glossary-item">
                            <strong>Participation Ratio (PR)</strong>
                            <p>Measures the number of single-particle orbital transitions contributing to the exciton. A higher PR indicates greater multiconfigurational character and delocalization.</p>
                        </div>
                        <div class="glossary-item">
                            <strong>True Exciton Size (d_eh)</strong>
                            <p>The root-mean-square distance between the electron and hole density distributions, factoring in both their spread and their spatial separation.</p>
                        </div>
                        <div class="glossary-item">
                            <strong>Particle Spread (σ_h, σ_e)</strong>
                            <p>The spatial standard deviation of the individual hole (σ_h) and electron (σ_e) probability densities around their respective centers of mass.</p>
                        </div>
                        <div class="glossary-item">
                            <strong>Charge Transfer Distance (d_CT)</strong>
                            <p>The absolute distance between the spatial center of mass of the hole and the center of mass of the electron. Large values indicate significant charge-transfer character.</p>
                        </div>
                        <div class="glossary-item">
                            <strong>Spatial Correlation (Pearson R)</strong>
                            <p>Evaluates how strictly the electron and hole positions are correlated. Positive values indicate they localize in the same regions; negative values indicate spatial separation.</p>
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"  Saved Exciton Analysis dashboard to {filename}")

    if show:
        fig.show()

