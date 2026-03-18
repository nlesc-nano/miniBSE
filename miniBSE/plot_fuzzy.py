import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re
from scipy.spatial import cKDTree

def load_fuzzy(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    Z = np.asarray(d["intensity"], dtype=float)
    centres = np.asarray(d["centres"], dtype=float)
    step = int(np.ceil((Z.size / 250_000) ** 0.5))
    if step > 1:
        Z = Z[::step, ::step]
        centres = centres[::step]
    return dict(
        centres=centres, Z=Z, 
        tick_positions=np.asarray(d.get("tick_positions", []), dtype=float), 
        tick_labels=[str(x) for x in d.get("tick_labels", [])],
        extent=np.asarray(d.get("extent", [0.0, float(Z.shape[1]-1), float(centres.min()), float(centres.max())]), dtype=float),
        ewin=np.asarray(d.get("ewin", [float(centres.min()), float(centres.max())]), dtype=float)
    )

def load_pdos_csv(csv_path):
    if not os.path.exists(csv_path): return np.array([]), [], np.empty((0,0))
    df = pd.read_csv(csv_path)
    return df.iloc[:, 0].to_numpy(dtype=float), list(df.columns[1:]), df.iloc[:, 1:].to_numpy(dtype=float)

def load_dict_csv(csv_path):
    if not os.path.exists(csv_path): return np.array([]), [], {}
    df = pd.read_csv(csv_path)
    return df.iloc[:, 0].to_numpy(dtype=float), list(df.columns[1:]), {p: df[p].to_numpy(dtype=float) for p in df.columns[1:]}

def load_ipr_csv(csv_path):
    if not os.path.exists(csv_path): return np.array([]), np.array([])
    df = pd.read_csv(csv_path)
    return df.iloc[:, 0].to_numpy(dtype=float), df.iloc[:, 1].to_numpy(dtype=float)

def build_wireframe_agnostic(atoms):
    if len(atoms) < 2: return [], [], []
    coords = np.array([[a[1], a[2], a[3]] for a in atoms])
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    local_radii = dists[:, 1] * 0.6
    xs, ys, zs = [], [], []
    max_r = np.max(local_radii)
    pairs = tree.query_pairs(max_r * 2.5)
    for i, j in pairs:
        d_ij = np.linalg.norm(coords[i] - coords[j])
        if d_ij <= 1.25 * (local_radii[i] + local_radii[j]):
            xs.extend([coords[i][0], coords[j][0], None])
            ys.extend([coords[i][1], coords[j][1], None])
            zs.extend([coords[i][2], coords[j][2], None])
    return xs, ys, zs

def parse_cube(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    n_atoms = int(lines[2].split()[0])
    origin = np.array([float(x) for x in lines[2].split()[1:4]])
    nx, dx, _, _ = [float(x) for x in lines[3].split()]
    ny, _, dy, _ = [float(x) for x in lines[4].split()]
    nz, _, _, dz = [float(x) for x in lines[5].split()]
    nx, ny, nz = int(nx), int(ny), int(nz)
    atoms = []
    for i in range(abs(n_atoms)):
        parts = lines[6+i].split()
        atoms.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    data_str = " ".join(lines[6+abs(n_atoms):])
    V = np.fromstring(data_str, sep=' ').reshape((nx, ny, nz))
    BOHR_TO_ANG = 0.529177249
    origin *= BOHR_TO_ANG
    dx *= BOHR_TO_ANG; dy *= BOHR_TO_ANG; dz *= BOHR_TO_ANG
    step_x = max(1, nx // 20); step_y = max(1, ny // 20); step_z = max(1, nz // 20)
    V = V[::step_x, ::step_y, ::step_z]
    nx_new, ny_new, nz_new = V.shape
    x = origin[0] + np.arange(nx_new) * (dx * step_x)
    y = origin[1] + np.arange(ny_new) * (dy * step_y)
    z = origin[2] + np.arange(nz_new) * (dz * step_z)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    atoms_ang = [(z, ax*BOHR_TO_ANG, ay*BOHR_TO_ANG, az*BOHR_TO_ANG) for z, charge, ax, ay, az in atoms]
    return X.flatten(), Y.flatten(), Z.flatten(), V.flatten(), atoms_ang

def generate_interactive_plot(prefix="sf", material="DEFAULT", ef=0.0, e_homo=None, e_lumo=None, normalize_coop=False):
    lbl = "SOC" if prefix == "soc" else "Spin-Free"
    print(f"  [Plotter] Generating elegant {lbl} HTML dashboard...")
    
    # =========================================================================
    # PART 1: 2D DASHBOARD
    # =========================================================================
    fig = make_subplots(
        rows=1, cols=5, 
        shared_yaxes=True, 
        column_widths=[0.34, 0.22, 0.11, 0.11, 0.22],
        horizontal_spacing=0.015, 
        subplot_titles=(f"{lbl} Fuzzy Bands", "PDOS", "IPR", "Surf/Core", "COOP")
    )
    
    fig.update_layout(
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", 
        height=900,  
        font=dict(family="Helvetica, Arial, sans-serif", size=24, color="#222"),
        margin=dict(l=100, r=40, t=220, b=100),
        legend=dict(title=dict(text="<b>Atoms</b>", font=dict(size=20)), orientation="h", x=0.45, xanchor="center", y=1.08, yanchor="bottom", font=dict(size=18), bgcolor="rgba(255,255,255,0)"),
        legend2=dict(title=dict(text="<b>Localization</b>", font=dict(size=20)), orientation="h", x=0.68, xanchor="center", y=1.08, yanchor="bottom", font=dict(size=18), bgcolor="rgba(255,255,255,0)"),
        legend3=dict(title=dict(text="<b>Bonds</b>", font=dict(size=20)), orientation="h", x=0.89, xanchor="center", y=1.08, yanchor="bottom", font=dict(size=18), bgcolor="rgba(255,255,255,0)")
    )

    for annotation in fig['layout']['annotations']: annotation['font'] = dict(size=28, family="Helvetica", color="#111")
    palette = ["#636EFA","#EF553B","#00CC96","#AB63FA","#FFA15A","#19D3F3","#FF6692"]
    
    fuzzy = load_fuzzy(f"fuzzy_data_{prefix}.npz")
    pdos_E, pdos_L, pdos_Y = load_pdos_csv(f"pdos_data_{prefix}.csv")
    coop_E, coop_P, coop_V = load_dict_csv(f"coop_data_{prefix}.csv")
    sc_E, sc_P, sc_V = load_dict_csv(f"surf_core_data_{prefix}.csv")
    ipr_E, ipr_V = load_ipr_csv(f"ipr_data_{prefix}.csv")

    Z, ewin = fuzzy["Z"], fuzzy["ewin"]
    kx = np.linspace(fuzzy["extent"][0], fuzzy["extent"][1], Z.shape[1])
    Zpos = Z[Z > 1e-9]
    vmax = float(np.percentile(Z, 99.9))
    vmin_base = float(np.percentile(Zpos, 5)) if Zpos.size else 1e-6
    Zm = Z.astype(np.float32); Zm[Zm <= 0] = np.nan
    
    fig.add_shape(type="rect", xref="x", yref="y", x0=kx[0], x1=kx[-1], y0=ewin[0], y1=ewin[1], fillcolor="black", line=dict(width=0), layer="below") 

    heat = go.Heatmap(
        z=np.log10(Zm), x=kx, y=fuzzy["centres"], colorscale="Inferno",
        zmin=float(np.log10(vmin_base)), zmax=float(np.log10(vmax)), showscale=True,
        zsmooth="best",  # <--- ADD THIS EXACT LINE
        colorbar=dict(title=dict(text="<b>log₁₀(I)</b>", font=dict(size=20)), orientation="h", len=0.30, thickness=20, x=0.17, xanchor="center", y=1.08, yanchor="bottom", tickfont=dict(size=18)),
        hovertemplate="k=%{x:.3f} Å⁻¹<br>E=%{y:.3f} eV<br>log10(I)=%{z:.2f}<extra></extra>"
    )

    fig.add_trace(heat, row=1, col=1)
   
    if fuzzy["tick_positions"].size:
        # FIX: Scale ticks based on the original physical width, not the downsampled pixel count
        original_width = fuzzy["extent"][1] - fuzzy["extent"][0]
        scale = (kx[-1] - kx[0]) / original_width if original_width > 0 else 1.0
        tpos_plot = kx[0] + fuzzy["tick_positions"] * scale
        
        for x in tpos_plot: fig.add_vline(x=x, line_color="rgba(255,255,255,0.4)", line_width=2, row=1, col=1)
        fig.update_xaxes(tickmode="array", tickvals=tpos_plot, ticktext=fuzzy["tick_labels"], row=1, col=1)
 
    if len(pdos_L) > 0:
        for j, lab in enumerate(pdos_L):
            fig.add_trace(go.Scatter(x=pdos_Y[:, j], y=pdos_E, mode="lines", fill="tonextx" if j > 0 else "tozerox", line=dict(width=1.0, color="rgba(0,0,0,0)"), fillcolor=palette[j % len(palette)], name=lab, showlegend=True, legend="legend", hovertemplate=f"{lab}: %{{x:.3f}}<br>E=%{{y:.3f}} eV<extra></extra>"), row=1, col=2)
        total = pdos_Y[:, -1]
        fig.add_trace(go.Scatter(x=total, y=pdos_E, mode="lines", line=dict(color="black", width=3), name="Total DOS", showlegend=False), row=1, col=2)
        fig.update_xaxes(range=[0, float(max(total.max(), 1e-12)) * 1.05], row=1, col=2)

    if len(ipr_E) > 0:
        fig.add_trace(go.Scatter(x=ipr_V, y=ipr_E, mode="markers", marker=dict(size=6, color="#440154", line=dict(width=1, color="black")), showlegend=False, hovertemplate="IPR: %{x:.4f}<br>E: %{y:.3f} eV<extra></extra>"), row=1, col=3)
        fig.update_xaxes(range=[0, 1.05], row=1, col=3)

    if len(sc_E) > 0:
        surf, core = sc_V["Surface"], sc_V["Core"]
        xs_c, ys_c, xs_s, ys_s = [], [], [], []
        for yi, cv, sv in zip(sc_E, core, surf):
            xs_c.extend([0.0, float(cv), None]); ys_c.extend([float(yi), float(yi), None])
            xs_s.extend([float(cv), float(cv+sv), None]); ys_s.extend([float(yi), float(yi), None])
        fig.add_trace(go.Scattergl(x=xs_c, y=ys_c, mode="lines", line=dict(color="#1f77b4", width=3), name="Core", legend="legend2"), row=1, col=4)
        fig.add_trace(go.Scattergl(x=xs_s, y=ys_s, mode="lines", line=dict(color="#ff7f0e", width=3), name="Surface", legend="legend2"), row=1, col=4)
        fig.update_xaxes(range=[0, 1.05], row=1, col=4)

    if len(coop_P) > 0:
        mask = (coop_E >= ewin[0]) & (coop_E <= ewin[1])
        E_sticks = coop_E[mask]
        scale, vmax_abs = 1.0, 1.0
        if normalize_coop:
            gmax = max((np.nanmax(np.abs(coop_V[p][mask])) for p in coop_P if coop_V[p][mask].size), default=0.0)
            scale = (1.0 / gmax) if gmax > 0 else 1.0
            fig.update_xaxes(range=[-1.05, 1.05], row=1, col=5)
        else:
            vmax_abs = max((np.nanmax(np.abs(coop_V[p][mask])) for p in coop_P if coop_V[p][mask].size), default=1.0)
            fig.update_xaxes(range=[-(vmax_abs*1.1), (vmax_abs*1.1)], row=1, col=5)
        for i, p in enumerate(coop_P):
            v = (coop_V[p][mask] * scale).astype(np.float32)
            if v.size == 0: continue
            xs, ys = [], []
            for yi, xv in zip(E_sticks, v):
                xs.extend([0.0, float(xv), None]); ys.extend([float(yi), float(yi), None])
            fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color=palette[i % len(palette)], width=4), name=p, legend="legend3"), row=1, col=5)

    for col in range(1, 6):
        line_col_ef = "white" if col == 1 else "rgba(0,0,0,0.4)"
        fig.add_hline(y=ef, line_dash="dash", line_color=line_col_ef, line_width=2.5, row=1, col=col, layer="above")
        if e_homo is not None: fig.add_hline(y=e_homo, line_dash="dot", line_color="royalblue", line_width=3, row=1, col=col, layer="above")
        if e_lumo is not None: fig.add_hline(y=e_lumo, line_dash="dot", line_color="crimson", line_width=3, row=1, col=col, layer="above")

    if e_homo is not None and e_lumo is not None:
        gap = e_lumo - e_homo
        fig.add_annotation(x=0.03, y=ef, xref="x domain", yref="y", text=f"<b>E<sub>g</sub> = {gap:.3f} eV</b>", showarrow=False, font=dict(color="white", size=24), xanchor="left", yanchor="bottom", yshift=8)
        fig.add_annotation(x=0.97, y=e_homo, xref="x domain", yref="y", text="<b>HOMO</b>", showarrow=False, font=dict(color="royalblue", size=22), xanchor="right", yanchor="bottom", yshift=6)
        fig.add_annotation(x=0.97, y=e_lumo, xref="x domain", yref="y", text="<b>LUMO</b>", showarrow=False, font=dict(color="crimson", size=22), xanchor="right", yanchor="top", yshift=-6)

    for col in range(1, 6):
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, ticks="outside", gridcolor='rgba(0,0,0,0.1)', zeroline=False, row=1, col=col)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, ticks="outside", gridcolor='rgba(0,0,0,0.1)', zeroline=False, row=1, col=col)
        
    fig.update_yaxes(range=[ewin[0], ewin[1]], title_text="<b>Energy (eV)</b>", title_font=dict(size=28), row=1, col=1)
    fig.update_xaxes(title_text="<b>k-Path</b>", title_font=dict(size=28), tickangle=0, row=1, col=1)
    fig.update_xaxes(title_text="<b>DOS</b>", title_font=dict(size=28), row=1, col=2)
    fig.update_xaxes(title_text="<b>IPR</b>", title_font=dict(size=28), tickvals=[0, 0.5, 1.0], row=1, col=3)
    fig.update_xaxes(title_text="<b>Char</b>", title_font=dict(size=28), tickvals=[0, 0.5, 1.0], row=1, col=4)
    fig.update_xaxes(title_text="<b>COOP</b>", title_font=dict(size=28), row=1, col=5)

    opts = [1, 10, 100, 1000]
    buttons = [dict(label=f"Contrast: {s}x", method="restyle", args=[{"zmin": [float(np.log10(max(vmin_base, vmax/s)))]}, [0]]) for s in opts]
    fig.update_layout(updatemenus=[dict(type="dropdown", buttons=buttons, x=0.17, y=1.24, xanchor="center", yanchor="bottom", font=dict(size=20), bgcolor="#f8f9fa", bordercolor="black")])

    plot_2d_html = fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True, 'displaylogo': False})


    # =========================================================================
    # PART 2: 3D MOLECULAR ORBITAL (CUBE) VISUALIZATION
    # =========================================================================
    if prefix == "soc":
        cube_files = [f for f in os.listdir('.') if f.lower().startswith('spinor_') and f.lower().endswith('.cube')]
    else:
        cube_files = [f for f in os.listdir('.') if f.lower().startswith('spatial_') and f.lower().endswith('.cube')]

    def extract_idx(name):
        name = name.upper()
        if "HOMO" in name:
            match = re.search(r'HOMO-(\d+)', name)
            return -int(match.group(1)) if match else 0
        elif "LUMO" in name:
            match = re.search(r'LUMO\+(\d+)', name)
            return 1 + (int(match.group(1)) if match else 0)
            
        # Fallback for other numbered files
        match = re.search(r'\d+', name)
        return int(match.group()) if match else 0
    
    cube_files.sort(key=extract_idx)
    
    plot_3d_html = ""
    first_iso_val = 0.0001
    
    if len(cube_files) > 0:
        if len(cube_files) >= 4:
            mid = len(cube_files) // 2
            cube_files = cube_files[mid-2 : mid+2]
            cube_titles = ["HOMO-1", "HOMO", "LUMO", "LUMO+1"]
        else:
            cube_titles = [os.path.basename(f).replace('.cube', '') for f in cube_files]
            
        fig_3d = make_subplots(
            rows=1, cols=len(cube_files),
            specs=[[{'type': 'scene'}] * len(cube_files)],
            subplot_titles=cube_titles,
            horizontal_spacing=0.02
        )
        
        for idx, cfile in enumerate(cube_files):
            print(f"    -> Parsing and mapping {cfile} to 3D grid...")
            X_grid, Y_grid, Z_grid, V_data, atoms = parse_cube(cfile)
            
            v_max = float(np.max(V_data))
            v_min = float(np.min(V_data))
            is_density = (v_min >= -1e-6)
            
            if prefix == "soc" or is_density:
                iso_val = 0.0001
                iso_pos = min(iso_val, v_max * 0.95)
                if iso_pos < 1e-6: iso_pos = v_max * 0.50
                fig_3d.add_trace(go.Isosurface(
                    x=X_grid, y=Y_grid, z=Z_grid, value=V_data,
                    isomin=iso_pos, isomax=iso_pos, surface_count=1,
                    colorscale=[[0, 'blue'], [1, 'blue']], showscale=False,
                    caps=dict(x_show=False, y_show=False, z_show=False), opacity=0.6, name="Density"
                ), row=1, col=idx+1)
            else:
                iso_val = max(abs(v_max), abs(v_min)) * 0.15
                iso_pos = min(iso_val, v_max * 0.95)
                iso_neg = max(-iso_val, v_min * 0.95)
                
                if iso_pos > 1e-6:
                    fig_3d.add_trace(go.Isosurface(
                        x=X_grid, y=Y_grid, z=Z_grid, value=V_data,
                        isomin=iso_pos, isomax=iso_pos, surface_count=1,
                        colorscale=[[0, 'blue'], [1, 'blue']], showscale=False,
                        caps=dict(x_show=False, y_show=False, z_show=False), opacity=0.35, name="Pos Lobe",
                        flatshading=False,
                    ), row=1, col=idx+1)
                
                if iso_neg < -1e-6:
                    fig_3d.add_trace(go.Isosurface(
                        x=X_grid, y=Y_grid, z=Z_grid, value=V_data,
                        isomin=iso_neg, isomax=iso_neg, surface_count=1,
                        colorscale=[[0, 'red'], [1, 'red']], showscale=False,
                        caps=dict(x_show=False, y_show=False, z_show=False), opacity=0.35, name="Neg Lobe",
                        flatshading=False,
                    ), row=1, col=idx+1)
                    
            if idx == 0: first_iso_val = iso_val
            
            b_xs, b_ys, b_zs = build_wireframe_agnostic(atoms)
            fig_3d.add_trace(go.Scatter3d(x=b_xs, y=b_ys, z=b_zs, mode='lines', line=dict(color='#555555', width=3), showlegend=False, hoverinfo='skip'), row=1, col=idx+1)
            
            atom_colors = {1: '#FFFFFF', 6: '#777777', 7: '#0000FF', 8: '#FF0000', 16: '#CCCC00', 14: '#FFC0CB', 15: '#FFA500'}
            ax, ay, az, ac = [], [], [], []
            for a in atoms:
                ax.append(a[1]); ay.append(a[2]); az.append(a[3]); ac.append(atom_colors.get(a[0], '#A0A0A0'))
            fig_3d.add_trace(go.Scatter3d(x=ax, y=ay, z=az, mode='markers', marker=dict(size=4, color=ac, line=dict(width=1, color='black')), showlegend=False, hoverinfo='skip'), row=1, col=idx+1)
            
        scene_config = dict(xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=''), yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=''), zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title=''), aspectmode='data')
        layout_update = {f"scene{i+1}" if i > 0 else "scene": scene_config for i in range(len(cube_files))}
        
        fig_3d.update_layout(
            **layout_update, paper_bgcolor="#fdfdfd", plot_bgcolor="#fdfdfd",
            margin=dict(l=10, r=10, t=50, b=10), height=500, font=dict(family="Helvetica, Arial, sans-serif", size=24, color="#222")
        )
        
        plot_3d_html = fig_3d.to_html(full_html=False, include_plotlyjs=False, div_id="mo_3d_plot")

    # =========================================================================
    # PART 3: WRAPPING HTML TEMPLATE AND GLOSSARY WITH NATIVE JS INPUT
    # =========================================================================
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>miniBSE {lbl} Electronic Structure Analysis</title>
        <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
        <style>
            body {{ font-family: 'Helvetica', 'Arial', sans-serif; background-color: #f8f9fa; margin: 0; padding: 20px; color: #333; }}
            .dashboard-container {{ max-width: 2400px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
            .plot-container {{ width: 100%; margin-bottom: 20px; }}
            .mo-container {{ width: 100%; background: #fdfdfd; border: 1px solid #eaeaea; border-radius: 8px; padding: 10px 0; margin-bottom: 40px; position: relative; }}
            .iso-control-bar {{ background: white; padding: 12px 25px; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: inline-flex; align-items: center; gap: 15px; border: 1px solid #eaeaea; margin: 15px 0 0 25px; z-index: 10; }}
            .iso-control-bar input {{ padding: 6px 12px; font-size: 16px; border: 1px solid #ccc; border-radius: 4px; width: 120px; }}
            .iso-control-bar button {{ padding: 8px 16px; font-size: 16px; background: #1f77b4; color: white; border: none; border-radius: 4px; cursor: pointer; transition: 0.2s; }}
            .iso-control-bar button:hover {{ background: #155d8f; }}
            .explanation-box {{ background: #fdfdfd; border: 1px solid #eaeaea; border-radius: 8px; padding: 30px; margin-top: 10px; }}
            .explanation-box h2 {{ margin-top: 0; color: #222; font-size: 26px; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-bottom: 20px; }}
            .glossary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 25px; }}
            .glossary-item {{ background: #fff; padding: 20px; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); border-left: 4px solid #1f77b4; }}
            .glossary-item > strong {{ display: block; font-size: 20px; margin-bottom: 10px; color: #111; }}
            .glossary-item p {{ margin: 0; font-size: 16px; line-height: 1.5; color: #555; }}
            .trap-indicator {{ border-left-color: #ff7f0e; }}
            .cube-indicator {{ border-left-color: #d62728; }}
        </style>
    </head>
    <body>
        <div class="dashboard-container">
            <div class="plot-container">
                {plot_2d_html}
            </div>
            
            {f'''
            <div class="mo-container">
                <div class="iso-control-bar">
                    <label for="isoInput"><strong>Isosurface Threshold (±):</strong></label>
                    <input type="number" id="isoInput" step="0.0001" value="{first_iso_val:.5f}">
                    <button onclick="updateIso()">Apply</button>
                </div>
                {plot_3d_html}
            </div>
            ''' if plot_3d_html else ''}
            
            <div class="explanation-box">
                <h2>Glossary of Electronic Structure Plots</h2>
                <div class="glossary-grid">
                    <div class="glossary-item">
                        <strong>1. Fuzzy Bands (Reciprocal Space)</strong>
                        <p>Projects the finite-size real-space Molecular Orbitals onto a bulk-like momentum ($k$) grid. A sharp, continuous band structure indicates bulk-like delocalized states, while flat, smeared lines across the Brillouin zone indicate highly localized molecules or defects.</p>
                    </div>
                    <div class="glossary-item">
                        <strong>2. PDOS (Projected Density of States)</strong>
                        <p>Shows how much specific elements (or orbitals) contribute to the overall electronic density at a given energy. The peaks correspond to available molecular orbitals.</p>
                    </div>
                    <div class="glossary-item trap-indicator">
                        <strong>3. IPR (Inverse Participation Ratio)</strong>
                        <p>A mathematical measure of spatial localization running from 0 to 1. An IPR near <strong>0.0</strong> indicates a delocalized "bulk" state spread across many atoms. An IPR near <strong>1.0</strong> indicates a state strictly localized onto a single atomic site (a strong indicator of a trap state).</p>
                    </div>
                    <div class="glossary-item trap-indicator">
                        <strong>4. Surface vs. Core Character</strong>
                        <p>Evaluates whether the electron density of a specific state resides in the inner core of the nanostructure or on the outer 25% of the radius. <em>Note: A state with both a high IPR and >90% Surface Character is definitively an unpassivated surface trap.</em></p>
                    </div>
                    <div class="glossary-item">
                        <strong>5. COOP (Crystal Orbital Overlap Population)</strong>
                        <p>Quantifies the chemical bonding interactions between specific pairs of atoms. <strong>Positive</strong> values (plotted to the right) indicate stabilizing bonding interactions, while <strong>negative</strong> values (plotted to the left) indicate destabilizing anti-bonding interactions.</p>
                    </div>
                    <div class="glossary-item cube-indicator">
                        <strong>6. 3D Molecular Orbitals (MOs)</strong>
                        <p>Interactive 3D representations of the electron density limits for the key frontier orbitals (HOMO-1, HOMO, LUMO, LUMO+1). Red and blue lobes represent positive and negative phase domains of the electron wavefunctions.</p>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
        function updateIso() {{
            var val = parseFloat(document.getElementById('isoInput').value);
            if (isNaN(val) || val <= 0) return;
            var plotDiv = document.getElementById('mo_3d_plot');
            if (!plotDiv) return;
            
            for (var i = 0; i < plotDiv.data.length; i++) {{
                var trace = plotDiv.data[i];
                if (trace.type === 'isosurface') {{
                    if (trace.isomin > 0) {{
                        Plotly.restyle(plotDiv, {{'isomin': val, 'isomax': val}}, [i]);
                    }} else if (trace.isomin < 0) {{
                        Plotly.restyle(plotDiv, {{'isomin': -val, 'isomax': -val}}, [i]);
                    }}
                }}
            }}
        }}
        </script>
    </body>
    </html>
    """
    
    out_html = f"fuzzy_dashboard_{prefix}.html"
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html_template)
        
    print(f"  [Plotter] Successfully saved elegant HTML dashboard to {out_html}")

