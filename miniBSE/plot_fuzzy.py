import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def load_fuzzy(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    Z = np.asarray(d["intensity"], dtype=float)
    centres = np.asarray(d["centres"], dtype=float)
    step = int(np.ceil((Z.size / 2_000_000) ** 0.5))
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

def load_coop_csv(csv_path):
    if not os.path.exists(csv_path): return np.array([]), [], {}
    df = pd.read_csv(csv_path)
    return df.iloc[:, 0].to_numpy(dtype=float), list(df.columns[1:]), {p: df[p].to_numpy(dtype=float) for p in df.columns[1:]}

def generate_interactive_plot(has_soc=False, material="DEFAULT", ef=None, e_homo=None, e_lumo=None, normalize_coop=False):
    print(f"  [Plotter] Generating elegant HTML dashboard...")
    
    # Defaults in case older scripts call this
    if ef is None: ef = {"sf": 0.0}
    if e_homo is None: e_homo = {"sf": None}
    if e_lumo is None: e_lumo = {"sf": None}

    nrows = 2 if has_soc else 1
    titles = ("Spin-Free Fuzzy Bands", "SF PDOS", "SF COOP")
    if has_soc: titles += ("SOC Fuzzy Bands", "SOC PDOS", "SOC COOP")

    fig = make_subplots(
        rows=nrows, cols=3, 
        shared_xaxes="columns", shared_yaxes="rows", 
        column_widths=[0.40, 0.30, 0.30],
        horizontal_spacing=0.04, vertical_spacing=0.08, 
        subplot_titles=titles
    )
    
    # --- GLOBAL TYPOGRAPHY: STRICTLY 24pt ---
    y_top = 1.05 + (0.04 if nrows == 1 else 0)
    fig.update_layout(
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", 
        height=900 * nrows,  
        font=dict(family="Helvetica, Arial, sans-serif", size=24, color="#222"),
        margin=dict(l=100, r=40, t=160, b=100),
        
        legend=dict(
            title=dict(text="<b>Atoms</b>", font=dict(size=24)),
            orientation="h", x=0.60, xanchor="center", y=y_top, yanchor="bottom",
            font=dict(size=24), bgcolor="rgba(255,255,255,0)"
        ),
        legend2=dict(
            title=dict(text="<b>Bonds</b>", font=dict(size=24)),
            orientation="h", x=0.85, xanchor="center", y=y_top, yanchor="bottom",
            font=dict(size=24), bgcolor="rgba(255,255,255,0)"
        )
    )

    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=28, family="Helvetica", color="#111")

    palette = ["#636EFA","#EF553B","#00CC96","#AB63FA","#FFA15A","#19D3F3","#FF6692"]
    runs = [("sf", 1)]
    if has_soc: runs.append(("soc", 2))
    
    heatmap_indices, vmin_bases, vmaxes = [], [], []

    for prefix, row in runs:
        fuzzy = load_fuzzy(f"fuzzy_data_{prefix}.npz")
        pdos_E, pdos_L, pdos_Y = load_pdos_csv(f"pdos_data_{prefix}.csv")
        coop_E, coop_P, coop_V = load_coop_csv(f"coop_data_{prefix}.csv")

        ef_val = ef.get(prefix, 0.0)
        e_homo_val = e_homo.get(prefix, None)
        e_lumo_val = e_lumo.get(prefix, None)

        # 1. Fuzzy Map
        Z, ewin = fuzzy["Z"], fuzzy["ewin"]
        kx = np.linspace(fuzzy["extent"][0], fuzzy["extent"][1], Z.shape[1])
        
        Zpos = Z[Z > 1e-9]
        vmax = float(np.percentile(Z, 99.9))
        vmin_base = float(np.percentile(Zpos, 5)) if Zpos.size else 1e-6
        vmin_bases.append(vmin_base); vmaxes.append(vmax)
        
        Zm = Z.astype(np.float32)
        Zm[Zm <= 0] = np.nan
        
        # Black Background
        axis_num = 3 * (row - 1) + 1
        x_str = f"x{axis_num if axis_num > 1 else ''}"
        y_str = f"y{axis_num if axis_num > 1 else ''}"
        fig.add_shape(type="rect", xref=x_str, yref=y_str, x0=kx[0], x1=kx[-1], y0=ewin[0], y1=ewin[1], fillcolor="black", line=dict(width=0), layer="below") 
        heat = go.Heatmap(
            z=np.log10(Zm), x=kx, y=fuzzy["centres"], colorscale="Inferno",
            zmin=float(np.log10(vmin_base)), zmax=float(np.log10(vmax)),
            showscale=(row == 1),
            colorbar=dict(
                title=dict(text="<b>log₁₀(I)</b>", font=dict(size=24)), 
                orientation="h", len=0.35, thickness=20, 
                x=0.20, xanchor="center", y=y_top, yanchor="bottom",
                tickfont=dict(size=24)
            ) if row == 1 else None,
            hovertemplate="k=%{x:.3f} Å⁻¹<br>E=%{y:.3f} eV<br>log10(I)=%{z:.2f}<extra></extra>"
        )
        fig.add_trace(heat, row=row, col=1)
        heatmap_indices.append(len(fig.data) - 1)
        
        # Ticks and Lines
        if fuzzy["tick_positions"].size:
            scale = (kx[-1] - kx[0]) / (Z.shape[1] - 1)
            tpos_plot = kx[0] + fuzzy["tick_positions"] * scale
            for x in tpos_plot:
                fig.add_vline(x=x, line_color="rgba(255,255,255,0.4)", line_width=2, row=row, col=1)
            if row == nrows:
                fig.update_xaxes(tickmode="array", tickvals=tpos_plot, ticktext=fuzzy["tick_labels"], row=row, col=1)
            else:
                fig.update_xaxes(showticklabels=False, row=row, col=1)

        # HOMO/LUMO Lines
        for col in (1, 2, 3):
            fig.add_hline(y=ef_val, line_dash="dash", line_color="white" if col==1 else "rgba(0,0,0,0.4)", line_width=2.5, row=row, col=col)
            if e_homo_val is not None: fig.add_hline(y=e_homo_val, line_dash="dot", line_color="royalblue", line_width=3, row=row, col=col)
            if e_lumo_val is not None: fig.add_hline(y=e_lumo_val, line_dash="dot", line_color="crimson", line_width=3, row=row, col=col)

        # --- Band Gap & HOMO/LUMO Annotations ---
        if e_homo_val is not None and e_lumo_val is not None:
            gap = e_lumo_val - e_homo_val
            
            axis_num = 3 * (row - 1) + 1
            x_str = f"x{axis_num if axis_num > 1 else ''}"
            y_str = f"y{axis_num if axis_num > 1 else ''}"
            
            # E_g text (Left side, white text over black)
            fig.add_annotation(
                x=0.03, y=ef_val, 
                xref=f"{x_str} domain", yref=y_str,
                text=f"<b>E<sub>g</sub> = {gap:.3f} eV</b>",
                showarrow=False, font=dict(color="white", size=24),
                xanchor="left", yanchor="bottom", yshift=8 
            )
            
            # HOMO text (Right side, anchored above the blue line)
            fig.add_annotation(
                x=0.97, y=e_homo_val, 
                xref=f"{x_str} domain", yref=y_str,
                text="<b>HOMO</b>",
                showarrow=False, font=dict(color="royalblue", size=22),
                xanchor="right", yanchor="bottom", yshift=6 
            )

            # LUMO text (Right side, anchored below the red line)
            fig.add_annotation(
                x=0.97, y=e_lumo_val, 
                xref=f"{x_str} domain", yref=y_str,
                text="<b>LUMO</b>",
                showarrow=False, font=dict(color="crimson", size=22),
                xanchor="right", yanchor="top", yshift=-6 
            )

        # 2. PDOS
        if len(pdos_L) > 0:
            for j, lab in enumerate(pdos_L):
                fig.add_trace(go.Scatter(
                    x=pdos_Y[:, j], y=pdos_E, mode="lines", fill="tonextx" if j > 0 else "tozerox",
                    line=dict(width=1.0, color="rgba(0,0,0,0)"), fillcolor=palette[j % len(palette)], 
                    name=lab, showlegend=(row==1), legend="legend", hovertemplate=f"{lab}: %{{x:.3f}}<br>E=%{{y:.3f}} eV<extra></extra>"
                ), row=row, col=2)
            
            total = pdos_Y[:, -1]
            fig.add_trace(go.Scatter(x=total, y=pdos_E, mode="lines", line=dict(color="black", width=3), name="Total DOS", showlegend=False), row=row, col=2)
            fig.update_xaxes(range=[0, float(max(total.max(), 1e-12)) * 1.05], row=row, col=2)
            if row < nrows: fig.update_xaxes(showticklabels=False, row=row, col=2)

        # 3. COOP
        if len(coop_P) > 0:
            mask = (coop_E >= ewin[0]) & (coop_E <= ewin[1])
            E_sticks = coop_E[mask]
            scale, vmax_abs = 1.0, 1.0
            
            if normalize_coop:
                gmax = max((np.nanmax(np.abs(coop_V[p][mask])) for p in coop_P if coop_V[p][mask].size), default=0.0)
                scale = (1.0 / gmax) if gmax > 0 else 1.0
                fig.update_xaxes(range=[-1.05, 1.05], row=row, col=3)
            else:
                vmax_abs = max((np.nanmax(np.abs(coop_V[p][mask])) for p in coop_P if coop_V[p][mask].size), default=1.0)
                fig.update_xaxes(range=[-(vmax_abs*1.1), (vmax_abs*1.1)], row=row, col=3)
            
            if row < nrows: fig.update_xaxes(showticklabels=False, row=row, col=3)

            for i, p in enumerate(coop_P):
                v = (coop_V[p][mask] * scale).astype(np.float32)
                if v.size == 0: continue
                xs, ys = [], []
                for yi, xv in zip(E_sticks, v):
                    xs.extend([0.0, float(xv), None])
                    ys.extend([float(yi), float(yi), None])
                fig.add_trace(go.Scattergl(
                    x=xs, y=ys, mode="lines", line=dict(color=palette[i % len(palette)], width=4), 
                    name=p, hoverinfo="skip", showlegend=(row==1), legend="legend2"
                ), row=row, col=3)

        # Apply robust Axis formatting at 24pt
        fig.update_yaxes(range=[ewin[0], ewin[1]], title_text="<b>Energy (eV)</b>", title_font=dict(size=28), tickfont=dict(size=24), row=row, col=1)
        fig.update_yaxes(tickfont=dict(size=24), row=row, col=2)
        fig.update_yaxes(tickfont=dict(size=24), row=row, col=3)
        if row == nrows:
            fig.update_xaxes(title_text="<b>k-Path</b>", title_font=dict(size=28), tickfont=dict(size=24), tickangle=0, row=row, col=1)
            fig.update_xaxes(title_text="<b>DOS (a.u.)</b>", title_font=dict(size=28), tickfont=dict(size=24), row=row, col=2)
            fig.update_xaxes(title_text="<b>COOP (a.u.)</b>", title_font=dict(size=28), tickfont=dict(size=24), row=row, col=3)

    if heatmap_indices:
        opts = [1, 10, 100, 1000]
        buttons = []
        for sval in opts:
            zmins = [float(np.log10(max(vmin_bases[i], vmaxes[i]/sval))) for i in range(len(heatmap_indices))]
            buttons.append(dict(label=f"Contrast: {sval}x", method="restyle", args=[{"zmin": zmins}, heatmap_indices]))
        fig.update_layout(updatemenus=[
            dict(type="dropdown", buttons=buttons, x=0.01, y=y_top + 0.05, 
                 xanchor="left", yanchor="bottom", font=dict(size=20, color="#333"), bgcolor="#f8f9fa", bordercolor="#ddd")
        ])

    out_html = f"fuzzy_dashboard_{material}.html"
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True, config={"responsive": True, "displaylogo": False})
    print(f"  [Plotter] Successfully saved elegant HTML dashboard to {out_html}")

