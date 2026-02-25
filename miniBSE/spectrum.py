import numpy as np

def generate_spectrum(energies, f_strengths, e_min=0.0, e_max=10.0, n_points=1000, sigma=0.1, profile='gaussian'):
    """
    Convolutes discrete transitions with a broadening function.
    Returns: x_grid (eV), y_grid (Intensity)
    """
    x_grid = np.linspace(e_min, e_max, n_points)
    y_grid = np.zeros_like(x_grid)
    
    for E_i, f_i in zip(energies, f_strengths):
        if profile.lower() == 'gaussian':
            y_grid += f_i * np.exp(-0.5 * ((x_grid - E_i) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
        elif profile.lower() == 'lorentzian':
            y_grid += f_i * (sigma / np.pi) / ((x_grid - E_i)**2 + sigma**2)
            
    return x_grid, y_grid

def plot_spectrum(x_grid, y_grid, energies, f_strengths, filename="spectrum.png", show=False):
    """
    Plots the UV-Vis spectrum with a nice aesthetic using matplotlib.
    Generates two plots: one in eV and one in nm.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [Warning] matplotlib is not installed. Skipping plot generation.")
        return

    min_e = np.min(energies)
    max_e = np.max(energies)
    mask = np.array(f_strengths) > 1e-4

    # [FIXED]: Plot exactly up to the final excitation energy so it doesn't visually chop the tail
    right_cutoff_eV = max_e 

    # ==========================================
    # 1. PLOT IN ENERGY (eV)
    # ==========================================
    fig_eV, ax1_eV = plt.subplots(figsize=(8, 5), dpi=150)
    
    ax1_eV.plot(x_grid, y_grid, color='black', linewidth=1.5, label='Absorption Envelope')
    ax1_eV.fill_between(x_grid, y_grid, color='royalblue', alpha=0.3)
    ax1_eV.set_ylabel("Intensity (arb. units)", fontsize=12, fontweight='bold')
    
    ax2_eV = ax1_eV.twinx()
    ax2_eV.vlines(np.array(energies)[mask], ymin=0, ymax=np.array(f_strengths)[mask], 
               color='crimson', linewidth=1.5, alpha=0.8, label='Oscillator Strengths')
    ax2_eV.set_ylabel("Oscillator Strength ($f$)", fontsize=12, fontweight='bold', color='crimson')
    ax2_eV.tick_params(axis='y', labelcolor='crimson')
    
    ax1_eV.set_xlim(max(0.0, min_e - 0.5), right_cutoff_eV)
    
    ax1_eV.set_ylim(0, np.max(y_grid) * 1.1)
    if np.any(mask):
        ax2_eV.set_ylim(0, np.max(np.array(f_strengths)[mask]) * 1.1)
    
    ax1_eV.set_xlabel("Energy (eV)", fontsize=12, fontweight='bold')
    ax1_eV.set_title("Simulated UV-Vis Spectrum", fontsize=14, fontweight='bold')
    
    ax1_eV.tick_params(axis='x', which='major', labelsize=10)
    ax1_eV.grid(True, linestyle='--', alpha=0.6, color='gray')
    
    lines_1, labels_1 = ax1_eV.get_legend_handles_labels()
    lines_2, labels_2 = ax2_eV.get_legend_handles_labels()
    ax1_eV.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True, shadow=True)
    
    fig_eV.tight_layout()
    if filename: fig_eV.savefig(filename, bbox_inches='tight')

    # ==========================================
    # 2. PLOT IN WAVELENGTH (nm)
    # ==========================================
    fig_nm, ax1_nm = plt.subplots(figsize=(8, 5), dpi=150)
    
    valid_idx = x_grid > 0.1
    x_nm = 1240.0 / x_grid[valid_idx]
    y_nm = y_grid[valid_idx]
    energies_nm = 1240.0 / np.array(energies)

    ax1_nm.plot(x_nm, y_nm, color='black', linewidth=1.5, label='Absorption Envelope')
    ax1_nm.fill_between(x_nm, y_nm, color='mediumseagreen', alpha=0.3)
    ax1_nm.set_ylabel("Intensity (arb. units)", fontsize=12, fontweight='bold')
    
    ax2_nm = ax1_nm.twinx()
    ax2_nm.vlines(energies_nm[mask], ymin=0, ymax=np.array(f_strengths)[mask], 
               color='crimson', linewidth=1.5, alpha=0.8, label='Oscillator Strengths')
    ax2_nm.set_ylabel("Oscillator Strength ($f$)", fontsize=12, fontweight='bold', color='crimson')
    ax2_nm.tick_params(axis='y', labelcolor='crimson')
    
    min_nm = 1240.0 / right_cutoff_eV
    max_nm = 1240.0 / max(0.1, min_e - 0.5)
    ax1_nm.set_xlim(min_nm, max_nm)
    
    ax1_nm.set_ylim(0, np.max(y_nm) * 1.1)
    if np.any(mask):
        ax2_nm.set_ylim(0, np.max(np.array(f_strengths)[mask]) * 1.1)
    
    ax1_nm.set_xlabel("Wavelength (nm)", fontsize=12, fontweight='bold')
    ax1_nm.set_title("Simulated UV-Vis Spectrum", fontsize=14, fontweight='bold')
    
    ax1_nm.tick_params(axis='x', which='major', labelsize=10)
    ax1_nm.grid(True, linestyle='--', alpha=0.6, color='gray')
    ax1_nm.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', frameon=True, shadow=True)
    
    fig_nm.tight_layout()
    if filename:
        nm_filename = filename.replace(".png", "_nm.png")
        fig_nm.savefig(nm_filename, bbox_inches='tight')
        print(f"  Plots saved to '{filename}' and '{nm_filename}'")

    if show: plt.show()
    plt.close(fig_eV)
    plt.close(fig_nm)

