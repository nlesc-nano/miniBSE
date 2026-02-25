import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
from scipy.integrate import simps

# Constants
KB_EV = 8.617333262145e-5  # Boltzmann constant in eV/K

def main():
    parser = argparse.ArgumentParser(description="miniBSE MD Trajectory Post-Processor")
    parser.add_argument("--pattern", type=str, default="frame_*/exciton_results.csv", help="Glob pattern for CSVs")
    parser.add_argument("--dt", type=float, default=2.5, help="Time step between frames in fs")
    parser.add_argument("--sigma", type=float, default=0.05, help="Broadening for ensemble spectrum (eV)")
    parser.add_argument("--temp", type=float, default=300.0, help="Simulation temperature in K for Boltzmann weighting")
    args = parser.parse_args()

    # 1. Gather all data
    files = sorted(glob.glob(args.pattern), key=lambda x: "".join(filter(str.isdigit, x)) or 0)
    if not files:
        print(f"No CSV files found matching: {args.pattern}")
        return

    frames = [pd.read_csv(f) for f in files]
    n_frames = len(frames)
    
    # 2. Extract S1 Trajectory Data
    times = np.array([df['Time'].iloc[0] for df in frames])
    energies = np.array([df[df['State'] == 1]['Energy_eV'].iloc[0] for df in frames])
    f_osc = np.array([df[df['State'] == 1]['f_osc'].iloc[0] for df in frames])
    
    # Vectors & Spatial Metrics
    mu_x = np.array([df[df['State'] == 1]['mu_x'].iloc[0] for df in frames])
    mu_y = np.array([df[df['State'] == 1]['mu_y'].iloc[0] for df in frames])
    mu_z = np.array([df[df['State'] == 1]['mu_z'].iloc[0] for df in frames])
    mu_vectors = np.vstack((mu_x, mu_y, mu_z)).T

    sigma_h = np.array([df[df['State'] == 1]['sigma_h_A'].iloc[0] for df in frames])
    sigma_e = np.array([df[df['State'] == 1]['sigma_e_A'].iloc[0] for df in frames])
    d_eh = np.array([df[df['State'] == 1]['d_eh_A'].iloc[0] for df in frames])
    d_ct = np.array([df[df['State'] == 1]['d_CT_A'].iloc[0] for df in frames])

    # 3. Calculations
    # Kubo Dipole Autocorrelation C_mu(t)
    c_mu = np.zeros(n_frames)
    for t in range(n_frames):
        dot_products = np.sum(mu_vectors[:n_frames-t] * mu_vectors[t:], axis=1)
        c_mu[t] = np.mean(dot_products)
    c_mu /= c_mu[0]

    # Energy Autocorrelation C_E(t)
    de = energies - np.mean(energies)
    c_e = np.correlate(de, de, mode='full')[n_frames-1:] / np.correlate(de, de, mode='valid')[0]

    # Radiative Lifetimes (ns)
    k_inst = (f_osc * (energies**2)) / 1.499
    
    # Boltzmann
    E_min = np.min(energies)
    boltz_weights = np.exp(-(energies - E_min) / (KB_EV * args.temp))
    k_boltz = np.sum(k_inst * boltz_weights) / np.sum(boltz_weights)
    tau_boltz = 1.0 / k_boltz if k_boltz > 0 else np.nan

    # Strickler-Berg
    e_grid = np.linspace(np.min(energies)-0.5, np.max(energies)+0.5, 1000)
    total_spec = np.zeros_like(e_grid)
    for e, f in zip(energies, f_osc):
        total_spec += f * np.exp(-0.5 * ((e_grid - e) / args.sigma)**2)
    total_spec /= n_frames
    
    integral_abs = simps(total_spec / e_grid, e_grid)
    avg_E3_inv = simps(total_spec / (e_grid**3), e_grid) / simps(total_spec, e_grid)
    k_sb = (1.0 / 1.499) * (integral_abs / avg_E3_inv) * 0.01
    tau_sb = 1.0 / k_sb if k_sb > 0 else np.nan

    # Calculate 1/e decay times for dynamics
    def get_decay_time(corr_array, dt):
        idx = np.where(corr_array < np.exp(-1))[0]
        return idx[0] * dt if len(idx) > 0 else float('inf')
        
    tau_c_e = get_decay_time(c_e, args.dt)
    tau_c_mu = get_decay_time(c_mu, args.dt)

    # ---------------------------------------------------------
    # 4. COMPREHENSIVE CONSOLE OUTPUT
    # ---------------------------------------------------------
    print("==========================================================")
    print(f" miniBSE MD Trajectory Analysis (@ {args.temp} K)")
    print("==========================================================")
    print(f"[Trajectory Data]")
    print(f"  Frames processed : {n_frames}")
    print(f"  Time span        : {times[0]:.2f} to {times[-1]:.2f} fs (dt = {args.dt} fs)")
    print("")
    print(f"[Energetics (S1)]")
    print(f"  Mean Energy      : {np.mean(energies):.4f} ± {np.std(energies):.4f} eV")
    print(f"  Min / Max Energy : {np.min(energies):.4f} / {np.max(energies):.4f} eV")
    print("")
    print(f"[Exciton Spatial Analysis & Traps]")
    print(f"  Mean Hole Size (sigma_h)     : {np.mean(sigma_h):.2f} A")
    print(f"  Mean Electron Size (sigma_e) : {np.mean(sigma_e):.2f} A")
    print(f"  Hole Trapped (<3.0 A)        : {(np.sum(sigma_h < 3.0) / n_frames) * 100:.1f} % of frames")
    print(f"  Electron Trapped (<3.0 A)    : {(np.sum(sigma_e < 3.0) / n_frames) * 100:.1f} % of frames")
    print(f"  Mean e-h Separation (d_eh)   : {np.mean(d_eh):.2f} A")
    print("")
    print(f"[Photophysics & Lifetimes]")
    print(f"  Mean f_osc                   : {np.mean(f_osc):.4f}")
    print(f"  Dark Frames (f_osc < 0.05)   : {(np.sum(f_osc < 0.05) / n_frames) * 100:.1f} %")
    print(f"  Naive Mean Tau               : {np.mean(1.0/k_inst):.2f} ns")
    print(f"  Boltzmann-Weighted Tau       : {tau_boltz:.2f} ns")
    print(f"  Strickler-Berg Tau           : {tau_sb:.2f} ns")
    print("")
    print(f"[Dynamical Decoherence (1/e decay)]")
    print(f"  Energy Gap Memory, C_E(t)    : {tau_c_e:.1f} fs")
    print(f"  Dipole Coherence, C_mu(t)    : {tau_c_mu:.1f} fs")
    print("==========================================================\n")

    # ---------------------------------------------------------
    # 5. Multi-Panel Visualizations
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2)

    # Plot 1: Ensemble Spectrum
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(e_grid, total_spec, color='black', lw=2)
    ax1.fill_between(e_grid, total_spec, color='gray', alpha=0.3)
    ax1.set_title(f"Absorption Spectrum ({args.temp} K)")
    ax1.set_xlabel("Energy (eV)")

    # Plot 2: Dreuw/Plasser Spatial Tracking
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(times, d_eh, label='d_eh (e-h dist)', color='purple', alpha=0.7)
    ax2.plot(times, sigma_h, label='sigma_h (Hole)', color='blue', lw=2)
    ax2.plot(times, sigma_e, label='sigma_e (Electron)', color='green', lw=2)
    ax2.axhline(y=3.0, color='red', linestyle='--', label='Trap Threshold')
    ax2.set_title("Exciton Spatial Evolution")
    ax2.set_ylabel("Distance (A)")
    ax2.legend()

    # Plot 3: Energy Gap Autocorrelation
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(times[:n_frames//2], c_e[:n_frames//2], color='darkred')
    ax3.set_title("Energy Gap Autocorrelation $C_E(t)$")
    ax3.set_xlabel("Lag Time (fs)")

    # Plot 4: Kubo Dipole Autocorrelation
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(times[:n_frames//2], c_mu[:n_frames//2], color='teal', lw=2)
    ax4.axhline(0, color='black', lw=0.5)
    ax4.set_title("Dipole Autocorrelation $C_\mu(t)$ (Kubo)")
    ax4.set_xlabel("Lag Time (fs)")

    # Plot 5: Dual Brightness vs Carrier Size
    ax5 = fig.add_subplot(gs[2, 0])
    sc1 = ax5.scatter(sigma_h, f_osc, c=energies, cmap='coolwarm', marker='o', s=40, alpha=0.7, label='Hole')
    sc2 = ax5.scatter(sigma_e, f_osc, c=energies, cmap='coolwarm', marker='^', s=40, alpha=0.7, label='Electron')
    ax5.set_title("Brightness vs. Carrier Sizes")
    ax5.set_xlabel("Carrier Size (A)")
    ax5.set_ylabel("f_osc")
    ax5.legend()
    plt.colorbar(sc1, ax=ax5, label='Energy (eV)')

    # Plot 6: Exciton-Phonon Coupling (PSD)
    ax6 = fig.add_subplot(gs[2, 1])
    freqs = np.fft.rfftfreq(n_frames, d=args.dt)
    psd = np.abs(np.fft.rfft(de))**2
    ax6.plot(freqs * 33356.4, psd, color='indigo')
    ax6.set_xlim(0, 500)
    ax6.set_title("Exciton-Phonon Coupling (Spectral Density)")
    ax6.set_xlabel("Frequency (cm^-1)")

    plt.tight_layout()
    out_name = f"md_analysis_{int(args.temp)}K.png"
    plt.savefig(out_name, dpi=300)
    print(f"Full analysis plot saved to: {out_name}")
    # plt.show() # Uncomment if running interactively

if __name__ == "__main__":
    main()

