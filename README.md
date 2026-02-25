# miniBSE

**miniBSE** is a high-performance, lightweight post-DFT exciton solver designed for calculating and analyzing the excited states of molecules and semiconductor nanoclusters. 

By combining the ease of a Python interface with a lightning-fast C++ backend powered by `Libint2` and `Eigen3`, `miniBSE` offers researchers a scalable tool to investigate light-matter interactions, exciton delocalization, and charge-transfer (CT) characteristics without the overhead of massive quantum chemistry suites.

## What it Does

At its core, `miniBSE` solves the **Bethe-Salpeter Equation (BSE)** under the Tamm-Dancoff Approximation (TDA). It constructs an active-space electron-hole Hamiltonian using:
1. **DFT Ground State Data**: Takes Molecular Orbitals (MOs) and orbital energies from a prior ground-state DFT calculation.
2. **Analytic Integrals**: Uses `Libint2` to instantly compute Gaussian basis set overlaps and dipole transition matrices in real-space.
3. **Screened Coulomb Interactions**: Implements a distance-dependent Ohno-Klopman Coulomb kernel, screened by automated or user-defined bulk dielectric constants. Optional sTDA-like exchange kernels are also supported.
4. **Iterative Diagonalization**: Deploys a Davidson algorithm (or full dense diagonalization) to extract the lowest-lying excited states and their oscillator strengths.

Beyond calculating energies, `miniBSE` performs **extensive wavefunction analysis** based on the Dreuw/Plasser framework, outputting physical descriptors such as exciton radii ($d_{eh}$), spatial correlation (Pearson $R$), and volumetric transition densities.

---

## Installation

Because `miniBSE` relies on C++ extensions, **Conda is the highly recommended installation method**. Our `environment.yml` handles the installation of C++ compilers, `CMake`, `Eigen3`, and `Libint2`, saving you the hassle of system-level configurations.

### Method 1: Conda (Recommended)

1. Clone the repository:
   ```bash
   git clone [https://github.com/nlesc-nano/miniBSE.git](https://github.com/nlesc-nano/miniBSE.git)
   cd miniBSE
   ```
2. Create and activate the environment:
   ```bash
   conda env create -f environment.yml
   conda activate minibse_env
   ```
   *(Note: The `environment.yml` automatically installs the `miniBSE` package in editable mode via pip at the end of the process).*

### Method 2: Standard Pip

If you already have `CMake` (>= 3.16), a C++17 compiler, `Eigen3`, and `Libint2` installed natively on your OS:

```bash
git clone [https://github.com/nlesc-nano/miniBSE.git](https://github.com/nlesc-nano/miniBSE.git)
cd miniBSE
pip install -r requirements.txt
pip install -e .
```

---

## How to Use It

Once installed, the solver is accessible globally via the `minibse` command-line interface. 

### Typical Calculation

Below is a standard example for calculating the excited states of an Indium Arsenide (InAs) semiconductor cluster, computing the lowest states within a 2 eV threshold, and plotting the results:

```bash
minibse \
  --mo_file MOs_cleaned.txt \
  --xyz last_opt.xyz \
  --basis_txt BASIS_MOLOPT \
  --basis_name DZVP-MOLOPT-SR-GTH \
  --e_thresh 2 \
  --qp_gap 3.0278 \
  --sigma 0.03 \
  --plot \
  --full-diag \
  --nthreads 8 \
  --material INAS \
  --exchange \
  --alpha 0.2
```

### CLI Argument Breakdown

**Inputs & Structure:**
* `--mo_file`: Path to your molecular orbitals (supports `.txt` or `.npz` arrays).
* `--xyz`: The Cartesian coordinates of your system.
* `--basis_txt` & `--basis_name`: The basis set file and the specific basis name (e.g., CP2K MOLOPT format) used to generate the C++ integrals.

**Physics & Truncation:**
* `--qp_gap`: The target quasi-particle gap (in eV). `miniBSE` uses this to apply a "scissor shift" to the raw DFT HOMO-LUMO gap.
* `--e_thresh`: Energy threshold (in eV). Truncates the active space by discarding electron-hole transitions that exceed this gap.
* `--material`: Uses a built-in material database to estimate dielectric screening.
* `--alpha`: Manually overrides the Coulomb screening factor ($\alpha$).
* `--exchange`: Toggles the inclusion of the sTDA-like exchange matrix.

**Solver & Output Controls:**
* `--full-diag`: Forces full dense diagonalization. Omit this to use the iterative Davidson solver for larger systems.
* `--nthreads`: Number of CPU threads dedicated to C++ integral generation and PyTorch matrix contractions.
* `--sigma`: Broadening width (in eV) for the generated UV-Vis spectrum.
* `--plot`: Generates PNG spectra and an interactive HTML diagnostic dashboard.
* `--cube`: (Optional) Generates 3D volumetric `.cube` files of the brightest exciton's electron/hole densities.

---

## Outputs

A successful run of `miniBSE` will yield several outputs in your working directory:

1. **Standard Output**: A console table listing the excited states, energies, oscillator strengths, and primary orbital transitions (e.g., `HOMO -> LUMO+1`).
2. **`spectrum.png` & `spectrum_nm.png`**: UV-Vis absorption spectra utilizing your requested broadening (`--sigma`). 
3. **`exciton_analysis.html`**: An interactive Plotly dashboard. This visualizes exciton spatial correlations, sizes, and charge-transfer ratios across the energy spectrum.
4. **`exciton_results.csv`**: (If `--write-csv` is used) Tabular data containing detailed spatial metrics ($d_{CT}$, $\sigma_h$, $\sigma_e$) for post-processing or tracking across MD trajectories.
5. **Cube Files**: (If `--cube` is used) Volumetric densities ready to be visualized in software like VMD, PyMOL, or ChimeraX.

