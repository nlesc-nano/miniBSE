# miniBSE

**miniBSE** is a lightweight, high-performance post-DFT exciton solver designed for the study of excited states in molecular and nanocluster systems. It leverages a C++ backend for fast integral evaluation and Python for advanced wavefunction analysis and diagonalization.

## Key Features

* **Fast Integral Engine**: Utilizes a C++ backend powered by `Libint2` and `Eigen3` to compute AO overlaps, dipole matrices, and analytic Fourier transforms.
* **Flexible Solver**: Supports both full dense diagonalization and the iterative **Davidson algorithm** for handling large transition spaces.
* **Advanced Exciton Descriptors**: Includes comprehensive spatial analysis based on the Dreuw/Plasser framework:
    * Exciton size ($d_{eh}$) and participation ratio (PR).
    * Electron-hole spatial correlation (Pearson $R$).
    * Charge transfer distance ($d_{CT}$) and particle RMS spread ($\sigma_h$, $\sigma_e$).
* **Physics-Based Screening**: Implements various screening models including SOS-polarizability, geometric, and dielectric confinement.
* **3D Visualizations**: Generates volumetric `.cube` files for electron, hole, and transition densities.
* **Automated Broadening**: Produces UV-Vis spectra with Gaussian or Lorentzian broadening.

## Installation

### Prerequisites

* **CMake** (>= 3.16)
* **C++17** compatible compiler
* **Libint2** and **Eigen3** libraries
* **Python 3.12+**

### Setup

Clone the repository and install it in editable mode using `pip`. The `setup.py` script will automatically invoke CMake to compile the C++ bindings:

```bash
git clone https://github.com/nlesc-nano/miniBSE.git
cd miniBSE
pip install -e .
```

## Usage

After installation, the package provides a command-line interface tool named `minibse`.

### Typical Command

Below is a standard example for calculating the excited states of a semiconductor cluster (e.g., InAs):

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

### Argument Overview

* `--mo_file`: Path to the molecular orbital coefficients.
* `--qp_gap`: Target quasi-particle gap (eV) to determine the scissor shift.
* `--e_thresh`: Energy threshold (eV) for CI space truncation.
* `--material`: Pre-defined material properties for automated screening calculations.
* `--exchange`: Activates the sTDA-like exchange term.
* `--cube`: Generates 3D transition density files.

