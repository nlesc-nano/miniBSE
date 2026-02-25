# miniBSE

Lightweight post-DFT exciton solver based on:

- KS orbitals
- Scissor operator
- Screened Coulomb kernel (hardness model)
- TDA approximation
- Optional sTDA-like kernel

## Features

- AO-based Löwdin orthogonalization
- Subspace selection (HOMO ± N)
- Davidson solver
- Libint-backed integrals

## Build libint backend

```bash
cd libint
mkdir build && cd build
cmake ..
make


