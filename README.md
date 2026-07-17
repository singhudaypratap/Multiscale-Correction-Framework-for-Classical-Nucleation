# Multiscale Correction Framework for Classical Nucleation Theory

Curvature-dependent surface tension, cooperative polarization, and electric field effects in water nanocluster formation.

**Authors:** Uday Pratap Singh\*, Mukesh Chandra, Bersha Kumari, Ebtasam Ahmad Siddiqui

**Target journal:** Journal of Molecular Liquids (Elsevier)

## Overview

This repository provides the complete computational framework and reproducibility materials for a multiscale correction to Classical Nucleation Theory (CNT) applied to water nanocluster formation under external electric fields. The framework incorporates three corrections absent from standard CNT: curvature-dependent surface tension via the Tolman length (including its sign), cooperative molecular polarization in finite clusters, and explicit electric field coupling through the Debye-Langevin polarizability.

The corrected nucleation barrier is

$$\Delta G^* = \frac{16\pi\sigma(r)^3}{3\Delta G_v^2}, \quad \sigma(r) = \frac{\sigma_\infty}{1 + 2\delta/r}$$

where $\delta \approx -0.05$ nm (TIP4P/2005) and the volumetric driving force includes field-dependent polarization contributions. Stockmayer fluid Monte Carlo simulations validate the field-induced stabilization predictions across 120 conditions (5 temperatures, 4 field strengths, 6 cluster sizes).

## Repository structure

```
.
├── README.md
├── Nanoscale_Water_Clusters_JMolLiq.docx        # Manuscript
├── Nanoscale_Water_Clusters_JMolLiq_Final.ipynb  # Reproducibility notebook
├── mc_stockmayer.c                               # Stockmayer MC engine (C)
├── Graphical_Abstract.png                        # Graphical abstract
├── Competing_Interests_Declaration.docx          # COI declaration
└── figures_nano_trends/                          # Publication figures (600 DPI)
    ├── fig1.png   # MC results: binding energy + dipole alignment
    ├── fig2.png   # Standard vs corrected CNT + Tolman sign effect
    ├── fig3.png   # Correction mechanisms (4 panels)
    ├── fig4.png   # Validation summary (4 panels)
    ├── fig5.png   # NP surface fields + rate enhancement
    └── fig6.png   # Dimensionless framework + anti-icing design
```

## Quick start

### Run the Monte Carlo simulation

```bash
gcc -O3 -o mc_stockmayer mc_stockmayer.c -lm
./mc_stockmayer 273    # temperature in Kelvin
```

Output is CSV: `T(K), E(V/m), N, E_total(ε), E_std(ε), E/N(ε), alignment, acc_rate`

### Reproduce all figures and tables

Open `Nanoscale_Water_Clusters_JMolLiq_Final.ipynb` in Jupyter and run all cells. Requirements: Python 3.8+, NumPy, Matplotlib.

```bash
pip install numpy matplotlib
jupyter notebook Nanoscale_Water_Clusters_JMolLiq_Final.ipynb
```

The notebook generates all 6 figures at 600 DPI into `figures_nano_trends/` and prints both manuscript tables.

## Key parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Tolman length δ | −0.05 nm | TIP4P/2005 MD (Refs. 9,10) |
| Cooperative parameter c | 0.15 | Calibrated to DFT cluster data |
| LJ σ (Stockmayer) | 3.166 Å | SPC/E water mapping |
| LJ ε/k_B | 78.2 K | SPC/E water mapping |
| Reduced dipole μ* | 3.16 | SPC/E water mapping |
| Supersaturation S | 5.0 | Typical atmospheric conditions |
| MC equilibration | 3000 + 20N sweeps | Convergence tested |
| MC production | 5000 + 30N sweeps | Convergence tested |

## Key results

- **30/30** MC conditions at E = 10⁹ V/m show field-induced stabilization
- Negative Tolman length **increases** the nucleation barrier by 15–25% relative to the classical positive assumption
- Sensitivity to cooperative parameter c remains below ±4% across the range [0.05, 0.25]
- Framework executes ~10⁷× faster than equivalent MD simulations

## Citation

If you use this code or framework, please cite:

> U.P. Singh, M. Chandra, B. Kumari, E.A. Siddiqui, "Multiscale Correction Framework for Classical Nucleation Theory: Curvature-Dependent Surface Tension, Cooperative Polarization, and Electric Field Effects in Water Nanocluster Formation," *Journal of Molecular Liquids* (2026). Submitted.

## License

This code is provided for academic reproducibility. Please contact the corresponding author (udaypratap@allduniv.ac.in) for reuse permissions.
