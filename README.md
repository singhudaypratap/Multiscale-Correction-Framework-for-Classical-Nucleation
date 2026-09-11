# A Multiscale Correction Framework for Classical Nucleation Theory

Curvature-dependent surface tension, cooperative polarization, and electric-field effects in water vapor–liquid nucleation.

**Authors:** Uday Pratap Singh*, Bersha Kumari, Mukesh Chandra, Ebtasam Ahmad Siddiqui

**Target journal:** *Colloids and Surfaces A: Physicochemical and Engineering Aspects* (Elsevier)

## Overview

This repository provides the computational framework and reproducibility materials for a multiscale correction framework for Classical Nucleation Theory (CNT) applied to nanoscale water vapor–liquid clusters. The framework examines three nanoscale contributions that are not explicitly represented in standard CNT: curvature-dependent surface tension through the Tolman length, size-dependent cooperative molecular polarization, and electric-field coupling.

The curvature-dependent surface tension is written as

$$
\sigma(r)=\frac{\sigma_\infty}{1+\frac{2\delta}{r}}
$$

and the corresponding curvature-corrected CNT barrier is

$$
\Delta G^=\frac{16\pi\left[\sigma(r^)\right]^3}{3\left(\Delta G_{v,\mathrm{eff}}\right)^2}
$$

where the effective volumetric driving force includes the modelled polarization and electric-field contribution.

The framework is intended as a physically motivated screening and correction approach rather than a replacement for molecular simulation of nucleation kinetics. In particular, the Stockmayer Monte Carlo calculations are used as an **energetic consistency check of the field response of preformed dipolar clusters**, not as direct validation of nucleation free energies or nucleation rates.

The present study focuses on **water vapor–liquid nucleation/condensation**. It does not model ice nucleation.

## Reproducibility scope

The production Stockmayer Monte Carlo campaign contains:

- **120 matched E = 0 versus E = 10^9 V/m comparisons**
- **240 individual trajectories** in total
- Cluster sizes: **N = 20, 50, 100**
- Temperatures: **T = 233, 253, 273, 293, 313 K**
- **8 independent replicas** for each `(T, N, E)` condition
- Equilibration: **3000 sweeps**
- Production: **5000 sweeps**
- Ewald electrostatics for the dipolar interactions
- GPU execution using JAX

Energy reduction was defined as

$$
R_E=100\\frac{E_0-E_E}{|E_0|},
$$

so that positive values indicate lower configurational energy in the applied-field case.

Across the 120 matched pairs, the mean energy reduction was **20.93%** (95% CI: **20.53–21.34%**), with a range of **12.88–24.53%**. All 120 matched comparisons showed lower configurational energy under the applied field. A paired comparison gave **t = 103.01, p < 0.001**.

These results demonstrate a reproducible energetic field response within the Stockmayer model. They should not be interpreted as direct evidence for a corresponding quantitative change in real-water nucleation rates.

## Repository structure

The repository is organized around the manuscript, analytical CNT calculations, Monte Carlo simulations, and publication figures.

```text
.
├── README.md
├── manuscript / manuscript source files
├── reproducibility notebook(s)
├── stockmayer_ewald_gpu_VERIFIED.py      # GPU Stockmayer MC production calculation
├── mc_stockmayer.c                       # Legacy/CPU Stockmayer MC implementation
├── Graphical_Abstract_CNT_BW_Polished.png
├── Graphical_Abstract_CNT_BW_Polished.pptx
└── figures_nano_trends/
    └── publication figures and supporting plots
```

File names may differ between repository releases; the production dataset should be identified by the matched-pair Stockmayer MC output containing **240 trajectories / 120 matched pairs**.

## Quick start

### Analytical CNT calculations

The analytical calculations can be reproduced with Python using NumPy and Matplotlib.

```bash
pip install numpy matplotlib jupyter
jupyter notebook
```

Open the reproducibility notebook included with the repository and run the cells sequentially.

The notebook calculates the CNT parameter grid, curvature-dependent surface tension, polarization contribution, critical radius, critical cluster size, and nucleation-barrier quantities reported in the manuscript.

### Stockmayer Monte Carlo

The verified production implementation uses JAX with GPU acceleration and Ewald electrostatics.

The production simulation parameters are:

```text
Cluster sizes:       N = 20, 50, 100
Temperatures:        233, 253, 273, 293, 313 K
Electric fields:     E = 0, 1e9 V/m
Replicas:             8 per condition
Equilibration:       3000 sweeps
Production:           5000 sweeps
```

The code should be run in an environment with a compatible JAX/CUDA installation. GPU availability can be checked with:

```python
import jax
print(jax.devices())
```

The production calculation writes a matched-pair CSV containing the individual trajectory results.

## Key model parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Molecular polarizability, α_e | 1.45 × 10⁻³⁰ m³ | Molecular polarizability volume used in the field term |
| SI polarizability conversion | 4π ε₀ α_e | Conversion from polarizability volume to SI polarizability |
| Cooperative parameter, c | 0.15 | Phenomenological parameter used in the model |
| LJ σ | 3.166 Å | Stockmayer water mapping |
| LJ ε/k_B | 78.2 K | Stockmayer water mapping |
| Reduced dipole, μ* | 4.0148 | Value used in the verified Stockmayer implementation |
| Dipole scale, μ_scale | 1.95247 × 10⁻³⁰ C m | Conversion used for the Stockmayer field scale |
| Reference supersaturation, S | 5.0 | Representative CNT screening condition |
| Applied field | 10⁹ V/m | Field used for the matched-pair MC consistency test |

The polarizability is treated dimensionally as a volume in the input formulation, with the SI conversion

$$
\\alpha_{\\mathrm{SI}}=4\\pi\\epsilon_0\\alpha_e.
$$

The electric-field interaction is therefore kept dimensionally consistent with an induced-dipole energy term.

## Curvature correction

The framework uses the Tolman form

$$
\\sigma(r)=\\frac{\\sigma_\\infty}{1+2\\delta/r}.
$$

The sign of the Tolman length is important because it changes the curvature dependence of the surface tension and, consequently, the predicted CNT barrier.

The manuscript does not treat a particular Tolman-length value or sign as a universal property of all water models. Reported values depend on the molecular model, cluster definition, temperature, and fitting procedure. The repository therefore preserves the parameter choices used in the manuscript and separates model assumptions from general conclusions.

## Polarization and electric-field treatment

The framework includes a size-dependent molecular polarization contribution and an electric-field coupling term. The field contribution is used to examine how an applied electric field can alter the energetic balance entering the CNT screening calculation.

The Stockmayer simulations provide an independent molecular-level check of the **energetic response** to the applied field. They do not calculate a nucleation free-energy surface, critical nucleus through rare-event sampling, or absolute nucleation rate.

The field effect should also be interpreted with care because the thermodynamic constraint matters. A field-induced change at fixed vapor chemical potential is not automatically equivalent to a change at fixed supersaturation or fixed pressure.

## CNT parameter-grid screening

The manuscript evaluates a corrected CNT parameter grid containing **1,344 combinations**.

The resulting critical cluster size spans approximately:

```text
N* ≈ 1.9–289.8 molecules
```

For reporting purposes, the study divides the results into operational screening categories:

- **Preferred CNT-screening range:** 899 cases (66.9%)
- **Caution range:** 257 cases (19.1%)
- **Below the adopted continuum-screening boundary:** 188 cases (14.0%)

These categories are **study-specific reporting boundaries**, not universal literature-established limits for CNT validity.

## Interpretation of the Monte Carlo results

The matched-pair simulations give the following aggregate field response:

| Quantity | Result |
|----------|--------|
| Matched pairs | 120 |
| Individual trajectories | 240 |
| Field comparison | E = 0 vs. 10⁹ V/m |
| Cluster sizes | N = 20, 50, 100 |
| Temperatures | 233–313 K |
| Mean energy reduction | 20.93% |
| 95% CI | 20.53–21.34% |
| Range | 12.88–24.53% |
| Pairs with lower field energy | 120/120 |
| Paired t-test | t = 103.01, p < 0.001 |

The response varied across temperature and cluster size, but remained positive for every matched pair in this model campaign. The result is best described as **Stockmayer-model energetic stabilization under the applied field**.

It should not be used to claim that real water necessarily exhibits the same percentage stabilization or that the result quantitatively predicts an experimental nucleation-rate enhancement.

## Limitations

The following limitations are important for interpreting the repository results:

1. **Stockmayer representation:** the model contains point dipoles and Lennard-Jones interactions and does not reproduce the full hydrogen-bonding physics of real water.
2. **Preformed clusters:** the Monte Carlo calculations examine energetic/structural response of clusters of specified size rather than directly simulating spontaneous nucleation.
3. **No direct nucleation kinetics:** nucleation free-energy profiles and absolute nucleation rates are not obtained from the Stockmayer simulations.
4. **Phenomenological parameters:** the cooperative parameter is a model parameter and should not be regarded as a universal material constant.
5. **Continuum limitations:** CNT and Tolman-type corrections become increasingly model-dependent as the cluster size decreases.
6. **Electric-field representation:** the nanoparticle surface-field calculation is a vacuum point-charge screening estimate, not a material-specific electrostatic field map.
7. **Water phase scope:** the present framework addresses vapor-to-liquid condensation and should not be interpreted as a quantitative model of ice nucleation.
8. **Validation:** quantitative prediction for real systems requires independent molecular simulation and/or experimental validation outside the calibration and screening calculations presented here.

## Figures

Publication figures are provided separately from the computational source files. The main figure set covers:

- Stockmayer Monte Carlo convergence and energetic response
- Matched-pair field response
- Temperature and polarization effects
- Standard and curvature-dependent CNT barriers
- Tolman-length sign dependence
- Parameter-grid sensitivity
- Critical cluster-size distribution
- Nanoparticle surface-field estimate
- Relative nucleation-enhancement proxy
- Comparison and positioning of nucleation approaches
- Proposed calibration workflow
- Physical mechanism of the framework

## Citation

If you use the code, data, or framework, please cite the associated manuscript:

> U.P. Singh, B. Kumari, M. Chandra, E.A. Siddiqui, “A Multiscale Correction Framework for Classical Nucleation Theory: Curvature-Dependent Surface Tension, Cooperative Polarization, and Electric-Field Effects in Water Vapor–Liquid Nucleation,” *Colloids and Surfaces A: Physicochemical and Engineering Aspects* (2026). Manuscript submitted.

Please update the bibliographic information after publication.

## Data and reproducibility

The repository is intended to preserve the numerical settings required to reproduce the reported calculations. Random seeds, simulation scripts, parameter files, and generated CSV data should be retained with the corresponding repository release whenever possible.

The reported Monte Carlo results are based on the verified production campaign described above. Earlier exploratory simulations are not used for the aggregate results reported in the manuscript.

## License

This code and associated materials are provided for academic research and reproducibility. Please contact the corresponding author for reuse, redistribution, or incorporation into derivative software where required.

**Corresponding author:** Uday Pratap Singh  
**Email:** udaypratap@allduniv.ac.in
