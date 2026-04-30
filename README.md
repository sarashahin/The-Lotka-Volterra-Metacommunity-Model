# A modelling technique unifying four paradigms of metacommunity theory

This repository contains the code accompanying the manuscript:

> Shahin, S., O'Sullivan, J. D., & Rossberg, A. G.
> *A modelling technique unifying four paradigms of metacommunity theory.*
> Submitted to *Ecology Letters*.

The Probabilistic-Stochastic-Deterministic (PSD) framework approximates
individual-based metacommunity dynamics using a hybrid scheme in which
each species occupies one of three states (Stochastic, Propagule,
Deterministic), with transitions between states governed by sign changes
of the local growth rate and Poisson-clock establishment events.

---

## Repository contents

This repository provides two executables, corresponding to the two
implementations described in the manuscript:

| Folder | Contents |
|---|---|
| `single_patch/` | Single-patch model with event-driven variable-step-size ODE integration (CVode/SUNDIALS via Assimulo). Reproduces Figs. 2–3 and Table 1. |
| `metacommunity/` | Multi-patch (metacommunity) model with fixed-step Euler integration and dispersal. Reproduces Figs. 4–6 (RPS metacommunity). |

Each folder is self-contained and runs independently.

---

## Requirements

- Python 3.10 or 3.11
- NumPy ≥ 1.24
- SciPy ≥ 1.10
- Matplotlib ≥ 3.7
- Assimulo ≥ 3.5 (provides the CVode/SUNDIALS interface)
- SUNDIALS ≥ 5.0 (system library required by Assimulo)
- statsmodels (only for the optional ANOVA scripts)

A `requirements.txt` is provided. Assimulo and SUNDIALS are most
easily installed via conda:

```bash
conda create -n psd python=3.11
conda activate psd
conda install -c conda-forge assimulo sundials numpy scipy matplotlib statsmodels
```

---

## Reproducing the results

### Single-patch validation (Figs. 2–3, Table 1)

```bash
cd single_patch
python main.py config.py
```

This integrates IBM, ODE and PSD models for `TMAX = 10000` time units
with `S = 300` species. Outputs:
- `model_outputs.npz`  – trajectories used by figure scripts
- `Trajectory.png`, `All_Models_Trajectory.png`, `Biomass_Distribution_Histogram.png`

To switch between body-mass regimes, edit `BODY_MASS` in `config.py`:
- Low regime:           `BODY_MASS = 1e-11`
- Intermediate regime:  `BODY_MASS = 2.5e-8`
- High regime:          `BODY_MASS = 1e-4`

The PSD2 model in `single_patch/models_psd2.py` uses the CVode
variable-step-size solver from SUNDIALS (via Assimulo) with state events
for state-transition detection — exactly as described in the
manuscript's pseudocode.

### Metacommunity / RPS dynamics (Figs. 4–6)

```bash
cd metacommunity
python run_all_rps.py
```

This runs the three-species rock-paper-scissors (RPS) metacommunity on
a 50×50 grid with non-local dispersal (LONG_DISTANCE_PROB = 1.0).
Outputs are written to `results/data/`, `results/plots/`, `results/movies/`.

The metacommunity PSD2 implementation uses fixed-step Euler integration
(`Δt = 1`, see manuscript Section 4.2) because the size of the state
vector (2 · S · N_x · N_y) makes event-driven integration impractical at
this scale.

### Generating manuscript figures

The `figures/` folder contains the scripts used to produce the figures
in the manuscript:
- `make_mosaic_movie_panels.py` — RPS spatial mosaics (Fig. 5)
- `make_hist_panel.py` — biomass histograms by body-mass regime (Fig. 2)
- `vis_trajectories.py` — biomass trajectories (Fig. 3)
- `anova_richness.py` (optional) — ANOVA across body-mass regimes

Each script accepts an `--npz` argument pointing to a previously
generated `model_outputs.npz`.

---

## Code-manuscript correspondence

The code is implemented to match the manuscript's pseudocode exactly.
In particular, the local growth rate of species *i* is computed as

    g_i = r_i − Σ_j C_ij B_j

where the sum runs over **all** species, including those currently in
the S- (waiting) and P- (propagule) states. This is implemented in:

- `single_patch/models_psd2.py`, function `_derivatives()`
- `metacommunity/models_psd2.py`, function `_derivatives()`

State transitions (S→D, S→P, P→S, P→D, D→P) are implemented in
`_handle_event_fn()` (single-patch, called by CVode at each detected
state event) and inline in the Euler loop (metacommunity).

---

## License

MIT License — see `LICENSE`.

---

## Citation

If you use this code, please cite the published paper (DOI to be added
upon acceptance) and the archived release on Zenodo (DOI: TBA).

## Contact

Sara Shahin — s.shahin@qmul.ac.uk
Axel G. Rossberg — a.rossberg@qmul.ac.uk
