# OpenCRS

A closed-loop cardiorespiratory model: a lumped-parameter cardiovascular system
coupled to respiratory mechanics and gas exchange, under baroreflex and
chemoreflex control, simulated at rest and during exercise.

This repository holds the model implementation and the sensitivity, history
matching and Bayesian calibration analyses built on it.

---

## Where the paper's code is

> ### **[`Entire_system/DGSM_Union_Paper_gas/`](Entire_system/DGSM_Union_Paper_gas/)**
>
> All code, configuration and analysis scripts for the accompanying manuscript
> live in this one folder, which has its own detailed
> [README](Entire_system/DGSM_Union_Paper_gas/README.md) covering the
> environment, the pipeline stage by stage, and which script produces which
> figure.

Start there. The rest of this repository is the development history behind it
and is **not** needed to reproduce the paper.

---

## Repository layout

| Path | What it is |
|---|---|
| `Entire_system/DGSM_Union_Paper_gas/` | **The manuscript's analysis.** Union (rest + exercise) DGSM, 8-wave history matching, NUTS calibration, MAP refinement, figures |
| `Entire_system/` (root files) | The full-system model: cardiovascular, respiratory, gas-exchange and control equations (`All_derivatives_njit.py`, `All_Cardiovascular_system.py`, `All_Respiratory_controller.py`, `All_Gas_exchange.py`), plus earlier rest-only and exercise-only calibration pipelines |
| `Entire_system/DGSM_Union_Paper_Final_50/`, `DGSM_Union_Paper_Longer_Delay/`, `DGSM_Union/`, `DGSM_Pericardium/` | Earlier union-calibration variants, kept for provenance |
| `Entire_system/DGSM_Exercise_Paper/`, `DGSM_Exercise_Paper_Final_20/` | Exercise-only calibration variants |
| `Entire_system/Union_Calibration/`, `Plot_abstract/` | Supporting calibration and figure scripts |
| Root `.py` files | The original single-file model implementation that the `Entire_system` version grew out of |
| `Just Resp/`, `Reduced/` | Respiratory-only and reduced-order model experiments |

The `DGSM_*` folders are deliberate snapshots rather than branches: each one
pins the model and parameter ranges used for that run, so results stay
reproducible. They share filenames but are not interchangeable.

---

## Quick start

```bash
git clone https://github.com/Sheng-Ya/OpenCRS.git
cd OpenCRS/Entire_system/DGSM_Union_Paper_gas
```

Then follow that folder's README. It lists pinned dependencies (Python 3.12,
numpy, scipy, numba, SALib, torch, gpytorch, pyro, autoemulate), the seven
pipeline stages with their commands, and the figure-to-script map.

---

## Data

The repository contains source code only. The numerical arrays behind the
published results — the DGSM design and model outputs, the history-matching
NROY region, the fitted Gaussian-process emulators and the MCMC posterior
samples — total roughly 19 GB, and will be deposited in Zenodo under CC BY 4.0,
with the DOI added here prior to publication.

See the [Data section](Entire_system/DGSM_Union_Paper_gas/README.md#data) of the
paper folder's README for the file-by-file breakdown and for the ~2.8 GB subset
that reproduces the main results.

---

## License

MIT — see [`LICENSE`](LICENSE). You may use, modify and redistribute this code,
including commercially, provided the copyright notice is retained. It is
supplied without warranty.

This is a research model. It is not validated for clinical use and must not be
used to inform patient care.

## Citation

If you use this code, please cite the accompanying manuscript.

## Contact

Please open an issue for questions about reproduction.
