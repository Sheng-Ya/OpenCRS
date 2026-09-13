# Union DGSM, History Matching and Bayesian Calibration of a Closed-Loop Cardiopulmonary Model

Analysis code for the manuscript's rest + exercise ("union") workflow: a
derivative-based global sensitivity analysis (DGSM) over the full closed-loop
cardiovascular-respiratory ODE model, followed by history matching (HM) with
Gaussian-process emulators and NUTS MCMC calibration against 50 clinical
targets (25 observables x {rest, exercise}).

Everything needed to reproduce the reported results is in this folder. Large
numerical arrays are not stored in git - see [Data](#data).

Part of the [OpenCRS repository](../../README.md); the sibling `DGSM_*` folders
are earlier development variants and are not used by the manuscript.

---

## Model and problem size

| Quantity | Value |
|---|---|
| ODE integration | numba-compiled RK23 (Bogacki-Shampine) driver, `rk23_njit.py` |
| DGSM parameters | 272 (`Samples_for_DGSM_Union.py`, `ProblemSpec`) |
| DGSM design | SALib `finite_diff`, 500 base points -> 136,500 model evaluations |
| Raw outputs per condition | 32; union vector = 64 (rest ++ exercise) |
| Calibration targets | 50 (25 per condition) |
| Calibrated parameters | 72 (`Overlap` 43 + `Rest_only` 14 + `Exercise_only` 15); all others fixed at their nominal midpoint |
| Parameter ranges | nominal +/- 50% unless narrowed explicitly |
| History matching | 8 waves, implausibility threshold 3.25, 400,000 test points per wave |
| MCMC | Pyro NUTS, 4 chains x 3,000 draws after 500 warmup, copula + log-spline NROY prior |

---

## Layout

```
DGSM_Union_Paper_gas/
├── All_derivatives_njit.py           # njit RHS of the full closed-loop model
├── rk23_njit.py                      # njit RK23 driver (bit-identical to scipy RK23)
├── Activation_Functions.py           # cardiac / valve activation functions
├── All_Next_Conditions.py            # per-beat and per-breath output extraction
├── Resp_Control_Breath_Optimiser.py  # breath-pattern optimiser (t1/t2, driving pressure)
├── fixed_params.py                   # nominal parameter values
├── Initial_Conditions_after_running_again.py   # converged initial state used by every run
├── Samples_for_DGSM_Union.py         # STAGE 1: builds the DGSM design and runs the simulations
├── Derivative-based GSA Union.py     # STAGE 2: DGSM analysis + convergence + figures
├── Overlap_in_Parameters_Union.py    # STAGE 3: rest/exercise sensitive-parameter overlap
├── DGSM_Union_Rest.txt               # ranked DGSM output, rest      (input to stages 3-4)
├── DGSM_Union_Exercise.txt           # ranked DGSM output, exercise
├── Edge_bundling/                    # hierarchical edge-bundling figures (per subsystem)
├── plots/                            # cached-array replotting for the DGSM figures
└── HM_MCMC_3/                        # STAGES 4-7: HM, MCMC, MAP refinement, appendices
    ├── Simulator_Union.py                  # autoemulate Simulator base class
    ├── AutoEmulate_Simulator_Union.py      # model wrapper exposing rest+exercise outputs
    ├── History_matching_function_union_all.py  # HM workflow (emulator fit, implausibility, NROY)
    ├── Bayesian_Calibration_Union_all.py   # STAGE 4 entry point: observations + 8 HM waves
    ├── MCMC_Union.py                       # STAGE 5: NUTS posterior
    ├── Refine_MAP_Union_all.py             # STAGE 6: L-BFGS-B refinement of the MAP
    ├── Run_model.py                        # STAGE 7: MAP -> full simulation + target traces
    ├── Plot_Union_Refined_MAP_With_Simulation.py
    ├── Appendix_D_corrected/               # emulator R2 / predictive-variance appendix
    ├── DGSM_Compare_with_Sobol/            # DGSM re-run restricted to the wave-8 NROY region
    └── Run_Sobol_Paper/                    # dependent-input (DLR / conditional) GSA on NROY
```

---

## Requirements

Python 3.12. The reported results were produced with these pinned versions:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install numpy==1.26.4 scipy==1.14.1 numba==0.61.2 \
            SALib==1.5.1 joblib==1.4.2 scikit-learn==1.5.2 \
            torch==2.5.1 gpytorch==1.13 linear-operator==0.5.3 botorch==0.12.0 \
            pyro-ppl==1.9.1 autoemulate==1.1.1 \
            matplotlib==3.10.0 pandas==2.2.3 seaborn==0.12.2 \
            tqdm==4.67.1 tqdm_joblib==0.0.4
```

No GPU is required; the emulators were trained and evaluated on CPU.

Stage 2 imports `dgsm_edited.py` (a copy of SALib's DGSM analyzer that also
returns the standard error `vi_std`), which lives one directory up. Put the
parent directory on the path before running it:

```bash
export PYTHONPATH="$(cd .. && pwd):$PYTHONPATH"
```
```powershell
$env:PYTHONPATH = (Resolve-Path ..).Path + ";" + $env:PYTHONPATH
```

Run each script from the directory that contains it - paths are resolved
relative to the script or to the working directory.

---

## Reproducing the analysis

Stage 1 is the expensive step (136,500 converged rest+exercise simulations, run
on a 128-core cluster node); everything downstream runs from its saved arrays.

**1. DGSM design and simulations** - writes `DGSM_500_X_union_50_27_08_gas.npy`
(136500 x 272) and `DGSM_500_Result_union_50_27_08_gas.npy` (136500 x 64).

```bash
python Samples_for_DGSM_Union.py
```

Uncomment the `finite_diff.sample` lines at the bottom of the file to regenerate
the design rather than load the saved one. On a cluster, use the array-job form
(the commented `argparse` block, plus `HM_MCMC_3/DGSM_Compare_with_Sobol/dgsm.sh`
as a PBS template for 4 x 128 cores) and concatenate the `Result_task_*.npy`
outputs.

**2. DGSM analysis** - writes `DGSM_Union_Rest.txt`, `DGSM_Union_Exercise.txt`,
`DGSM_50_reduced_tidal_cache.npz`, `plots/convergence_metrics.npz` and the
corresponding figures.

```bash
python "Derivative-based GSA Union.py"
```

**3. Rest/exercise parameter overlap** - prints the `Overlap`, `Rest_only` and
`Exercise_only` sets that define the 72 calibrated parameters. These are pasted
into `HM_MCMC_3/Bayesian_Calibration_Union_all.py`.

```bash
python Overlap_in_Parameters_Union.py
```

**4. History matching (8 waves)** - writes `NROY_Points_union_all_50.npy`,
`NROY_Params_union_all_50.npy`, `NROY_Implaus_union_all_50.npy`, the per-wave
`nroy_*` / `impl_scores_*` / `test_params_*` arrays, and the
`Emulator_union_all_wave_*/` GP directories.

```bash
cd HM_MCMC_3
python Bayesian_Calibration_Union_all.py
```

**5. NUTS calibration** - writes `MCMC_Union_50_28_08_copula_prior/` containing
`posterior_samples.npy`, `posterior_chains.npy`, `pred_check_matrix.npy`,
`mcmc_diagnostics.json`, `copula_prior.joblib`, per-chain logs and KDE figures.

```bash
python MCMC_Union.py                            # 4 chains concurrently, then aggregate
python MCMC_Union.py --chain-id 0 --n-chains 4  # one chain (array-job form)
python MCMC_Union.py --aggregate-only --n-chains 4
```

**6. MAP refinement** - L-BFGS-B from the top posterior draws; writes
`refined_map_parameters.npy`, `refined_map_predictions.npy` and the refinement
summary into the run directory.

```bash
python Refine_MAP_Union_all.py MCMC_Union_50_28_08_copula_prior
```

**7. Full simulation of the refined MAP** - runs the refined parameters through
the same simulator to convergence at rest then exercise, and writes the residual
figure and the rest/exercise target traces to `<run_dir>/Target_Trace/`.

```bash
python Run_model.py --run-dir MCMC_Union_50_28_08_copula_prior
```

### Supplementary analyses

```bash
cd HM_MCMC_3/Run_Sobol_Paper
python GSA_from_HM_conditional.py       # dependent-input GSA (DLR main effects, conditional totals)
python plot_conditional_gsa_results.py  # figures from the saved GSA result directory

cd ../DGSM_Compare_with_Sobol
python "Derivative-based GSA Union.py"  # DGSM restricted to the wave-8 NROY region

cd ../Appendix_D_corrected
python regenerate_appendix_d.py         # emulator R2 / predictive-variance panels + audit JSON
```

---

## Figures

Every figure can be regenerated without repeating the expensive stages, because
the analysis caches its summary arrays.

| Figure | Script | Cache it reads |
|---|---|---|
| DGSM rest / exercise sensitivity | `plots/plot_dgsm_reduced_tidal_from_cache.py` | `DGSM_50_reduced_tidal_cache.npz` |
| DGSM convergence appendix | `plots/plot_dgsm_convergence_summary_from_cache.py` | `plots/convergence_metrics.npz` |
| Parameter-output edge bundling | `Edge_bundling/{LA,LV,RA,RV,Resp}_edge_bundling*_flat.py` | `*_Groups*.txt`, `DGSM_Union_*.txt` |
| Prior / NROY / posterior KDEs | `plots/KDE_plot_paper_MCMC_HM.py` | `Y_train_union_all.pt`, MCMC run dir |
| Copula prior vs NROY vs posterior | `plots/Plot_Copula_Marginals_vs_NROY_Posterior_logspline.py` | `posterior_samples.npy` |
| Refined-MAP residuals and target traces | `HM_MCMC_3/Run_model.py`, `plots/Run_model_four_panel.py` | `<run_dir>/Target_Trace/` cache pickle |
| NROY / posterior dependence | `HM_MCMC_3/plot_nroy_posterior_selected_dependence.py` | NROY + posterior arrays |
| Emulator R2 and predictive variance (Appendix D) | `HM_MCMC_3/Appendix_D_corrected/regenerate_appendix_d.py` | per-wave test arrays |
| Dependent-input GSA | `HM_MCMC_3/Run_Sobol_Paper/plot_conditional_gsa_results.py` | `GSA_conditional_union_wave8/` |

---

## Data

The complete set of intermediate arrays for this workflow is about 19 GB - too
large for git - so this repository tracks source code only. The dataset
underlying the reported results will be deposited in Zenodo under CC BY 4.0,
with the DOI added here prior to publication. It contains:

| File | Contents | Size |
|---|---|---|
| `DGSM_500_X_union_50_27_08_gas.npy` | 136500 x 272 DGSM design | 283 MB |
| `DGSM_500_Result_union_50_27_08_gas.npy` | 136500 x 64 model outputs | 67 MB |
| `DGSM_Union_{Rest,Exercise}.txt` | ranked DGSM indices per output | 23 KB each |
| `DGSM_50_reduced_tidal_cache.npz`, `plots/convergence_metrics.npz` | figure caches | < 1 MB |
| `HM_MCMC_3/X_train_union_all.pt`, `Y_train_union_all.pt` | initial LHS design and outputs | 3.5 MB |
| `HM_MCMC_3/{nroy,impl_scores,test_params,train_points}_*_wave_{1..8}.npy` | per-wave HM state | ~2.8 GB |
| `HM_MCMC_3/NROY_{Points,Params,Implaus}_union_all_50.npy`, `test_param_union_all_50.npy` | final NROY region and its test set | 327 MB |
| `HM_MCMC_3/Emulator_union_all_wave/` | GP emulators refit on the last wave - **this is the set MCMC_Union.py loads** | 2.0 GB |
| `HM_MCMC_3/Emulator_union_all_wave_{1..8}/` | per-wave GP emulators (joblib); wave 8 is used by Appendix D and the Sobol comparison | ~9.8 GB |
| `HM_MCMC_3/MCMC_Union_50_28_08_copula_prior/` | chains, posterior samples, diagnostics, MAP | 159 MB |

The **minimal dataset** needed to reproduce the main reported results (posterior,
refined MAP, and the main-text figures) without re-running the simulator is the
DGSM design/result pair, the figure caches, the final NROY arrays,
`Emulator_union_all_wave/`, and the MCMC run directory - about 2.8 GB. Note that
`MCMC_Union.py` loads `Emulator_union_all_wave/`, the refit set, not
`Emulator_union_all_wave_8/`; the run's `config.json` records which directory was
used. Reproducing Appendix D and the Sobol comparison additionally needs the
per-wave emulators for waves 2, 4, 6 and 8 and the per-wave test arrays.

Clinical target values and their variances are literature-derived and listed
inline as the `observation` dictionaries in
`HM_MCMC_3/Bayesian_Calibration_Union_all.py` and
`HM_MCMC_3/Refine_MAP_Union_all.py`; sources are given in the manuscript.

---

## Notes and known caveats

- The four pre-atrial-contraction volume outputs are emulated as raw volumes but
  constrained as an active-emptying *fraction*,
  `r = (V_pre - V_min) / (V_max - V_min)`, targeted at 0.25 with variance
  0.0025. The HM gate accepts a point when `P(r in [0.20, 0.30]) >= 0.06`, so
  these four columns filter the NROY region only weakly.
- `rk23_njit.solve_rk23` is a line-by-line transcription of SciPy 1.14's RK23
  and reproduces `solve_ivp(..., method="RK23")` bit for bit. Set
  `USE_NJIT_RK23 = False` in `Samples_for_DGSM_Union.py` to fall back to SciPy
  for regression checks.
- Set `OMP_NUM_THREADS=1` and the MKL / OpenBLAS / NumExpr equivalents before
  any parallel run. The scripts do this themselves, but a nested BLAS thread
  pool will otherwise oversubscribe the machine.
- Filenames carry the run date (`..._27_08_gas`, `..._28_08_copula_prior`).
  Lines that must be edited for a new run are marked `# change`.

---

## License

Released under the MIT License - see [`LICENSE`](../../LICENSE) at the
repository root. You may use, modify and redistribute this code, including
commercially, provided the copyright notice is retained. It is supplied without
warranty.

The dataset archived separately (see [Data](#data)) is released under
CC BY 4.0.

This is a research model. It is not validated for clinical use and must not be
used to inform patient care.

## Citation

If you use this code, please cite the accompanying manuscript.

## Contact

Questions about reproduction: please open an issue on this repository.
