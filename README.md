# Predicting FIFA Penalty Trajectories

Code, inputs, results, and figures for 2026 University Physics Competition Problem B.

## Repository layout

```text
scripts/
  reproduce.py          Run the Python calculations and render figures
  common/               Flight dynamics and plot utilities
  part01/               Observation and goal geometry
  part02/               Analytical baselines
  part03/               Still-air aerodynamic fits
  part04/               Steady-wind refits and fixed-release predictions
  part05/               Observation uncertainty and aerodynamic sensitivity
  part06/               Given-shot prediction and validation
  part07/               One-bounce trajectories
  part08/               Complete browser solver and prediction workflow figure
data/
  part02/–part08/       Section inputs, reference results, and numerical tables
figures/
  part01/–part08/       Archived paper figures and diagnostics
```

## Quick start

Use Python 3.14. From the repository root, run:

```sh
python -m pip install -r requirements.txt
python scripts/reproduce.py --mode figures --out-dir reproduced
```

This renders the plots from the archived results and copies the reference photograph and solver screenshot. Outputs are written to `reproduced/data/` and `reproduced/figures/`.

To recompute the Python results in a separate directory:

```sh
python scripts/reproduce.py --mode full --out-dir recomputed --jobs 8
```

Full mode reruns the 4,000-draw ensembles, the synthetic benchmarks, and the 600 coverage trials. Use the same output directory with `--resume` to reuse completed coverage trials when their inputs, code, and software versions match. Historical coverage provenance retains the original calculation paths; the current source locations are listed below.

The archived Part 5 runtime is Python 3.12.10, NumPy 2.5.4 and SciPy 1.18.1. Part 6 and coverage record Python 3.14.7, NumPy 2.5.2 and SciPy 1.18.0. These are the environments reported by the saved runs, rather than a claim that every result was computed in one environment. `requirements.txt` pins the Part 6 reproduction environment and plotting dependencies.

## Browser solver

The [live solver](https://kimonlu.github.io/penalty-trajectory-solver/) and the complete archived interface share the same numerical core. With Node.js 24.18.0 and npm available, run:

```sh
cd scripts/part08/solver
npm ci
npm run dev -- --host 127.0.0.1
```

Open the local URL printed by Vite, including the `/penalty-trajectory-solver/` base path. After dependencies are installed, the interface can run locally without the hosted site. `npm run build` creates a static build; `npm run preview` serves it locally. Run `npm test` to repeat the 20 wind comparisons, the two 2,000-draw uncertainty comparisons, and the event/region checks. Tests read the archived JSON files under `data/part08/`; they do not require a Python rerun. The separate `npm run test:browser` suite requires Playwright's Chromium installation.

## Results and figures

| Part | Entry point under `scripts/` | Main outputs under the matching `data/` or `figures/` folder |
| --- | --- | --- |
| 1 | `part01.geometry` | Observed positions and goal geometry, `part1_fig1.png` |
| 2 | `part02.part2_baseline` | Analytical fits, residuals, trajectories, and `part2_fig1.png` to `part2_fig3.png` |
| 3 | `part03.run_part3`, `part03.plot_part3_revision` | Fits, spin and drag profiles, trajectory and crossing plots |
| 4 | `part04.run_part4` | W1 refits and W2 fixed-release results, wind contours and coverage |
| 5 | `part05.run_part5`, `part05.run_part5_wake`, `part05.plot_part5_revision` | Ensembles, sensitivity cases, density and wake figures |
| 6 | `part06.run_part6`, `part06.summarize_benchmarks`, `part06.plot_part6`, `part06.plot_goal_plane` | Synthetic benchmarks, common-subset/error summaries, event confusion counts, coverage, and given-shot plot |
| 7 | `part07.run_part7`, `part07.write_part7_table` | Contact events and friction sweep, `part7_bounce_3d.png` |
| 8 | `part08.workflow`, `part08/solver/` | Workflow figure, full solver source, tests, archived screenshot and comparison data |

## Paper identifiers and source data

The paper uses family names B-CA, B-G, M-D and M-DM. Analysis identifiers describe a calculation or question; they are not section numbers. The same analysis identifiers are used in the code, data, figures and paper.

| Paper case | Meaning | Result key or source |
| --- | --- | --- |
| B1 / B2 | Constant acceleration / gravity-only baseline | `B1` / `B2` |
| S1 / S2 | Still-air drag, fixed / fitted coefficient | `S1` / `S2` |
| S3 | Still-air constant-spin reference | `S3` |
| S4 / S5 | Spin decay with 5 s / 1 s time constant | `S4` / `S5` |
| W1 / W2 | Prescribed-wind refit / fixed-release response | `W1/*` / `W2/*` |
| U1 / U2 | Coordinate / timing perturbations | `position/*` / `time/*` in Part 5 |
| U3 | Mass, density and prescribed drag scans | `U3` scenario family |
| U4 / U6 | Qualitative gust / spin-axis discussion | No new numerical experiment claimed |
| U5 | Zero-spin spectral wake refits | `wake_crossings.csv`, `wake_results.json` |
| V1 / V2 / V3 / V4 | Numerical checks / synthetic benchmark / model mismatch / coverage | `v1_*` / `v2_*` / `v3_*` / `coverage/` |
| R1 | Illustrative one-bounce friction sweep | `paper_bounce.csv`, `paper_trajectories.csv` |

## Reading the archived comparisons

`data/part06/benchmark_comparison.json` and `.csv` are derived only from `v2_predictions.csv`. Spatial errors are reported both on each model's paired-crossing subset and on the intersection where the truth and all four models cross. The common counts at coordinate noise 0, 1, 5 and 10 mm are 72, 1148, 1051 and 972. Event accuracy uses every observation-eligible trial, including failed predictions. `event_confusion.csv` records the corresponding truth/prediction counts. To refresh these tables and plots without fitting a trajectory:

```sh
python -m scripts.part06.summarize_benchmarks
python -m scripts.part06.plot_part6
```

`python -m scripts.part06.kde_sensitivity` evaluates the same stored crossing samples at Scott bandwidth multipliers 0.8, 1.0 and 1.2, using prefixes of 2,000 and 4,000 draws. It writes `data/part06/kde_sensitivity.json` without rerunning a flight model. At 4,000 draws the bandwidth variations move the conditional mode by less than 2 cm; halving the sample count changes its height by about 12 cm. The paper therefore keeps the 4,000-draw Scott mode as a smoothed point summary and reports the joint spatial region alongside it.

The supplied-shot KDE uses inside-goal samples, Scott's Gaussian bandwidth, a 65 by 65 search grid between the 0.1% and 99.9% coordinate quantiles, and BFGS refinement from the eight densest grid points. Its settings are implemented in `scripts/part05/run_part5.py::kde_mode`. The empirical ellipse instead uses all successful plane crossings. The 2,000-to-4,000-prefix stability statistics are in `data/part05/sample_stability.csv`; they do not measure KDE bandwidth sensitivity.

Coverage settings are in `data/part06/config.json`; each outer data set has 5 mm independent Gaussian coordinate noise and each region uses 4,000 fresh refits. The predictor knows each scenario's prescribed wind and assumes constant spin. Decay is introduced only into the third truth scenario, with a 1 s time constant. Missing predicted crossings are excluded from the spatial region; fit failures remain in the event counts. An absent region is counted as noncoverage. All 600 archived outer trials produced a valid region. Random streams are generated by `SeedSequence([20261107, 4, trial]).spawn(2)`; the same standard draws are paired across scenarios.

Core numerical functions are in `scripts/common/still_air.py` (forces, DOP853, bounded fits), `scripts/part05/part5_solver.py` (batched RK4 and inverse fits), and `scripts/part05/run_part5.py` (crossing summaries and KDE). `scripts/part06/part6_coverage.py` defines the repeated coverage experiment. The browser equivalents are `flight.ts`, `fit.ts`, and `uncertainty.ts` under `scripts/part08/solver/src/`.

`data/part08/paper_manifest.json` records the core-source hashes, function line ranges and exact repository-to-Overleaf figure names. Regenerate it with `python -m scripts.part08.paper_manifest` after changing a paper asset or core source.

## Sources

- The three ball positions come from [2026 UPC Problem B](https://www.uphysicsc.com/2026contest.html).
- `figures/part01/background_penalty.jpg` is [Football penalty](https://commons.wikimedia.org/wiki/File:Football_penalty.jpg) by dustpuppy, [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).
- `data/part05/wake_psd_30ms.csv` contains digitized side-force spectral values from [Asai et al., Fig. 7(a)](https://link.springer.com/article/10.1038/s41598-026-56275-1/figures/7). The input image hash and extraction settings are in `wake_psd_source.json`. To repeat digitization without editing the checked-in inputs, set `UPC_OUTPUT_ROOT` to a scratch directory and run `python -m scripts.part05.digitize_wake_spectrum <original-image>` from the repository root.
- `data/part06/drag_curves.json` records the wind-tunnel drag inputs from [Goff, Asai, and Hong](https://doi.org/10.1177/1754337114526173).
