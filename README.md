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
  part08/               Solver core and prediction workflow figure
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

Full mode reruns the 4,000-draw ensembles, the synthetic benchmarks, and the 600 coverage trials. Use the same output directory with `--resume` to reuse completed coverage trials when their inputs, code, and software versions match. The solver's TypeScript core and archived comparison data are included for reference.

## Results and figures

| Part | Entry point under `scripts/` | Main outputs under the matching `data/` or `figures/` folder |
| --- | --- | --- |
| 1 | `part01.geometry` | Observed positions and goal geometry, `part1_fig1.png` |
| 2 | `part02.part2_baseline` | Analytical fits, residuals, trajectories, and `part2_fig1.png` to `part2_fig3.png` |
| 3 | `part03.run_part3`, `part03.plot_part3_revision` | Fits, spin and drag profiles, trajectory and crossing plots |
| 4 | `part04.run_part4` | 4A refits and 4B fixed-release results, wind contours and coverage |
| 5 | `part05.run_part5`, `part05.run_part5_wake`, `part05.plot_part5_revision` | Ensembles, sensitivity cases, density and wake figures |
| 6 | `part06.run_part6`, `part06.plot_goal_plane` | Synthetic benchmarks, coverage, given-shot results, and goal-plane plot |
| 7 | `part07.run_part7`, `part07.write_part7_table` | Contact events and friction sweep, `part7_bounce_3d.png` |
| 8 | `part08.workflow`, `part08/solver/src/` | Workflow figure, core solver, archived screenshot and comparison data |

## Sources

- The three ball positions come from [2026 UPC Problem B](https://www.uphysicsc.com/2026contest.html).
- `figures/part01/background_penalty.jpg` is [Football penalty](https://commons.wikimedia.org/wiki/File:Football_penalty.jpg) by dustpuppy, [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).
- `data/part05/wake_psd_30ms.csv` contains digitized side-force spectral values from [Asai et al., Fig. 7(a)](https://link.springer.com/article/10.1038/s41598-026-56275-1/figures/7). The input image hash and extraction settings are in `wake_psd_source.json`. To repeat digitization without editing the checked-in inputs, set `UPC_OUTPUT_ROOT` to a scratch directory and run `python -m scripts.part05.digitize_wake_spectrum <original-image>` from the repository root.
- `data/part06/drag_curves.json` records the wind-tunnel drag inputs from [Goff, Asai, and Hong](https://doi.org/10.1177/1754337114526173).
