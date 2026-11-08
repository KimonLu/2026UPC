"""Recreate the data and figures for the 2026 UPC Problem B study."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from shutil import copy2, copytree
import subprocess
import sys

from common.paths import REPO_ROOT


FULL_INPUTS = {
    3: ("config.json",),
    4: ("config.json",),
    5: ("wake_psd_30ms.csv", "wake_psd_source.json"),
    6: ("config.json", "drag_curves.json"),
    7: ("config.json",),
}


def prepare_output(output, mode, resume):
    if (output == REPO_ROOT or output in REPO_ROOT.parents or
            any(output.is_relative_to(REPO_ROOT / folder) for folder in ("data", "figures", "scripts", ".git"))):
        raise ValueError("Choose a separate output directory.")
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError(f"{output} is not empty. Choose an empty directory or use --resume.")

    for part in range(1, 9):
        name = f"part{part:02d}"
        (output / "data" / name).mkdir(parents=True, exist_ok=True)
        (output / "figures" / name).mkdir(parents=True, exist_ok=True)
        source = REPO_ROOT / "data" / name
        target = output / "data" / name
        if mode == "figures":
            if source.is_dir() and part != 8:
                copytree(source, target, dirs_exist_ok=True)
        else:
            for file in FULL_INPUTS.get(part, ()):
                copy2(source / file, target / file)


def run_module(module, args, env):
    print(f"[{module}] {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-B", "-m", f"scripts.{module}", *args],
                   cwd=REPO_ROOT, env=env, check=True)


def run_full(jobs, resume, env):
    parallel = ["--jobs", str(jobs)]
    commands = [
        ("part02.part2_baseline", []),
        ("part03.run_part3", ["--no-plots"]),
        ("part04.run_part4", ["--no-plots"]),
        ("part05.run_part5", []),
        ("part05.verify_part5", []),
        ("part05.run_part5_wake", parallel),
        ("part06.run_part6", ["--no-plots", *parallel, *(["--resume"] if resume else [])]),
        ("part06.verify_part6", ["--check-provenance"]),
        ("part07.run_part7", ["--no-plots"]),
    ]
    for module, options in commands:
        run_module(module, options, env)


def draw_figures(mode, env):
    if mode == "figures":
        run_module("part02.part2_baseline", ["--plots-only"], env)
    commands = [
        ("part01.geometry", []),
        ("part03.run_part3", ["--plots-only"]),
        ("part03.plot_part3_revision", []),
        ("part04.run_part4", ["--plots-only"]),
        ("part04.run_part4", ["--paper-figures"]),
        ("part05.plot_part5_revision", []),
        ("part06.summarize_benchmarks", []),
        ("part06.kde_sensitivity", []),
        ("part06.plot_part6", []),
        ("part06.plot_goal_plane", []),
        ("part06.write_part6_tables", []),
        ("part07.run_part7", ["--plots-only"]),
        ("part07.write_part7_table", []),
        ("part08.workflow", []),
    ]
    for module, options in commands:
        run_module(module, options, env)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require_close(actual, expected, name, *, tolerance):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise ValueError(f"{name}: got {actual!r}, expected {expected!r}")


def verify_full_results(output):
    actual = output / "data"
    expected = REPO_ROOT / "data"
    baseline = "part02/part2_baseline_results.json"
    for got, ref in zip(load_json(actual / baseline)["models"], load_json(expected / baseline)["models"],
                        strict=True):
        for key in ("goal_time_s", "ground_contact_time_s", "position_rmse_m", "coordinate_rmse_m"):
            require_close(got[key], ref[key], key, tolerance=1e-8)
        for component, value in zip(got["goal_position_m"], ref["goal_position_m"], strict=True):
            require_close(component, value, "baseline crossing", tolerance=1e-8)

    for section in ("part03", "part04"):
        path = f"{section}/results.json"
        observed = load_json(actual / path)["cases"]
        archived = load_json(expected / path)["cases"]
        if observed.keys() != archived.keys():
            raise ValueError(f"{section}: case list differs.")
        for name, ref in archived.items():
            got = observed[name]
            if got["event"] != ref["event"]:
                raise ValueError(f"{name}: event differs from the archive.")
            for key, value in ref["goal"].items():
                require_close(got["goal"][key], value, f"{name} {key}", tolerance=1e-6)

    shot = load_json(actual / "part06/given_shot.json")["uncertainty"]["M-DM/calm"]
    if shot["n"] != 4000 or shot["events"]["goal_inside"] != 3886:
        raise ValueError(f"Unexpected still-air shot counts: {shot['events']}")
    for key, value in (("y_m", 2.56), ("z_m", 2.03)):
        require_close(shot["inside_kde_mode"][key], value, f"Conditional mode {key}", tolerance=.005)
    require_close(shot["ellipse_area_m2"], 1.16, "Still-air ellipse area", tolerance=.005)

    coverage = load_json(actual / "part06/coverage/summary.json")
    for scenario, hits in (("calm", 192), ("tailwind", 191), ("tailwind_spin_decay", 190)):
        summary = coverage[f"{scenario}/4000"]
        if (summary["hits"], summary["trials"]) != (hits, 200):
            raise ValueError(f"{scenario}: unexpected coverage summary: {summary}")

    for table in ("events.csv", "paper_bounce.csv"):
        def read_events(directory):
            with (directory / "part07" / table).open(encoding="utf-8", newline="") as stream:
                return {row["case"]: row["event"] for row in csv.DictReader(stream)}
        if read_events(actual) != read_events(expected):
            raise ValueError(f"Section 7 event classes differ in {table}.")

    print("Baseline, still air, steady winds, shot uncertainty, coverage and bounce checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("figures", "full"), default="figures")
    parser.add_argument("--out-dir", type=Path, default=Path("reproduced"))
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--resume", action="store_true", help="Reuse compatible coverage trials in full mode.")
    args = parser.parse_args()
    if args.jobs < 1 or (os.name == "nt" and args.jobs > 61):
        parser.error("--jobs must be at least 1 and at most 61 on Windows.")
    if args.resume and args.mode != "full":
        parser.error("--resume requires --mode full.")
    output = args.out_dir.resolve()
    try:
        prepare_output(output, args.mode, args.resume)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    env = dict(os.environ, UPC_OUTPUT_ROOT=str(output), OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    if args.mode == "full":
        run_full(args.jobs, args.resume, env)
        verify_full_results(output)
    draw_figures(args.mode, env)
    for name in ("wind_reference.json", "uncertainty_reference.json"):
        relative = Path("data/part08") / name
        copy2(REPO_ROOT / relative, output / relative)
    for part, name in ((1, "background_penalty.jpg"), (8, "solver.png")):
        relative = Path("figures") / f"part{part:02d}" / name
        copy2(REPO_ROOT / relative, output / relative)
    run_module("part08.paper_manifest", [], env)
    print(f"Outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
