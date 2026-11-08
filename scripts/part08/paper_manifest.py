"""Record the current core sources, analysis identifiers and paper assets."""

import ast
import hashlib
import json

from scripts.common.paths import OUTPUT_ROOT, REPO_ROOT, data_dir


CORE = {
    "scripts/common/still_air.py": ("_air_force_terms", "simulate", "fit_from_positions"),
    "scripts/part05/part5_solver.py": ("residual_jacobian", "fit_batch", "predict_batch"),
    "scripts/part05/run_part5.py": ("summarize", "kde_mode"),
    "scripts/part06/part6_coverage.py": ("coverage_draws", "run_trial"),
}
PAPER_FIGURES = {
    "part01/part1_fig1.png": "part1_fig1.png",
    "part02/part2_fig3.png": "part2_fig3.png",
    "part03/part3_traj_mdm_constant.png": "part3_traj_mdm_constant.png",
    "part03/part3_fig1.png": "part3_fig1.png",
    "part03/part3_goal_comparison.png": "part3_goal_comparison.png",
    "part03/part3_omega1_crossings.png": "part3_omega1_crossings.png",
    "part03/part3_cd_sensitivity.png": "part3_cd_sensitivity.png",
    "part04/part4_w1_fit_parameters.png": "part4_w1_fit_parameters.png",
    "part04/part4_w1_wind_contours.png": "part4_w1_wind_contours.png",
    "part04/part4_w2_wind_coverage.png": "part4_w2_wind_coverage.png",
    "part05/part5_position_density.png": "part5_position_density.png",
    "part05/part5_timing_density.png": "part5_timing_density.png",
    "part05/part5_cd_sensitivity.png": "part5_cd_sensitivity.png",
    "part05/part5_wake_spectrum_crossings.png": "part5_wake_spectrum_crossings.png",
    "part06/part6_goal_plane.png": "part6_goal_plane.png",
    "part06/part6_fig1.png": "part6_fig1.png",
    "part06/part6_fig2.png": "part6_fig2.png",
    "part07/part7_bounce_3d.png": "part7_bounce_3d.png",
    "part08/solver.png": "part8_solver.png",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = {}
    for relative, functions in CORE.items():
        path = REPO_ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        lines = {node.name: [node.lineno, node.end_lineno] for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in functions}
        if set(lines) != set(functions):
            raise ValueError(f"Missing documented core function in {relative}")
        sources[relative] = dict(sha256=digest(path), functions=lines)
    figures = {}
    for relative, paper_name in PAPER_FIGURES.items():
        path = OUTPUT_ROOT / "figures" / relative
        figures["figures/" + relative] = dict(overleaf="figures/" + paper_name, sha256=digest(path))
    result = dict(
        models=["B-CA", "B-G", "M-D", "M-DM"],
        cases={"B": ["B1", "B2"], "S": [f"S{i}" for i in range(1, 6)],
               "W": ["W1", "W2"], "U": [f"U{i}" for i in range(1, 7)],
               "V": [f"V{i}" for i in range(1, 5)], "R": ["R1"]},
        core_sources=sources, figures=figures,
        derived_data=["data/part06/benchmark_comparison.json", "data/part06/event_confusion.csv",
                      "data/part06/kde_sensitivity.json"],
    )
    target = data_dir(8) / "paper_manifest.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(sources)} core modules and {len(figures)} paper figures.")


if __name__ == "__main__":
    main()
