"""Describe KDE sensitivity using only the archived given-shot crossings."""

import json
import platform

import numpy as np
import scipy

from scripts.common.paths import data_dir
from scripts.part05.run_part5 import kde_mode


def main():
    archive = data_dir(5)
    summary = json.loads((archive / "monte_carlo_summary.json").read_text(encoding="utf-8"))
    index = summary["ensemble_conditions"].index("position/gaussian/0.005")
    with np.load(archive / "ensembles.npz", allow_pickle=False) as samples:
        crossings = samples[f"{index}_M-DM/calm_goal"]
        events = samples[f"{index}_M-DM/calm_event"]
    records = []
    for draws in (2000, 4000):
        inside = crossings[:draws][events[:draws] == "goal_inside"]
        for multiplier in (.8, 1., 1.2):
            bandwidth = len(inside) ** (-1 / 6) * multiplier
            mode = kde_mode(inside, bandwidth=bandwidth)
            mode.update(draws=draws, inside_samples=len(inside), scott_multiplier=multiplier,
                        conditioning="inside-goal crossings only")
            records.append(mode)
    result = dict(source="data/part05/ensembles.npz", condition="position/gaussian/0.005",
                  model="M-DM/calm", grid_per_axis=65, grid_quantiles=[.001, .999],
                  starts=8, optimizer="BFGS", gradient_tolerance=1e-7,
                  runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  modes=records)
    (data_dir(6) / "kde_sensitivity.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for record in records:
        print(record["draws"], record["scott_multiplier"], record["y_m"], record["z_m"])


if __name__ == "__main__":
    main()
