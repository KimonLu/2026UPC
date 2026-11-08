"""Draw the given-shot crossings in still air and their joint 95% ellipse."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from scripts.common.paths import data_dir, figures_dir
from scripts.common.plot_style import GRID, INK, panel, save
from scripts.part06.part6_common import read_json


def plot_goal_plane(path):
    part5 = data_dir(5)
    shot = read_json(data_dir(6) / "given_shot.json")
    archive = read_json(part5 / "monte_carlo_summary.json")
    index = archive["ensemble_conditions"].index("position/gaussian/0.005")
    with np.load(part5 / "ensembles.npz", allow_pickle=False) as samples:
        events = samples[f"{index}_M-DM/calm_event"]
        yz = samples[f"{index}_M-DM/calm_goal"][:, 1:]
    inside = yz[events == "goal_inside"]
    outside = yz[events == "goal_outside"]
    model = shot["uncertainty"]["M-DM/calm"]
    goal = shot["cases"][shot["main_case"]]["goal"]
    mode = model["inside_kde_mode"]
    settings = read_json(data_dir(3) / "config.json")["integration"]
    # Use the same rectangular ball-centre criterion as the simulation.
    half_width = settings["goal_half_width"]
    height = settings["goal_height"]
    if len(outside) and (np.any(outside[:, 1] <= height) or np.any(abs(outside[:, 0]) >= half_width)):
        raise ValueError("The crossbar label requires every outside crossing to be above the bar.")

    angles = np.linspace(0, 2 * np.pi, 361)
    radii, basis = np.linalg.eigh(model["covariance_yz"])
    circle = np.array([np.cos(angles), np.sin(angles)])
    ellipse = np.asarray(model["mean"][1:])[:, None] + basis @ (
        np.sqrt(radii * model["ellipse_radius_squared"])[:, None] * circle
    )

    scored = "#6a9fbd"
    high = "#bb4a26"
    gold = "#ffbf3c"
    face = "#f0f6f9"
    edge = "#45535e"
    with plt.rc_context({"font.size": 10, "axes.labelcolor": INK, "xtick.color": edge,
                         "ytick.color": edge, "axes.spines.top": False, "axes.spines.right": False}):
        fig, (full, detail) = plt.subplots(
            1, 2, figsize=(11.4, 4.15), gridspec_kw={"width_ratios": [2, 1]}, layout="constrained"
        )
        panel(full, "(a)", "Full goal mouth")
        panel(detail, "(b)", "Detail near the crossbar")
        full.add_patch(Rectangle((-half_width, 0), 2 * half_width, height,
                                 facecolor=face, edgecolor=edge, linewidth=1.5))
        full.add_patch(Rectangle((1.6, 1.0), 2.0, 1.9, fill=False,
                                 edgecolor="#8797a5", linestyle=":", linewidth=1.0))
        full.set(xlim=(-4, 4), ylim=(-.18, 3.05), xticks=(-half_width, -2, 0, 2, half_width),
                 yticks=(0, 1, 2, height, 3))
        detail.axhspan(0, height, color=face)
        detail.axhline(height, color=edge, linewidth=1.4)
        detail.text(1.66, height + .04, "Crossbar", color=edge, fontsize=9)
        detail.set(xlim=(1.6, 3.6), ylim=(1.0, 2.91), xticks=(1.75, 2, 2.5, 3, 3.5),
                   yticks=(1.25, 1.5, 2, height, 2.75))
        for ax, size in ((full, 3), (detail, 7)):
            ax.scatter(*inside.T, color=scored, s=size, alpha=.24, linewidths=0, zorder=3)
            ax.scatter(*outside.T, color=high, s=size * 1.8, alpha=.72, linewidths=0, zorder=4)
            ax.plot(*ellipse, color=INK, linewidth=1.6, linestyle="--", zorder=5)
            ax.scatter(mode["y_m"], mode["z_m"], marker="D", s=size * 8, facecolor=gold,
                       edgecolor=INK, linewidth=1, zorder=6)
            ax.scatter(goal["y_m"], goal["z_m"], s=size * 7, facecolor="white",
                       edgecolor=INK, linewidth=1.5, zorder=7)
            ax.set(xlabel=r"Lateral position, $y_g$ (m)", ylabel=r"Height, $z_g$ (m)")
            ax.set_aspect("equal", adjustable="box")
            ax.set_axisbelow(True)
            ax.grid(color=GRID, linewidth=.7)
        handles = [
            Line2D([], [], marker="o", linestyle="", color=scored, label=f"Inside goal ({len(inside):,})"),
            Line2D([], [], marker="o", linestyle="", color=high, label=f"Above crossbar ({len(outside):,})"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="white", color=INK,
                   label="Nominal crossing"),
            Line2D([], [], marker="D", linestyle="", markerfacecolor=gold, color=INK,
                   label="Most likely entry"),
            Line2D([], [], color=INK, linestyle="--", label="Joint 95% ellipse"),
        ]
        fig.legend(handles=handles, loc="outside lower center", ncols=5, frameon=False)
        save(fig, path)


if __name__ == "__main__":
    plot_goal_plane(figures_dir(6) / "part6_goal_plane.png")
