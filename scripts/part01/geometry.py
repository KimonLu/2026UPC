"""Plot the observed positions and goal in the horizontal and vertical planes."""

from matplotlib.lines import Line2D

from scripts.common.paths import data_dir, figures_dir
from scripts.common.plot_style import plt
from scripts.part06.part6_common import read_json


def main():
    configuration = read_json(data_dir(3) / "config.json")
    times = configuration["times_s"]
    x, y, z = zip(*configuration["positions_m"])
    goal = configuration["integration"]
    goal_x = goal["goal_x"]
    width = goal["goal_half_width"]
    height = goal["goal_height"]
    teal, blue = "#147e74", "#204362"
    fig, (top, side) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for ax, observed, axis, title, opening, bounds in (
        (top, y, "y", "(a) Horizontal projection", (-width, width), (-4.2, 4.5)),
        (side, z, "z", "(b) Vertical projection", (0, height), (-.15, 2.95)),
    ):
        ax.axvspan(0, 3, color="#eaf0f5")
        ax.set_axisbelow(True)
        ax.grid(alpha=.25)
        ax.axvline(goal_x, color=teal, linewidth=1.8)
        ax.vlines(goal_x, *opening, color=teal, linewidth=5, zorder=3)
        ax.plot(x, observed, "--", color="#7d8b9c", linewidth=1)
        ax.scatter(x, observed, color=blue, s=44, zorder=4)
        for t, xx, yy in zip(times, x, observed):
            label = f"{t:.2f} s" if t else "0 s"
            ax.annotate(label, (xx, yy), xytext=(3, 7), textcoords="offset points")
        ax.set(title=title, xlabel="$x$ (m)", ylabel=f"${axis}$ (m)",
               xlim=(-.4, 12), ylim=bounds, xticks=(0, 3, 6, 9, goal_x))
        ax.spines[["top", "right"]].set_visible(False)
    for sign in (-1, 1):
        top.annotate(f"{sign * width:+.2f}", (goal_x, sign * width), xytext=(-7, 0),
                     textcoords="offset points", ha="right", va="center", color=teal)
    side.annotate(f"{height:.2f} m", (goal_x, height), xytext=(-7, 0),
                  textcoords="offset points", ha="right", va="bottom", color=teal)
    side.axhline(0, color="#909090", linewidth=.8)
    side.legend(handles=[Line2D([], [], color=teal, lw=2, label="Goal plane"),
                         Line2D([], [], color=blue, marker="o", ls="", label="Observations")],
                loc="upper left", frameon=False)
    fig.savefig(figures_dir(1) / "part1_fig1.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
