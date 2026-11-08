"""Draw the prediction procedure described in the paper."""

from scripts.common.paths import figures_dir
from scripts.common.plot_style import plt


def workflow_figure(path):
    figure = plt.figure(figsize=(8.4, 1.48), facecolor="white")
    steps = (
        ("Inputs", "Positions and wind"),
        ("Fit", "Release parameters"),
        ("Integrate", "Ball-centre trajectory"),
        ("Classify", "First flight event"),
    )
    positions = (0.12, 0.37, 0.63, 0.88)
    for position, (label, detail) in zip(positions, steps):
        figure.text(position, 0.77, label, ha="center", fontsize=11,
                    fontweight="bold", color="#18292b")
        figure.text(position, 0.56, detail, ha="center", fontsize=9, color="#37474a")
    for left, right in zip(positions, positions[1:]):
        figure.text((left + right) / 2, 0.72, "→", ha="center", fontsize=14,
                    color="#496264")
    figure.add_artist(plt.Line2D([0.035, 0.965], [0.41, 0.41],
                                transform=figure.transFigure, color="#cad4d5", linewidth=0.8))
    figure.text(0.5, 0.20,
                "With position noise: perturb the observations, repeat the fit and flight, "
                "and aggregate the crossings.",
                ha="center", fontsize=9, color="#18292b")
    figure.savefig(path, dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    workflow_figure(figures_dir(8) / "workflow.png")
