"""Plot colors, labels and PNG export shared by the section figures."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = '#233344'
GRID = '#d9e2eb'
TEAL = '#007b6e'
VIOLET = '#7846a0'
STYLES = {
    'B-G': ('#6c7581', 'o', 'B-G'),
    'B-CA': ('#344962', 's', 'B-CA'),
    'M-D20': ('#c58535', 'X', r'M-D ($C_D=0.20$)'),
    'M-D': ('#b84327', 'D', 'M-D (fitted)'),
    'M-DM': (TEAL, 'o', 'M-DM'),
    'M-DM5': ('#2965a5', 'P', r'M-DM ($5\,\mathrm{s}$)'),
    'M-DM1': (VIOLET, '^', r'M-DM ($1\,\mathrm{s}$)'),
}


def panel(ax, label, title):
    ax.set_title(f'{label}   {title}', loc='left', pad=9, fontweight='bold', color=INK)


def save(fig, path):
    # A sibling file inherits the figures directory's Windows permissions.
    staged = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix='.png', delete=False) as output:
            staged = Path(output.name)
            fig.savefig(output, format='png', dpi=300, facecolor='white')
        staged.replace(path)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
        plt.close(fig)
