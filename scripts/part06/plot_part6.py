"""Draw the two Section 6 paper figures from exported results."""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from scripts.common.paths import figures_dir
from scripts.part06.part6_common import CONFIG, DATA, read_json
from scripts.common.plot_style import GRID, INK, STYLES, TEAL, VIOLET, panel, save


def prediction_errors(summary, path):
    models = CONFIG['synthetic']['models']
    noise = CONFIG['synthetic']['noise_sigma_m']
    x = np.arange(len(noise))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.35), layout='constrained')
    for ax, metric, label, factor in (
        (axes[0], 'position_m', 'Goal-plane error (m)', 1),
        (axes[1], 'arrival_s', 'Absolute arrival-time error (ms)', 1000),
    ):
        for i, model in enumerate(models):
            color, marker, _ = STYLES[model]
            records = [summary[f'reference/{model}/{sigma:g}'][metric] for sigma in noise]
            medians = np.array([row['median'] for row in records])*factor
            p90 = np.array([row['p90'] for row in records])*factor
            offset = (i-(len(models)-1)/2)*.16
            ax.errorbar(x+offset, medians, yerr=[np.zeros(len(x)), p90-medians],
                        fmt=marker, color=color, capsize=3, markersize=5, label=model)
        ax.set_xticks(x, [f'{1000*sigma:g}' for sigma in noise])
        ax.set_xlabel(r'Coordinate noise $\sigma$ (mm)')
        ax.set_ylabel(label)
        ax.grid(axis='y', color=GRID, linewidth=.8)
        ax.set_axisbelow(True)
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, -.035*hi), hi)
    panel(axes[0], '(a)', 'Crossing position')
    panel(axes[1], '(b)', 'Arrival time')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncols=4, frameon=False)
    save(fig, path)


def coverage_plot(summary, path):
    fig, ax = plt.subplots(figsize=(7.7, 2.9), layout='constrained')
    prefix = CONFIG['coverage']['prefix_draws']
    full = CONFIG['coverage']['inner_draws']
    scenarios = ('calm', 'tailwind', 'tailwind_spin_decay')
    labels = ('Calm, constant spin', 'Tailwind, constant spin', 'Tailwind, decaying spin')
    colors = (INK, TEAL, VIOLET)
    all_limits = [summary[f'{name}/{full}']['wilson_95'] for name in scenarios]
    x0 = min(.90, min(lo for lo, _ in all_limits)-.012)
    x1 = max(.99, max(hi for _, hi in all_limits)+.01)
    for i, (name, color) in enumerate(zip(scenarios, colors)):
        main = summary[f'{name}/{full}']
        short = summary[f'{name}/{prefix}']
        p = main['coverage']
        lo, hi = main['wilson_95']
        ax.errorbar(p, i, xerr=[[p-lo], [hi-p]], fmt='o', color=color, capsize=4,
                    markersize=6, label=f'{full} refits and Wilson interval' if i == 0 else None)
        ax.scatter(short['coverage'], i+.19, s=23, marker='D', facecolors='white',
                   edgecolors=color, zorder=3, label=f'{prefix} refits' if i == 0 else None)
        ax.text(1.018, i, f'{main["hits"]}/{main["trials"]}',
                transform=ax.get_yaxis_transform(), color=color, va='center', clip_on=False)
    ax.axvline(.95, color='#7f8994', linestyle='--', linewidth=1, label='Nominal 95%')
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(2.45, -.45)
    ax.set_xlim(x0, x1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel('Empirical joint coverage')
    ax.grid(axis='x', color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncols=3, frameon=False)
    save(fig, path)


def make_figures():
    figures = figures_dir(6)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'text.color': INK, 'axes.labelcolor': INK,
                         'xtick.color': INK, 'ytick.color': INK})
    prediction_errors(read_json(DATA / 'v2_summary.json'), figures / 'part6_fig1.png')
    coverage_plot(read_json(DATA / 'coverage' / 'summary.json'), figures / 'part6_fig2.png')


if __name__ == '__main__':
    make_figures()
