"""Create the Section 7 paper PNGs from the saved event and trajectory data."""

import csv
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from scripts.common.plot_style import GRID, INK, TEAL, VIOLET, panel, save
from scripts.common.paths import figures_dir
from .run_part7 import DATA, GOAL_EVENTS, case_id, read_json


COLORS = ('#2965a5', TEAL, VIOLET, '#b84327')
MARKS = ('o', 's', '^', 'D')


def read_paths(path, columns=('x_m', 'z_m')):
    paths = defaultdict(list)
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            paths[row['case'], int(row['leg'])].append(tuple(float(row[column]) for column in columns))
    return {key: np.asarray(points).T for key, points in paths.items()}


def draw_leg(ax, paths, key, leg, radius, color, label=None):
    x, z = paths[key, leg]
    ax.plot(x, 1000 * (z - radius), color=color, lw=1.8, label=label)


def axes_style(ax):
    ax.set_axisbelow(True)
    ax.grid(color=GRID, linewidth=.8)


def restitution_plot(result, paths, filename):
    config, cases = result['config'], result['cases']
    e_values = config['restitution']
    nominal = config['reference_e_n']
    radius = result['physics']['radius']
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.6), layout='constrained')
    nominal_key = case_id('low', 'air', nominal, 0)
    draw_leg(left, paths, nominal_key, 1, radius, INK, 'Before impact')
    first = cases[nominal_key]['impact']['position'][0]
    left.scatter(first, 0, s=30, color=INK, zorder=5)
    for e_n, color, mark in zip(e_values, COLORS, MARKS):
        key = case_id('low', 'air', e_n, 0)
        row = cases[key]
        draw_leg(left, paths, key, 2, radius, color, rf'$e_n={e_n:g}$')
        left.scatter(row['state'][0], 0, s=35, marker=mark, facecolor='white',
                     edgecolor=color, zorder=5)
    left.axhline(0, color='#8a95a0', lw=.8)
    left.set(xlabel=r'Downfield distance $x$ (m)', ylabel=r'Clearance $z-R$ (mm)',
             xlim=(-.2, 7.55), ylim=(-4, 58))
    left.legend(frameon=False, ncols=2, loc='upper right', fontsize=8)
    panel(left, '(a)', 'Low shot with drag')
    axes_style(left)

    rows = [cases[case_id('low', 'air', e_n, 0)] for e_n in e_values]
    rows.append(cases[case_id('low', 'gravity', nominal)])
    labels = [rf'Air, $e_n={e_n:g}$' for e_n in e_values] + [rf'Gravity, $e_n={nominal:g}$']
    for i, (row, color, mark) in enumerate(zip(rows, (*COLORS[:3], INK), (*MARKS[:3], 's'))):
        t1, t2 = 1000 * row['impact']['time'], 1000 * row['time_s']
        right.hlines(i, t1, t2, lw=2, color=color)
        right.scatter(t1, i, s=26, color=color, zorder=5)
        right.scatter(t2, i, s=35, marker=mark, facecolor='white', edgecolor=color, zorder=5)
    right.set_yticks(range(len(rows)), labels)
    right.set(xlabel='Time from release (ms)', xlim=(189, 383), ylim=(3.5, -.5))
    panel(right, '(b)', 'Time between contacts')
    axes_style(right)
    handles = [
        Line2D([], [], color=INK, marker='o', ls='', label='First contact'),
        Line2D([], [], color=INK, marker='o', markerfacecolor='white', ls='', label='Second contact'),
    ]
    fig.legend(handles=handles, loc='outside lower center', ncols=2, frameon=False)
    save(fig, filename)


def friction_plot(result, paths, filename):
    config, cases = result['config'], result['cases']
    e_values, mu_values = config['restitution'], config['friction']
    nominal = config['reference_e_n']
    radius = result['physics']['radius']
    goal_x = result['integration']['goal_x']
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.55), layout='constrained')
    common_key = case_id('near', 'air', nominal, 0)
    draw_leg(left, paths, common_key, 1, radius, INK)
    left.scatter(cases[common_key]['impact']['position'][0], 0, color=INK, s=28, zorder=5)
    for mu_f, color, mark in zip(mu_values, COLORS, MARKS):
        key = case_id('near', 'air', nominal, mu_f)
        row = cases[key]
        draw_leg(left, paths, key, 2, radius, color, rf'$\mu_f={mu_f:.2f}$')
        x, _, z = row['state'][:3]
        left.scatter(x, 1000 * (z - radius), s=38, marker=mark, zorder=5,
                     facecolor=color if row['event'] in GOAL_EVENTS else 'white', edgecolor=color)
    left.axhline(0, color='#8a95a0', lw=.8)
    left.axvline(goal_x, color='#8a95a0', lw=1, ls='--')
    left.set(xlabel=r'Downfield distance $x$ (m)', ylabel=r'Clearance $z-R$ (mm)',
             xlim=(6.7, 11.27), ylim=(-6, 76))
    left.legend(frameon=False, fontsize=8, ncols=2, loc='upper left')
    panel(left, '(a)', rf'Near the goal at $e_n={nominal:g}$')
    axes_style(left)

    for i, e_n in enumerate(e_values):
        for j, mu_f in enumerate(mu_values):
            row = cases[case_id('near', 'air', e_n, mu_f)]
            reaches = row['event'] in GOAL_EVENTS
            background = '#d9efea' if reaches else '#eff2f6'
            right.add_patch(Rectangle((j - .5, i - .5), 1, 1, fc=background, ec='white', lw=2))
            if reaches:
                label = f'Goal\n+{1000 * (row["state"][2] - radius):.1f} mm'
            else:
                label = f'Contact\n{row["state"][0]:.2f} m'
            right.text(j, i, label, ha='center', va='center', color=INK, fontsize=8.7)
    right.set(xlim=(-.5, len(mu_values) - .5), ylim=(len(e_values) - .5, -.5),
              xlabel=r'Friction coefficient $\mu_f$', ylabel=r'Restitution $e_n$')
    right.set_xticks(range(len(mu_values)), [f'{mu:.2f}' for mu in mu_values])
    right.set_yticks(range(len(e_values)), [f'{e:g}' for e in e_values])
    right.tick_params(length=0)
    for spine in right.spines.values():
        spine.set_visible(False)
    panel(right, '(b)', 'First event after rebound')
    save(fig, filename)


def make_figures(result):
    style = {
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': INK, 'ytick.color': INK,
    }
    paths = read_paths(DATA / 'trajectories.csv')
    figures = figures_dir(7)
    figures.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(style):
        restitution_plot(result, paths, figures / 'part7_fig1.png')
        friction_plot(result, paths, figures / 'part7_fig2.png')


if __name__ == '__main__':
    make_figures(read_json(DATA / 'results.json'))
