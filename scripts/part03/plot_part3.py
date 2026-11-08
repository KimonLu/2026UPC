"""Create the four paper PNGs from the exported Section 3 data."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scripts.common.plot_style import GRID, INK, STYLES, TEAL, VIOLET, panel, save


def fit_residuals(data, path):
    cases = data['cases']
    keys = list(cases)
    values = 1000 * np.array([cases[key]['residuals_m'] for key in keys])
    extent = max(6., np.abs(values).max() * 1.10)
    screen = 1000 * data['config']['coordinate_screen_m']
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True, layout='constrained')
    rows = np.arange(len(keys))[::-1]
    for i, ax in enumerate(axes):
        ax.axvspan(-screen, screen, color='#e5f2ed', zorder=0)
        ax.axvline(0, color='#8795a4', lw=0.9)
        for key, row, residual in zip(keys, rows, values[:, i]):
            color = STYLES[key][0]
            for j, marker in enumerate(('o', 's', '^')):
                y = row + (1 - j) * 0.21
                ax.hlines(y, 0, residual[j], color=color, alpha=0.6, linewidth=0.9)
                ax.scatter(residual[j], y, color=color, marker=marker, s=28, linewidth=0.35,
                           edgecolor='white', zorder=3)
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-0.55, len(keys) - 0.45)
        ax.set_yticks(rows, [STYLES[key][2] for key in keys])
        ax.set_xlabel('Predicted minus observed (mm)')
        ax.grid(axis='x', color=GRID, alpha=0.62)
        time_ms = data['config']['times_s'][i + 1] * 1000
        panel(ax, f'({chr(97+i)})', f'{time_ms:g} ms after release')
    components = [Line2D([], [], marker=marker, color=INK, linestyle='none',
                         markersize=5.2, label=rf'$\epsilon_{axis}$')
                  for marker, axis in zip(('o', 's', '^'), 'xyz')]
    components.append(Rectangle((0, 0), 1, 1, fc='#e5f2ed', label=f'±{screen:g} mm screen'))
    fig.legend(handles=components, loc='outside lower center', ncols=4, frameon=False)
    save(fig, path)


def identifiability(data, path):
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.0), layout='constrained')
    ax = axes[0, 0]
    singular = np.array(data['svd']['singular_values_m']) * 1000
    index = np.arange(1, len(singular) + 1)
    ax.semilogy(index, singular, 'o-', color=TEAL, linewidth=1.6, markersize=4.8)
    ax.set(xticks=index, xlabel='Singular-value index', ylabel='Singular value (mm)')
    ax.text(0.05, 0.12, rf'$\kappa_2 = {data["svd"]["condition_number"]:.0f}$',
            transform=ax.transAxes, color=INK)
    panel(ax, '(a)', 'Scaled observation Jacobian')

    drag = [row for row in data['profiles'] if row['case'] == 'cd']
    chord = [row for row in data['profiles'] if row['case'] == 'omega_parallel']
    cd = np.array([row['value'] for row in drag])
    cd_max = np.array([row['max_residual_m'] for row in drag]) * 1000
    cd_best = data['cases']['M-DM']['cd']
    screen = data['config']['coordinate_screen_m'] * 1000
    ax = axes[0, 1]
    ax.axhspan(0, screen, color='#e5f2ed', zorder=0)
    ax.axhline(screen, color='#648176', lw=0.9, ls='--', label=f'{screen:g} mm screen')
    ax.plot(cd, cd_max, '.-', color=TEAL, linewidth=1.4, markersize=3.2)
    ax.scatter(cd_best, 0, marker='*', color=INK, s=75, zorder=3, label='Joint fit')
    ax.set(xlabel=r'Fixed $C_D$', ylabel='Largest coordinate residual (mm)', ylim=(-0.65, None))
    ax.legend(frameon=False, loc='upper right', fontsize=8.5)
    panel(ax, '(b)', 'Drag profile')

    goal = data['cases']['M-DM']['goal']
    for ax, profile, name, xlabel, baseline in (
        (axes[1, 0], drag, 'Drag', r'Fixed $C_D$', cd_best),
        (axes[1, 1], chord, 'Chord-axis spin', r'Fixed $\omega_1$ (rad s$^{-1}$)', 0.),
    ):
        values = np.array([row['value'] for row in profile])
        passes = np.array([row['passes_screen'] for row in profile])
        ax.fill_between(values, 0, 1, where=passes, transform=ax.get_xaxis_transform(),
                        color='#e5f2ed', zorder=0)
        ax.axhline(0, color='#8795a4', lw=0.8)
        ax.axvline(baseline, color='#8795a4', lw=0.8, ls=':')
        for coordinate, color, marker in (('y', TEAL, 'o'), ('z', VIOLET, 's')):
            delta = [100 * (row['goal'][f'{coordinate}_m'] - goal[f'{coordinate}_m'])
                     if row['goal'] else np.nan for row in profile]
            ax.plot(values, delta, color=color, marker=marker, markersize=3,
                    markevery=max(1, len(values) // 12), linewidth=1.45,
                    label=rf'$\Delta {coordinate}_g$')
        ax.legend(frameon=False, loc='upper right', fontsize=8.5)
        ax.set(xlabel=xlabel, ylabel='Goal shift (cm)')
        panel(ax, '(c)' if name == 'Drag' else '(d)', f'{name} and goal shift')
    for ax in axes.flat:
        ax.grid(color=GRID, alpha=0.62)
        ax.set_axisbelow(True)
    save(fig, path)


def goal_and_spin(data, trajectories, path):
    cases = data['cases']
    settings = data['config']['integration']
    width, height = settings['goal_half_width'], settings['goal_height']
    fig = plt.figure(figsize=(8.2, 5.0), layout='constrained')
    spec = fig.add_gridspec(2, 2, width_ratios=(1.48, 1.0), height_ratios=(1., 1.16))
    goal_ax = fig.add_subplot(spec[0, 0])
    zoom_ax = fig.add_subplot(spec[1, 0])
    spin_ax = fig.add_subplot(spec[:, 1])
    goals = [record['goal'] for record in cases.values() if record['goal']]
    ys = [goal['y_m'] for goal in goals]
    zs = [goal['z_m'] for goal in goals]
    y_bounds = (min(ys) - 0.11, max(ys) + 0.11)
    z_bounds = (min(zs) - 0.12, max(zs) + 0.12)
    for ax in (goal_ax, zoom_ax):
        ax.set_facecolor('#f7f9fb')
        ax.add_patch(Rectangle((-width, 0), 2 * width, height,
                               fc='#e5f2ed', ec='#44695d', lw=1.2))
        for key, record in cases.items():
            if record['goal'] is not None:
                goal = record['goal']
                color, marker, _ = STYLES[key]
                ax.scatter(goal['y_m'], goal['z_m'], c=color, marker=marker,
                           s=60 if ax is zoom_ax else 32, ec='white', lw=0.65, zorder=3)
        ax.set_aspect('equal', adjustable='box')
        ax.set(xlabel=r'$y_g$ (m)', ylabel=r'$z_g$ (m)')
        ax.grid(color='white', alpha=0.8)
        ax.set_axisbelow(True)
    goal_ax.set(xlim=(-width - 0.2, width + 0.2), ylim=(-0.12, height + 0.16))
    goal_ax.add_patch(Rectangle((y_bounds[0], z_bounds[0]), np.ptp(y_bounds), np.ptp(z_bounds),
                                fc='none', ec=INK, linewidth=1, ls='--', zorder=4))
    zoom_ax.set(xlim=y_bounds, ylim=z_bounds)
    panel(goal_ax, '(a)', 'Goal plane at true aspect ratio')
    panel(zoom_ax, '(b)', 'Crossings in the boxed region')

    time_end = data['config']['times_s'][-1]
    spin_ax.axvspan(0., time_end, color='#e7eff6', zorder=0)
    spin_ax.text(time_end / 2, 0.04, 'Fit window', transform=spin_ax.get_xaxis_transform(),
                 color=INK, ha='center', va='bottom', fontsize=8)
    for key in ('M-DM', 'M-DM5', 'M-DM1'):
        values = trajectories[key]
        color, marker, _ = STYLES[key]
        spin_ax.plot(values[:, 0], values[:, 4], color=color, lw=1.8)
        spin_ax.scatter(*values[-1, [0, 4]], color=color, marker=marker, s=45, ec='white', lw=0.6, zorder=4)
    spin_ax.set(xlabel='Time after release (s)', ylabel=r'Angular speed $|\boldsymbol{\omega}|$ (rad s$^{-1}$)',
                xlim=(0., max(cases[key]['event_time_s'] for key in ('M-DM', 'M-DM5', 'M-DM1')) + 0.03))
    spin_ax.grid(color=GRID)
    spin_ax.set_axisbelow(True)
    panel(spin_ax, '(c)', 'Refitted spin histories')
    handles = [Line2D([], [], marker=marker, color=color, markeredgecolor='white',
                      linestyle='none', markersize=7, label=label)
               for color, marker, label in STYLES.values()]
    fig.legend(handles=handles, loc='outside lower center', ncols=4, frameon=False)
    save(fig, path)


def trajectories_3d(data, trajectories, path):
    config = data['config']
    field = config['integration']
    goal_x, half_width, height = field['goal_x'], field['goal_half_width'], field['goal_height']
    observed = np.array(config['positions_m'])
    last_t = config['times_s'][-1]
    keys = ('B-G', 'M-D', 'M-DM', 'M-DM1')
    fig = plt.figure(figsize=(8.2, 4.65))
    ax = fig.add_axes([0.045, 0.19, 0.91, 0.80], projection='3d', proj_type='ortho',
                      computed_zorder=False)
    ground = [(0., -half_width, 0.), (goal_x, -half_width, 0.),
              (goal_x, half_width, 0.), (0., half_width, 0.)]
    gate = [(goal_x, -half_width, 0.), (goal_x, half_width, 0.),
            (goal_x, half_width, height), (goal_x, -half_width, height)]
    ax.add_collection3d(Poly3DCollection([ground], facecolor='#e1eee7', edgecolor='#abc6b6',
                                         alpha=0.63, linewidth=0.8, zorder=1))
    ax.add_collection3d(Poly3DCollection([gate], facecolor='#cbdfe7', edgecolor='none',
                                         alpha=0.38, zorder=2))
    frame = np.array(gate + [gate[0]])
    ax.plot(*frame.T, color='#416579', linewidth=1.55, zorder=3)
    ax.plot([0, goal_x], [0, 0], [0, 0], color='#97ac9e', lw=0.85, ls=':', zorder=3)
    ax.text(goal_x, -half_width, height + 0.14, 'Goal plane', color=INK, fontsize=9, zorder=8)
    patterns = [(0, (5, 2)), (1, (4, 2)), (0, (7, 2)), (3, (2, 2))]
    for key, pattern in zip(keys, patterns):
        points = trajectories[key]
        color, marker, _ = STYLES[key]
        for selection, style in ((points[:, 0] <= last_t, '-'), (points[:, 0] >= last_t, pattern)):
            ax.plot(*points[selection, 1:4].T, color=color, lw=1.75, ls=style, zorder=4)
        end = points[-1, 1:4]
        ax.scatter(*end, color=color, marker=marker, s=38, depthshade=False,
                   ec='white', lw=0.65, zorder=5)
    ax.scatter(*observed.T, color=INK, marker='o', s=34, depthshade=False,
               ec='white', lw=0.85, zorder=6)
    for point, time in zip(observed, config['times_s']):
        ax.text(point[0] - 0.08, point[1] + 0.06, point[2] + 0.24, f'{time * 1000:g} ms',
                color=INK, fontsize=8, zorder=8)
    limits = [(-0.6, goal_x + 0.65), (-half_width - 0.45, half_width + 0.45), (0, height + 0.44)]
    ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2])
    ax.set_xlabel(r'$x$ (m)', labelpad=9)
    ax.set_ylabel(r'$y$ (m)', labelpad=9)
    ax.set_zlabel(r'$z$ (m)', labelpad=7)
    ax.set_xticks([0, 3, 6, 9, 11])
    ax.set_yticks([-3, 0, 3])
    ax.set_zticks([0, 1, 2])
    ax.set_box_aspect([end - start for start, end in limits], zoom=1.0)
    ax.view_init(elev=34, azim=-116)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
    ax.grid(False)
    model_legend = [
        Line2D([], [], color=STYLES[key][0], marker=STYLES[key][1], markeredgecolor='white',
               linewidth=1.6, label=STYLES[key][2]) for key in keys
    ]
    model_legend.append(Line2D([], [], color=INK, marker='o', linestyle='none',
                               label='Observed position'))
    phase_legend = [Line2D([], [], color=INK, lw=1.5, ls=style, label=label)
                    for style, label in (('-', 'Observation window'), ('--', 'Extrapolation'))]
    fig.legend(handles=model_legend, loc='lower center', bbox_to_anchor=(0.5, 0.047),
               ncols=len(model_legend), frameon=False)
    fig.legend(handles=phase_legend, loc='lower center', bbox_to_anchor=(0.5, 0.005),
               ncols=2, frameon=False)
    save(fig, path)


def make_figures(data_directory, figure_directory):
    results = json.loads((data_directory / 'results.json').read_text(encoding='utf-8'))
    raw = defaultdict(list)
    with (data_directory / 'trajectories.csv').open(encoding='utf-8', newline='') as file:
        for row in csv.DictReader(file):
            raw[row['case']].append([float(row[column]) for column in
                                     ('time_s', 'x_m', 'y_m', 'z_m', 'spin_rad_s')])
    trajectories = {key: np.array(samples) for key, samples in raw.items()}
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelcolor': INK,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#a4b1bf',
        'xtick.color': INK, 'ytick.color': INK, 'legend.labelcolor': INK,
        'mathtext.fontset': 'dejavusans', 'savefig.transparent': False,
    })
    fit_residuals(results, figure_directory / 'part3_fig1.png')
    identifiability(results, figure_directory / 'part3_fig2.png')
    goal_and_spin(results, trajectories, figure_directory / 'part3_fig3.png')
    trajectories_3d(results, trajectories, figure_directory / 'part3_fig4.png')
