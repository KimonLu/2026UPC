"""Render revised Section 3 figures from the existing fitted trajectories."""
import csv
import json
from collections import defaultdict

from scripts.common.plot_style import GRID, INK, STYLES, TEAL, VIOLET, panel, save
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from scripts.common.paths import data_dir, figures_dir

DATA = data_dir(3)
OUTPUT = figures_dir(3)
CASES = {
    'M-D20': 'md_fixed', 'M-D': 'md_fit', 'M-DM': 'mdm_constant',
    'M-DM5': 'mdm_tau5', 'M-DM1': 'mdm_tau1',
}
FIT, PRED = '#b58a39', '#3a6478'
PASS = '#e5f2ed'


def read_trajectories():
    rows = defaultdict(list)
    with (DATA / 'trajectories.csv').open(newline='', encoding='utf-8') as source:
        for row in csv.DictReader(source):
            rows[row['case']].append([float(row[k]) for k in ('time_s', 'x_m', 'y_m', 'z_m')])
    return {case: np.array(points) for case, points in rows.items()}


def trajectory(points, config, path):
    observed = np.asarray(config['positions_m'])
    last_t = config['times_s'][-1]
    field = config['integration']
    x, y, z = field['goal_x'], field['goal_half_width'], field['goal_height']
    fig = plt.figure(figsize=(7.1, 3.75))
    ax = fig.add_axes([0.025, 0.12, 0.93, 0.88], projection='3d', proj_type='ortho',
                      computed_zorder=False)
    gate = np.array([(x, -y, 0), (x, y, 0), (x, y, z), (x, -y, z), (x, -y, 0)])
    ax.add_collection3d(Poly3DCollection([gate[:-1]], facecolor='#e6eef2',
                                         edgecolor='none', alpha=0.58, zorder=0))
    ax.plot(*gate.T, color=INK, linewidth=1.05, zorder=2)
    ax.plot([0, x], [0, 0], [0, 0], color='#a5aeb7', lw=.65, ls=':', zorder=1)
    ax.plot(*points[points[:, 0] <= last_t, 1:].T, color=FIT, lw=2.1, zorder=3)
    ax.plot(*points[points[:, 0] >= last_t, 1:].T, color=PRED, lw=2.1, zorder=3)
    ax.scatter(*observed.T, facecolor='white', edgecolor=INK, s=21, depthshade=False,
               linewidth=.75, zorder=4)
    ax.scatter(*points[-1, 1:], color=PRED, s=32, edgecolor='white', linewidth=.55,
               depthshade=False, zorder=4)
    ax.set(xlim=(-.25, x + .28), ylim=(-y - .12, y + .12), zlim=(0, z + .18),
           xticks=[0, 3, 6, 9, 11], yticks=[-3.66, 0, 3.66], zticks=[0, 1, 2, 2.44],
           xlabel='$x$ (m)', ylabel='$y$ (m)', zlabel='$z$ (m)')
    ax.set_yticklabels(['−3.66', '0', '3.66'])
    ax.set_zticklabels(['0', '1', '2', '2.44'])
    ax.tick_params(pad=1, labelsize=9)
    ax.set_box_aspect((11, 7.32, 3.9), zoom=1.03)
    ax.view_init(elev=24, azim=-58)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
    legend = [Line2D([], [], color=FIT, lw=2, label='Fit window'),
              Line2D([], [], color=PRED, lw=2, label='Extrapolation'),
              Line2D([], [], color=INK, markerfacecolor='white', marker='o',
                     ls='none', label='Observed position'),
              Line2D([], [], color=PRED, marker='o', ls='none', label='Goal crossing')]
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .006),
               ncols=4, frameon=False, fontsize=9)
    save(fig, path)


def goal_comparison(data, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.65), layout='constrained',
                             gridspec_kw={'width_ratios': [1.26, 1]})
    field = data['config']['integration']
    w, h = field['goal_half_width'], field['goal_height']
    zoom = (2.40, 3.34, 1.82, 2.04)
    for i, ax in enumerate(axes):
        ax.add_patch(Rectangle((-w, 0), 2*w, h, fc=PASS, ec='#44695d', lw=1.2))
        for key, (color, marker, _) in STYLES.items():
            goal = data['cases'][key]['goal']
            ax.scatter(goal['y_m'], goal['z_m'], color=color, marker=marker,
                       s=31 if i else 25, ec='white', lw=.65, zorder=3)
        ax.set(xlabel=r'$y_g$ (m)', ylabel=r'$z_g$ (m)')
        ax.set_aspect('equal')
        ax.grid(color='white', alpha=.85)
        ax.set_axisbelow(True)
    axes[0].set(xlim=(-w-.16, w+.16), ylim=(-.10, h+.14))
    axes[0].add_patch(Rectangle((zoom[0], zoom[2]), zoom[1]-zoom[0], zoom[3]-zoom[2],
                                fill=False, ec=INK, ls='--', lw=1, zorder=5))
    axes[1].set(xlim=zoom[:2], ylim=zoom[2:], xticks=[2.4, 2.6, 2.8, 3.0, 3.2],
                yticks=[1.85, 1.90, 1.95, 2.00])
    panel(axes[0], '(a)', 'Full goal mouth')
    panel(axes[1], '(b)', 'Crossing detail')
    handles = [Line2D([], [], marker=marker, color=color, markeredgecolor='white',
                      ls='none', markersize=7, label=label)
               for color, marker, label in STYLES.values()]
    fig.legend(handles=handles, loc='outside lower center', ncols=4, frameon=False, fontsize=9)
    save(fig, path)


def chord_scatter(chord, path):
    fig, ax = plt.subplots(figsize=(6.6, 4.3), layout='constrained')
    y, z = ([row['goal'][f'{axis}_m'] for row in chord] for axis in 'yz')
    spin = np.array([row['value'] for row in chord])
    marks = ax.scatter(y, z, c=spin, cmap='coolwarm', vmin=-100, vmax=100, s=55,
                       edgecolor=INK, linewidth=.45, zorder=3)
    middle = np.flatnonzero(spin == 0).item()
    ax.scatter(y[middle], z[middle], s=158, facecolor='none', edgecolor=INK,
               linewidth=1.2, label=r'Constant-spin reference ($\omega_1=0$)', zorder=4)
    ax.set(xlabel=r'$y_g$ (m)', ylabel=r'$z_g$ (m)',
           xlim=(2.555, 2.664), ylim=(1.914, 2.013))
    ax.set_aspect('equal')
    ax.grid(color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    fig.colorbar(marks, ax=ax, pad=.04, ticks=[-100, -50, 0, 50, 100],
                 label=r'Fixed $\omega_1$ (rad s$^{-1}$)')
    save(fig, path)


def drag_profile(drag, best, screen, path):
    fig, axes = plt.subplots(1, 2, figsize=(8.65, 3.4), layout='constrained')
    cd = [row['value'] for row in drag]
    passes = [row['value'] for row in drag if row['passes_screen']]
    for ax in axes:
        ax.axvspan(min(passes), max(passes), color=PASS, zorder=0)
        ax.set(xlabel=r'Fixed $C_D$', xlim=(.04, .61))
        ax.grid(color=GRID, alpha=.75)
        ax.set_axisbelow(True)
    axes[0].plot(cd, [1000*row['max_residual_m'] for row in drag], color=TEAL, lw=1.6)
    axes[0].axhline(screen, color=INK, ls='--', lw=.9, label='5 mm screen')
    axes[0].scatter(best['cd'], 1000*best['max_residual_m'], marker='*', color=INK,
                    s=80, zorder=3, label='Joint fit')
    axes[0].set(ylabel='Largest coordinate residual (mm)', ylim=(-.75, None))
    axes[0].legend(frameon=False, fontsize=8.5)
    panel(axes[0], '(a)', 'Fit residual')
    axes[1].axhline(0, color='#8795a4', lw=.8)
    for axis, color in (('y', TEAL), ('z', VIOLET)):
        key = f'{axis}_m'
        axes[1].plot(cd, [100*(row['goal'][key]-best['goal'][key]) for row in drag],
                     color=color, lw=1.55, label=rf'$\Delta {axis}_g$')
    axes[1].axvline(best['cd'], color=INK, ls=':', lw=.9)
    axes[1].set_ylabel('Crossing shift (cm)')
    axes[1].legend(frameon=False, fontsize=9)
    panel(axes[1], '(b)', 'Shift from the joint fit')
    save(fig, path)


def main():
    plt.rcParams.update({'font.size': 10, 'axes.labelcolor': INK, 'axes.edgecolor': '#a4b1bf',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'xtick.color': INK, 'ytick.color': INK})
    data = json.loads((DATA / 'results.json').read_text(encoding='utf-8'))
    paths = read_trajectories()
    with (DATA / 'cases.csv').open(newline='', encoding='utf-8') as stream:
        table = {row['case']: row for row in csv.DictReader(stream)}
    assert set(data['cases']) == set(STYLES) == set(table)
    for key, rows in paths.items():
        reference = table[key]
        np.testing.assert_allclose(rows[-1], [float(reference['t_goal_s']), 11.,
                                              float(reference['y_goal_m']),
                                              float(reference['z_goal_m'])], atol=1e-10, rtol=0)
        assert np.count_nonzero(rows[:, 0] == data['config']['times_s'][-1]) == 1
        assert reference['event'] == 'goal_inside'
    chord = [row for row in data['profiles'] if row['case'] == 'omega_parallel']
    drag = [row for row in data['profiles'] if row['case'] == 'cd']
    assert [row['value'] for row in chord] == list(range(-100, 101, 20))
    assert all(row['event'] == 'goal_inside' and row['max_residual_m'] < 1e-10 for row in chord)
    screen = data['config']['coordinate_screen_m']
    passed = [row['value'] for row in drag if row['max_residual_m'] <= screen]
    np.testing.assert_allclose(passed, np.arange(.28, .49, .01))
    OUTPUT.mkdir(exist_ok=True)
    for key, name in CASES.items():
        trajectory(paths[key], data['config'], OUTPUT / f'part3_traj_{name}.png')
    goal_comparison(data, OUTPUT / 'part3_goal_comparison.png')
    chord_scatter(chord, OUTPUT / 'part3_omega1_crossings.png')
    drag_profile(drag, data['cases']['M-DM'], 1000*screen, OUTPUT / 'part3_cd_sensitivity.png')
    print(f'Checked {len(table)} crossings, {len(chord)} chord-spin samples and '
          f'{len(drag)} drag samples. Screened drag interval {min(passed):.2f}–{max(passed):.2f}.')


if __name__ == '__main__':
    main()
