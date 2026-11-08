"""Fit and plot the constant-acceleration and gravity-only baselines."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import scipy

from scripts.common.paths import data_dir, figures_dir


DATA = data_dir(2)
FIGURES = figures_dir(2)
TIMES = np.array([0., .05, .10])
OBSERVATIONS = np.array([[0., 0., .11], [1.50, .45, .44], [2.96, .86, .74]])
G = 9.81
R = .11
GOAL_X = 11.
COLORS = ['#126782', '#a34924']


def position(model: dict, times) -> np.ndarray:
    """Ball-center positions at scalar time or an array of times (SI units)."""
    t = np.asarray(times, dtype=float)[..., None]
    return OBSERVATIONS[0] + t * np.array(model['v0_m_s']) + .5 * t**2 * np.array(model['a_m_s2'])


def compute_baselines() -> list[dict]:
    """Fit the two prescribed models and analytically locate their events."""
    h = TIMES[1]
    a_ca = (OBSERVATIONS[2] - 2 * OBSERVATIONS[1] + OBSERVATIONS[0]) / h**2
    v_ca = (OBSERVATIONS[1] - OBSERVATIONS[0]) / h - .5 * a_ca * h
    gravity = np.array([0., 0., -G])
    t = TIMES[1:, None]
    v_g = np.sum(t * (OBSERVATIONS[1:] - OBSERVATIONS[0] - .5 * gravity * t**2), axis=0) / np.sum(t**2)
    models = []
    for name, case, velocity, acceleration in [('B-CA', '1A', v_ca, a_ca), ('B-G', '1B', v_g, gravity)]:
        # The numerically stable smaller root; x0=0, v0x>0 for this dataset.
        discriminant = velocity[0]**2 + 2 * acceleration[0] * GOAL_X
        tg = 2 * GOAL_X / (velocity[0] + np.sqrt(discriminant))
        # Initial z=R is excluded: this is the later downward return to z=R.
        t_contact = -2 * velocity[2] / acceleration[2]
        model = {'model': name, 'case': case, 'v0_m_s': velocity.tolist(),
                 'a_m_s2': acceleration.tolist(), 'goal_time_s': float(tg),
                 'ground_contact_time_s': float(t_contact),
                 'fit_status': 'analytic_interpolation' if case == '1A' else 'analytic_least_squares'}
        crossing = position(model, tg)
        residual = position(model, TIMES[1:]) - OBSERVATIONS[1:]
        sse = float(np.sum(residual**2))
        model.update(goal_position_m=crossing.tolist(), residuals_m=residual.tolist(),
                     residual_definition='predicted minus observed at t=0.05 and 0.10 s',
                     sse_m2=sse, position_rmse_m=float(np.sqrt(sse / 2)),
                     coordinate_rmse_m=float(np.sqrt(sse / 6)),
                     goal_velocity_m_s=(velocity + acceleration * tg).tolist())
        model['event'] = ('ground_first' if t_contact <= tg else
                          'inside_goal' if abs(crossing[1]) < 3.66 and 0 < crossing[2] < 2.44 else
                          'outside_goal')
        models.append(model)
    return models


def write_results(models: list[dict]) -> None:
    out = DATA
    out.mkdir(parents=True, exist_ok=True)
    delta = np.array(models[1]['goal_position_m']) - np.array(models[0]['goal_position_m'])
    payload = {
        'scope': 'Cases 1A and 1B',
        'input': {'times_s': TIMES.tolist(), 'observations_m': OBSERVATIONS.tolist(),
                  'fixed_initial_position': True, 'gravity_m_s2': G, 'radius_m': R,
                  'goal_x_m': GOAL_X, 'goal_y_bounds_m': [-3.66, 3.66], 'goal_z_bounds_m': [0., 2.44]},
        'provenance': {'observations': 'https://uphysicsc.com/2026contest.html',
                       'models': 'Constant acceleration and gravity only',
                       'gravity_and_radius': 'Fixed model parameters in SI units',
                       'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                                   'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__},
                       'fit_weights': 'equal coordinate weights; initial position fixed',
                       'randomness': 'none'},
        'models': models,
        'comparison': {'B_G_minus_B_CA_goal_position_m': delta.tolist(),
                       'goal_separation_m': float(np.linalg.norm(delta[1:])),
                       'B_CA_minus_B_G_time_s': models[0]['goal_time_s'] - models[1]['goal_time_s'],
                       'interpretation': 'model spread, not a confidence interval'},
    }
    (out / 'part2_baseline_results.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (out / 'part2_baseline_summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['case','model','v0x_m_s','v0y_m_s','v0z_m_s','ax_m_s2','ay_m_s2','az_m_s2',
                         'goal_time_s','goal_x_m','goal_y_m','goal_z_m','ground_contact_time_s',
                         'sse_m2','position_rmse_m','coordinate_rmse_m','event','fit_status'])
        for m in models:
            writer.writerow([m['case'],m['model'],*m['v0_m_s'],*m['a_m_s2'],m['goal_time_s'],
                             *m['goal_position_m'],m['ground_contact_time_s'],m['sse_m2'],
                             m['position_rmse_m'],m['coordinate_rmse_m'],m['event'],m['fit_status']])
    with (out / 'part2_baseline_residuals.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model','time_s','observed_x_m','observed_y_m','observed_z_m',
                         'fitted_x_m','fitted_y_m','fitted_z_m','residual_x_m','residual_y_m','residual_z_m'])
        for m in models:
            for time, obs, pred in zip(TIMES, OBSERVATIONS, position(m, TIMES)):
                writer.writerow([m['model'],time,*obs,*pred,*(pred-obs)])
    with (out / 'part2_baseline_trajectories.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model','time_s','x_m','y_m','z_m','vx_m_s','vy_m_s','vz_m_s','phase'])
        for m in models:
            times = np.unique(np.r_[np.linspace(0, m['goal_time_s'], 601), TIMES])
            for time, p in zip(times, position(m, times)):
                v = np.array(m['v0_m_s']) + np.array(m['a_m_s2']) * time
                writer.writerow([m['model'],time,*p,*v,'observed_window' if time <= .1 else 'prediction'])
    # Tables are generated from the same full-precision calculations as the CSV/JSON.
    rows = []
    for m in models:
        p = m['goal_position_m']
        rows.append(f"{m['model']} & {m['goal_time_s']:.6f} & {p[1]:.6f} & {p[2]:.6f} & "
                    f"{1000*m['position_rmse_m']:.3f} & {m['ground_contact_time_s']:.6f} \\\\")
    table = r'''% Generated by scripts/part02/part2_baseline.py; do not edit numeric rows.
\noindent\begin{minipage}{\linewidth}
\centering
\captionsetup{hypcap=false,skip=4pt}
\captionof{table}{Baseline predictions.}
\label{tab:part2-results}
\small
\begin{tabular}{lrrrrr}
\toprule
Model & $t_g$ (s) & $y_g$ (m) & $z_g$ (m) & RMSE (mm) & $t_{\rm c}$ (s) \\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{minipage}
\par\smallskip
'''
    (out / 'part2_baseline_table.tex').write_text(table, encoding='utf-8')


def plot_style() -> None:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.labelsize': 13, 'xtick.labelsize': 11,
                         'ytick.labelsize': 11, 'savefig.facecolor': 'white',
                         'axes.edgecolor': '#73808a'})


def save_tight_plot(fig, path: Path) -> None:
    """Export the rendered plot's content bounds, excluding empty 3D margins."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rows, columns = np.where(np.any(rgba[:, :, :3] < 248, axis=2))
    pad = 4
    height = rgba.shape[0]
    bounds = Bbox.from_extents((columns.min() - pad) / fig.dpi,
                              (height - rows.max() - pad) / fig.dpi,
                              (columns.max() + pad) / fig.dpi,
                              (height - rows.min() + pad) / fig.dpi)
    fig.savefig(path, dpi=360, bbox_inches=bounds, pad_inches=0)


def draw_figure(models: list[dict], index: int) -> None:
    """A clean 3D path: axes, observations, prediction, and goal frame only."""
    plot_style()
    model = models[index]
    color = COLORS[index]
    fig = plt.figure(figsize=(7.4, 4.25), facecolor='white')
    ax = fig.add_axes([.005, .025, .99, .97], projection='3d')
    early = position(model, np.linspace(0, .10, 120))
    late = position(model, np.linspace(.10, model['goal_time_s'], 500))
    ax.plot(*early.T, color='#d3981e', lw=2.8)
    ax.plot(*late.T, color=color, lw=2.1)
    frame = np.array([[11, -3.66, 0], [11, -3.66, 2.44],
                      [11, 3.66, 2.44], [11, 3.66, 0], [11, -3.66, 0]])
    ax.add_collection3d(Poly3DCollection([frame[:4]], facecolors='#9cbcc7',
                                       alpha=.09, edgecolors='none'))
    ax.plot(*frame.T, color='#394b59', lw=1.1)
    ax.plot(late[:, 0], late[:, 1], np.zeros(len(late)), color=color,
            alpha=.25, lw=.9, ls='--')
    ax.scatter(*OBSERVATIONS.T, s=22, c='white', edgecolors='#202b33',
               linewidths=.9, depthshade=False)
    crossing = np.array(model['goal_position_m'])
    ax.scatter(*crossing, s=40, c=color, edgecolor='white',
               linewidth=.7, depthshade=False)
    ax.plot([11, 11], [crossing[1], crossing[1]], [0, crossing[2]],
            ls=':', color=color, lw=.9)
    ax.set(xlim=(0, 11.5), ylim=(-3.9, 3.9), zlim=(0, 2.75))
    ax.set_xlabel('$x$ (m)', labelpad=9)
    ax.set_ylabel('$y$ (m)', labelpad=10)
    ax.set_zlabel('$z$ (m)', labelpad=9)
    ax.set_xticks([0, 3, 6, 9, 11])
    ax.set_yticks([-3.66, 0, 3.66])
    ax.set_zticks([0, 1, 2, 2.44])
    ax.tick_params(pad=1)
    ax.set_box_aspect((11.5, 7.8, 3.8), zoom=1.16)
    ax.view_init(elev=21, azim=-59)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis._axinfo['grid'].update(color=(.75, .79, .82, .4), linewidth=.5)
    for artist in [*ax.lines, *ax.collections]:
        artist.set_clip_on(False)
    path = FIGURES / f'part2_fig{index+1}.png'
    path.parent.mkdir(exist_ok=True)
    save_tight_plot(fig, path)
    plt.close(fig)


def draw_comparison(models: list[dict]) -> None:
    """Goal-plane geometry and the two named predictions, without commentary."""
    plot_style()
    fig, ax = plt.subplots(figsize=(7.4, 3.1), facecolor='white')
    fig.subplots_adjust(left=.11, right=.985, bottom=.22, top=.985)
    ax.add_patch(Rectangle((-3.66, 0), 7.32, 2.44, facecolor='#f5f8f9',
                           edgecolor='#394b59', lw=1.2))
    ax.axvline(0, color='#b2bbc2', ls=':', lw=.8, zorder=0)
    for index, model in enumerate(models):
        p = model['goal_position_m']
        ax.scatter(p[1], p[2], s=42, facecolors=COLORS[index] if index == 0 else 'white',
                   edgecolors=COLORS[index], linewidths=1.2, zorder=5)
        label = f"{model['model']}\n({p[1]:.3f}, {p[2]:.3f}) m"
        ax.annotate(label, (p[1], p[2]),
                    xytext=(1.05, 1.32) if index == 0 else (2.85, .63),
                    textcoords='data', ha='center', va='top',
                    color=COLORS[index], fontsize=11, linespacing=1.3,
                    arrowprops={'arrowstyle': '-', 'color': COLORS[index], 'lw': .75})
    ax.set(xlim=(-3.95, 3.95), ylim=(-.10, 2.70), xlabel='$y$ (m)', ylabel='$z$ (m)')
    ax.set_xticks([-3.66, 0, 3.66])
    ax.set_yticks([0, 1.22, 2.44])
    ax.set_aspect('equal', adjustable='box')
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(FIGURES / 'part2_fig3.png', dpi=360,
                bbox_inches='tight', pad_inches=.035)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--plots-only', action='store_true', help='Plot the saved fit results.')
    modes.add_argument('--no-plots', action='store_true', help='Export results and table only.')
    args = parser.parse_args()

    if args.plots_only:
        models = json.loads((DATA / 'part2_baseline_results.json').read_text(encoding='utf-8'))['models']
    else:
        models = compute_baselines()
        write_results(models)
    if not args.no_plots:
        FIGURES.mkdir(parents=True, exist_ok=True)
        for index in range(len(models)):
            draw_figure(models, index)
        draw_comparison(models)
    for m in models:
        print(f"{m['model']}: v0={m['v0_m_s']}, a={m['a_m_s2']}, "
              f"tg={m['goal_time_s']:.12f}, goal={m['goal_position_m']}, "
              f"RMSE={m['position_rmse_m']:.12f} m, contact={m['ground_contact_time_s']:.12f} s")
    print('Goal-plane separation:', np.linalg.norm(np.array(models[1]['goal_position_m'])[1:] - np.array(models[0]['goal_position_m'])[1:]), 'm')


if __name__ == '__main__':
    main()
