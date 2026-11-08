"""Draw the Section 4 figures directly from the exported results."""

import csv
import json
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator

from scripts.common.plot_style import GRID, INK, TEAL, VIOLET, panel, save

ALONG = '#2965a5'
ACROSS = '#b84327'
COVERAGE = '#c2dedc'
FIGURE_STYLE = {
    'font.size': 11, 'axes.labelcolor': INK, 'axes.edgecolor': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.axisbelow': True, 'grid.color': GRID,
    'grid.linewidth': 0.7, 'xtick.color': INK, 'ytick.color': INK,
}


def wind_label(case):
    wx, wy, _ = case['wind_m_s']
    return f'$({wx:g},{wy:g})$'


def coverage_mesh(path):
    """Return the simulated polar cells and outer ring in goal-plane metres."""
    with path.open(encoding='utf-8', newline='') as source:
        rows = sorted(csv.DictReader(source),
                      key=lambda row: (float(row['speed_m_s']), int(row['angle_deg'])))
    points = np.array([[float(row['y_m']), float(row['z_m'])] for row in rows])
    rings = points[1:].reshape(16, 72, 2)
    adjacent = np.roll(rings, -1, axis=1)
    hub = np.broadcast_to(points[0], rings[0].shape)
    triangles = np.stack((hub, rings[0], adjacent[0]), axis=1)
    cells = np.stack((rings[:-1], rings[1:], adjacent[1:], adjacent[:-1]), axis=2)
    return [*triangles, *cells.reshape(-1, 4, 2)], np.vstack((rings[-1], rings[-1, 0]))


def goal_plane(ref, settings, contour_file, path, *, fixed_release=False,
               show_full=True, coverage_file=None):
    reference = np.array([ref['y_m'], ref['z_m']])
    with contour_file.open(encoding='utf-8', newline='') as source:
        samples = list(csv.DictReader(source))
    rings = defaultdict(list)
    axis_sweeps = defaultdict(list)
    offsets = []
    for row in samples:
        speed, angle = float(row['speed_m_s']), int(row['angle_deg'])
        point = 100 * (np.array([float(row['y_m']), float(row['z_m'])]) - reference)
        offsets.append(point)
        if row['on_ring'] == '1':
            rings[speed].append((angle, point))
        if angle % 90 == 0:
            axis_sweeps[angle].append((speed, point))

    if coverage_file is not None:
        cells, boundary = coverage_mesh(coverage_file)
    if show_full:
        fig, (full, zoom) = plt.subplots(1, 2, figsize=(9.2, 3.4), width_ratios=(1.04, 1))
        half_width = settings['goal_half_width']
        height = settings['goal_height']
        full.add_patch(Rectangle((-half_width, 0), 2 * half_width, height,
                                 edgecolor=INK, linewidth=1.1, facecolor='#f3f7fa', zorder=0))
        absolute = reference + np.asarray(offsets) / 100
        full.set(xlim=(min(-half_width, absolute[:, 0].min()) - 0.25,
                       max(half_width, absolute[:, 0].max()) + 0.25),
                 ylim=(min(0, absolute[:, 1].min()) - 0.17, max(height, absolute[:, 1].max()) + 0.17),
                 xlabel=r'$y_g$ (m)', ylabel=r'$z_g$ (m)', aspect='equal')
        full.set_anchor('N')
        panel(full, '(a)', 'Full goal plane')
        if coverage_file is None:
            full.scatter(*absolute.T, s=4, color=TEAL, zorder=3)
        else:
            full.add_collection(PolyCollection(cells, facecolor=COVERAGE, edgecolor='none',
                                               antialiased=False, zorder=2))
            full.plot(*boundary.T, color=TEAL, lw=1.0, zorder=3)
        full.annotate('Crossing region', reference, xytext=(-1.55, 1.57), fontsize=9,
                      color=INK, va='center',
                      arrowprops=dict(arrowstyle='->', color=INK, lw=0.9, shrinkB=4))
        panel(zoom, '(b)', 'Equal wind speeds (4B)' if fixed_release else 'Equal wind speeds')
    else:
        fig, zoom = plt.subplots(figsize=(6.8, 4.65))
        zoom.set_title('Equal wind speeds (4A)', loc='left', pad=9, fontweight='bold', color=INK)
    zoom.set(xlabel=r'$\Delta y_g$ (cm)', ylabel=r'$\Delta z_g$ (cm)')
    if fixed_release:
        zoom.set_box_aspect(0.50)
        zoom.text(0.025, 0.055, 'Expanded vertical scale', transform=zoom.transAxes,
                  fontsize=8, color=INK)
    else:
        zoom.set_aspect('equal')
    zoom.set_anchor('N')
    if coverage_file is not None:
        offset_cells = [100 * (cell - reference) for cell in cells]
        zoom.add_collection(PolyCollection(offset_cells, facecolor=COVERAGE, edgecolor='none',
                                           antialiased=False, zorder=1))
    speed_lines = []
    for (speed, ring), shade in zip(sorted(rings.items()), np.linspace(0.60, 0.12, len(rings))):
        points = [point for _, point in sorted(ring)]
        closed = np.vstack([points, points[0]])
        line, = zoom.plot(*closed.T, lw=1.55, color=plt.cm.viridis(shade),
                          label=f'{speed:g} m/s')
        speed_lines.append(line)
    label_offsets = (
        ((6, -9), (-4, 6), (5, -8), (-6, -5)) if fixed_release
        else ((4, 3), (-4, -7), (5, -8), (-7, 6))
    )
    for (angle, label, color, alignment), offset in zip((
        (0, r'$+x$', ALONG, 'left'),
        (180, r'$-x$', ALONG, 'right'),
        (90, r'$+y$', ACROSS, 'left'),
        (270, r'$-y$', ACROSS, 'right'),
    ), label_offsets):
        path_points = np.vstack([[0, 0], *[point for _, point in sorted(axis_sweeps[angle])]])
        zoom.plot(*path_points.T, color=color, ls='--', lw=1.2, zorder=4)
        end = path_points[-1]
        zoom.scatter(*end, marker='D', s=16, color=color, zorder=5)
        zoom.annotate(label, end, xytext=offset, textcoords='offset points',
                      color=color, fontsize=10, ha=alignment, va='center')

    zoom.scatter(0, 0, marker='o', facecolors='white', edgecolors=INK, lw=1.2, s=36, zorder=6)
    padding = [8.0, 0.7] if fixed_release else [0.63, 0.34]
    lower = np.min(offsets, axis=0) - padding
    upper = np.max(offsets, axis=0) + padding
    zoom.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
    zoom.xaxis.set_major_locator(MultipleLocator(20 if fixed_release else 1))
    zoom.yaxis.set_major_locator(MultipleLocator(2 if fixed_release else 0.5))
    fig.legend(handles=speed_lines, loc='lower center', ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 0.105))
    direction_handles = [
        Line2D([], [], color=ALONG, ls='--', label=r'Pure $x$ wind'),
        Line2D([], [], color=ACROSS, ls='--', label=r'Pure $y$ wind'),
        Line2D([], [], color=INK, ls='', marker='o', mfc='white', label='Calm'),
    ]
    if coverage_file is not None:
        direction_handles.append(Patch(facecolor=COVERAGE, label='Sampled coverage'))
    fig.legend(handles=direction_handles, loc='lower center', ncol=len(direction_handles),
               frameon=False, bbox_to_anchor=(0.5, 0.016))
    fig.subplots_adjust(left=0.075 if show_full else 0.13, right=0.975,
                        bottom=0.24 if show_full else 0.29, top=0.89, wspace=0.32)
    save(fig, path)


def axial_fit_parameters(calm, contour_file, path):
    reference = dict(cd=calm['cd'], omega2_rad_s=calm['omega_basis_rad_s'][1],
                     omega3_rad_s=calm['omega_basis_rad_s'][2])
    reference.update({f'v0{axis}_m_s': value for axis, value in zip('xyz', calm['v0_m_s'])})
    with contour_file.open(encoding='utf-8', newline='') as source:
        samples = list(csv.DictReader(source))
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.0), sharex=True, layout='constrained')
    metrics = (
        ('cd', '(a)', 'Effective drag', r'$C_D$'),
        ('omega2_rad_s', '(b)', 'First fitted spin component', r'$\omega_2$ (rad/s)'),
        ('omega3_rad_s', '(c)', 'Second fitted spin component', r'$\omega_3$ (rad/s)'),
    )
    velocity_axis = axes[1, 1]
    component_styles = ('-', '--', ':')
    for axis, other, color in (('x', 'y', ALONG), ('y', 'x', ACROSS)):
        sweep = [(0., reference)]
        sweep.extend((float(row[f'w{axis}_m_s']), row) for row in samples
                     if float(row[f'w{other}_m_s']) == 0.)
        speeds, fits = zip(*sorted(sweep))
        for ax, (key, _, _, _) in zip(axes.flat, metrics):
            ax.plot(speeds, [float(fit[key]) for fit in fits], color=color, lw=1.6)
        for component, style in zip('xyz', component_styles):
            key = f'v0{component}_m_s'
            delta = [1000 * (float(fit[key]) - reference[key]) for fit in fits]
            velocity_axis.plot(speeds, delta, color=color, ls=style, lw=1.6)

    for ax, (key, letter, title, units) in zip(axes.flat, metrics):
        panel(ax, letter, title)
        ax.axhline(reference[key], color=INK, ls='--', alpha=0.55, lw=0.9)
        ax.set_ylabel(units)
    axes[0, 0].axhline(0.60, color=INK, ls=':', alpha=0.65, lw=0.9)
    axes[0, 0].text(-7.7, 0.61, 'Original upper bound', fontsize=8, color=INK)
    panel(velocity_axis, '(d)', 'Change in release velocity')
    velocity_axis.axhline(0, color=INK, ls='--', alpha=0.55, lw=0.9)
    velocity_axis.set_ylabel(r'$\Delta v_0$ (mm/s)')
    velocity_axis.legend(handles=[
        Line2D([], [], color=INK, ls=style, label=fr'$\Delta v_{{0{component}}}$')
        for component, style in zip('xyz', component_styles)
    ], frameon=False, loc='upper left', ncol=3, fontsize=9)
    for ax in axes.flat:
        ax.set(xlim=(-8.25, 8.25), xticks=(-8, -4, 0, 4, 8))
        ax.yaxis.set_major_locator(MaxNLocator(5))
    for ax in axes[1]:
        ax.set_xlabel('Signed wind speed (m/s)')
    fig.legend(handles=[
        Line2D([], [], color=ALONG, label=r'Pure $x$ wind'),
        Line2D([], [], color=ACROSS, label=r'Pure $y$ wind'),
        Line2D([], [], color=INK, ls='--', alpha=0.55, label='Calm reference'),
    ], loc='outside lower center', ncol=3, frameon=False)
    save(fig, path)


def fitted_parameters(cases, path):
    ref = cases[0]
    x = np.arange(len(cases))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.2), sharex=True, layout='constrained')
    labels = [wind_label(case) for case in cases]
    metrics = (
        (axes[0, 0], '(a)', 'Effective drag', r'$C_D$',
         [case['cd'] for case in cases], ref['cd']),
        (axes[0, 1], '(b)', 'First fitted spin component', r'$\omega_2$ (rad/s)',
         [case['omega_basis_rad_s'][1] for case in cases], ref['omega_basis_rad_s'][1]),
        (axes[1, 0], '(c)', 'Second fitted spin component', r'$\omega_3$ (rad/s)',
         [case['omega_basis_rad_s'][2] for case in cases], ref['omega_basis_rad_s'][2]),
    )
    for ax, letter, title, units, values, reference in metrics:
        panel(ax, letter, title)
        ax.axhline(reference, color=INK, ls='--', lw=0.8, zorder=1)
        for mask, color in ((slice(1, 5), ALONG), (slice(5, 9), ACROSS)):
            ax.plot(x[mask], values[mask], color=color, marker='o', lw=1.1, ms=4.5)
        ax.scatter(x[[0, 9]], np.array(values)[[0, 9]], c=[INK, VIOLET], zorder=4, s=23)
        ax.set_ylabel(units)
    ax = axes[1, 1]
    panel(ax, '(d)', 'Change in release velocity')
    ax.axhline(0, color=INK, ls='--', lw=0.8, zorder=1)
    delta_v = 1000 * (np.array([case['v0_m_s'] for case in cases]) - ref['v0_m_s'])
    for i, (component, color, marker) in enumerate(zip('xyz', (ALONG, ACROSS, TEAL), ('o', 's', 'D'))):
        ax.scatter(x + 0.17 * (i - 1), delta_v[:, i], s=18, c=color, marker=marker,
                   label=fr'$\Delta v_{{0{component}}}$')
    ax.legend(ncol=3, frameon=False, fontsize=9, loc='upper right')
    ax.margins(y=0.25)
    ax.set_ylabel(r'$\Delta v_0$ (mm/s)')
    for ax in axes.flat:
        ax.axvline(4.5, lw=0.75, color=GRID)
        ax.axvline(8.5, lw=0.75, color=GRID)
        ax.set_xticks(x, labels, rotation=34, ha='right')
        ax.yaxis.set_major_locator(MaxNLocator(4))
    for ax in axes[1]:
        ax.set_xlabel(r'Wind $(w_x,w_y)$ (m/s)')
    save(fig, path)


def displacement_comparison(pairs, path):
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 4.6), sharey=True, layout='constrained')
    rows = np.arange(len(pairs))
    for ax, (letter, title, key, scale, unit) in zip(axes, (
        ('(a)', 'Lateral crossing', 'y_m', 100, r'$\Delta y_g$ (cm)'),
        ('(b)', 'Crossing height', 'z_m', 100, r'$\Delta z_g$ (cm)'),
        ('(c)', 'Arrival time', 't_s', 1000, r'$\Delta t_g$ (ms)'),
    )):
        panel(ax, letter, title)
        refit, fixed = (np.array([pair[i]['delta_goal'][key] * scale
                                 if pair[i]['delta_goal'] is not None else np.nan
                                 for pair in pairs]) for i in (0, 1))
        ax.axvline(0, color=INK, ls='--', lw=0.85)
        ax.hlines(rows, refit, fixed, color='#9caebe', lw=1, zorder=2)
        ax.scatter(fixed, rows, color=VIOLET, marker='s', s=42, zorder=3, label='4B  Fixed release')
        ax.scatter(refit, rows, color=TEAL, marker='o', s=32, zorder=4, label='4A  Refit')
        ax.set_xlabel(unit)
        ax.xaxis.set_major_locator(MaxNLocator(5))
    axes[0].set_yticks(rows, [wind_label(pair[0]) for pair in pairs])
    axes[0].invert_yaxis()
    axes[0].set_ylabel(r'Wind $(w_x,w_y)$ (m/s)')
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='outside lower center', ncol=2, frameon=False)
    save(fig, path)


def trajectories_3d(cases, observations, settings, trajectory_file, path):
    scenarios = (
        ('x-5', 'Headwind'), ('x+5', 'Tailwind'),
        ('y+5', 'Crosswind'), ('diagonal', 'Oblique wind'),
    )
    keys = {f'{experiment}/{wind}' for wind, _ in scenarios for experiment in ('4A', '4B')}
    tracks = defaultdict(list)
    with trajectory_file.open(encoding='utf-8', newline='') as source:
        for row in csv.DictReader(source):
            if row['case'] in keys:
                tracks[row['case']].append([float(row[axis]) for axis in ('x_m', 'y_m', 'z_m')])
    goal_x, half_width, height = (settings[key] for key in
                                 ('goal_x', 'goal_half_width', 'goal_height'))
    fig = plt.figure(figsize=(9.2, 6.9))
    for index, (wind, title) in enumerate(scenarios, 1):
        ax = fig.add_subplot(2, 2, index, projection='3d')
        wx, wy, _ = cases[f'4A/{wind}']['wind_m_s']
        ax.set_title(fr'({chr(96 + index)})  {title}  $({wx:g},{wy:g})$ m/s',
                     loc='left', color=INK, fontweight='bold', pad=0)
        ax.plot([goal_x] * 5, [-half_width, half_width, half_width, -half_width, -half_width],
                [0, 0, height, height, 0], color=INK, linewidth=1.1, alpha=0.68)
        for experiment, color, style in (('4A', TEAL, '-'), ('4B', VIOLET, '--')):
            positions = np.array(tracks[f'{experiment}/{wind}'])
            ax.plot(*positions.T, color=color, linestyle=style, linewidth=1.9)
            ax.plot(*positions[-1], marker='o', color=color, markersize=3.2)
        ax.plot(*np.array(observations).T, marker='o', color=INK, linestyle='none',
                markersize=3.4)
        # The horizontal arrow lies in the ground plane and uses ground-frame components.
        dx, dy = 0.53 * wx, 0.53 * wy
        ax.quiver(6.0 - dx / 2, -0.9 - dy / 2, 0, dx, dy, 0,
                  color=ACROSS, linewidth=2.1, arrow_length_ratio=0.21)
        ax.set(xlim=(0, goal_x + 0.35), ylim=(-half_width, half_width), zlim=(0, 2.75),
               xticks=(0, 5, 11), yticks=(-3, 0, 3), zticks=(0, 1, 2))
        ax.set_xlabel('$x$ (m)', labelpad=1)
        ax.set_ylabel('$y$ (m)', labelpad=1)
        ax.set_zlabel('$z$ (m)', labelpad=1)
        ax.tick_params(pad=0, labelsize=9)
        ax.set_box_aspect((goal_x + 0.35, 2 * half_width, 2.75), zoom=1.10)
        ax.view_init(elev=23, azim=-62)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.96, 0.97, 0.98, 0.65))
    fig.legend(handles=[
        Line2D([], [], color=TEAL, lw=2, label='4A  Refit'),
        Line2D([], [], color=VIOLET, lw=2, ls='--', label='4B  Fixed release'),
        Line2D([], [], color=INK, marker='o', ls='', ms=4, label='Observed positions'),
        Line2D([], [], color=ACROSS, lw=2, marker='>', markevery=[1], label='Wind direction'),
    ], loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.015))
    fig.subplots_adjust(left=0.035, right=0.94, bottom=0.085, top=0.96, wspace=0.0, hspace=0.06)
    save(fig, path)


def make_figures(data_directory, figure_directory, *, crossings_only=False,
                 fixed_contours_only=False):
    result = json.loads((data_directory / 'results.json').read_text(encoding='utf-8'))
    cases = result['cases']
    names = result['config']['winds']
    with plt.rc_context(FIGURE_STYLE):
        reference = cases['4A/calm']['goal']
        settings = result['part3_config']['integration']
        if not fixed_contours_only:
            goal_plane(reference, settings, data_directory / 'wind_contours.csv',
                       figure_directory / 'part4_fig1.png')
        goal_plane(reference, settings, data_directory / 'fixed_release_contours.csv',
                   figure_directory / 'part4_fig5.png', fixed_release=True)
        if crossings_only or fixed_contours_only:
            return
        fitted_parameters([cases[f'4A/{name}'] for name in names], figure_directory / 'part4_fig2.png')
        displacement_comparison([(cases[f'4A/{name}'], cases[f'4B/{name}']) for name in names],
                                figure_directory / 'part4_fig3.png')
        base = result['part3_config']
        trajectories_3d(cases, base['positions_m'], base['integration'],
                        data_directory / 'trajectories.csv', figure_directory / 'part4_fig4.png')


def make_paper_figures(data_directory, figure_directory):
    result = json.loads((data_directory / 'results.json').read_text(encoding='utf-8'))
    calm = result['cases']['4A/calm']
    settings = result['part3_config']['integration']
    refits = data_directory / 'wind_contours.csv'
    with plt.rc_context(FIGURE_STYLE):
        goal_plane(calm['goal'], settings, refits, figure_directory / 'part4_4a_wind_contours.png',
                   show_full=False)
        axial_fit_parameters(calm, refits, figure_directory / 'part4_4a_fit_parameters.png')
        goal_plane(calm['goal'], settings, data_directory / 'fixed_release_contours.csv',
                   figure_directory / 'part4_4b_wind_coverage.png', fixed_release=True,
                   coverage_file=data_directory / 'fixed_release_coverage.csv')
