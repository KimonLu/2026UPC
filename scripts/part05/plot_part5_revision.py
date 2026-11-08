"""Plot observation-error densities, the drag scan and spectral wake crossings."""

import csv
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

from scripts.common.paths import data_dir, figures_dir
from scripts.part06.part6_common import GOAL_EVENTS, read_json, write_json


DATA = data_dir(5)
FIGURES = figures_dir(5)
MODEL = 'M-DM/wind'
POSITION = [(f'position/gaussian/{value/1000:g}', f'Coordinate SD = {value} mm')
            for value in (1, 5, 10)]
TIMING = [(f'time/{kind}/{value/1000:g}', f'{name}, SD = {value:g} ms')
          for kind, name in [('independent', 'Independent'), ('common', 'Shared offset')]
          for value in (.5, 1)]


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def load_observations():
    source = read_json(DATA / 'monte_carlo_summary.json')
    conditions = {key for key, _ in POSITION + TIMING}
    samples = defaultdict(list)
    for row in read_rows(DATA / 'monte_carlo_samples.csv'):
        if row['model'] == MODEL and row['condition'] in conditions:
            samples[row['condition']].append(row)
    result = {}
    for key in sorted(conditions):
        rows = samples[key]
        cloud = np.array([[float(r[c]) for c in ('t_g_s', 'y_g_m', 'z_g_m')]
                          for r in rows if r['event'] in GOAL_EVENTS])
        saved = source['summaries'][f'{key}/{MODEL}']
        events = Counter(row['event'] for row in rows)
        bounds = sum(row['active_bound'] == 'True' for row in rows)
        assert len(rows) == 4000 == len({r['sample'] for r in rows})
        assert dict(events) == {k: v for k, v in saved['events'].items() if v}
        assert bounds == saved['active_bound']
        np.testing.assert_allclose(cloud.std(axis=0, ddof=1), saved['sd'])
        result[key] = dict(saved, goal_fraction=events['goal_inside']/len(rows))
        samples[key] = cloud[:, 1:]
    return samples, result, np.array(source['deterministic'][MODEL]['mean'][1:])


def density_figure(samples, summaries, cases, baseline, shape, filename):
    ellipses = {}
    angle = np.linspace(0, 2*np.pi, 241)
    circle = np.array([np.cos(angle), np.sin(angle)])
    for key, _ in cases:
        info = summaries[key]
        values, vectors = np.linalg.eigh(info['covariance_yz'])
        mean = np.array(info['mean'][1:])
        ellipses[key] = mean[:, None]+vectors @ (
            np.sqrt(values*info['ellipse_radius_squared'])[:, None]*circle)
    all_points = np.concatenate([samples[key] for key, _ in cases] +
                                [ellipse.T for ellipse in ellipses.values()])
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    padding = .055*(high-low)
    low, high = low-padding, high+padding
    # Resolve the tightest panel even when it shares a wide range with the others.
    y, z = (np.linspace(a, b, 225) for a, b in zip(low, high))
    mesh = np.array(np.meshgrid(y, z))
    if shape[0] == 1:
        fields = [gaussian_kde(samples[key].T)(mesh.reshape(2, -1)).reshape(mesh.shape[1:])
                  for key, _ in cases]
    else:
        # Timing-only clouds can be thinner than a pixel. Use the same
        # small display smoothing for the four conditional histograms.
        dy, dz = y[1]-y[0], z[1]-z[0]
        y_edges, z_edges = np.r_[y-dy/2, y[-1]+dy/2], np.r_[z-dz/2, z[-1]+dz/2]
        fields = []
        for key, _ in cases:
            counts, *_ = np.histogram2d(*samples[key].T, bins=(y_edges, z_edges))
            fields.append(gaussian_filter(counts.T, 1.4) / (counts.sum()*dy*dz))
    peak = max(np.max(field) for field in fields)
    norm = LogNorm(peak/2000, peak)
    fig, axes = plt.subplots(*shape, figsize=(10.3, 3.45) if shape[0] == 1 else (8.1, 5.4),
                             sharex=True, sharey=True, squeeze=False, layout='constrained')
    cmap = plt.colormaps['viridis'].with_extremes(under='#f7f9fc', bad='#f7f9fc')
    for ax, (key, title), field in zip(axes.flat, cases, fields):
        image = ax.pcolormesh(y, z, np.ma.masked_less(field, norm.vmin), cmap=cmap, norm=norm,
                              shading='auto', rasterized=True)
        if low[1] <= 2.44 <= high[1]:
            ax.axhline(2.44, ls='--', color='#5d5d5d', lw=1, zorder=2)
        for edge in (-3.66, 3.66):
            if low[0] <= edge <= high[0]:
                ax.axvline(edge, ls='--', color='#5d5d5d', lw=1, zorder=2)
        ax.plot(*ellipses[key], color='#ef6228', lw=1.1, alpha=.9)
        ax.scatter(*baseline, marker='*', c='white', edgecolors='black', lw=.65, s=55, zorder=5)
        ax.set_title(title, fontsize=10)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]))
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$Y_g$ (m)')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$Z_g$ (m)')
    fig.colorbar(image, ax=axes, label=r'Conditional density (m$^{-2}$)', shrink=.82)
    handles = [Line2D([], [], marker='*', color='k', markerfacecolor='w', lw=0, label='No observation error'),
               Line2D([], [], color='#ef6228', label='Empirical 95% ellipse')]
    if low[1] <= 2.44 <= high[1] or low[0] <= 3.66 <= high[0] or low[0] <= -3.66 <= high[0]:
        handles.append(Line2D([], [], color='#5d5d5d', ls='--', label='Goal frame'))
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles), frameon=False, fontsize=9)
    fig.savefig(FIGURES / filename, dpi=250)
    plt.close(fig)


def cd_figure():
    rows = [r for r in read_rows(DATA / 'scenarios.csv') if r['family'] == '5C']
    nominal = next(r for r in rows if r['parameter'] == 'mass' and r['value'] == '0.43')
    fixed = [r for r in rows if r['parameter'] == 'cd']
    assert [float(r['cd']) for r in fixed] == [.10, .20, .30, .50]
    fig, (comp, scatter) = plt.subplots(1, 2, figsize=(8.5, 3.55), layout='constrained')
    for key, label, marker in [('mass', 'Mass scan', 'o'), ('density', 'Density scan', 's')]:
        group = [r for r in rows if r['parameter'] == key]
        cd = np.array([float(r['cd']) for r in group])
        ratio = [float(r['rho_cd_over_m'])/c for r, c in zip(group, cd)]
        comp.scatter(ratio, cd, marker=marker, label=label)
    ratio_grid = np.linspace(2.49, 3.08, 100)
    comp.plot(ratio_grid, float(nominal['rho_cd_over_m'])/ratio_grid, '--', color='#555555',
              label=r'Constant $\rho C_D/m$')
    comp.set(xlabel=r'$\rho/m$ (m$^{-3}$)', ylabel=r'Fitted $C_D$', title='(a) Free fit')
    comp.legend(frameon=False, fontsize=8)
    scatter.axhline(0, color='0.82', lw=.8)
    scatter.axvline(0, color='0.82', lw=.8)
    scatter.scatter(0, 0, marker='*', color='#e56225', s=95, label='Free fit (0.432)')
    for r in fixed:
        point = np.array([float(r['delta_y_m']), float(r['delta_z_m'])])*1e3
        scatter.scatter(*point, color='#206495')
        leftmost = point[0] < 0
        scatter.annotate(f"{float(r['cd']):.2f}", point, xytext=(5 if leftmost else -5, 4),
                         textcoords='offset points', ha='left' if leftmost else 'right', fontsize=9)
    scatter.set(xlabel=r'$\Delta y_g$ (mm)', ylabel=r'$\Delta z_g$ (mm)',
                title='(b) Prescribed drag coefficient')
    scatter.legend(loc='lower right', frameon=False, fontsize=8)
    fig.savefig(FIGURES / 'part5_cd_sensitivity.png', dpi=250)
    plt.close(fig)


def wake_figure():
    path = DATA / 'wake_crossings.csv'
    if not path.exists():
        return
    report = read_json(DATA / 'wake_results.json')
    base = np.array([report['baseline'][k] for k in ('y_g_m', 'z_g_m')])
    clouds = defaultdict(list)
    for row in read_rows(path):
        if row['event'] in GOAL_EVENTS:
            clouds[row['ball']].append(1e3*(np.array([float(row[k]) for k in ('y_g_m', 'z_g_m')])-base))
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.1), sharex=True, sharey=True, layout='constrained')
    for ax, (ball, points), colour in zip(axes.flat, clouds.items(),
                                         ('#b85850', '#00814a', '#24629f', '#1883a6')):
        ax.axhline(0, color='0.8', lw=.8)
        ax.axvline(0, color='0.8', lw=.8)
        ax.scatter(*np.array(points).T, s=11, color=colour, alpha=.52, edgecolor='none')
        ax.scatter(0, 0, marker='*', color='white', edgecolor='black', lw=.65, s=73, zorder=4)
        ax.set_title(ball, fontsize=10)
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$\Delta y_g$ (mm)')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\Delta z_g$ (mm)')
    fig.savefig(FIGURES / 'part5_wake_spectrum_crossings.png', dpi=250)
    plt.close(fig)


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    samples, results, baseline = load_observations()
    density_figure(samples, results, POSITION, baseline, (1, 3), 'part5_position_density.png')
    density_figure(samples, results, TIMING, baseline, (2, 2), 'part5_timing_density.png')
    cd_figure()
    wake_figure()
    write_json(DATA / 'observation_density_summary.json', results)
