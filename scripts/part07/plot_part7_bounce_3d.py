"""Reproduce the one-bounce friction comparison for Section 4.6."""

import csv

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from scripts.common.paths import figures_dir
from scripts.common.plot_style import INK, TEAL, VIOLET, save
from .plot_part7 import read_paths
from .run_part7 import DATA, read_json
from scripts.common.still_air import Settings


COLORS = ('#2965a5', TEAL, VIOLET, '#b84327')
STYLES = ('-', '--', '-.', ':')
OCHRE = '#b58a39'
OUTPUT = figures_dir(7) / 'part7_bounce_3d.png'


def plot(cases, paths, settings, radius):
    goal_x, goal_width = settings.goal_x, settings.goal_half_width
    z_top = 0.32
    fig = plt.figure(figsize=(7.1, 4.1))
    ax = fig.add_axes([.025, .16, .93, .84], projection='3d',
                      proj_type='ortho', computed_zorder=False)
    corners = [(goal_x, -goal_width, 0), (goal_x, goal_width, 0),
               (goal_x, goal_width, z_top), (goal_x, -goal_width, z_top)]
    ax.add_collection3d(Poly3DCollection([corners], facecolor='#e6eef2',
                                         edgecolor='none', alpha=.58, zorder=0))
    # The omitted crossbar is above the displayed height range.
    ax.plot(*np.asarray([corners[i] for i in (3, 0, 1, 2)]).T,
            color=INK, lw=1.05, zorder=2)
    ax.plot([0, goal_x], [0, 0], [0, 0], color='#a5aeb7', lw=.65, ls=':', zorder=1)
    ax.plot([goal_x, goal_x], [-goal_width, goal_width], [radius, radius],
            color='#a5aeb7', lw=.8, ls='--', zorder=1)

    first = cases[0]
    contact = [float(first[f'first_{axis}_m']) for axis in 'xyz']
    ax.plot(*paths[first['case'], 1], color=OCHRE, lw=2.2, zorder=3)
    ax.scatter(*contact, color=OCHRE, edgecolor='white', marker='D',
               linewidth=.7, s=33, depthshade=False, zorder=6)
    rebound_keys = []
    for case, color, style in zip(cases, COLORS, STYLES, strict=True):
        goal = case['event'] == 'goal_inside'
        friction = float(case['mu_f'])
        terminal = [float(case[f'end_{axis}_m']) for axis in 'xyz']
        ax.plot(*paths[case['case'], 2], color=color, ls=style, lw=2.0, zorder=4)
        ax.scatter(*terminal, facecolor=color if goal else 'white',
                   edgecolor=color, linewidth=.9, s=14, depthshade=False, zorder=5)
        rebound_keys.append(Line2D([], [], color=color, ls=style, lw=2,
                                   label=rf'$\mu_f={friction:.2f}$'))

    ax.set(xlim=(-.25, goal_x + .28), ylim=(-goal_width - .12, goal_width + .12),
           zlim=(0, z_top), xticks=[0, 3, 6, 9, goal_x],
           yticks=[-goal_width, 0, goal_width], zticks=[0, radius, .20, .30],
           xlabel='$x$ (m)', ylabel='$y$ (m)', zlabel='$z$ (m)')
    ax.set_yticklabels(['−3.66', '0', '3.66'])
    ax.set_zticklabels(['0', '0.11', '0.20', '0.30'])
    ax.tick_params(pad=1, labelsize=9)
    ax.set_box_aspect((11, 7.32, 3.9), zoom=1.03)
    ax.view_init(elev=24, azim=-58)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
    event_keys = [
        Line2D([], [], color=OCHRE, marker='D', lw=2, label='Flight to first contact'),
        Line2D([], [], color=INK, marker='o', ls='none', label='Inside goal'),
        Line2D([], [], color=INK, marker='o', markerfacecolor='white', ls='none',
               label='Second contact'),
    ]
    fig.legend(handles=rebound_keys, loc='lower center', bbox_to_anchor=(.5, .059),
               ncols=4, frameon=False, fontsize=9)
    fig.legend(handles=event_keys, loc='lower center', bbox_to_anchor=(.5, .005),
               ncols=3, frameon=False, fontsize=9)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save(fig, OUTPUT)


def make_paper_figure(result):
    restitution = result['config']['reference_e_n']
    with (DATA / 'paper_bounce.csv').open(newline='', encoding='utf-8') as stream:
        cases = [row for row in csv.DictReader(stream) if float(row['e_n']) == restitution]
    paths = read_paths(DATA / 'paper_trajectories.csv', columns=('x_m', 'y_m', 'z_m'))
    plot(cases, paths, Settings(**result['integration']), result['physics']['radius'])


if __name__ == '__main__':
    make_paper_figure(read_json(DATA / 'results.json'))
