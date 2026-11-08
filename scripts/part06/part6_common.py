"""Shared inputs and records for the Section 6 experiments."""

import csv
import json
import math
from dataclasses import replace

import numpy as np

from scripts.common.still_air import Motion, Physics, Settings

from scripts.common.paths import data_dir

DATA = data_dir(6)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


CONFIG = read_json(DATA / 'config.json')
BASE = read_json(data_dir(3) / 'config.json')
TIMES = np.array(BASE['times_s'])
OBS = np.array(BASE['positions_m'])
R0 = OBS[0]
PHYSICS = Physics(**BASE['physics'])
SETTINGS = Settings(**BASE['integration'])
TRUTH_SETTINGS = replace(SETTINGS, **CONFIG['truth_integration'])
GOAL_EVENTS = ('goal_inside', 'goal_outside')


def launch_motion(speed, elevation, azimuth, vertical_spin, *, cd=None, tau=None):
    theta, phi = np.deg2rad([elevation, azimuth])
    velocity = speed * np.array([np.cos(theta) * np.cos(phi),
                                 np.cos(theta) * np.sin(phi), np.sin(theta)])
    synthetic = CONFIG['synthetic']
    return Motion(velocity, synthetic['cd'] if cd is None else cd,
                  (0., 0., float(vertical_spin)), synthetic['k_l'],
                  math.inf if tau is None else tau)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
