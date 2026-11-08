"""Separate zero-spin stress test using the wind-tunnel authors' drag fits."""

import numpy as np
from scipy.special import expit

from scripts.part06.part6_common import DATA, PHYSICS, TIMES, launch_motion, read_json, write_csv, write_json
from scripts.part06.part6_experiments import (
    build_grid, build_prediction_row, endpoint, event_metrics, predict_observations, rhs_flight,
)

CURVES = read_json(DATA / 'drag_curves.json')


def drag_coefficient(ball, speed):
    q_min, q_max = CURVES['valid_q_m_s']
    if np.any((np.asarray(speed) < q_min) | (np.asarray(speed) > q_max)):
        raise ValueError(f'Air speed is outside the wind-tunnel domain [{q_min}, {q_max}] m/s')
    parameters = CURVES['balls'][ball]
    return parameters['a'] + parameters['b'] * expit(
        (parameters['q_c'] - speed) / parameters['q_s'])


def variable_drag_flight(ball, launch):
    if launch[-1] != 0:
        raise ValueError('Published curve applies to non-spinning balls')
    motion = launch_motion(*launch)
    scale = PHYSICS.density * PHYSICS.area / (2 * PHYSICS.mass)

    def rhs(_time, state):
        velocity = state[3:]
        q = np.linalg.norm(velocity)
        acceleration = -scale*drag_coefficient(ball, q)*q*velocity
        acceleration[2] -= PHYSICS.effective_gravity
        return np.r_[velocity, acceleration]

    flight = rhs_flight(rhs, motion.v0)
    return flight, np.linalg.norm(flight.states[:, 3:], axis=1)


def run_drag_curves(target=DATA):
    all_rows = []
    summary = {}
    for ball in CURVES['balls']:
        rows = []
        for i, launch in enumerate(build_grid()):
            if launch[0] >= CURVES['valid_q_m_s'][1] or launch[-1] != 0:
                continue
            flight, speed = variable_drag_flight(ball, launch)
            truth = endpoint(flight)
            eligible = flight.time > TIMES[-1] and flight.event != 'numerical_failure'
            positions = flight.sample(TIMES)[:, :3] if eligible else None
            if eligible:
                prediction = predict_observations(positions[None], 'M-D')[0]
            else:
                prediction = dict(event='not_observed', goal=None, stop_time=None)
            row = build_prediction_row(i, ball, 'M-D', 0., 0, truth, prediction, positions)
            cd = drag_coefficient(ball, speed)
            rows.append(dict(**row, speed_m_s=launch[0], elevation_deg=launch[1],
                             azimuth_deg=launch[2], q_min_m_s=float(speed.min()),
                             q_max_m_s=float(speed.max()), cd_start=float(cd[0]),
                             cd_end=float(cd[-1])))
        all_rows.extend(rows)
        for speed in (20, 30):
            group = [row for row in rows if row['speed_m_s'] == speed]
            summary[f'{ball}/{speed}'] = event_metrics(group)
    write_csv(target / 'v3_drag_curves.csv', all_rows)
    write_json(target / 'v3_drag_curves_summary.json', summary)


if __name__ == '__main__':
    run_drag_curves()
