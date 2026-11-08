"""Numerical checks and held-out synthetic flight predictions."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import product
from math import inf

import numpy as np
from scipy.integrate import simpson, solve_ivp
from scipy.optimize import least_squares

from scripts.common.paths import data_dir
from scripts.part06.part6_common import (
    CONFIG, DATA, GOAL_EVENTS, OBS, PHYSICS, R0, SETTINGS, TIMES,
    TRUTH_SETTINGS, launch_motion, read_json, write_csv,
)
from scripts.part05 import part5_solver
from scripts.part03.run_part3 import baseline_flight
from scripts.common.still_air import (
    Flight, Motion, PARAMETERS, aerodynamic_acceleration, baseline_initials,
    fit_from_positions, inside_goal, simulate,
)
from scripts.part05.wake_spectrum import (
    PSD_PATH, VALIDATION_DRAWS_PER_SPECTRUM, VALIDATION_SEED, generate_traces, wake_rhs,
)

SYNTHETIC = CONFIG['synthetic']
MISMATCH = CONFIG['mismatch']
EVENTS = (*part5_solver.EVENTS, 'not_observed')
COORDINATES = ('x', 'y', 'z')


def saved_shots():
    part3 = read_json(data_dir(3) / 'results.json')['cases']
    part4 = read_json(data_dir(4) / 'results.json')['cases']
    return {**part3, **{key: case for key, case in part4.items()
                        if key.startswith('W1/') and key != 'W1/calm'}}


def run_v1():
    checks = []

    def check(name, value, limit, unit=''):
        checks.append(dict(check=name, value=float(value), limit=limit, unit=unit,
                           passed=bool(value <= limit)))

    case_map = saved_shots()
    velocity = np.asarray(case_map['B2']['v0_m_s'])
    for name, physics in [('gravity', replace(PHYSICS, density=0.)),
                          ('gravity_buoyancy', PHYSICS)]:
        flight = simulate(R0, Motion(velocity, 0.), physics, SETTINGS)
        t_goal = SETTINGS.goal_x / velocity[0]
        acceleration = np.array([0., 0., -physics.effective_gravity])
        times = np.linspace(0., min(flight.time, t_goal), 31)[:, None]
        exact = np.column_stack([R0 + times*velocity + .5*times**2*acceleration,
                                 velocity + times*acceleration])
        error = flight.sample(times.ravel()) - exact
        check(f'{name}/position', np.abs(error[:, :3]).max(), 1e-10, 'm')
        check(f'{name}/velocity', np.abs(error[:, 3:]).max(), 1e-10, 'm/s')
        check(f'{name}/time', abs(flight.time-t_goal), 1e-11, 's')
        check(f'{name}/event', flight.event != 'goal_inside', 0)

    motion = Motion([30., 4., 6.], .2, [7., -5., -40.])
    no_lift = simulate(R0, replace(motion, k_l=0.), PHYSICS, SETTINGS)
    horizontal = no_lift.states[:, 3:5]
    unit = horizontal / np.linalg.norm(horizontal, axis=1)[:, None]
    check('drag_only/horizontal_direction', np.abs(unit-unit[0]).max(), 1e-12)

    for wind in [(0., 0., 0.), (2., 3., 0.)]:
        name = 'calm' if not any(wind) else 'wind'
        physics = replace(PHYSICS, wind=wind)
        flight = simulate(R0, motion, physics, SETTINGS)
        times = np.linspace(0., flight.time, 257)
        state = flight.sample(times)
        v = state[:, 3:]
        u = v - wind
        drag, lift = map(np.asarray, zip(*(aerodynamic_acceleration(t, vv, motion, physics)
                                           for t, vv in zip(times, v))))
        dot = lambda a, b: np.sum(a*b, axis=1)
        check(f'{name}/drag_collinearity', np.linalg.norm(np.cross(u, drag), axis=1).max(),
              1e-11, 'm2/s3')
        check(f'{name}/drag_opposes_flow', not np.all(dot(drag, u) < 0), 0)
        check(f'{name}/magnus_orthogonality', np.abs(dot(lift, u)).max(), 1e-11, 'm2/s3')
        for frame, velocity_in_frame in [('ground', v), ('air', u)]:
            energy = PHYSICS.mass*(.5*dot(velocity_in_frame, velocity_in_frame)
                                   + physics.effective_gravity*state[:, 2])
            power = PHYSICS.mass * dot(drag+lift, velocity_in_frame)
            work = simpson(power, x=times)
            check(f'{name}/{frame}_work_energy', abs(energy[-1]-energy[0]-work), 1e-6, 'J')
            if frame == 'air' or name == 'calm':
                check(f'{name}/{frame}_energy_monotone', not np.all(np.diff(energy) < 0), 0)
        if name == 'wind':
            check('wind/ground_and_air_power_differ',
                  np.allclose(dot(drag+lift, v), dot(drag+lift, u)), 0)

    gravity = replace(PHYSICS, density=0.)
    contact_vz = gravity.gravity / 4
    for name, v0, expected, time in [
        ('high', [22., 0., 9.], 'goal_outside', .5),
        ('wide', [22., 8., 5.], 'goal_outside', .5),
        ('ground', [22., 0., 1.], 'ground_first', 2/gravity.gravity),
        ('at_release', [22., 0., 0.], 'ground_first', 0.),
        ('side_edge', [22., 7.32, 5.], 'goal_outside', .5),
        ('top_edge', [22., 0., (2.44-.11)/.5 + contact_vz], 'goal_outside', .5),
        ('coincident', [22., 0., contact_vz], 'ground_first', .5),
        ('timeout', [1., 0., 20.], 'timeout', 2.),
    ]:
        flight = simulate(R0, Motion(v0, 0.), gravity, SETTINGS)
        check(f'events/{name}/type', flight.event != expected, 0)
        check(f'events/{name}/time', abs(flight.time-time), 1e-10, 's')

    refined = replace(SETTINGS, **CONFIG['refined_integration'])
    limits = CONFIG['convergence_limits']
    for case_id, case in case_map.items():
        if case['cd'] is None:
            continue
        physics = replace(PHYSICS, wind=tuple(case.get('wind_m_s', (0, 0, 0))))
        motion = Motion(case['v0_m_s'], case['cd'], case['omega_xyz_rad_s'],
                        case['k_l'], case['tau_s'] or inf)
        nominal = simulate(R0, motion, physics, SETTINGS)
        fine = simulate(R0, motion, physics, refined)
        check(f'{case_id}/refined_event', nominal.event != fine.event, 0)
        check(f'{case_id}/refined_time', abs(nominal.time-fine.time), limits['time_s'], 's')
        check(f'{case_id}/refined_position',
              np.linalg.norm(nominal.states[-1, 1:3]-fine.states[-1, 1:3]), limits['goal_m'], 'm')
        stored = case['goal']
        check(f'{case_id}/saved_position',
              np.linalg.norm(nominal.states[-1, 1:3]-[stored['y_m'], stored['z_m']]),
              limits['goal_m'], 'm')
        if case['tau_s'] is None:
            params = np.array([*case['v0_m_s'], case['cd'], *case['omega_basis_rad_s'][1:]])
            basis = part5_solver.basis_batch(OBS[None])
            rk4 = part5_solver.predict_batch(params[None], basis, physics.wind)
            check(f'{case_id}/rk4_event', rk4['event'][0] != nominal.event, 0)
            check(f'{case_id}/rk4_time', abs(rk4['goal'][0, 0]-nominal.time),
                  limits['time_s'], 's')
            check(f'{case_id}/rk4_position', np.linalg.norm(rk4['goal'][0, 1:]-nominal.states[-1, 1:3]),
                  limits['goal_m'], 'm')
    return checks


def build_grid():
    return list(product(*(SYNTHETIC[k] for k in ('speeds_m_s', 'elevation_deg',
                                                 'azimuth_deg', 'vertical_spin_rad_s'))))


def rhs_flight(rhs, v0, *, settings=TRUTH_SETTINGS):
    """Dense event-terminated flight for the prescribed wake and axis experiments."""
    def goal(_t, y):
        return y[0] - settings.goal_x

    def ground(t, y):
        return np.spacing(R0[2]) if t == 0 else y[2] - R0[2]

    goal.direction, ground.direction = 1, -1
    goal.terminal = ground.terminal = True
    solution = solve_ivp(rhs, (0, settings.t_max), np.r_[R0, v0],
                         method='DOP853', rtol=settings.rtol, atol=settings.atol,
                         max_step=settings.max_step, events=(goal, ground), dense_output=True)
    time = float(solution.t[-1])
    state = solution.y[:, -1]
    event = 'timeout'
    if not solution.success:
        event = 'numerical_failure'
    elif len(solution.t_events[1]) or (len(solution.t_events[0]) and
                                      state[2] <= R0[2]+1e-12 and state[5] < 0):
        event = 'ground_first'
    elif len(solution.t_events[0]):
        event = 'goal_inside' if inside_goal(*state[1:3], settings) else 'goal_outside'
    return Flight(event, time, solution.y.T, solution.t, solution.message, solution.sol)


def endpoint(flight):
    goal = None
    if flight.event in GOAL_EVENTS:
        goal = [flight.time, *flight.states[-1, 1:3]]
    return dict(event=flight.event, goal=goal, stop_time=flight.time)


def truth_flight(launch, options):
    motion = launch_motion(*launch, cd=options.get('cd'), tau=options.get('tau'))
    physics = replace(PHYSICS, wind=options.get('wind', PHYSICS.wind))
    if 'wake' in options:
        return rhs_flight(wake_rhs([*motion.v0, motion.cd], physics, options['wake']), motion.v0)
    if 'axis' in options:
        parameters = np.r_[motion.v0, motion.cd, 0., motion.omega0[2]]
        rhs = part5_solver.make_rhs(parameters, np.eye(3), physics=physics,
                                    axis=options.get('axis'),
                                    rate=options.get('rate', 0.))
        return rhs_flight(rhs, motion.v0)
    return simulate(R0, motion, physics, TRUTH_SETTINGS)


def fit_drag_or_fixed_spin(positions, *, fit_spin=False, cd=None):
    options = dict(fit_spin=fit_spin, cd=cd)
    fit = fit_from_positions(TIMES, positions, PHYSICS, SETTINGS, **options)
    expanded = fit.best is None or np.any(fit.best.active_mask)
    if expanded:
        fit = fit_from_positions(TIMES, positions, PHYSICS, SETTINGS, **options,
                                 expanded=PARAMETERS,
                                 warm_start=fit.best.motion if fit.best else None)
    if fit.best is None:
        return dict(event='fit_failure', goal=None, stop_time=None)
    motion = fit.best.motion
    flight = simulate(R0, motion, PHYSICS, SETTINGS)
    return dict(**endpoint(flight), v0=motion.v0, cd=motion.cd,
                omega=motion.omega0, residual=fit.best.residuals,
                expanded=bool(expanded), active_bound=bool(np.any(fit.best.active_mask)))


def fit_known_wake(positions, wake, *, settings=SETTINGS):
    # Known waveform is an oracle comparator. The initial velocity and drag are still fitted.
    velocity = baseline_initials(TIMES, positions, PHYSICS.gravity)[0]
    low, high = part5_solver.LOW[:4], part5_solver.HIGH[:4]
    scales = part5_solver.SCALES[:4]

    def make_rhs(p):
        return wake_rhs(p, PHYSICS, wake)

    def residual(scaled):
        p = scaled * scales
        sol = solve_ivp(make_rhs(p), (0., TIMES[-1]), np.r_[R0, p[:3]],
                        t_eval=TIMES[1:], method='DOP853', rtol=settings.rtol,
                        atol=settings.atol, max_step=settings.max_step)
        if not sol.success:
            raise FloatingPointError(sol.message)
        return (sol.y[:3].T-positions[1:]).ravel()

    try:
        fit = least_squares(residual, np.clip(np.r_[velocity, .2], low, high)/scales,
                            bounds=(low/scales, high/scales),
                            jac='3-point', diff_step=1e-4,
                            ftol=1e-12, xtol=1e-12, gtol=1e-13)
    except FloatingPointError:
        return dict(event='fit_failure', goal=None, stop_time=None)
    p = fit.x * scales
    flight = rhs_flight(make_rhs(p), p[:3], settings=settings)
    if not fit.success or flight.time <= TIMES[-1]:
        return dict(event='fit_failure', goal=None, stop_time=None)
    return dict(**endpoint(flight), v0=p[:3], cd=p[3], omega=np.zeros(3),
                residual=fit.fun.reshape(2, 3), active_bound=bool(np.any(fit.active_mask)))


def predict_observations(observations, model, *, wake=None):
    times = np.broadcast_to(TIMES, (len(observations), 3))
    if model in ('B-G', 'B-CA'):
        fitted = part5_solver.baseline(observations, times, model)
        output = []
        for v0, a, residual in zip(fitted['velocity'], fitted['acceleration'], fitted['residual']):
            if v0[2] <= 0:
                prediction = dict(event='ground_first', goal=None, stop_time=0.)
            else:
                prediction = endpoint(baseline_flight(R0, v0, a, PHYSICS, SETTINGS))
            output.append(dict(**prediction, v0=v0, acceleration=a, residual=residual))
        return output
    if model == 'M-D':
        return [fit_drag_or_fixed_spin(positions) for positions in observations]
    if model == 'M-DM fixed drag':
        return [fit_drag_or_fixed_spin(positions, fit_spin=True, cd=SYNTHETIC['cd'])
                for positions in observations]
    if model == 'M-D + known wake':
        return [fit_known_wake(positions, wake) for positions in observations]
    if model != 'M-DM':
        raise ValueError(model)
    fit = part5_solver.fit_batch(observations, times)
    predicted = part5_solver.predict_batch(fit['parameters'], fit['basis'])
    output = []
    for i, (event, goal, time) in enumerate(zip(predicted['event'], predicted['goal'],
                                               predicted['stop_time'])):
        if not fit['success'][i] or (event in (*GOAL_EVENTS, 'ground_first') and time <= TIMES[-1]):
            output.append(dict(event='fit_failure', goal=None, stop_time=None))
        else:
            v0, cd, spin = fit['parameters'][i, :3], fit['parameters'][i, 3], fit['parameters'][i, 4:]
            output.append(dict(event=event, goal=goal if event in GOAL_EVENTS else None,
                               stop_time=time, v0=v0, cd=cd, omega=fit['basis'][i] @ np.r_[0., spin],
                               residual=fit['residual'][i], expanded=bool(fit['expanded'][i]),
                               active_bound=bool(fit['active'][i])))
    return output


def build_prediction_row(case_id, scenario, model, sigma, rep, truth, prediction, positions):
    row = dict(case_id=case_id, scenario=scenario, model=model, sigma_m=sigma, replicate=rep,
               obs_eligible=positions is not None, truth_event=truth['event'],
               prediction_event=prediction['event'], truth_stop_s=truth['stop_time'],
               prediction_stop_s=prediction['stop_time'])
    for kind, outcome in [('truth', truth), ('prediction', prediction)]:
        if outcome['goal'] is not None:
            row.update({f'{kind}_{key}': value
                        for key, value in zip(('t_g_s', 'y_g_m', 'z_g_m'), outcome['goal'])})
    if truth['goal'] is not None and prediction['goal'] is not None:
        dt, dy, dz = np.asarray(prediction['goal'])-truth['goal']
        row.update(error_m=float(np.hypot(dy, dz)), time_error_s=dt, y_error_m=dy, z_error_m=dz)
    if positions is not None:
        for index, pos in enumerate(positions[1:], start=1):
            row.update({f'obs_{axis}{index}_m': val for axis, val in zip(COORDINATES, pos)})
    for name, unit in [('v0', 'm_s'), ('omega', 'rad_s'), ('acceleration', 'm_s2')]:
        if name in prediction:
            row.update({f'{name}_{axis}_{unit}': val
                        for axis, val in zip(COORDINATES, prediction[name])})
    for key in ('cd', 'expanded', 'active_bound'):
        if key in prediction:
            row[key] = prediction[key]
    if 'residual' in prediction:
        residual = prediction['residual']
        row['residual_rms_m'] = float(np.sqrt(np.mean(residual**2)))
        for index, error in enumerate(residual, start=1):
            row.update({f'residual_{axis}{index}_m': val for axis, val in zip(COORDINATES, error)})
    return row


def error_stats(values):
    if not len(values):
        return None
    return dict(median=float(np.median(np.abs(values))),
                p90=float(np.quantile(np.abs(values), .90)),
                rmse=float(np.sqrt(np.mean(np.square(values)))),
                bias=float(np.mean(values)))


def event_metrics(rows):
    eligible = [row for row in rows if row['obs_eligible']]
    paired = [row for row in eligible if row.get('error_m') is not None]
    confusion = {truth: dict.fromkeys(EVENTS, 0) for truth in EVENTS}
    for row in rows:
        confusion[row['truth_event']][row['prediction_event']] += 1
    correct = sum(row['truth_event'] == row['prediction_event'] for row in eligible)
    return dict(n=len(rows), n_eligible=len(eligible), n_ineligible=len(rows)-len(eligible),
                both_cross=len(paired), both_cross_fraction=len(paired)/len(eligible) if eligible else None,
                correct_events=correct, event_accuracy=correct/len(eligible) if eligible else None,
                truth_events={k: sum(row['truth_event'] == k for row in eligible) for k in EVENTS},
                prediction_events={k: sum(row['prediction_event'] == k for row in eligible) for k in EVENTS},
                confusion=confusion, position_m=error_stats([row['error_m'] for row in paired]),
                arrival_s=error_stats([row['time_error_s'] for row in paired]),
                y_m=error_stats([row.get('y_error_m', 0.) for row in paired]),
                z_m=error_stats([row.get('z_error_m', 0.) for row in paired]))


def run_case(task):
    case_id, launch, scenario, options, models, noise, repetitions = task
    flight = truth_flight(launch, options)
    truth = endpoint(flight)
    eligible = flight.time > TIMES[-1] and flight.event != 'numerical_failure'
    parameters = launch_motion(*launch, cd=options.get('cd'), tau=options.get('tau'))
    record = dict(case_id=case_id, scenario=scenario, speed_m_s=launch[0], elevation_deg=launch[1],
                  azimuth_deg=launch[2], spin_z_rad_s=launch[3], cd=parameters.cd,
                  wind_m_s=options.get('wind', PHYSICS.wind), tau_s=options.get('tau'),
                  wake_ball=options['wake'].ball if 'wake' in options else None,
                  wake_trace=options['wake'].index if 'wake' in options else None,
                  axis=options.get('axis'), axis_rate_rad_s=options.get('rate'),
                  obs_eligible=eligible, **truth)
    sample_times = np.unique(np.r_[np.arange(0., flight.time, .005),
                                   TIMES[TIMES <= flight.time], flight.time])
    samples = flight.sample(sample_times)
    trajectories = [dict(case_id=case_id, scenario=scenario, time_s=t, x_m=x, y_m=y, z_m=z,
                         vx_m_s=vx, vy_m_s=vy, vz_m_s=vz)
                    for t, (x, y, z, vx, vy, vz) in zip(sample_times, samples)]
    rows = []
    # The standard draws are common across noise levels and all models.
    rng = np.random.default_rng(np.random.SeedSequence([CONFIG['seed'], 2, case_id]))
    standard = rng.standard_normal((repetitions, 2, 3))
    if eligible:
        clean = flight.sample(TIMES)[:, :3]
    for sigma in noise:
        n = 1 if sigma == 0 else repetitions
        if eligible:
            inputs = np.repeat(clean[None], n, axis=0)
            inputs[:, 1:] += sigma*standard[:n]
        else:
            inputs = [None]*n
        for model in models:
            predictions = (predict_observations(inputs, model, wake=options.get('wake')) if eligible
                           else [dict(event='not_observed', goal=None, stop_time=None)]*n)
            rows.extend(build_prediction_row(case_id, scenario, model, sigma, rep,
                                             truth, prediction, positions)
                        for rep, (prediction, positions) in enumerate(zip(predictions, inputs)))
    return record, trajectories, rows


def synthetic_tasks(*, mismatch=False, pilot=False):
    grid = build_grid()
    indexed = list(enumerate(grid))
    if pilot:
        indexed = [(i, grid[i]) for i in (0, 39, 80)]
    if not mismatch:
        return [(i, launch, 'reference', {}, SYNTHETIC['models'], SYNTHETIC['noise_sigma_m'],
                 1 if pilot else SYNTHETIC['noisy_repetitions']) for i, launch in indexed]
    scenarios = [
        ('sidewind', dict(wind=tuple(MISMATCH['crosswind_m_s'])), ['M-DM']),
        ('higher_drag', dict(cd=MISMATCH['cd']), ['M-DM', 'M-DM fixed drag']),
        ('spin_decay', dict(tau=MISMATCH['tau_s']), ['M-DM']),
        ('axis_z_control', dict(axis=(0., 0., 1.), rate=np.deg2rad(MISMATCH['axis_rate_deg_s'])), ['M-DM']),
        ('axis_y', dict(axis=(0., 1., 0.), rate=np.deg2rad(MISMATCH['axis_rate_deg_s'])), ['M-DM']),
    ]
    tasks = [(i, launch, scenario, options, models, [0.], 1) for i, launch in indexed
             for scenario, options, models in scenarios]
    if PSD_PATH.exists():
        traces, _ = generate_traces(VALIDATION_DRAWS_PER_SPECTRUM, VALIDATION_SEED)
        for trace in traces:
            tasks.extend((i, launch, f'wake/{trace.ball}/{trace.index}',
                          dict(wake=trace), ['M-D', 'M-D + known wake'], [0.], 1)
                         for i, launch in indexed if launch[-1] == 0)
    return tasks


def run_experiment(which, jobs, *, pilot=False):
    target = DATA / ('pilot' if pilot else '')
    target.mkdir(exist_ok=True)
    tasks = synthetic_tasks(mismatch=which == 'V3', pilot=pilot)
    truths, trajectories, rows = [], [], []
    print(f'{which}: {len(tasks)} truth trajectories', flush=True)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for index, (truth, trajectory, predictions) in enumerate(pool.map(run_case, tasks), start=1):
            truths.append(truth)
            trajectories.extend(trajectory)
            rows.extend(predictions)
            if index % 20 == 0:
                print(f'{which}: {index}/{len(tasks)} truth trajectories processed', flush=True)
    prefix = which.lower()
    write_csv(target / f'{prefix}_truth.csv', truths)
    write_csv(target / f'{prefix}_trajectories.csv', trajectories)
    write_csv(target / f'{prefix}_predictions.csv', rows)
    groups = defaultdict(list)
    for row in rows:
        name = row['scenario'] if not row['scenario'].startswith('wake/') else 'wake'
        groups[f'{name}/{row["model"]}/{row["sigma_m"]:g}'].append(row)
    return {key: event_metrics(group) for key, group in groups.items()}
