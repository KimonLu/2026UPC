"""Reproduce the prescribed-wind fits and the fixed-release controls."""

import argparse
import csv
import json
import platform
from dataclasses import replace
from math import cos, radians, sin

import numpy as np
import scipy

from scripts.part03.run_part3 import attempt_record, case_record, flat_case, goal_record, trajectory, write_csv
from scripts.common.still_air import Motion, Physics, Settings, aerodynamic_acceleration, fit_from_positions, simulate

from scripts.common.paths import data_dir, figures_dir

DATA = data_dir(4)


def evaluate_case(experiment, name, times, positions, motion, physics, settings, basis):
    flight = simulate(positions[0], motion, physics, settings)
    observed = times[1:] <= flight.time
    residuals = np.full((2, 3), np.nan)
    residuals[observed] = flight.sample(times[1:][observed])[:, :3] - positions[1:][observed]
    samples = trajectory(flight, motion, times, physics)
    record = case_record(f'{experiment}/{name}', motion, flight, residuals, basis,
                         samples, aerodynamic=True)
    if not np.all(observed):
        record['residuals_m'] = [row.tolist() if available else [None] * 3
                                 for row, available in zip(residuals, observed)]
        record['rms_m'] = record['max_residual_m'] = None

    force_checks = {}
    for label, time in (('t1', times[1]), ('t2', times[2]), ('event', flight.time)):
        if time <= flight.time:
            velocity = flight.sample(time)[0, 3:]
            drag, magnus = aerodynamic_acceleration(time, velocity, motion, physics)
            force_checks[label] = dict(
                time_s=float(time), air_speed_m_s=float(np.linalg.norm(velocity - physics.wind)),
                drag_acceleration_m_s2=drag.tolist(), magnus_acceleration_m_s2=magnus.tolist(),
            )
    record.update(experiment=experiment, wind_id=name, wind_m_s=list(physics.wind),
                  integration_message=flight.message, aerodynamic_samples=force_checks)
    return record, samples


def fit_wind(times, positions, physics, settings, fit_options, random_starts):
    original = fit_from_positions(times, positions, physics, settings,
                                  random_starts=random_starts, **fit_options)
    if original.best is None:
        raise RuntimeError(f'No admissible fit for wind {physics.wind}.')
    selected = original
    parameters = original.best.parameters
    at_edge = (np.isclose(parameters, original.lower, atol=1e-6, rtol=0)
               | np.isclose(parameters, original.upper, atol=1e-6, rtol=0))
    expanded = tuple(name for name, edge in zip(original.parameter_names, at_edge) if edge)
    wider = None
    if expanded:
        wider = fit_from_positions(times, positions, physics, settings, expanded=expanded,
                                   warm_start=original.best.motion, random_starts=random_starts,
                                   **fit_options)
        if wider.best is not None and wider.best.sse < original.best.sse:
            selected = wider
    return selected, dict(
        names=original.parameter_names, scales=original.scales.tolist(),
        original_lower=original.lower.tolist(), original_upper=original.upper.tolist(),
        attempts=[attempt_record(item) for item in original.attempts],
        expanded_parameters=expanded, expanded_used=selected is not original,
        expanded_attempts=[] if wider is None else [attempt_record(item) for item in wider.attempts],
        selected_attempt=next(i for i, item in enumerate(selected.attempts) if item is selected.best),
    )


def check_convergence(record, refined, limits):
    if record['event'] != refined['event'] or record['event'] == 'numerical_failure':
        raise AssertionError(f"{record['case']}: inconsistent or failed event integration.")
    time_delta = abs(record['event_time_s'] - refined['event_time_s'])
    goal_delta = None
    if record['goal'] is not None:
        goal_delta = float(np.hypot(record['goal']['y_m'] - refined['goal']['y_m'],
                                    record['goal']['z_m'] - refined['goal']['z_m']))
    if (time_delta >= limits['event_time_s']
            or (goal_delta is not None and goal_delta >= limits['goal_displacement_m'])):
        raise AssertionError(f"{record['case']}: convergence limits exceeded.")
    return dict(event=refined['event'], time_delta_s=time_delta,
                goal_delta_m=goal_delta, refined_goal=refined['goal'])


def run():
    config = json.loads((DATA / 'config.json').read_text(encoding='utf-8'))
    base_path = DATA / config['base_config']
    base = json.loads(base_path.read_text(encoding='utf-8'))
    times, positions = np.array(base['times_s']), np.array(base['positions_m'])
    physics = Physics(**base['physics'])
    settings = Settings(**base['integration'])
    tight = replace(settings, **config['refined_integration'])
    fit_options = dict(k_l=base['k_l'], seed=base['seed'])
    cases, convergence, trajectories = {}, {}, {}

    for name, wind in config['winds'].items():
        print(f'Fitting {name} {tuple(wind)} m/s', flush=True)
        current = replace(physics, wind=tuple(wind))
        fit, audit = fit_wind(times, positions, current, settings, fit_options, base['random_starts'])
        record, samples = evaluate_case('W1', name, times, positions, fit.best.motion,
                                        current, settings, fit.basis)
        expanded = audit['expanded_parameters'] if audit['expanded_used'] else ()
        refined_fit = fit_from_positions(times, positions, current, tight, **fit_options,
                                         warm_start=fit.best.motion, expanded=expanded)
        if refined_fit.best is None:
            raise RuntimeError(f'No admissible refined fit for wind {wind}.')
        refined, _ = evaluate_case('W1', name, times, positions, refined_fit.best.motion,
                                    current, tight, fit.basis)
        record['fit'] = audit
        cases[record['case']], trajectories[record['case']] = record, samples
        convergence[record['case']] = check_convergence(record, refined, config['convergence_limits'])
        if name == 'calm':
            calm_motion, calm_basis = fit.best.motion, fit.basis
            refined_calm_motion = refined_fit.best.motion

    for name, wind in config['winds'].items():
        current = replace(physics, wind=tuple(wind))
        record, samples = evaluate_case('W2', name, times, positions, calm_motion,
                                        current, settings, calm_basis)
        refined, _ = evaluate_case('W2', name, times, positions, refined_calm_motion,
                                    current, tight, calm_basis)
        cases[record['case']], trajectories[record['case']] = record, samples
        convergence[record['case']] = check_convergence(record, refined, config['convergence_limits'])

    reference = json.loads(base_path.with_name('results.json').read_text(encoding='utf-8'))['cases']['S3']
    calm = cases['W1/calm']
    for case in cases.values():
        if case['wind_id'] == 'calm':
            assert case['event'] == reference['event']
            for key in ('v0_m_s', 'cd', 'omega_xyz_rad_s'):
                np.testing.assert_allclose(case[key], reference[key], atol=1e-7, rtol=0)
            np.testing.assert_allclose(list(case['goal'].values()), list(reference['goal'].values()),
                                       atol=1e-8, rtol=0)
        if case['experiment'] == 'W2':
            for key in ('v0_m_s', 'cd', 'omega_xyz_rad_s', 'k_l', 'tau_s'):
                assert case[key] == calm[key]
        case['delta_goal'] = None if case['goal'] is None else {
            key: value - calm['goal'][key] for key, value in case['goal'].items()
        }
    assert len(cases) == 20 and config['part5_reference'] in cases

    export = dict(config=config, part3_config=base, spin_basis_columns=calm_basis.tolist(),
                  runtime=dict(python=platform.python_version(), numpy=np.__version__,
                               scipy=scipy.__version__), cases=cases, convergence=convergence)
    (DATA / 'results.json').write_text(json.dumps(export, indent=2, allow_nan=False) + '\n',
                                       encoding='utf-8')
    write_csv(DATA / 'cases.csv', [
        dict(experiment=case['experiment'], wind_id=case['wind_id'],
             **{f'w{axis}_m_s': value for axis, value in zip('xyz', case['wind_m_s'])},
             **flat_case(case),
             **{f'delta_{key}': (case['delta_goal'] or {}).get(key)
                for key in ('t_s', 'y_m', 'z_m')})
        for case in cases.values()
    ])
    write_csv(DATA / 'trajectories.csv', [
        dict(case=key, time_s=t, x_m=state[0], y_m=state[1], z_m=state[2],
             vx_m_s=state[3], vy_m_s=state[4], vz_m_s=state[5], reynolds=re,
             spin_rad_s=spin, spin_perp=spin_perp)
        for key, (ts, states, res, spins, sps) in trajectories.items()
        for t, state, re, spin, spin_perp in zip(ts, states, res, spins, sps)
    ])
    print(f'Exported {len(cases)} trajectories and convergence records.', flush=True)


def run_wind_contours():
    """Refit the observed positions along constant-speed rings and pure-axis sweeps."""
    config = json.loads((DATA / 'config.json').read_text(encoding='utf-8'))
    base = json.loads((DATA / config['base_config']).read_text(encoding='utf-8'))
    times, positions = np.array(base['times_s']), np.array(base['positions_m'])
    physics, settings = Physics(**base['physics']), Settings(**base['integration'])
    fit_options = dict(k_l=base['k_l'], seed=base['seed'])
    ring_speeds = (2, 4, 6, 8)
    cardinal = {0: (1., 0.), 90: (0., 1.), 180: (-1., 0.), 270: (0., -1.)}
    samples = {(speed, angle) for speed in ring_speeds for angle in range(0, 360, 5)}
    samples.update((step / 2, angle) for step in range(1, 17) for angle in cardinal)

    rows = []
    for speed, angle in sorted(samples):
        theta = radians(angle)
        dx, dy = cardinal.get(angle, (cos(theta), sin(theta)))
        current = replace(physics, wind=(speed * dx, speed * dy, 0.))
        fit, audit = fit_wind(times, positions, current, settings, fit_options, random_starts=0)
        flight = simulate(positions[0], fit.best.motion, current, settings)
        goal = goal_record(flight)
        residual = float(np.max(np.abs(fit.best.residuals)))
        if goal is None or residual > 1e-8:
            raise RuntimeError(f'No interpolating goal-plane prediction for wind {current.wind}.')
        omega2, omega3 = (fit.basis.T @ fit.best.motion.omega0)[1:]
        rows.append(dict(
            speed_m_s=speed, angle_deg=angle, on_ring=int(speed in ring_speeds),
            wx_m_s=current.wind[0], wy_m_s=current.wind[1], cd=fit.best.motion.cd,
            **{f'v0{axis}_m_s': value for axis, value in zip('xyz', fit.best.motion.v0)},
            omega2_rad_s=omega2, omega3_rad_s=omega3, rms_m=np.sqrt(fit.best.sse / 6),
            expanded_parameters=' '.join(audit['expanded_parameters']) if audit['expanded_used'] else '',
            max_residual_m=residual, event=flight.event, **goal,
        ))

    write_csv(DATA / 'wind_contours.csv', rows)
    print(f'Exported {len(rows)} refitted winds for Figure 14.', flush=True)


def fixed_release_model():
    result = json.loads((DATA / 'results.json').read_text(encoding='utf-8'))
    base = result['part3_config']
    physics, settings = Physics(**base['physics']), Settings(**base['integration'])
    calm = result['cases']['W1/calm']
    motion = Motion(calm['v0_m_s'], calm['cd'], calm['omega_xyz_rad_s'],
                    k_l=calm['k_l'], tau=calm['tau_s'] if calm['tau_s'] is not None else np.inf)
    return base['positions_m'][0], motion, physics, settings, calm['goal']


def run_fixed_release_contours():
    """Evaluate the same wind samples with the calm release and drag held fixed."""
    release, motion, physics, settings, calm_goal = fixed_release_model()
    rows = []
    with (DATA / 'wind_contours.csv').open(encoding='utf-8', newline='') as source:
        for sample in csv.DictReader(source):
            wind = (float(sample['wx_m_s']), float(sample['wy_m_s']), 0.)
            flight = simulate(release, motion, replace(physics, wind=wind), settings)
            goal = goal_record(flight)
            if goal is None:
                raise RuntimeError(f'No fixed-release goal-plane crossing for wind {wind}: {flight.event}.')
            rows.append(dict(
                speed_m_s=sample['speed_m_s'], angle_deg=sample['angle_deg'], on_ring=sample['on_ring'],
                wx_m_s=wind[0], wy_m_s=wind[1], event=flight.event, **goal,
                **{f'delta_{key}': value - calm_goal[key] for key, value in goal.items()},
            ))
    write_csv(DATA / 'fixed_release_contours.csv', rows)
    print(f'Exported {len(rows)} fixed-release wind predictions.', flush=True)


def run_fixed_release_coverage():
    """Map a polar wind grid to the goal plane with the fixed calm release."""
    release, motion, physics, settings, _ = fixed_release_model()
    winds = [(0.0, 0)]
    winds.extend((step / 2, angle) for step in range(1, 17) for angle in range(0, 360, 5))
    rows = []
    for speed, angle in winds:
        theta = radians(angle)
        wind = (speed * cos(theta), speed * sin(theta), 0.0)
        flight = simulate(release, motion, replace(physics, wind=wind), settings)
        goal = goal_record(flight)
        if goal is None:
            raise RuntimeError(f'No fixed-release goal-plane crossing for wind {wind}: {flight.event}.')
        rows.append(dict(speed_m_s=speed, angle_deg=angle, wx_m_s=wind[0],
                         wy_m_s=wind[1], event=flight.event, **goal))
    write_csv(DATA / 'fixed_release_coverage.csv', rows)
    print(f'Exported {len(rows)} goal-plane coverage samples.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--no-plots', action='store_true')
    mode.add_argument('--plots-only', action='store_true')
    mode.add_argument('--contours-only', action='store_true', help='Recompute and draw both contour figures.')
    mode.add_argument('--fixed-contours-only', action='store_true',
                      help='Recompute and draw only the fixed-release contours.')
    mode.add_argument('--paper-figures', action='store_true',
                      help='Generate the three revised Section 4 figures from the saved calm fit.')
    args = parser.parse_args()
    if args.paper_figures:
        from scripts.part04.plot_part4 import make_paper_figures

        make_paper_figures(DATA, figures_dir(4))
        raise SystemExit(0)
    if not (args.plots_only or args.contours_only or args.fixed_contours_only):
        run()
    if not (args.plots_only or args.fixed_contours_only) or not (DATA / 'wind_contours.csv').is_file():
        run_wind_contours()
    if not args.plots_only or not (DATA / 'fixed_release_contours.csv').is_file():
        run_fixed_release_contours()
    if not (args.plots_only or args.contours_only or args.fixed_contours_only):
        run_fixed_release_coverage()
    if not args.no_plots:
        from scripts.part04.plot_part4 import make_figures

        make_figures(DATA, figures_dir(4), crossings_only=args.contours_only,
                     fixed_contours_only=args.fixed_contours_only)
