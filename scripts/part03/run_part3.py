"""Reproduce the fitted cases, sensitivity scans, tables and Section 3 figures."""

import argparse
import csv
import json
import platform
from dataclasses import replace
from math import inf

import numpy as np
import scipy

from scripts.common.still_air import (
    Flight, Motion, Physics, Settings, baseline_initials, fit_from_positions,
    inside_goal, position_residuals, simulate, spin_at,
)

from scripts.common.paths import data_dir, figures_dir

DATA = data_dir(3)
FIGURES = figures_dir(3)
GOAL_EVENTS = ('goal_inside', 'goal_outside')


def baseline_flight(r0, velocity, acceleration, physics, settings):
    """Analytical quadratic path with the same flight-ending events."""
    def sample(t):
        t = np.atleast_1d(t)[:, None]
        return np.column_stack((r0 + t * (velocity + t * acceleration / 2),
                                velocity + t * acceleration)).T

    def event_time(axis, target, direction):
        roots = np.roots([acceleration[axis] / 2, velocity[axis], r0[axis] - target])
        return min((root.real for root in roots
                    if abs(root.imag) < 1e-12 and root.real > 0
                    and direction * (velocity[axis] + acceleration[axis] * root.real) > 0),
                   default=inf)

    goal = event_time(0, settings.goal_x, 1)
    ground = event_time(2, physics.radius, -1)
    end = min(goal, ground, settings.t_max)
    state = sample(end)[:, 0]
    if end == ground:
        event = 'ground_first'
    elif end == goal:
        event = 'goal_inside' if inside_goal(*state[1:3], settings) else 'goal_outside'
    else:
        event = 'timeout'
    return Flight(event, float(end), state[None, :], np.array([end]), _interpolant=sample)


def goal_record(flight):
    if flight.event not in GOAL_EVENTS:
        return None
    return dict(t_s=flight.time, y_m=float(flight.states[-1, 1]), z_m=float(flight.states[-1, 2]))


def trajectory(flight, motion, observation_times, physics):
    times = np.unique(np.r_[np.linspace(0., flight.time, 241),
                            observation_times[observation_times <= flight.time]])
    states = flight.sample(times)
    speeds = np.linalg.norm(states[:, 3:] - physics.wind, axis=1)
    reynolds = 2 * physics.density * physics.radius * speeds / physics.viscosity
    spins = np.array([spin_at(time, motion) for time in times])
    spin_rates = np.linalg.norm(spins, axis=1)
    relative_velocity = states[:, 3:] - physics.wind
    perpendicular = np.linalg.norm(np.cross(spins, relative_velocity), axis=1)
    spin_parameters = np.divide(physics.radius * perpendicular, speeds**2,
                                out=np.zeros_like(speeds), where=speeds > 0)
    return times, states, reynolds, spin_rates, spin_parameters


def case_record(case_id, motion, flight, residuals, basis, samples, *, aerodynamic):
    _, _, reynolds, spin_rates, spin_parameters = samples
    return dict(
        case=case_id, v0_m_s=np.asarray(motion.v0).tolist(),
        cd=float(motion.cd) if aerodynamic else None,
        k_l=motion.k_l if aerodynamic else None,
        tau_s=motion.tau if np.isfinite(motion.tau) else None,
        omega_xyz_rad_s=np.asarray(motion.omega0).tolist(),
        omega_basis_rad_s=(basis.T @ motion.omega0).tolist(),
        spin_start_rad_s=float(spin_rates[0]),
        spin_goal_rad_s=float(spin_rates[-1]) if flight.event in GOAL_EVENTS else None,
        residuals_m=residuals.tolist(), rms_m=float(np.sqrt(np.mean(residuals**2))),
        max_residual_m=float(np.max(np.abs(residuals))),
        event=flight.event, event_time_s=flight.time, event_state=flight.states[-1].tolist(),
        goal=goal_record(flight),
        reynolds_range=[float(reynolds.min()), float(reynolds.max())] if aerodynamic else None,
        spin_parameter_range=[float(spin_parameters.min()), float(spin_parameters.max())] if aerodynamic else None,
    )


def attempt_record(attempt):
    return dict(initial=attempt.initial.tolist(), fitted=attempt.parameters.tolist(),
                success=bool(attempt.success), status=int(attempt.status), message=attempt.message,
                evaluations=attempt.evaluations, active_mask=attempt.active_mask.tolist(),
                residuals_m=attempt.residuals.tolist() if np.isfinite(attempt.sse) else None,
                sse_m2=attempt.sse if np.isfinite(attempt.sse) else None)


def scaled_svd(fit, times, positions, physics, settings):
    motion = fit.best.motion

    def derivative(step):
        columns = []
        for index, scale in enumerate(fit.scales):
            delta = step * scale
            if index < 3:
                shift = np.eye(3)[index] * delta
                plus = replace(motion, v0=motion.v0 + shift)
                minus = replace(motion, v0=motion.v0 - shift)
            elif index == 3:
                plus = replace(motion, cd=motion.cd + delta)
                minus = replace(motion, cd=motion.cd - delta)
            else:
                shift = fit.basis[:, index - 3] * delta
                plus = replace(motion, omega0=motion.omega0 + shift)
                minus = replace(motion, omega0=motion.omega0 - shift)
            plus_r = position_residuals(times, positions, plus, physics, settings)
            minus_r = position_residuals(times, positions, minus, physics, settings)
            columns.append(((plus_r - minus_r) / (2 * step)).ravel())
        return np.column_stack(columns)

    step = 1e-4
    jacobian = derivative(step)
    _, singular, right = np.linalg.svd(jacobian, full_matrices=False)
    check = np.linalg.svd(derivative(step / 10), compute_uv=False)
    np.testing.assert_allclose(singular, check, rtol=1e-5, atol=1e-9)
    return dict(parameter_names=fit.parameter_names, scales=fit.scales.tolist(),
                jacobian_m=jacobian.tolist(), singular_values_m=singular.tolist(),
                condition_number=float(singular[0] / singular[-1]),
                weakest_scaled_direction=right[-1].tolist(), step_scaled=step,
                check_singular_values_m=check.tolist())


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flat_case(record):
    goal = record['goal'] or dict(t_s=None, y_m=None, z_m=None)
    return {
        'case': record['case'],
        'event': record['event'],
        'event_time_s': record['event_time_s'],
        'v0x_m_s': record['v0_m_s'][0],
        'v0y_m_s': record['v0_m_s'][1],
        'v0z_m_s': record['v0_m_s'][2],
        'cd': record['cd'],
        'tau_s': record['tau_s'],
        'omega1_rad_s': record['omega_basis_rad_s'][0],
        'omega2_rad_s': record['omega_basis_rad_s'][1],
        'omega3_rad_s': record['omega_basis_rad_s'][2],
        'spin_start_rad_s': record['spin_start_rad_s'],
        'spin_goal_rad_s': record['spin_goal_rad_s'],
        **{f'residual_{time}_{axis}_m': record['residuals_m'][i][j]
           for i, time in enumerate(('t1', 't2')) for j, axis in enumerate('xyz')},
        'rms_m': record['rms_m'],
        'max_residual_m': record['max_residual_m'],
        't_goal_s': goal['t_s'],
        'y_goal_m': goal['y_m'],
        'z_goal_m': goal['z_m'],
    }


def run():
    config = json.loads((DATA / 'config.json').read_text(encoding='utf-8'))
    times = np.array(config['times_s'])
    positions = np.array(config['positions_m'])
    physics = Physics(**config['physics'])
    settings = Settings(**config['integration'])
    fit_args = dict(k_l=config['k_l'], seed=config['seed'])
    cases = {}
    trajectories = {}
    v_ca, v_g, a_ca = baseline_initials(times, positions, physics.gravity)
    for key, velocity, acceleration in (
        ('B-G', v_g, np.array([0., 0., -physics.gravity])),
        ('B-CA', v_ca, a_ca),
    ):
        motion = Motion(velocity, 0.)
        flight = baseline_flight(positions[0], velocity, acceleration, physics, settings)
        residuals = flight.sample(times[1:])[:, :3] - positions[1:]
        samples = trajectory(flight, motion, times, physics)
        cases[key] = case_record(key, motion, flight, residuals, np.eye(3), samples, aerodynamic=False)
        trajectories[key] = samples

    for key, reference in (('B-G', [0.3706199461, 3.2169811321, 1.95700643]),
                           ('B-CA', [0.4050083612, 2.4948244142, 1.9203670565])):
        np.testing.assert_allclose(list(cases[key]['goal'].values()), reference, rtol=0, atol=1e-8)

    models = {
        'M-D20': dict(fit_spin=False, cd=config['fixed_cd']),
        'M-D': dict(fit_spin=False),
        'M-DM': dict(fit_spin=True),
        **{f'M-DM{tau:g}': dict(fit_spin=True, tau=tau) for tau in config['finite_spin_decay_s']},
    }
    best_fits = {}
    convergence = {}
    tight = replace(settings, rtol=1e-10, atol=1e-12, max_step=settings.max_step / 2)
    for key, options in models.items():
        print(f'Fitting {key}', flush=True)
        original = fit_from_positions(times, positions, physics, settings, **options, **fit_args,
                                      random_starts=config['random_starts'])
        if original.best is None:
            raise RuntimeError(f'No admissible fit for {key}.')
        selected = original
        edge = (np.isclose(original.best.parameters, original.lower, atol=1e-6, rtol=0)
                | np.isclose(original.best.parameters, original.upper, atol=1e-6, rtol=0))
        expanded = tuple(name for name, at_edge in zip(original.parameter_names, edge) if at_edge)
        expanded_fit = None
        if expanded:
            expanded_fit = fit_from_positions(
                times, positions, physics, settings, **options, **fit_args, expanded=expanded,
                random_starts=config['random_starts'], warm_start=original.best.motion,
            )
            if expanded_fit.best is not None and expanded_fit.best.sse < original.best.sse:
                selected = expanded_fit
        best_fits[key] = selected
        best = selected.best
        flight = simulate(positions[0], best.motion, physics, settings)
        samples = trajectory(flight, best.motion, times, physics)
        record = case_record(key, best.motion, flight, best.residuals, selected.basis, samples,
                             aerodynamic=True)
        record['fit'] = dict(
            names=original.parameter_names, original_lower=original.lower.tolist(),
            original_upper=original.upper.tolist(), attempts=[attempt_record(x) for x in original.attempts],
            expanded_parameters=expanded, expanded_used=selected is expanded_fit,
            expanded_attempts=[attempt_record(x) for x in expanded_fit.attempts] if expanded_fit else [],
        )
        cases[key] = record
        trajectories[key] = samples
        refined = fit_from_positions(times, positions, physics, tight, **options, **fit_args,
                                     warm_start=best.motion, expanded=expanded if selected is expanded_fit else ())
        if refined.best is None:
            raise RuntimeError(f'No admissible refined fit for {key}.')
        precise_flight = simulate(positions[0], refined.best.motion, physics, tight)
        if flight.event != precise_flight.event:
            raise AssertionError(f'Event classification changed for {key}.')
        difference = None
        if record['goal'] is not None:
            difference = float(np.linalg.norm(flight.states[-1, 1:3] - precise_flight.states[-1, 1:3]))
            if difference >= 1e-4:
                raise AssertionError(f'Goal-plane discrepancy {difference} m in {key}.')
        convergence[key] = dict(event=flight.event, goal_delta_m=difference,
                                time_delta_s=abs(flight.time - precise_flight.time),
                                refined_goal=goal_record(precise_flight))

    print('Profiling drag and chord-axis spin', flush=True)
    nominal = best_fits['M-DM']
    diagnostics = scaled_svd(nominal, times, positions, physics, settings)
    profiles = []
    for name, config_key in (('cd', 'cd_grid'), ('omega_parallel', 'chord_spin_grid_rad_s')):
        low, high, step = config[config_key]
        grid = np.linspace(low, high, round((high - low) / step) + 1)
        for value in grid:
            fit = fit_from_positions(times, positions, physics, settings, **fit_args,
                                     **{name: value}, warm_start=nominal.best.motion)
            if fit.best is None:
                raise RuntimeError(f'No admissible profile fit for {name}={value:g}.')
            best = fit.best
            flight = simulate(positions[0], best.motion, physics, settings)
            samples = trajectory(flight, best.motion, times, physics)
            record = case_record(name, best.motion, flight, best.residuals, fit.basis, samples,
                                 aerodynamic=True)
            record.update(value=float(value),
                          passes_screen=bool(record['max_residual_m'] <= config['coordinate_screen_m']))
            profiles.append(record)

    export = dict(config=config, runtime=dict(python=platform.python_version(),
                  numpy=np.__version__, scipy=scipy.__version__), cases=cases,
                  spin_basis_columns=nominal.basis.tolist(), svd=diagnostics,
                  profiles=profiles, convergence=convergence)
    (DATA / 'results.json').write_text(json.dumps(export, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    write_csv(DATA / 'cases.csv', [flat_case(record) for record in cases.values()])
    write_csv(DATA / 'profiles.csv', [
        dict(profile=record['case'], value=record['value'], passes_screen=record['passes_screen'],
             **{key: val for key, val in flat_case(record).items() if key != 'case'})
        for record in profiles
    ])
    write_csv(DATA / 'trajectories.csv', [
        dict(case=key, time_s=t, x_m=s[0], y_m=s[1], z_m=s[2], vx_m_s=s[3],
             vy_m_s=s[4], vz_m_s=s[5], reynolds=re, spin_rad_s=spin, spin_perp=sp)
        for key, (ts, states, res, spins, sps) in trajectories.items()
        for t, s, re, spin, sp in zip(ts, states, res, spins, sps)
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--no-plots', action='store_true')
    mode.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    if not args.plots_only:
        run()
    if not args.no_plots:
        from scripts.part03.plot_part3 import make_figures

        make_figures(DATA, FIGURES)
