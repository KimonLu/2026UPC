"""Recompute the Section 7 cases and export states and verification in SI units."""

import argparse
import json
import platform
from dataclasses import asdict, replace
from math import inf

import numpy as np
import scipy

from .part7_bounce import TIE_TOL_S, simulate_bounce
from scripts.common.paths import OUTPUT_ROOT, data_dir
from scripts.part03.run_part3 import write_csv
from scripts.common.still_air import Motion, Physics, Settings, position_residuals, simulate

DATA = data_dir(7)
GOAL_EVENTS = ('goal_inside', 'goal_outside')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def case_id(shot, model, e_n, mu_f=None):
    prefix = f'{shot}/{model}/e{e_n:g}'
    return prefix if mu_f is None else f'{prefix}/mu{mu_f:g}'


def export_impact(impact):
    if impact is None:
        return None
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in asdict(impact).items()}


def export_case(shot, motion, physics, options, fit_record):
    return {
        'event': shot.event,
        'time_s': shot.time,
        'state': shot.state.tolist(),
        'goal': dict(zip(('t_s', 'y_m', 'z_m'), (shot.time, *shot.state[1:3])))
        if shot.event in GOAL_EVENTS else None,
        'tie_gap_s': shot.tie_gap,
        'impact': export_impact(shot.impact),
        'fit': fit_record,
        'launch': {
            'v0_m_s': list(motion.v0), 'omega0_rad_s': list(motion.omega0),
            'cd': motion.cd, 'k_l': motion.k_l, 'density_kg_m3': physics.density,
            'wind_m_s': list(physics.wind), 'tau_s': None if motion.tau == inf else motion.tau,
        },
        **options,
    }


def state_row(case, time, state, omega=None):
    row = {'case': case, 'time_s': time,
           **dict(zip(('x_m', 'y_m', 'z_m', 'vx_m_s', 'vy_m_s', 'vz_m_s'), state))}
    if omega is not None:
        row.update(zip(('omega_x_rad_s', 'omega_y_rad_s', 'omega_z_rad_s'), omega))
    return row


def write_paper_example(config, physics, settings, refined):
    """Export the near-threshold shot used in the paper's 3D figure."""
    example = config['paper_example']
    motion = Motion(example['velocity_m_s'], config['synthetic_cd'], k_l=config['synthetic_k_l'])
    release = [0., 0., physics.radius]
    restitution = config['reference_e_n']
    cases = [(restitution, friction) for friction in config['friction']]
    cases.append((example['contrast_e_n'], example['contrast_mu_f']))
    expected = ['goal_inside', 'goal_inside', 'second_ground', 'second_ground', 'second_ground']
    summary, trajectories = [], []

    for (e_n, mu_f), expected_event in zip(cases, expected, strict=True):
        key = case_id('R1', 'air', e_n, mu_f)
        options = dict(e_n=e_n, mu_f=mu_f)
        coarse = simulate_bounce(release, motion, physics, settings, **options)
        fine = simulate_bounce(release, motion, physics, refined, **options)
        if (coarse.event != fine.event or coarse.impact is None or fine.impact is None
                or len(fine.segments) != 2
                or np.linalg.norm(coarse.state[:3] - fine.state[:3]) >= 1e-6
                or abs(coarse.time - fine.time) >= 1e-7):
            raise RuntimeError(f'Unstable one-bounce result: {key}')
        if fine.event != expected_event:
            raise RuntimeError(f'{key}: expected {expected_event}, got {fine.event}')

        first_x, first_y, first_z = fine.impact.position
        summary.append({
            'case': key, 'e_n': e_n, 'mu_f': mu_f, 'event': fine.event,
            'first_time_s': fine.impact.time, 'first_x_m': first_x,
            'first_y_m': first_y, 'first_z_m': first_z,
            'end_time_s': fine.time, 'end_x_m': fine.state[0],
            'end_y_m': fine.state[1], 'end_z_m': fine.state[2],
            'bottom_clearance_mm': 1000 * (fine.state[2] - physics.radius),
        })
        for leg, segment in enumerate(fine.segments, 1):
            trajectories.extend({**state_row(key, t, state), 'leg': leg}
                                for t, state in zip(segment.times, segment.states))

    write_csv(DATA / 'paper_bounce.csv', summary)
    write_csv(DATA / 'paper_trajectories.csv', trajectories)


def calculate():
    config = read_json(DATA / 'config.json')
    base = read_json(OUTPUT_ROOT / config['source_config'])
    saved = read_json(OUTPUT_ROOT / config['given_fit'])['cases'][config['given_case']]
    crosscheck = read_json(OUTPUT_ROOT / config['given_crosscheck'])['cases'][config['given_case']]
    physics = Physics(**base['physics'])
    settings = Settings(**base['integration'])
    refined_settings = replace(settings, **config['refined_integration'])
    gravity_only = replace(physics, density=0.)
    times, observations = np.asarray(base['times_s']), np.asarray(base['positions_m'])
    r0 = observations[0]
    given = Motion(saved['v0_m_s'], saved['cd'], saved['omega_xyz_rad_s'], saved['k_l'])
    residual = position_residuals(times, observations, given, physics, settings)
    np.testing.assert_allclose(residual, saved['residuals_m'], atol=1e-9, rtol=0)
    fit_record = {'source_case': config['given_case'], 'residuals_m': residual.tolist(),
                  'rms_m': float(np.sqrt(np.mean(residual**2)))}
    cases, convergence, analytic = {}, {}, {}
    trajectory_rows, event_rows = [], []

    def run(key, motion, atmosphere, *, fit=None, e_n=.6, mu_f=0.,
            allow_bounce=True, post_impact_magnus=True):
        options = dict(e_n=e_n, mu_f=mu_f, allow_bounce=allow_bounce,
                       post_impact_magnus=post_impact_magnus)
        shot = simulate_bounce(r0, motion, atmosphere, settings, **options)
        refined = simulate_bounce(r0, motion, atmosphere, refined_settings, **options)
        if (shot.event == 'numerical_failure' or shot.event != refined.event
                or (shot.impact is None) != (refined.impact is None)):
            raise RuntimeError(f'{key}: unstable event or solver failure: {shot.message}')
        time_delta = abs(shot.time - refined.time)
        position_delta = float(np.linalg.norm(shot.state[:3] - refined.state[:3]))
        if time_delta >= 1e-7 or position_delta >= 1e-6:
            raise RuntimeError(f'{key}: event does not converge to the reporting precision.')
        check = {'terminal_time_delta_s': time_delta, 'terminal_position_delta_m': position_delta}
        impact = shot.impact
        if impact is not None:
            finer = refined.impact
            inertia = (2 / 3) * atmosphere.mass * atmosphere.radius**2
            loss = (.5 * atmosphere.mass * (impact.v_minus @ impact.v_minus
                                            - impact.v_plus @ impact.v_plus)
                    + .5 * inertia * (impact.omega_minus @ impact.omega_minus
                                      - impact.omega_plus @ impact.omega_plus))
            np.testing.assert_allclose(impact.energy_loss, loss, atol=1e-10, rtol=0)
            if loss < -1e-10:
                raise RuntimeError(f'{key}: impact added kinetic energy.')
            for value in ('position', 'v_minus', 'v_plus', 'omega_minus', 'omega_plus',
                          'energy_loss', 'time'):
                np.testing.assert_allclose(getattr(impact, value), getattr(finer, value),
                                           atol=1e-7, rtol=0, err_msg=f'{key}: {value}')
            check['impact_time_delta_s'] = abs(impact.time - finer.time)
            check['impact_energy_loss_delta_J'] = abs(impact.energy_loss - finer.energy_loss)
            for phase in ('minus', 'plus'):
                state = np.r_[impact.position, getattr(impact, f'v_{phase}')]
                event_rows.append({
                    **state_row(key, impact.time, state, getattr(impact, f'omega_{phase}')),
                    'event': f'impact_{phase}',
                })
        cases[key] = export_case(shot, motion, atmosphere, options, fit)
        convergence[key] = check
        event_rows.append({
            **state_row(key, shot.time, shot.state, shot.segments[-1].omega_end),
            'event': shot.event,
        })
        for leg, segment in enumerate(shot.segments, 1):
            trajectory_rows.extend({**state_row(key, t, state), 'leg': leg}
                                   for t, state in zip(segment.times, segment.states))
        return shot

    main = run('given/stop', given, physics, fit=fit_record, allow_bounce=False)
    baseline = simulate(r0, given, physics, settings)
    if main.event != baseline.event or main.event != crosscheck['event']:
        raise RuntimeError('Case 3A disagrees with the earlier event classification.')
    np.testing.assert_allclose(main.state[:3], baseline.states[-1, :3], atol=1e-6, rtol=0)
    np.testing.assert_allclose(main.time, baseline.time, atol=1e-7, rtol=0)
    np.testing.assert_allclose(
        main.state[1:3], [crosscheck['goal']['y_m'], crosscheck['goal']['z_m']], atol=1e-6, rtol=0)
    np.testing.assert_allclose(main.time, crosscheck['goal']['t_s'], atol=1e-7, rtol=0)
    for e_n in config['restitution']:
        bounce = run(f'given/e{e_n:g}', given, physics, fit=fit_record, e_n=e_n)
        if bounce.event != main.event or bounce.impact is not None:
            raise RuntimeError('Case 3A unexpectedly changes with the bounce option.')
        np.testing.assert_allclose(bounce.state[:3], main.state[:3], atol=1e-6, rtol=0)
        np.testing.assert_allclose(bounce.time, main.time, atol=1e-7, rtol=0)

    launch_low = np.array(config['synthetic_velocity_m_s']['low'])
    gravity = physics.gravity
    first = 2 * launch_low[2] / gravity
    for e_n in config['restitution']:
        key = case_id('low', 'gravity', e_n)
        shot = run(key, Motion(launch_low, 0.), gravity_only, e_n=e_n)
        expected = {'first_s': first, 'second_s': first * (1 + e_n),
                    'v_minus_m_s': [*launch_low[:2], -launch_low[2]],
                    'v_plus_m_s': [*launch_low[:2], e_n * launch_low[2]],
                    'energy_loss_J': .5 * physics.mass * launch_low[2]**2 * (1 - e_n**2)}
        if shot.event != 'second_ground':
            raise RuntimeError(f'{key}: expected a second ground contact.')
        np.testing.assert_allclose([shot.impact.time, shot.time, shot.impact.energy_loss],
                                   [first, expected['second_s'], expected['energy_loss_J']],
                                   atol=1e-10, rtol=0)
        np.testing.assert_allclose(shot.impact.v_minus, expected['v_minus_m_s'], atol=1e-10, rtol=0)
        np.testing.assert_allclose(shot.impact.v_plus, expected['v_plus_m_s'], atol=1e-10, rtol=0)
        expected['first_x_m'] = launch_low[0] * first
        expected['second_x_m'] = launch_low[0] * expected['second_s']
        np.testing.assert_allclose([shot.impact.position[0], shot.state[0]],
                                   [expected['first_x_m'], expected['second_x_m']],
                                   atol=1e-10, rtol=0)
        analytic[key] = expected

    for name, velocity in config['synthetic_velocity_m_s'].items():
        motion = Motion(velocity, config['synthetic_cd'], k_l=config['synthetic_k_l'])
        run(f'{name}/air/stop', motion, physics, allow_bounce=False)
        for e_n in config['restitution']:
            for mu_f in config['friction']:
                run(case_id(name, 'air', e_n, mu_f), motion, physics, e_n=e_n, mu_f=mu_f)
        e_n, mu_f = config['reference_e_n'], config['ablation_mu_f']
        ablation_key = case_id(name, 'air', e_n, mu_f)
        run(f'{ablation_key}/no_lift', motion, physics, e_n=e_n, mu_f=mu_f,
            post_impact_magnus=False)
        if cases[ablation_key]['impact'] != cases[f'{ablation_key}/no_lift']['impact']:
            raise RuntimeError(f'{name}: disabling post-impact lift changed the impact.')

    result = {
        'config': config, 'physics': asdict(physics), 'integration': asdict(settings),
        'tie_tolerance_s': TIE_TOL_S, 'observations_m': observations.tolist(),
        'observation_times_s': times.tolist(), 'cases': cases, 'convergence': convergence,
        'analytic': analytic,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'scipy': scipy.__version__},
    }
    (DATA / 'results.json').write_text(
        json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    write_csv(DATA / 'trajectories.csv', trajectory_rows)
    write_csv(DATA / 'events.csv', event_rows)
    write_paper_example(config, physics, settings, refined_settings)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--plots-only', action='store_true', help='Reuse saved results.')
    modes.add_argument('--no-plots', action='store_true', help='Export data only.')
    options = parser.parse_args()
    result = read_json(DATA / 'results.json') if options.plots_only else calculate()
    if not options.no_plots:
        from .plot_part7 import make_figures
        from .plot_part7_bounce_3d import make_paper_figure

        make_figures(result)
        make_paper_figure(result)
    print(f'{len(result["cases"])} cases with event and convergence records.')


if __name__ == '__main__':
    main()
