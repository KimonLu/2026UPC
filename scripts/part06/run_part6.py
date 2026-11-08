"""Run Section 6 validation and regenerate its results and figures."""

import argparse
import platform

import numpy as np
import scipy

from scripts.common.paths import data_dir
from scripts.part06.part6_common import CONFIG, DATA, write_csv, write_json, read_json
from scripts.part06.part6_coverage import run_coverage
from scripts.part06.part6_drag_curves import run_drag_curves
from scripts.part06.part6_experiments import run_experiment, run_v1, saved_shots
from scripts.part05.run_part5 import kde_mode, summarize


def given_shot():
    part5_data = data_dir(5)
    published = read_json(part5_data / 'monte_carlo_summary.json')
    condition = 'position/gaussian/0.005'
    index = published['ensemble_conditions'].index(condition)
    summaries = {}
    with np.load(part5_data / 'ensembles.npz', allow_pickle=False) as samples:
        for model in ('M-DM/calm', 'M-DM/wind'):
            goal = samples[f'{index}_{model}_goal']
            event = samples[f'{index}_{model}_event']
            summary = summarize(dict(goal=goal, event=event))
            reference = published['summaries'][f'{condition}/{model}']
            for key in ('events', 'n', 'mean', 'covariance_yz', 'ellipse_radius_squared',
                        'ellipse_area_m2', 'q025', 'q975'):
                if key in ('events', 'n'):
                    if summary[key] != reference[key]:
                        raise RuntimeError(f'Section 5 reconciliation failed: {model} {key}')
                else:
                    np.testing.assert_allclose(summary[key], reference[key], rtol=1e-12,
                                               atol=1e-14, err_msg=f'{model} {key}')
            if 'kde_mode' in reference:
                summary['kde_mode'] = reference['kde_mode']
            summary['inside_kde_mode'] = kde_mode(goal[event == 'goal_inside'])
            summary['inside_kde_mode']['conditioning'] = 'inside-goal crossings only'
            summaries[model] = summary
    keys = ('case', 'event', 'v0_m_s', 'cd', 'k_l', 'tau_s', 'omega_xyz_rad_s',
            'omega_basis_rad_s', 'residuals_m', 'rms_m', 'goal')
    cases = {case_id: dict(**{key: case[key] for key in keys},
                           wind_m_s=case.get('wind_m_s', [0, 0, 0]))
             for case_id, case in saved_shots().items()}
    cases['B-G']['acceleration_m_s2'] = [0, 0, -9.81]
    cases['B-CA']['acceleration_m_s2'] = [-16, -16, -12]
    return dict(cases=cases, uncertainty=summaries, main_case='4A/calm', sigma_m=.005)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--section', choices=('V1', 'V2', 'V3', 'V3-curve', 'V4', 'shot'),
                        action='append')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plots-only', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if args.plots_only:
        from scripts.part06.plot_part6 import make_figures
        make_figures()
        return
    target = DATA / ('pilot' if args.pilot else '')
    target.mkdir(exist_ok=True)
    sections = args.section or (('V1', 'V2', 'V3', 'V4', 'shot') if args.pilot
                                else ('V1', 'V2', 'V3', 'V3-curve', 'V4', 'shot'))
    for section in sections:
        if section == 'V1':
            checks = run_v1()
            write_csv(target / 'v1_checks.csv', checks)
            failures = [check for check in checks if not check['passed']]
            if failures:
                raise RuntimeError(f'Numerical checks failed: {failures}')
            print(f'V1: {len(checks)} checks passed', flush=True)
        elif section in ('V2', 'V3'):
            result = run_experiment(section, args.jobs, pilot=args.pilot)
            write_json(target / f'{section.lower()}_summary.json', result)
        elif section == 'V3-curve':
            run_drag_curves(target)
        elif section == 'V4':
            count = CONFIG['coverage']['pilot_trials' if args.pilot else 'trials']
            run_coverage(count, args.jobs, resume=args.resume)
        else:
            write_json(target / 'given_shot.json', given_shot())
    write_json(target / 'runtime.json', dict(python=platform.python_version(),
                                            numpy=np.__version__, scipy=scipy.__version__))
    if not args.no_plots and not args.pilot and args.section is None:
        from scripts.part06.plot_part6 import make_figures
        make_figures()


if __name__ == '__main__':
    main()
