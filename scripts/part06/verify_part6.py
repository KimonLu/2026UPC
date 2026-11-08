"""Reconcile Section 6 export totals, forecasts, and ellipse membership."""

import argparse
import csv
from collections import defaultdict

import numpy as np

from scripts.part06.part6_common import CONFIG, DATA, GOAL_EVENTS, read_json
from scripts.part06.part6_coverage import coverage_summary, provenance
from scripts.part06.part6_experiments import event_metrics, synthetic_tasks


def csv_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def audit_predictions(section, mismatch=False):
    tasks = synthetic_tasks(mismatch=mismatch)
    truth = csv_rows(DATA / f'{section}_truth.csv')
    assert len(truth) == len(tasks)
    truth_ids = {(row['case_id'], row['scenario']) for row in truth}
    assert len(truth_ids) == len(tasks)
    rows = csv_rows(DATA / f'{section}_predictions.csv')
    expected_count = sum(len(models)*sum(1 if sigma == 0 else repetitions for sigma in noise)
                         for _, _, _, _, models, noise, repetitions in tasks)
    assert len(rows) == expected_count
    keys = {(row['case_id'], row['scenario'], row['model'], row['sigma_m'], row['replicate'])
            for row in rows}
    assert len(keys) == expected_count
    groups = defaultdict(list)
    for row in rows:
        assert (row['case_id'], row['scenario']) in truth_ids
        row['obs_eligible'] = row['obs_eligible'] == 'True'
        paired = row['truth_event'] in GOAL_EVENTS and row['prediction_event'] in GOAL_EVENTS
        for key in ('error_m', 'time_error_s', 'y_error_m', 'z_error_m'):
            row[key] = float(row[key]) if row[key] else None
            assert (row[key] is not None) == paired
        if paired:
            dt, dy, dz = [float(row['prediction_'+key]) - float(row['truth_'+key])
                          for key in ('t_g_s', 'y_g_m', 'z_g_m')]
            np.testing.assert_allclose([row['time_error_s'], row['y_error_m'], row['z_error_m'],
                                        row['error_m']], [dt, dy, dz, np.hypot(dy, dz)])
        name = 'wake' if row['scenario'].startswith('wake/') else row['scenario']
        groups[f'{name}/{row["model"]}/{float(row["sigma_m"]):g}'].append(row)
    expected = read_json(DATA / f'{section}_summary.json')
    assert expected == {key: event_metrics(group) for key, group in groups.items()}
    print(f'{section.upper()}: {len(truth)} truth trajectories, {len(rows)} scored rows reconciled')


def audit_coverage(check_provenance=False):
    settings = CONFIG['coverage']
    target = DATA / 'coverage'
    if check_provenance:
        assert read_json(target / 'provenance.json') == provenance()
    summaries = read_json(target / 'summary.json')
    count = settings['trials']
    for name in settings['scenarios']:
        rows = [read_json(target / f'{name}_{index:03d}.json') for index in range(count)]
        for draws in (settings['prefix_draws'], settings['inner_draws']):
            for trial, row in enumerate(rows):
                assert row['scenario'] == name and row['trial'] == trial
                region = row['regions'][str(draws)]
                assert sum(region['events'].values()) == draws
                assert region['n_crossings'] == sum(region['events'][key] for key in GOAL_EVENTS)
                if region['covered'] is not None:
                    delta = np.array(row['truth_crossing'][1:]) - region['mean'][1:]
                    covariance = np.asarray(region['covariance_yz'])
                    radius = region['ellipse_radius_squared']
                    assert np.all(np.linalg.eigvalsh(covariance) > 0)
                    assert np.isclose(region['ellipse_area_m2'], np.pi*radius*np.sqrt(np.linalg.det(covariance)))
                    assert (delta @ np.linalg.inv(covariance) @ delta <= radius) == region['covered']
            assert summaries[f'{name}/{draws}'] == coverage_summary(rows, draws)
    print(f'V4: {count * len(settings["scenarios"])} trial regions and their prefixes reconciled')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-provenance", action="store_true")
    args = parser.parse_args()
    assert all(row['passed'] == 'True' for row in csv_rows(DATA / 'v1_checks.csv'))
    audit_predictions('v2')
    audit_predictions('v3', mismatch=True)
    audit_coverage(args.check_provenance)
    curves = csv_rows(DATA / 'v3_drag_curves.csv')
    assert len(curves) == 36
    assert all(7 <= float(row['q_min_m_s']) <= float(row['q_max_m_s']) <= 35 for row in curves)
    shot = read_json(DATA / 'given_shot.json')
    assert len(shot['cases']) == 16
    assert all(case['event'] == 'goal_inside' for case in shot['cases'].values())
    assert all(summary['n'] == 4000 for summary in shot['uncertainty'].values())
    print('V1, wind-tunnel stress cases and given-shot exports reconciled')


if __name__ == '__main__':
    main()
