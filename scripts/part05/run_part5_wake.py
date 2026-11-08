"""Refit the given shot under the measured 30 m/s side-force spectra."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from scripts.common.paths import data_dir
from scripts.part06.part6_common import GOAL_EVENTS, OBS, SETTINGS, read_json, write_csv, write_json
from scripts.part06.part6_experiments import fit_known_wake
from scripts.part05.wake_spectrum import SHOT_SEED, WakeTrace, generate_traces


DATA = data_dir(5)


def fit_trace(trace, settings=SETTINGS):
    result = fit_known_wake(OBS, trace, settings=settings)
    goal = result['goal'] if result['goal'] is not None else [None]*3
    rms = (1e3*float(np.sqrt(np.mean(result['residual']**2)))
           if 'residual' in result else None)
    return dict(ball=trace.ball, trace=trace.index, event=result['event'],
                t_g_s=goal[0], y_g_m=goal[1], z_g_m=goal[2], rms_mm=rms,
                cd=result.get('cd'), active_bound=result.get('active_bound'))


def endpoint(row):
    return np.array([row['t_g_s'], row['y_g_m'], row['z_g_m']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    traces, checks = generate_traces(256, SHOT_SEED)
    base = fit_trace(WakeTrace('No wake', 0, None))
    stored = read_json(data_dir(3) / 'results.json')['cases']['S2']
    target = stored['goal']
    difference = endpoint(base)-[target['t_s'], target['y_m'], target['z_m']]
    if np.max(np.abs(difference)) > 1e-6 or abs(base['cd']-stored['cd']) > 1e-6:
        raise AssertionError(f'Zero wake differs from Section 3: {difference}')
    print(f'Zero-wake control agrees with Section 3: {difference}', flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(fit_trace, traces, chunksize=4))
    results = {}
    for ball, check in checks.items():
        if (abs(check['sampled_variance_n2']/check['target_variance_n2']-1) > .10 or
                check['rms_relative_bin_error'] > .09 or check['out_of_band_variance_n2'] > 1e-12):
            raise AssertionError(f'Spectral synthesis check failed for {ball}: {check}')
        cases = [row for row in rows if row['ball'] == ball]
        crossings = [row for row in cases if row['event'] in GOAL_EVENTS]
        shifts = np.array([endpoint(row)-endpoint(base) for row in crossings])*1e3
        r = np.linalg.norm(shifts[:, 1:], axis=1)
        fit_rms = [row['rms_mm'] for row in cases if row['rms_mm'] is not None]
        results[ball] = dict(events=Counter(row['event'] for row in cases),
                             offset_median_mm=float(np.median(r)),
                             offset_p95_mm=float(np.quantile(r, .95)),
                             delta_time_q025_q50_q975_ms=np.quantile(shifts[:, 0], [.025,.5,.975]).tolist(),
                             rms_median_mm=float(np.median(fit_rms)),
                             rms_q025_q975_mm=np.quantile(fit_rms, [.025,.975]).tolist())
    fine = replace(SETTINGS, max_step=SETTINGS.max_step/2, rtol=1e-11, atol=1e-13)
    refinements = []
    for trace, row in zip(traces, rows):
        if trace.index % 32 != 0:
            continue
        refined = fit_trace(trace, fine)
        if refined['event'] != row['event']:
            raise AssertionError(f'Event changed under refinement: {trace.ball}/{trace.index}')
        if refined['event'] in GOAL_EVENTS:
            refinements.append(np.abs(endpoint(refined)-endpoint(row)))
    max_delta = np.max(refinements, axis=0)
    if np.any(max_delta > [1e-5, 1e-4, 1e-4]):
        raise AssertionError(f'Timestep refinement differs by {max_delta}')
    write_csv(DATA / 'wake_crossings.csv', rows)
    write_json(DATA / 'wake_results.json', dict(seed=SHOT_SEED, baseline=base, spectra=checks,
               control_delta_s_m_m=difference.tolist(), refined_crossings=len(refinements),
               refinement_max_delta_s_m_m=max_delta.tolist(), results=results))


if __name__ == '__main__':
    main()
