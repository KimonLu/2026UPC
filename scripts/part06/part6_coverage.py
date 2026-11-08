"""Repeated coverage experiment using the published Section 5 procedure."""

import argparse
import hashlib
import os
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from math import sqrt
from statistics import NormalDist

import numpy as np
import scipy

from scripts.common.paths import REPO_ROOT, data_dir
from scripts.part06.part6_common import (
    CONFIG, DATA, GOAL_EVENTS, PHYSICS, R0, TIMES, TRUTH_SETTINGS,
    launch_motion, read_json, write_csv, write_json,
)
from scripts.part05 import part5_solver
from scripts.part05.run_part5 import summarize
from scripts.common.still_air import simulate

COVERAGE = CONFIG['coverage']


def coverage_draws(seed, trial, inner_count):
    outer_seed, inner_seed = np.random.SeedSequence([seed, 4, trial]).spawn(2)
    outer = np.random.default_rng(outer_seed).standard_normal((2, 3))
    inner = np.random.default_rng(inner_seed).standard_normal((inner_count, 2, 3))
    return outer, inner


def region_result(prediction, truth_yz, n=None):
    result = summarize(prediction, n=n)
    result.update(covered=None, truth_distance_squared=None)
    if result.get('region_dimension') == 2:
        delta = np.asarray(truth_yz) - result['mean'][1:]
        distance = float(delta @ np.linalg.solve(result['covariance_yz'], delta))
        result.update(covered=distance <= result['ellipse_radius_squared'],
                      truth_distance_squared=distance)
    return result


def wilson_interval(hits, count):
    if count == 0:
        return [None, None]
    p = hits / count
    z = NormalDist().inv_cdf(.975)
    scale = 1 + z*z / count
    center = (p + z*z / (2*count)) / scale
    radius = z * sqrt(p * (1 - p) / count + z*z / (4*count*count)) / scale
    return [0. if hits == 0 else center - radius,
            1. if hits == count else center + radius]


def coverage_truth(name):
    scenario = COVERAGE['scenarios'][name]
    wind = tuple(scenario['wind_m_s'])
    motion = launch_motion(*COVERAGE['launch'], tau=scenario['truth_tau_s'])
    truth = simulate(R0, motion, replace(PHYSICS, wind=wind), TRUTH_SETTINGS)
    if truth.event not in GOAL_EVENTS or truth.time <= TIMES[-1]:
        raise RuntimeError(f'Coverage truth has no eligible crossing: {name}')
    positions = truth.sample(TIMES)[:, :3]
    crossing = [truth.time, *truth.states[-1, 1:3]]
    return wind, positions, crossing


def run_trial(name, trial, wind, truth_positions, truth_crossing):
    n = COVERAGE['inner_draws']
    outer, inner = coverage_draws(CONFIG['seed'], trial, n)
    observed = truth_positions.copy()
    sigma = COVERAGE['sigma_m']
    observed[1:] += sigma * outer
    inputs = np.repeat(observed[None], n, axis=0)
    inputs[:, 1:] += sigma * inner
    fit = part5_solver.fit_batch(inputs, np.broadcast_to(TIMES, (n, 3)), wind)
    prediction = part5_solver.predict_batch(fit['parameters'], fit['basis'], wind)
    failed = ~fit['success']
    prediction['event'][failed] = 'fit_failure'
    prediction['goal'][failed] = np.nan
    prefixes = [COVERAGE['prefix_draws'], n]
    return dict(scenario=name, trial=trial, observation_m=observed.tolist(),
                truth_crossing=truth_crossing, expanded=int(fit['expanded'].sum()),
                active_bounds=int(fit['active'].sum()), solver_fallbacks=int(fit['fallback'].sum()),
                regions={str(size): region_result(prediction, truth_crossing[1:], size)
                         for size in prefixes})


def coverage_summary(rows, draws):
    regions = [row['regions'][str(draws)] for row in rows]
    valid = [r for r in regions if r['covered'] is not None]
    hits = sum(r['covered'] for r in valid)
    n = len(rows)
    return dict(hits=hits, trials=n, valid_regions=len(valid), failed_regions=n-len(valid),
                coverage=hits / n, wilson_95=wilson_interval(hits, n),
                conditional_coverage=hits / len(valid) if valid else None,
                conditional_wilson_95=wilson_interval(hits, len(valid)),
                median_area_m2=float(np.median([r['ellipse_area_m2'] for r in valid]))
                if valid else None)


def provenance():
    sources = [
        "scripts/part05/part5_solver.py",
        "scripts/part05/run_part5.py",
        "scripts/common/still_air.py",
        "scripts/part06/part6_common.py",
        "scripts/part06/part6_coverage.py",
    ]
    paths = {name: REPO_ROOT / name for name in sources}
    paths.update({"data/part06/config.json": DATA / "config.json",
                  "data/part03/config.json": data_dir(3) / "config.json"})
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                hashes={name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for name, path in paths.items()})


def run_coverage(count, jobs, *, resume):
    target = DATA / ('coverage_pilot' if count == COVERAGE['pilot_trials'] else 'coverage')
    target.mkdir(exist_ok=True)
    manifest = target / 'provenance.json'
    current = provenance()
    if manifest.exists() and (not resume or read_json(manifest) != current):
        raise RuntimeError(f'Existing results in {target} require matching provenance and --resume')
    write_json(manifest, current)

    completed = {}
    pending = []
    for name in COVERAGE['scenarios']:
        wind, positions, crossing = coverage_truth(name)
        for trial in range(count):
            key = name, trial
            existing = target / f'{name}_{trial:03d}.json'
            pilot = DATA / 'coverage_pilot'
            pilot_row = pilot / existing.name
            if existing.exists():
                completed[key] = read_json(existing)
            elif target != pilot and pilot_row.exists() and read_json(pilot / manifest.name) == current:
                completed[key] = read_json(pilot_row)
                write_json(existing, completed[key])
            else:
                pending.append((name, trial, wind, positions, crossing))

    # These are small linear systems. One BLAS thread per worker avoids oversubscription.
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    print(f'Coverage {count}: {len(completed)} resumed, {len(pending)} pending', flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(run_trial, *args) for args in pending]
            for future in as_completed(futures):
                row = future.result()
                name, trial = row['scenario'], row['trial']
                completed[name, trial] = row
                write_json(target / f'{name}_{trial:03d}.json', row)
                if len(completed) % 10 == 0:
                    print(f'Coverage: {len(completed)}/{count * len(COVERAGE["scenarios"])} regions', flush=True)
    summary = {}
    flat = []
    for name in COVERAGE['scenarios']:
        rows = [completed[name, trial] for trial in range(count)]
        for draws in (COVERAGE['prefix_draws'], COVERAGE['inner_draws']):
            summary[f'{name}/{draws}'] = coverage_summary(rows, draws)
        for row in rows:
            for draws, region in row['regions'].items():
                flat.append(dict(scenario=name, trial=row['trial'], inner_draws=int(draws),
                                 covered=region['covered'], crossings=region['n_crossings'],
                                 area_m2=region['ellipse_area_m2'],
                                 truth_distance_squared=region['truth_distance_squared'],
                                 threshold=region.get('ellipse_radius_squared')))
    write_csv(target / 'trials.csv', flat)
    write_json(target / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    run_coverage(COVERAGE['pilot_trials'] if args.pilot else COVERAGE['trials'],
                 args.jobs, resume=args.resume)
