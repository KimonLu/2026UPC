"""Generate zero-mean, wind-tunnel-calibrated horizontal wake-force traces."""

import csv
from dataclasses import dataclass
from math import hypot

import numpy as np
from scipy.interpolate import CubicSpline

from scripts.common.paths import data_dir


PSD_PATH = data_dir(5) / 'wake_psd_30ms.csv'
REFERENCE_SPEED = 30.
REFERENCE_DENSITY = 1.2
REFERENCE_AREA = np.pi * .11**2
# Oversample the synthesized signal for interpolation, retaining only
# the published frequency band. The period exceeds the 2 s ODE horizon.
SAMPLE_HZ = 800
N_SAMPLES = 4096
SHOT_SEED = 20261108
VALIDATION_SEED = 20261109
VALIDATION_DRAWS_PER_SPECTRUM = 4


@dataclass
class WakeTrace:
    ball: str
    index: int
    coefficients: np.ndarray | None

    def force_n(self, time):
        if self.coefficients is None:
            return 0.
        interval = min(int(time*SAMPLE_HZ), N_SAMPLES-1)
        t = time - interval/SAMPLE_HZ
        a, b, c, d = self.coefficients[interval]
        return ((a*t+b)*t+c)*t+d


def load_spectra(path=PSD_PATH):
    by_ball = {}
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            by_ball.setdefault(row['ball'], []).append(
                (float(row['frequency_hz']), float(row['psd_n2_per_hz'])))
    return {ball: np.asarray(rows).T for ball, rows in by_ball.items()}


def force_ensemble(frequency, spectrum, count, rng):
    grid = np.fft.rfftfreq(N_SAMPLES, 1/SAMPLE_HZ)
    psd = np.interp(grid, frequency, spectrum, left=0., right=0.)
    coefficients = (rng.standard_normal((count, len(grid))) +
                    1j*rng.standard_normal((count, len(grid))))
    # E[2 |X_k|^2 / (N fs)] = S(f_k) for positive interior bins.
    coefficients *= np.sqrt(N_SAMPLES * SAMPLE_HZ * psd / 4)
    samples = np.fft.irfft(coefficients, n=N_SAMPLES)
    return samples, grid, psd


def generate_traces(count, seed):
    rng = np.random.default_rng(seed)
    time = np.arange(N_SAMPLES + 1) / SAMPLE_HZ
    traces, checks = [], {}
    for ball, (frequency, spectrum) in load_spectra().items():
        forces, grid, psd = force_ensemble(frequency, spectrum, count, rng)
        actual = 2*np.mean(np.abs(np.fft.rfft(forces))**2, axis=0)/(N_SAMPLES*SAMPLE_HZ)
        used = psd > 0
        df = grid[1] - grid[0]
        checks[ball] = dict(
            n=count, band_hz=[float(frequency[0]), float(frequency[-1])],
            target_variance_n2=float(np.sum(psd)*df),
            sampled_variance_n2=float(np.mean(forces**2)),
            rms_relative_bin_error=float(np.sqrt(np.mean((actual[used]/psd[used]-1)**2))),
            out_of_band_variance_n2=float(actual[~used].sum()*df),
            max_trace_mean_n=float(np.max(np.abs(forces.mean(axis=1)))),
        )
        for idx, force in enumerate(forces):
            spline = CubicSpline(time, np.r_[force, force[0]], bc_type='periodic')
            traces.append(WakeTrace(ball, idx, spline.c.T.copy()))
    return traces, checks


def wake_rhs(velocity_drag, physics, trace):
    """Return the zero-spin equations with a transverse wind-tunnel side force."""
    drag = physics.density * physics.area * velocity_drag[3] / (2*physics.mass)
    wake_scale = (physics.density * physics.area /
                  (REFERENCE_DENSITY * REFERENCE_AREA * physics.mass * REFERENCE_SPEED**2))
    wx, wy, wz = physics.wind

    def rhs(time, state):
        vx, vy, vz = state[3:]
        ux, uy, uz = vx-wx, vy-wy, vz-wz
        u2 = ux*ux + uy*uy + uz*uz
        damping = -drag * np.sqrt(u2)
        u_horizontal = hypot(ux, uy)
        side = wake_scale * u2 * float(trace.force_n(time)) / u_horizontal if u_horizontal else 0.
        return [vx, vy, vz, damping*ux-side*uy, damping*uy+side*ux,
                damping*uz-physics.effective_gravity]

    return rhs
