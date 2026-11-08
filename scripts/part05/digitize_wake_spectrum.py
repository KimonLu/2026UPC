"""Extract the positive-frequency side-force spectrum from Asai et al., Fig. 7(a)."""

import argparse
import csv
import json
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.common.paths import data_dir


DATA = data_dir(5)
SOURCE = 'https://link.springer.com/article/10.1038/s41598-026-56275-1/figures/7'
BALLS = ('Trionda A', 'Al Rihla', 'Telstar', 'Brazuca')
DF = 200 / 256


def digitize(path):
    rgb = np.asarray(Image.open(path).convert('RGB'), dtype=float)
    if rgb.shape != (670, 2008, 3):
        raise ValueError('Use the full-resolution Fig. 7 image (2008 x 670).')
    x0, x100 = 147.5, 943.5
    y10, ymin = 41.5, 594.5
    # Interior samples from the four legend strokes. Subtracting from
    # white lets antialiased pixels be classified by colour and darkness.
    pigments = 255 - rgb[[96, 129, 162, 195], 667]
    directions = pigments / np.linalg.norm(pigments, axis=1)[:, None]
    ink = 255 - rgb[43:593, 150:943]
    projection = ink @ directions.T
    residual = np.sqrt(np.maximum(0, (ink**2).sum(axis=-1)[..., None] - projection**2))
    nearest = np.argmin(residual, axis=-1)
    yy, xx = np.indices(ink.shape[:2])
    yy, xx = yy + 43, xx + 150
    # Suppress the legend inside panel (a), outside the trajectory curves.
    usable = (np.ptp(ink, axis=-1) >= 38) & ~((xx >= 640) & (yy < 205))
    frequency = DF * np.arange(1, 128)
    x_bins = x0 + frequency * (x100 - x0) / 100
    rows = []
    for idx, ball in enumerate(BALLS):
        mask = usable & (nearest == idx) & (residual[..., idx] < 45)
        samples = []
        for x in x_bins:
            sel = mask & (np.abs(xx - x) <= 1.5)
            if not np.any(sel):
                samples.append((np.nan, np.nan))
                continue
            # Dark pixels near the calibrated column carry more weight.
            weights = projection[..., idx][sel] ** 2 * np.exp(-((xx[sel] - x)/.8)**2)
            pixels = yy[sel]
            order = np.argsort(pixels)
            cdf = np.cumsum(weights[order]) / weights.sum()
            p10, p50, p90 = np.interp([.1, .5, .9], cdf, pixels[order])
            samples.append((p50, max(p50 - p10, p90 - p50, 1.5)))
        y, uncertainty = np.asarray(samples).T
        missing = ~np.isfinite(y)
        observed = np.flatnonzero(~missing)
        if observed[0] != 0 or observed[-1] != len(frequency) - 1 or np.diff(observed).max() > 5:
            raise ValueError(f'{ball}: spectrum has unresolved intervals at {frequency[missing]}.')
        y[missing] = np.interp(frequency[missing], frequency[~missing], y[~missing])
        log_psd = 1 - 6 * (y - y10) / (ymin - y10)
        for f, value, dy, hidden in zip(frequency, log_psd, uncertainty, missing):
            rows.append(dict(ball=ball, frequency_hz=f, psd_n2_per_hz=10**value,
                             log10_reading_halfspan=None if hidden else 6*dy/(ymin-y10),
                             interpolated=bool(hidden)))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    args = parser.parse_args()
    rows = digitize(args.image)
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "wake_psd_30ms.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        'url': SOURCE, 'doi': '10.1038/s41598-026-56275-1', 'panel': '7(a), side force',
        'image_sha256': hashlib.sha256(args.image.read_bytes()).hexdigest(),
        'sample_hz': 200, 'welch_segment_samples': 256, 'frequency_bin_hz': DF,
        'axes': 'frequency linear 0 to 100 Hz, PSD logarithmic 1e-5 to 10 N^2/Hz',
        'used_band_hz': [DF, 127*DF],
        'exclusions': 'DC, frequencies below first positive bin, and Nyquist endpoint',
        'digitization': 'Weighted centre of colour stroke; 10th to 90th percentile pixel span. '
                        'Isolated covered bins are interpolated between adjacent visible bins in log PSD.',
        'balls': {ball: {
            'hidden_bins_hz': [row['frequency_hz'] for row in rows
                               if row['ball'] == ball and row['interpolated']],
            'first_bin_reading_halfspan_dex': next(row['log10_reading_halfspan']
                                                   for row in rows if row['ball'] == ball),
        } for ball in BALLS},
    }
    (DATA / "wake_psd_source.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
