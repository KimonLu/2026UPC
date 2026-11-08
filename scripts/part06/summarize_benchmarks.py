"""Tabulate saved benchmark predictions on model-specific and common subsets."""

import csv
import hashlib
import json
from collections import Counter
from math import isfinite

import numpy as np

from scripts.common.paths import data_dir


def statistics(values):
    if not values:
        return None
    values = np.asarray(values, dtype=float)
    return dict(median=float(np.median(np.abs(values))),
                p90=float(np.quantile(np.abs(values), .9)),
                rmse=float(np.sqrt(np.mean(values ** 2))), bias=float(values.mean()))


def numeric(row, column):
    value = row.get(column, "")
    return float(value) if value != "" and isfinite(float(value)) else None


def summarize_rows(rows, models):
    groups = {}
    for row in rows:
        if row["scenario"] != "reference" or row["model"] not in models:
            continue
        if str(row["obs_eligible"]).lower() != "true":
            continue
        sigma = f'{float(row["sigma_m"]):g}'
        group = groups.setdefault(sigma, {model: {} for model in models})
        key = (row["case_id"], row["replicate"])
        indexed = group[row["model"]]
        if key in indexed:
            raise ValueError(f'Duplicate prediction: {sigma}, {row["model"]}, {key}')
        indexed[key] = row

    output = {}
    for sigma, group in sorted(groups.items(), key=lambda item: float(item[0])):
        keys = set(group[models[0]])
        if any(set(group[model]) != keys for model in models):
            raise ValueError(f'Model trial keys differ at sigma={sigma}')
        for key in keys:
            if len({group[model][key]["truth_event"] for model in models}) != 1:
                raise ValueError(f'Truth events differ at {sigma}, {key}')
        common = {key for key in keys
                  if all(numeric(group[model][key], "error_m") is not None for model in models)}
        truth_counts = dict(sorted(Counter(row["truth_event"]
                                          for row in group[models[0]].values()).items()))
        summary = dict(trials=len(keys), common_crossings=len(common),
                       truth_event_counts=truth_counts, models={})
        for model in models:
            records = list(group[model].values())
            paired = [row for row in records if numeric(row, "error_m") is not None]
            confusion = {}
            for row in records:
                counts = confusion.setdefault(row["truth_event"], {})
                event = row["prediction_event"]
                counts[event] = counts.get(event, 0) + 1
            correct = sum(row["truth_event"] == row["prediction_event"] for row in records)
            shared = [group[model][key] for key in sorted(common)]
            summary["models"][model] = dict(
                trials=len(records), paired_crossings=len(paired),
                paired_position_m=statistics([numeric(row, "error_m") for row in paired]),
                paired_arrival_s=statistics([numeric(row, "time_error_s") for row in paired]),
                common_position_m=statistics([numeric(row, "error_m") for row in shared]),
                common_arrival_s=statistics([numeric(row, "time_error_s") for row in shared]),
                correct_events=correct, event_accuracy=correct / len(records), confusion=confusion,
            )
        output[sigma] = summary
    return output


def main():
    directory = data_dir(6)
    source = directory / "v2_predictions.csv"
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    with source.open(encoding="utf-8", newline="") as stream:
        summary = summarize_rows(csv.DictReader(stream), tuple(config["synthetic"]["models"]))
    result = dict(source="data/part06/v2_predictions.csv",
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  common_subset="Truth and every model reach the goal plane on the same case and replicate.",
                  event_denominator="All observation-eligible trials, including failed predictions.",
                  by_noise_m=summary)
    (directory / "benchmark_comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    compact, confusion = [], []
    for sigma, group in summary.items():
        for model, entry in group["models"].items():
            compact.append(dict(sigma_m=sigma, model=model, trials=entry["trials"],
                                paired_crossings=entry["paired_crossings"],
                                common_crossings=group["common_crossings"],
                                paired_median_m=entry["paired_position_m"]["median"],
                                paired_p90_m=entry["paired_position_m"]["p90"],
                                common_median_m=entry["common_position_m"]["median"],
                                common_p90_m=entry["common_position_m"]["p90"],
                                event_accuracy=entry["event_accuracy"]))
            for truth, counts in sorted(entry["confusion"].items()):
                for prediction, count in sorted(counts.items()):
                    confusion.append(dict(sigma_m=sigma, model=model, truth_event=truth,
                                          prediction_event=prediction, count=count))
    for name, records in (("benchmark_comparison.csv", compact), ("event_confusion.csv", confusion)):
        with (directory / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps(summary["0.005"], indent=2))


if __name__ == "__main__":
    main()
