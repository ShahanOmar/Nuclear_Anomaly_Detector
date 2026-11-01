"""Fit detectors on normal data, set thresholds on separate normal data,
then evaluate on test runs containing injected faults."""

import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import sklearn

from data import FEATURE_WINDOW, make_features, simulate_run
from detectors import (IsolationForestDetector, MahalanobisDetector,
                       RuleBasedDetector, calibrate_threshold, make_alarms,
                       validation_far)
from evaluation import (RECOVERY_MIN, WARMUP_MIN, evaluate_run, scored_mask,
                        summarise)
from plots import plot_delays, plot_scores, plot_telemetry

TRAIN_SEEDS = [0, 1, 2]
VAL_SEEDS = [10, 11]
TEST_SEEDS = list(range(100, 110))
TARGET_FAR = 0.002   # share of normal validation samples with the final alarm on
PERSISTENCE = 3      # samples in a row (30 s) before an alarm is raised
OUT = Path("results")


def main():
    OUT.mkdir(exist_ok=True)

    train = [simulate_run(s, with_faults=False) for s in TRAIN_SEEDS]
    val = [simulate_run(s, with_faults=False) for s in VAL_SEEDS]
    test = [simulate_run(s, with_faults=True) for s in TEST_SEEDS]

    # Normal runs have no faults, so only the warm-up is excluded
    X_train = np.vstack(
        [make_features(r)[scored_mask(r, faults=[])] for r in train])
    val_X = [make_features(r) for r in val]
    val_valid = [scored_mask(r, faults=[]) for r in val]

    detectors = {
        "Rule-based": RuleBasedDetector(),
        "Isolation Forest": IsolationForestDetector(seed=0),
        "Mahalanobis": MahalanobisDetector(),
    }

    thresholds, achieved_far = {}, {}
    for name, det in detectors.items():
        det.fit(X_train)
        val_scores = [det.score(X) for X in val_X]
        thr = calibrate_threshold(
            val_scores, val_valid, TARGET_FAR, PERSISTENCE)
        thresholds[name] = float(thr)
        achieved_far[name] = float(validation_far(
            val_scores, val_valid, thr, PERSISTENCE))
        print(f"{name}: threshold {thr:.4f}, "
              f"validation false-alarm rate {achieved_far[name]:.4%}")

    point_rows, event_rows = [], []
    example_scores, example_alarms = {}, {}
    for seed, run in zip(TEST_SEEDS, test):
        X = make_features(run)
        past_warmup = scored_mask(run, faults=[])   # gates the alarms
        # also drops post-fault recovery
        scored = scored_mask(run)
        for name, det in detectors.items():
            scores = det.score(X)
            alarms = make_alarms(
                scores, thresholds[name], PERSISTENCE, past_warmup)
            point, events = evaluate_run(run, alarms, scored)
            point_rows.append({"detector": name, "seed": seed, **point})
            event_rows += [{"detector": name, "seed": seed, **e}
                           for e in events]
            if seed == TEST_SEEDS[0]:
                example_scores[name] = scores
                example_alarms[name] = alarms

    point_df = pd.DataFrame(point_rows)
    event_df = pd.DataFrame(event_rows)
    point_summary, event_summary = summarise(point_df, event_df)

    point_df.to_csv(OUT / "point_metrics_per_run.csv", index=False)
    event_df.to_csv(OUT / "events_per_run.csv", index=False)
    point_summary.to_csv(OUT / "point_metrics_summary.csv")
    event_summary.to_csv(OUT / "event_summary.csv")

    info = {
        "train_seeds": TRAIN_SEEDS, "val_seeds": VAL_SEEDS, "test_seeds": TEST_SEEDS,
        "target_far": TARGET_FAR, "persistence": PERSISTENCE,
        "feature_window": FEATURE_WINDOW, "warmup_min": WARMUP_MIN,
        "recovery_min": RECOVERY_MIN,
        "thresholds": thresholds, "validation_far_achieved": achieved_far,
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2))

    pd.set_option("display.width", 160)
    print("\nPoint-wise metrics over test runs (mean, std)")
    print(point_summary)
    print("\nPer-fault detection")
    print(event_summary)

    plot_telemetry(test[0], example_alarms, OUT / "telemetry_and_alarms.png")
    plot_scores(test[0], example_scores, thresholds,
                OUT / "anomaly_scores.png")
    plot_delays(event_summary, OUT / "detection_delay.png")


if __name__ == "__main__":
    main()
