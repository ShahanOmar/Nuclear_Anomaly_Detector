import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from data import FAULTS, FEATURE_WINDOW, STEPS_PER_MIN

# until the rolling window is full
WARMUP_MIN = FEATURE_WINDOW / STEPS_PER_MIN
# smoothing lag after a fault ends
RECOVERY_MIN = FEATURE_WINDOW / STEPS_PER_MIN


def scored_mask(run, faults=FAULTS, warmup_min=WARMUP_MIN, recovery_min=RECOVERY_MIN):
    """Samples that count towards point-wise metrics.

    Excludes the warm-up, when the rolling mean covers fewer readings than the
    full window, and the period just after each fault, when the smoothed
    features still contain faulty readings but the label already says normal.
    Pass faults=[] for runs without faults.
    """
    t = run["time_min"].to_numpy()
    mask = t >= warmup_min
    for f in faults:
        mask &= ~((t >= f["end"]) & (t < f["end"] + recovery_min))
    return mask


def alarm_starts(alarms):
    """Indices where the alarm switches from off to on."""
    a = np.asarray(alarms, dtype=int)
    return np.flatnonzero(np.diff(a, prepend=0) == 1)


def evaluate_run(run, alarms, scored, faults=FAULTS):
    y = run["is_anomaly"].to_numpy()
    t = run["time_min"].to_numpy()
    alarms = np.asarray(alarms, dtype=int)
    scored = np.asarray(scored, dtype=bool)
    sample_hours = (t[1] - t[0]) / 60

    ys, als = y[scored], alarms[scored]
    normal = ys == 0
    starts = alarm_starts(alarms)
    false_starts = starts[scored[starts] & (y[starts] == 0)]
    normal_hours = normal.sum() * sample_hours

    point = {
        # A run with no alarms gets precision 0 (sklearn's zero_division=0).
        "precision": precision_score(ys, als, zero_division=0),
        "recall": recall_score(ys, als, zero_division=0),
        "f1": f1_score(ys, als, zero_division=0),
        "false_alarm_rate": als[normal].mean() if normal.any() else np.nan,
        "false_alarms_per_hour": len(false_starts) / normal_hours if normal_hours else np.nan,
        "alarm_samples_unscored": int(alarms[~scored].sum()),
    }

    events = []
    for f in faults:
        # first sample at or after onset
        onset = np.searchsorted(t, f["start"])
        in_window = (t[starts] >= f["start"]) & (t[starts] < f["end"])
        hits = starts[in_window]
        detected = hits.size > 0
        events.append({
            "fault": f["name"],
            "detected": bool(detected),
            "delay_min": float(t[hits[0]] - f["start"]) if detected else np.nan,
            "alarm_on_at_onset": bool(onset > 0 and alarms[onset - 1] == 1),
        })
    return point, events


def summarise(point_df, event_df):
    point_summary = (point_df.drop(columns="seed")
                     .groupby("detector")
                     .agg(["mean", "std"])
                     .round(4))
    event_summary = (event_df.groupby(["fault", "detector"])
                     .agg(n_runs=("detected", "size"),
                          n_detected=("detected", "sum"),
                          detection_rate=("detected", "mean"),
                          median_delay_if_detected=("delay_min", "median"),
                          median_delay_misses_as_inf=(
                              "delay_min", lambda d: float(np.median(d.fillna(np.inf)))),
                          runs_alarm_on_at_onset=("alarm_on_at_onset", "sum"))
                     .round(2))
    return point_summary, event_summary
