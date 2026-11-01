"""Three detectors with the same interface: fit() on normal data,
then score() new data, where a higher score means more unusual."""

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest


class RuleBasedDetector:
    """Alarm if any channel leaves its normal band.

    The score is the largest number of standard deviations any channel sits
    from its training mean, so a threshold of k is the rule
    "alarm if any sensor is outside mean +/- k * std".
    """

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        if np.any(self.std_ == 0):
            raise ValueError("A channel is constant in the training data, "
                             "so its normal band is undefined.")
        return self

    def score(self, X):
        return np.abs((X - self.mean_) / self.std_).max(axis=1)


class IsolationForestDetector:
    def __init__(self, seed=0):
        self.model = IsolationForest(n_estimators=200, random_state=seed)

    def fit(self, X):
        self.model.fit(X)
        return self

    def score(self, X):
        # score_samples is higher for normal points, so flip the sign
        return -self.model.score_samples(X)


class MahalanobisDetector:
    """Distance from the centre of the normal data, measured in a way that
    accounts for how the channels normally move together."""

    def fit(self, X):
        self.cov_ = EmpiricalCovariance().fit(X)
        return self

    def score(self, X):
        return self.cov_.mahalanobis(X)   # squared distance


def apply_persistence(raw_alarms, k):
    """Alarm only while the raw alarm has been on for the last k samples."""
    s = pd.Series(np.asarray(raw_alarms, dtype=int))
    return (s.rolling(k, min_periods=k).sum() == k).astype(int).to_numpy()


def make_alarms(scores, threshold, k, valid):
    """The final alarm signal. Used for both calibration and evaluation so the
    two can never drift apart. Samples outside `valid` (the warm-up) cannot
    raise an alarm or count towards the k consecutive samples."""
    raw = (np.asarray(scores) > threshold) & np.asarray(valid, dtype=bool)
    return apply_persistence(raw, k)


def validation_far(score_runs, valid_runs, threshold, k):
    """Fraction of valid validation samples with the final alarm on."""
    alarms = [make_alarms(s, threshold, k, v)[v]
              for s, v in zip(score_runs, valid_runs)]
    return np.concatenate(alarms).mean()


def calibrate_threshold(score_runs, valid_runs, target_far, k):
    """Lowest threshold whose final alarms on normal validation data stay at
    or below target_far.

    Raising the threshold can only remove raw alarms, and persistence can only
    remove alarms further, so the false-alarm rate never rises as the threshold
    rises. That makes a binary search over the observed scores valid.
    """
    candidates = np.unique(np.concatenate(
        [s[v] for s, v in zip(score_runs, valid_runs)]))
    lo, hi = 0, len(candidates) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if validation_far(score_runs, valid_runs, candidates[mid], k) <= target_far:
            hi = mid
        else:
            lo = mid + 1
    return candidates[lo]
