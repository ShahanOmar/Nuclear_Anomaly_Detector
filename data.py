"""Synthetic sensor data for a simplified PWR primary coolant loop.

This is not a physics simulation. Each channel is built from a few linear
relationships so that faults affect several sensors in a qualitatively
realistic way. Values are of the right order of magnitude for a pressurised
water reactor but are illustrative, not taken from any real plant.
"""

import numpy as np
import pandas as pd

DT_SECONDS = 10                     # one reading every 10 seconds
STEPS_PER_MIN = 60 // DT_SECONDS
RUN_MINUTES = 1000
FEATURE_WINDOW = 30                 # samples, i.e. 5 minutes

CHANNELS = ["power", "T_hot", "T_cold", "pressure", "radiation"]

# Simplified plant model
T_AVG_ZERO_POWER = 292.0   # degC, average coolant temperature at zero power
T_AVG_SLOPE = 16.5         # degC increase in average temperature from 0 to 100% power
DELTA_T_FULL = 33.0        # degC rise across the core at full power and full flow
P_SETPOINT = 15.5          # MPa, pressuriser pressure setpoint
P_PER_DEGC = 0.05          # MPa per degC of unplanned average-temperature rise
RAD_FULL_POWER = 1.0       # uSv/h, area radiation monitor at full power

# Standard deviation of measurement noise on each sensor
NOISE = {"power": 0.5, "T_hot": 0.2, "T_cold": 0.2,
         "pressure": 0.01, "radiation": 0.05}

# Faults injected into every test run (times in minutes)
FAULTS = [
    {"name": "Coolant flow loss", "kind": "flow",
        "start": 350, "end": 430, "ramp": 20},
    {"name": "T_hot sensor drift", "kind": "drift",
        "start": 550, "end": 650, "ramp": 100},
    {"name": "Small coolant leak", "kind": "leak",
        "start": 780, "end": 860, "ramp": 30},
]
TEST_MANEUVER_START = 150  # minutes; kept clear of the faults in test runs


def trapezoid(t, start, ramp, hold):
    """0 before start, rises to 1 over `ramp`, holds for `hold`, then falls back to 0."""
    rise = np.clip((t - start) / ramp, 0, 1)
    fall = np.clip((t - start - ramp - hold) / ramp, 0, 1)
    return rise - fall


def fault_profile(t, start, end, ramp):
    """0 outside [start, end). Inside, rises linearly to 1 over `ramp` minutes."""
    p = np.clip((t - start) / ramp, 0, 1)
    p[t >= end] = 0.0
    return p


def mean_reverting_noise(n, theta, sigma, rng, target=None):
    """Random wandering that is pulled back towards `target` (an
    Ornstein-Uhlenbeck process). Unlike white noise, neighbouring values
    are correlated, which is how real process variables behave."""
    if target is None:
        target = np.zeros(n)
    x = np.empty(n)
    x[0] = target[0]
    shocks = rng.normal(0, sigma, n)
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (target[i] - x[i - 1]) + shocks[i]
    return x


def simulate_run(seed, with_faults):
    rng = np.random.default_rng(seed)
    n = RUN_MINUTES * STEPS_PER_MIN
    t = np.arange(n) / STEPS_PER_MIN

    # A planned power reduction (load following). This is normal operation,
    # included so we can see whether detectors raise false alarms on it.
    depth = rng.uniform(0.05, 0.15)
    start = TEST_MANEUVER_START if with_faults else rng.uniform(
        50, RUN_MINUTES - 200)
    maneuver = trapezoid(t, start, ramp=30, hold=60)
    power = mean_reverting_noise(n, theta=0.02, sigma=0.001, rng=rng,
                                 target=1.0 - depth * maneuver)

    flow = np.ones(n)
    t_avg_upset = np.zeros(n)
    pressure_shift = np.zeros(n)
    radiation_shift = np.zeros(n)
    t_hot_bias = np.zeros(n)
    fault_name = np.full(n, "normal", dtype=object)

    if with_faults:
        for f in FAULTS:
            size = rng.uniform(0.8, 1.2)   # fault severity varies between runs
            p = size * fault_profile(t, f["start"], f["end"], f["ramp"])
            if f["kind"] == "flow":
                flow -= 0.12 * p           # up to ~12% loss of coolant flow
                t_avg_upset += 0.5 * p     # control system lags behind
            elif f["kind"] == "drift":
                t_hot_bias += 3.0 * p      # sensor reads up to ~3 degC high
            elif f["kind"] == "leak":
                pressure_shift -= 0.12 * p
                radiation_shift += 0.4 * p
            fault_name[(t >= f["start"]) & (t < f["end"])] = f["name"]

    # Heat balance: Q = m_dot * c_p * dT, so dT is proportional to power / flow
    delta_t = DELTA_T_FULL * power / flow
    t_avg = T_AVG_ZERO_POWER + T_AVG_SLOPE * power + t_avg_upset
    pressure = (P_SETPOINT
                + mean_reverting_noise(n, theta=0.05, sigma=0.003, rng=rng)
                + P_PER_DEGC * t_avg_upset
                + pressure_shift)

    true_values = {
        "power": 100 * power,
        "T_hot": t_avg + delta_t / 2,
        "T_cold": t_avg - delta_t / 2,
        "pressure": pressure,
        "radiation": RAD_FULL_POWER * power + radiation_shift,
    }

    data = {"time_min": t}
    for ch in CHANNELS:
        data[ch] = true_values[ch] + rng.normal(0, NOISE[ch], n)
    # the sensor is wrong, not the coolant
    data["T_hot"] += t_hot_bias
    data["fault"] = fault_name
    data["is_anomaly"] = (fault_name != "normal").astype(int)
    data["maneuver"] = (maneuver > 0).astype(int)
    return pd.DataFrame(data)


def make_features(run, window=FEATURE_WINDOW):
    """Trailing rolling mean of each channel. Only past readings are used,
    so the same calculation could run live on a plant."""
    return run[CHANNELS].rolling(window, min_periods=1).mean().to_numpy()
