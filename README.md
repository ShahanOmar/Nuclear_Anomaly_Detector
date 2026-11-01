# Anomaly detection on simulated reactor coolant-loop sensor data

A small experiment comparing a rule-based alarm system with two learned
detectors on synthetic sensor data from a simplified pressurised water
reactor (PWR) primary loop.

**This is a learning project. The data is simulated with simple linear
relationships, and nothing here says anything about the safety of real
reactors.**

## Question

At the same false-alarm rate, does a detector that learns how sensors relate
to each other catch faults earlier than fixed limits on each sensor?

## Simulated plant

Five channels sampled every 10 s: reactor power, hot-leg and cold-leg
temperature, pressuriser pressure, and an area radiation monitor. They are
linked through the heat balance (ΔT ∝ power / flow), a simplified
temperature control program, and pressure control. Process variables follow
a mean-reverting random process; each sensor adds measurement noise.

Each run lasts 1000 minutes and includes a planned power reduction of 5–15%,
which is normal operation and tests for false alarms.

## Injected faults (test runs only)

| Fault | What happens | Sensors affected |
|---|---|---|
| Coolant flow loss | Flow falls ~12% over 20 min | T_hot up, T_cold down, pressure slightly up |
| T_hot sensor drift | Reading drifts ~3 °C high over 100 min | T_hot only |
| Small coolant leak | Gradual over 30 min | Pressure down, radiation up |

Fault sizes vary ±20% between runs.

## Method

- Training: 3 normal runs. Validation: 2 normal runs. Test: 10 runs with faults.
- Features: 5-minute trailing mean of each channel.
- Detectors: per-channel z-score limits (rule-based), Isolation Forest,
  Mahalanobis distance.
- Every threshold is set so 0.2% of validation samples exceed it, giving
  each detector the same false-alarm budget.
- An alarm needs 3 consecutive samples (30 s) above threshold.

## Results

[Paste point_metrics_summary and event_summary here.]

![Telemetry and alarms](results/telemetry_and_alarms.png)
![Anomaly scores](results/anomaly_scores.png)
![Detection delay](results/detection_delay.png)

[Describe what you see: which detector caught which fault, how quickly,
and why you think that happened.]

## False alarms

[Where did they occur? During the power reduction? Right after faults
ended because of the smoothing window? Explain the cause of each.]

## Limitations

[Your own list. See the limitations discussion in your notes.]

## What I learned

[Write this yourself. What surprised you? What did you get wrong at first?
What would you do differently?]

## How to run

    pip install -r requirements.txt
    python run_experiment.py

## Acknowledgements

I used an AI assistant to review my original code and help redesign the
experiment. I ran the experiments and wrote the analysis myself.