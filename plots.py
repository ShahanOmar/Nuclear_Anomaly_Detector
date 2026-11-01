import matplotlib.pyplot as plt
import numpy as np

from data import CHANNELS, FAULTS

LABELS = {
    "power": "Power (%)",
    "T_hot": "T hot (°C)",
    "T_cold": "T cold (°C)",
    "pressure": "Pressure (MPa)",
    "radiation": "Radiation (µSv/h)",
}


def maneuver_span(run):
    t = run.loc[run["maneuver"] == 1, "time_min"]
    return (t.iloc[0], t.iloc[-1]) if len(t) else None


def shade_events(ax, run):
    for f in FAULTS:
        ax.axvspan(f["start"], f["end"], color="tab:red", alpha=0.12, lw=0)
    span = maneuver_span(run)
    if span:
        ax.axvspan(*span, color="tab:grey", alpha=0.15, lw=0)


def label_events(ax, run):
    labels = [((f["start"] + f["end"]) / 2, f["name"]) for f in FAULTS]
    span = maneuver_span(run)
    if span:
        labels.append((sum(span) / 2, "Power reduction (normal)"))
    for x, text in labels:
        ax.text(x, 1.03, text, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8)


def plot_telemetry(run, alarms, path):
    t = run["time_min"].to_numpy()
    fig, axes = plt.subplots(len(CHANNELS) + 1, 1, figsize=(12, 12),
                             sharex=True, layout="constrained")
    for ax, ch in zip(axes, CHANNELS):
        ax.plot(t, run[ch], lw=0.6, color="black")
        ax.set_ylabel(LABELS[ch])
        shade_events(ax, run)
    label_events(axes[0], run)

    ax = axes[-1]
    for i, a in enumerate(alarms.values()):
        on = t[a == 1]
        ax.plot(on, np.full(len(on), i), "|", markersize=10)
    ax.set_yticks(range(len(alarms)), list(alarms))
    ax.set_ylim(-0.5, len(alarms) - 0.5)
    ax.set_ylabel("Alarms")
    ax.set_xlabel("Time (min)")
    shade_events(ax, run)

    fig.suptitle(
        "Example test run: sensor readings, injected faults (red) and alarms")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scores(run, scores, thresholds, path):
    t = run["time_min"].to_numpy()
    fig, axes = plt.subplots(len(scores), 1, figsize=(12, 7),
                             sharex=True, layout="constrained")
    for ax, (name, s) in zip(axes, scores.items()):
        ax.plot(t, s, lw=0.7, color="black")
        ax.axhline(thresholds[name], color="tab:blue",
                   ls="--", lw=1, label="Threshold")
        ax.set_yscale("log")
        ax.set_ylabel(name)
        shade_events(ax, run)
    axes[0].legend(loc="upper left")
    axes[-1].set_xlabel("Time (min)")
    fig.suptitle("Anomaly scores for the example test run")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_delays(event_summary, path):
    faults = list(dict.fromkeys(event_summary.index.get_level_values("fault")))
    detectors = list(dict.fromkeys(
        event_summary.index.get_level_values("detector")))
    x = np.arange(len(faults))
    width = 0.8 / len(detectors)

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for i, det in enumerate(detectors):
        rows = event_summary.xs(det, level="detector").reindex(faults)
        heights = rows["median_delay_if_detected"].fillna(0).to_numpy()
        bars = ax.bar(x - 0.4 + width * (i + 0.5), heights, width, label=det)
        labels = [f"{int(d)}/{int(n)}" for d,
                  n in zip(rows["n_detected"], rows["n_runs"])]
        ax.bar_label(bars, labels=labels, fontsize=8, padding=2)
    ax.set_xticks(x, faults)
    ax.set_ylabel("Median delay after onset (min)")
    ax.set_title("Detection delay; labels show runs detected out of runs tested\n"
                 "(a zero-height bar labelled 0/N means never detected)")
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)
