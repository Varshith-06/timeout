"""Generate the figures used by the benchmark report and timeout.pdf.

Reads the built clip's recommendations.json and the model-metric JSONs, and
writes self-contained PNG charts into out/benchmark/. Pure matplotlib, no network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "benchmark"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1b2733"; ACCENT = "#e2571e"; BASE = "#9aa7b4"; GRID = "#e6ebf0"


def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)


def submodel_brier():
    m = json.loads((ROOT / "out/real/real_full_metrics.json").read_text())["brier"]
    names = ["shot", "pass", "drive"]
    model = [m[n]["model"] for n in names]
    base = [m[n]["baseline"] for n in names]
    x = np.arange(len(names)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=150)
    _style(ax)
    ax.bar(x - w/2, base, w, label="baseline (base rate)", color=BASE)
    ax.bar(x + w/2, model, w, label="model", color=ACCENT)
    for xi, v in zip(x - w/2, base): ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    for xi, v in zip(x + w/2, model): ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=8, color=ACCENT)
    ax.set_xticks(x); ax.set_xticklabels([n.title() for n in names])
    ax.set_ylabel("Brier score  (lower = better)")
    ax.set_title("Sub-model calibration on real 2015-16 data", color=INK, fontsize=11, weight="bold")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "submodel_brier.png"); plt.close(fig)


def coverage_timeline(built="out/webapp_heatnets_fixed"):
    data = json.loads((ROOT / built / "recommendations.json").read_text())
    recs = data["recommendations"]; dur = data["duration"]
    fig, ax = plt.subplots(figsize=(6.6, 1.7), dpi=150)
    _style(ax); ax.yaxis.grid(False)
    for r in recs:
        t = r["video_time"]
        shown = bool(r.get("actions"))
        ax.axvspan(t - 0.25, t + 0.25, color=ACCENT if shown else GRID, lw=0)
    ax.set_xlim(0, dur); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_xlabel("video time (s)")
    ax.set_title("Recommendation coverage across the clip "
                 "(orange = live play drawn, grey = withheld/uncalibrated)",
                 color=INK, fontsize=9.5, weight="bold")
    fig.tight_layout(); fig.savefig(OUT / "coverage_timeline.png"); plt.close(fig)


def action_mix(built="out/webapp_heatnets_fixed"):
    from collections import Counter
    data = json.loads((ROOT / built / "recommendations.json").read_text())
    shown = [r for r in data["recommendations"] if r.get("actions")]
    mix = Counter(r["actions"][0]["action"] for r in shown)
    labels = [k for k, _ in mix.most_common()]
    vals = [mix[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=150)
    _style(ax); ax.xaxis.grid(True, color=GRID); ax.yaxis.grid(False)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=ACCENT)
    for yi, v in zip(y, vals): ax.text(v + 0.5, yi, str(v), va="center", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([l.replace("_", " ").title() for l in labels])
    ax.set_xlabel("times ranked #1")
    ax.set_title("Top-ranked action mix on the clip", color=INK, fontsize=11, weight="bold")
    fig.tight_layout(); fig.savefig(OUT / "action_mix.png"); plt.close(fig)


def pipeline():
    fig, ax = plt.subplots(figsize=(6.8, 2.5), dpi=150)
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    boxes = [
        (0.2, "Broadcast\nvideo"), (2.1, "Perception\n(detect, track,\ncalibrate)"),
        (4.2, "State\n(10 players +\nball, in feet)"), (6.3, "Value model\n(rank actions\nby EPV)"),
        (8.4, "Overlay +\nrationale\non pause"),
    ]
    for x, label in boxes:
        c = ACCENT if label.startswith("State") else INK
        ax.add_patch(plt.Rectangle((x, 1.3), 1.5, 1.5, fill=False, ec=c, lw=1.6, joinstyle="round"))
        ax.text(x + 0.75, 2.05, label, ha="center", va="center", fontsize=8.2, color=c)
    for x in (1.75, 3.85, 5.95, 8.05):
        ax.annotate("", xy=(x + 0.32, 2.05), xytext=(x, 2.05),
                    arrowprops=dict(arrowstyle="-|>", color=BASE, lw=1.6))
    ax.text(4.95, 0.6, "one State schema — identical from broadcast CV or SportVU tracking",
            ha="center", fontsize=7.5, color=BASE, style="italic")
    fig.tight_layout(); fig.savefig(OUT / "pipeline.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    submodel_brier(); coverage_timeline(); action_mix(); pipeline()
    print("charts ->", OUT)
