#!/usr/bin/env python
"""Plot per-episode FTT distributions and ROC curves from a run_detector.py JSON.

For each (attack, suite, role) group in the file, draws:
  - a histogram of clean vs. trigger FTT values, so the separation (or overlap)
    between the two populations is visible directly, and a candidate threshold
    can be read off by eye
  - the ROC curve for that group, with the point closest to (0,1) marked as a
    candidate operating threshold (Youden's J statistic: maximizes TPR - FPR)

Works on any of this repo's detector output JSONs -- BadVLA or GoBA, any
suite, any role -- since it only reads the generic {label, ftt_used} pairs
every group's "samples" list carries, plus the group's own recorded AUROC.

Usage:
    python analysis/plot_threshold.py results/ftt_badvla_desc_only.json
    python analysis/plot_threshold.py results/ftt_goba_desc_only.json --out-dir figs/
    python analysis/plot_threshold.py results/*.json   # one figure per file
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def best_threshold(clean, trig):
    """Youden's J: the score cutoff that maximizes TPR - FPR.

    Low FTT = backdoor (this repo's fixed polarity), so "trigger" is the
    positive class and a sample is flagged when ftt_used <= threshold.
    """
    scores = np.concatenate([clean, trig])
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(trig))])
    thresholds = np.unique(scores)
    best_j, best_t = -np.inf, thresholds[0]
    for t in thresholds:
        pred_trigger = scores <= t
        tp = np.sum(pred_trigger & (labels == 1))
        fn = np.sum(~pred_trigger & (labels == 1))
        fp = np.sum(pred_trigger & (labels == 0))
        tn = np.sum(~pred_trigger & (labels == 0))
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t
    return best_t, best_j


def roc_points(clean, trig):
    """ROC curve (FPR, TPR) swept over score thresholds, low-FTT-is-positive."""
    scores = np.concatenate([clean, trig])
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(trig))])
    thresholds = np.concatenate(([-np.inf], np.sort(np.unique(scores)), [np.inf]))
    fpr, tpr = [], []
    for t in thresholds:
        pred_trigger = scores <= t
        tp = np.sum(pred_trigger & (labels == 1))
        fn = np.sum(~pred_trigger & (labels == 1))
        fp = np.sum(pred_trigger & (labels == 0))
        tn = np.sum(~pred_trigger & (labels == 0))
        tpr.append(tp / (tp + fn) if (tp + fn) else 0.0)
        fpr.append(fp / (fp + tn) if (fp + tn) else 0.0)
    return np.array(fpr), np.array(tpr)


def plot_group(ax_kde, ax_hist, ax_roc, key, group):
    clean = np.array([s["ftt_used"] for s in group["samples"] if s["label"] == 0])
    trig = np.array([s["ftt_used"] for s in group["samples"] if s["label"] == 1])
    lo, hi = min(clean.min(), trig.min()), max(clean.max(), trig.max())
    t, j = best_threshold(clean, trig)

    # Panel 1: smoothed distribution (KDE) + rug. Same information as the
    # histogram next to it, just drawn as a curve so the SHAPE (tight spike
    # vs. wide spread) is easy to see at a glance. y-axis is "density," not a
    # count -- both curves' areas are fixed to 1 regardless of n, so a
    # taller/narrower curve only means that group's scores are more tightly
    # bunched, not that it has more episodes. For "how many episodes," read
    # the histogram panel to its right -- same data, plain counts.
    pad = 0.05 * (hi - lo)
    xs = np.linspace(lo - pad, hi + pad, 400)
    for scores, label, color in ((clean, "clean", "#4C72B0"), (trig, "trigger", "#C44E52")):
        density = gaussian_kde(scores)(xs)
        ax_kde.plot(xs, density, color=color, linewidth=1.8, label=f"{label} (n={len(scores)})")
        ax_kde.fill_between(xs, density, alpha=0.25, color=color)
        ax_kde.plot(scores, np.full_like(scores, -0.02 * density.max()), "|",
                    color=color, alpha=0.6, markersize=8, clip_on=False)
    ax_kde.axvline(t, color="black", linestyle="--", linewidth=1.5, label=f"threshold={t:.4f}")
    ax_kde.set_title(f"{key}\nAUROC={group['auroc']:.4f}", fontsize=10)
    ax_kde.set_xlabel("FTT score (used)")
    ax_kde.set_ylabel("density (shape, not count)")
    ax_kde.set_ylim(bottom=0)
    ax_kde.legend(fontsize=8)

    # Panel 2: histogram. y-axis is literally "how many episodes" -- a real,
    # directly readable count, nothing normalized or smoothed. Same bins used
    # for both groups so bar heights are directly comparable.
    bins = np.linspace(lo, hi, 25)
    ax_hist.hist(clean, bins=bins, alpha=0.6, label=f"clean (n={len(clean)})", color="#4C72B0")
    ax_hist.hist(trig, bins=bins, alpha=0.6, label=f"trigger (n={len(trig)})", color="#C44E52")
    ax_hist.axvline(t, color="black", linestyle="--", linewidth=1.5, label=f"threshold={t:.4f}")
    ax_hist.set_title("same data, plain counts", fontsize=10)
    ax_hist.set_xlabel("FTT score (used)")
    ax_hist.set_ylabel("number of episodes")
    ax_hist.legend(fontsize=8)

    fpr, tpr = roc_points(clean, trig)
    ax_roc.plot(fpr, tpr, color="#4C72B0")
    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
    # mark the same Youden's-J point on the ROC curve
    pred_trigger = np.concatenate([clean, trig]) <= t
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(trig))])
    tp = np.sum(pred_trigger & (labels == 1)); fn = np.sum(~pred_trigger & (labels == 1))
    fp = np.sum(pred_trigger & (labels == 0)); tn = np.sum(~pred_trigger & (labels == 0))
    j_tpr = tp / (tp + fn) if (tp + fn) else 0.0
    j_fpr = fp / (fp + tn) if (fp + tn) else 0.0
    ax_roc.scatter([j_fpr], [j_tpr], color="black", zorder=5, s=30)
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.set_title(f"ROC (AUROC={group['auroc']:.4f})", fontsize=10)

    return t, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_files", nargs="+", help="run_detector.py output JSON(s)")
    ap.add_argument("--out-dir", default="analysis/figs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.json_files:
        data = json.load(open(path))
        # clean_baseline is the negative control (same non-backdoored model,
        # patch on vs off) -- there is no backdoor to threshold against, so
        # only plot the attack rows, where clean-vs-trigger separation is the
        # actual question.
        keys = [k for k in data if data[k]["role"] == "attack"]
        n = len(keys)
        if n == 0:
            print(f"\n{path}: no 'attack' role groups found, skipping")
            continue

        fig, axes = plt.subplots(n, 3, figsize=(15, 3.4 * n))
        if n == 1:
            axes = axes.reshape(1, 3)

        print(f"\n{'='*70}\n{path}\n{'='*70}")
        print(f"{'group':<36}{'AUROC':>8}{'threshold':>11}{'youden_J':>10}")
        for i, key in enumerate(keys):
            t, j = plot_group(axes[i, 0], axes[i, 1], axes[i, 2], key, data[key])
            print(f"{key:<36}{data[key]['auroc']:>8.4f}{t:>11.5f}{j:>10.3f}")

        fig.tight_layout()
        out_path = out_dir / (Path(path).stem + "_threshold.png")
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"\nsaved figure -> {out_path}")


if __name__ == "__main__":
    main()
