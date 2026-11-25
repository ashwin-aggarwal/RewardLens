"""Experiment 2 confirmatory evaluation figure. Reads the recorded result; evaluates nothing.

Usage: python -m scripts.exp2_visualize_final_eval
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from scripts.exp2_visualize_phase6 import MARK, SEED_COLORS, SEEDS, _save, _style

RESULT = Path("results/exp2/final_eval/final_eval.json")
HACK, GOOD, EDIT = "#DC2626", "#16A34A", "#111827"


def render_final_eval():
    r = json.load(open(RESULT))
    crit = r["repair_criterion"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    fig.patch.set_facecolor("white")

    # Panel 1: outcome shares on the reserved layouts, per seed.
    ax = axes[0]
    _style(ax, "Reserved layouts: what each network does")
    groups = [("Hacking\n(unedited)", lambda s: [r["baselines"]["per_seed"][s]["hacking"]]),
              ("Hacking + rank-8 edit\n(3 runs per seed)", lambda s: [m["final_eval"] for m in r["models"] if str(m["dqn_seed"]) == s]),
              ("Good\n(reference)", lambda s: [r["baselines"]["per_seed"][s]["good"]])]
    for gi, (label, rows) in enumerate(groups):
        for si, s in enumerate(SEEDS):
            vals = rows(s)
            x = gi + (si - 1) * 0.26
            direct = np.mean([v["direct_goal"] for v in vals])
            farm = np.mean([v["farming"] for v in vals])
            ax.bar(x, direct, 0.24, color=SEED_COLORS[s], edgecolor="white")
            ax.bar(x, farm, 0.24, bottom=direct, color=SEED_COLORS[s], alpha=0.3, hatch="///", edgecolor="white")
    ax.set_xticks(range(3), [g[0] for g in groups])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Share of 1,500 reserved layouts")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#6B7280", label="Reaches the goal directly"),
                       plt.Rectangle((0, 0), 1, 1, facecolor="#6B7280", alpha=0.3, hatch="///", edgecolor="white", label="Farms the reward tile")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=9)

    # Panels 2 and 3: development reporting half against the reserved set, model by model.
    for ax, key, bound, title in ((axes[1], "direct_goal", crit["min_direct_goal"], "Direct goal rate"),
                                  (axes[2], "path_efficiency_direct", crit["min_path_efficiency"], "Path efficiency")):
        _style(ax, f"{title}: development vs reserved")
        for m in r["models"]:
            s = str(m["dqn_seed"])
            d, f = m["dev_report_750"][key], m["final_eval"][key]
            jitter = (m["opt_seed"] - 1) * 0.05
            ax.plot([0 + jitter, 1 + jitter], [d, f], color=SEED_COLORS[s], lw=1.3, alpha=0.7, zorder=3)
            ax.scatter([0 + jitter, 1 + jitter], [d, f], color=SEED_COLORS[s], marker=MARK[s], s=48, edgecolor="white", lw=0.6, zorder=4)
        ax.axhline(bound, color="#6B7280", ls="--", lw=1.2, zorder=2)
        ax.text(1.38, bound + 0.002, f"criterion {bound:.2f}", ha="right", va="bottom", fontsize=9, color="#6B7280")
        ax.set_xticks([0, 1], ["Development\n(750 reporting layouts)", "Reserved\n(1,500 layouts, run once)"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylim(0.88, 1.005)
    axes[1].legend(handles=[Line2D([], [], marker=MARK[s], ls="-", color=SEED_COLORS[s], ms=8, label=f"DQN seed {s}") for s in SEEDS],
                   loc="upper center", bbox_to_anchor=(1.1, -0.2), ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"Confirmatory evaluation: {r['n_meeting_criterion']} of {r['n_models']} rank-8 edits meet the repair criterion on layouts never used before",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, "exp2_final_eval.png")


if __name__ == "__main__":
    render_final_eval()
