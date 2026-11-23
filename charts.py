"""charts.py (Person 4) - figures for the report, built from the runner's output.

Usage (from the repo root):
    python3 charts.py --tag quick              # figures from results/results_quick.csv
    python3 charts.py --tag full --maps tiny college

Reads  results/results_<tag>.csv and results/details_<tag>.json  (written by runner.py)
Writes results/figures_<tag>/*.png

Reminder: the `score` column is always the TRUE-weighted coverage of the final placement,
whatever weighting the optimiser used. `weighting` says what the optimiser was shown.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import interfaces as I
from map import build_map

ALGO_COLORS = {"random": "#888888", "greedy": "#1f77b4", "hill_climbing": "#2ca02c",
               "hill_climbing_restarts": "#ff7f0e", "annealing": "#d62728"}
WEIGHT_COLORS = {"uniform": "#888888", "predicted": "#1f77b4", "true": "#d62728"}
SCORE_LABEL = "Weighted coverage (true priority)"


# ------------------------------------------------------------------ loading
def load_results(tag, results_dir=None):
    d = Path(results_dir or config.RESULTS_DIR)
    df = pd.read_csv(d / f"results_{tag}.csv")
    details_path = d / f"details_{tag}.json"
    details = json.loads(details_path.read_text()) if details_path.exists() else {}
    return df, details


def _key(algo, map_name, K, weighting, seed):
    return f"{algo}|{map_name}|{K}|{weighting}|{seed}"


def _color(algo):
    return ALGO_COLORS.get(algo)


def _mean_std(d, by):
    """Mean and std of score per `by`. Across maps if several, otherwise across seeds."""
    if d["map"].nunique() > 1:  # first average the seeds of each map, then compare maps
        d = d.groupby(by + ["map"], as_index=False)["score"].mean()
    g = d.groupby(by)["score"].agg(["mean", "std"]).fillna(0.0).reset_index()
    return g


def best_algorithm(df, map_name, K, weighting="true"):
    d = df[(df["map"] == map_name) & (df["K"] == K) & (df["weighting"] == weighting)]
    return d.groupby("algorithm")["score"].mean().idxmax()


# ------------------------------------------------------------------ charts
def plot_priority_maps(map_name):
    """True vs predicted priority map side by side."""
    g = build_map(map_name)
    maps = [("True priority", I.true_priority(g)), ("Predicted priority", I.predict_priority(g))]
    vmax = max(m.max() for _, m in maps)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (title, w) in zip(axes, maps):
        ax.set_facecolor("#444444")
        im = ax.imshow(np.ma.masked_where(g.grid == 1, w), cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_title(f"{map_name}: {title}", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8, label="priority weight")
    return fig


def plot_score_vs_k(df, map_name=None, weighting="true", algorithms=None):
    """Mean true-weighted coverage against number of cameras, one line per algorithm."""
    d = df[df["weighting"] == weighting]
    if map_name:
        d = d[d["map"] == map_name]
    algorithms = algorithms or [a for a in config.ALGORITHMS if a in set(d["algorithm"])]
    g = _mean_std(d, ["algorithm", "K"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a in algorithms:
        s = g[g["algorithm"] == a].sort_values("K")
        ax.plot(s["K"], s["mean"], marker="o", label=a, color=_color(a))
        ax.fill_between(s["K"], s["mean"] - s["std"], s["mean"] + s["std"], alpha=0.15, color=_color(a))
    ax.set_xlabel("Number of cameras (K)")
    ax.set_ylabel(SCORE_LABEL)
    ax.set_xticks(sorted(d["K"].unique()))
    where = map_name or f"mean of {d['map'].nunique()} maps"
    ax.set_title(f"Coverage vs K ({where}; optimiser used {weighting} weights)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_weighting_vs_k(df, algorithm, map_name=None):
    """Does knowing the priority help? One line per weighting for one algorithm."""
    d = df[df["algorithm"] == algorithm]
    if map_name:
        d = d[d["map"] == map_name]
    g = _mean_std(d, ["weighting", "K"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for w in config.WEIGHTINGS:
        s = g[g["weighting"] == w].sort_values("K")
        if len(s):
            ax.plot(s["K"], s["mean"], marker="o", label=w, color=WEIGHT_COLORS[w])
            ax.fill_between(s["K"], s["mean"] - s["std"], s["mean"] + s["std"], alpha=0.15, color=WEIGHT_COLORS[w])
    ax.set_xlabel("Number of cameras (K)")
    ax.set_ylabel(SCORE_LABEL)
    ax.set_xticks(sorted(d["K"].unique()))
    where = map_name or f"mean of {d['map'].nunique()} maps"
    ax.set_title(f"{algorithm}: effect of the weighting used by the optimiser ({where})", fontsize=10)
    ax.legend(title="optimiser weights", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_weighting_bars(df, K=None, map_name=None):
    """Grouped bars: algorithm x weighting at one K."""
    K = K if K is not None else int(sorted(df["K"].unique())[len(df["K"].unique()) // 2])
    d = df[df["K"] == K]
    if map_name:
        d = d[d["map"] == map_name]
    g = _mean_std(d, ["algorithm", "weighting"])
    algos = [a for a in config.ALGORITHMS if a in set(g["algorithm"])]
    weights = [w for w in config.WEIGHTINGS if w in set(g["weighting"])]
    width = 0.8 / len(weights)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j, w in enumerate(weights):
        vals = [g[(g["algorithm"] == a) & (g["weighting"] == w)]["mean"].iloc[0] for a in algos]
        errs = [g[(g["algorithm"] == a) & (g["weighting"] == w)]["std"].iloc[0] for a in algos]
        ax.bar(np.arange(len(algos)) + j * width, vals, width, yerr=errs, capsize=2,
               label=w, color=WEIGHT_COLORS[w])
    ax.set_xticks(np.arange(len(algos)) + width * (len(weights) - 1) / 2)
    ax.set_xticklabels(algos, fontsize=8)
    ax.set_ylabel(SCORE_LABEL)
    lo = g["mean"].min()
    ax.set_ylim(max(0, lo - 0.1), min(1.0, g["mean"].max() + 0.08))
    where = map_name or f"mean of {d['map'].nunique()} maps"
    ax.set_title(f"Uniform vs predicted vs true weights, K={K} ({where})", fontsize=10)
    ax.legend(title="optimiser weights", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_convergence(details, map_name, K, weighting="true", algorithms=None):
    """Best score found so far against evaluations (in the optimiser's own weighting)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    algorithms = algorithms or config.ALGORITHMS
    for a in algorithms:
        curves, n_max = [], 0
        for key, v in details.items():
            alg, m, k, w, _ = key.split("|")
            if (alg, m, int(k), w) == (a, map_name, K, weighting):
                curves.append(v["history"])
                n_max = max(n_max, v["history"]["n"])
        if not curves:
            continue
        x = np.arange(n_max)
        ys = np.array([np.interp(x, h["idx"], h["best"]) for h in curves])
        mean, std = ys.mean(0), ys.std(0)
        ax.plot(x, mean, label=a, color=_color(a))
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=_color(a))
    ax.set_xlabel("Score evaluations")
    ax.set_ylabel(f"Best score so far ({weighting} weights)")
    ax.set_title(f"Convergence on {map_name}, K={K}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_placements(df, details, map_name, K, algorithm=None):
    """Chosen cameras on top of the true priority map, one panel per weighting."""
    g = build_map(map_name)
    true_w = I.true_priority(g)
    algorithm = algorithm or best_algorithm(df, map_name, K)
    candidates, masks = I.build_candidates(g)
    index = {tuple(c): i for i, c in enumerate(candidates)}
    weights = [w for w in config.WEIGHTINGS if w in set(df["weighting"])]
    fig, axes = plt.subplots(1, len(weights), figsize=(5 * len(weights), 5.2), squeeze=False)
    for ax, w in zip(axes[0], weights):
        rows = df[(df["algorithm"] == algorithm) & (df["map"] == map_name) &
                  (df["K"] == K) & (df["weighting"] == w)]
        ax.set_facecolor("#444444")
        ax.imshow(np.ma.masked_where(g.grid == 1, true_w), cmap="YlOrRd", vmin=0, vmax=true_w.max())
        if rows.empty:
            ax.set_title(f"{w}: no runs", fontsize=9)
            continue
        best = rows.loc[rows["score"].idxmax()]
        placement = details.get(_key(algorithm, map_name, K, w, int(best["seed"])), {}).get("placement", [])
        idx = [index[tuple(p)] for p in placement if tuple(p) in index]
        if idx:
            covered = masks[idx].any(axis=0)
            faded = np.ma.masked_where(covered | (g.grid == 1), np.ones(covered.shape))
            ax.imshow(faded, cmap="gray_r", alpha=0.0, vmin=0, vmax=1)          # keeps axes identical
            ax.imshow(faded, cmap=matplotlib.colors.ListedColormap(["white"]), alpha=0.6, vmin=0, vmax=1)
        for r, c, d in placement:
            dr, dc = config.DIRECTION_VECTORS[d]
            ax.scatter([c], [r], s=70, c="black", marker="o", zorder=3)
            ax.annotate("", xy=(c + 2.0 * dc, r + 2.0 * dr), xytext=(c, r),
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6), zorder=3)
        ax.set_title(f"{w} weights, score {best['score']:.3f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{algorithm} on {map_name}, K={K}: cameras over the TRUE priority map "
                 f"(faded cells = not covered)", fontsize=11)
    fig.tight_layout()
    return fig


def plot_runtime(df):
    """Mean runtime per algorithm (log scale)."""
    g = df.groupby("algorithm")["runtime"].mean()
    algos = [a for a in config.ALGORITHMS if a in g.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(algos, [g[a] for a in algos], color=[_color(a) for a in algos])
    ax.set_yscale("log")
    ax.set_ylabel("Mean runtime per run (s, log scale)")
    ax.set_title("Runtime per algorithm", fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ all
def make_all(tag, maps=None, out_dir=None, results_dir=None):
    df, details = load_results(tag, results_dir)
    out = Path(out_dir or Path(results_dir or config.RESULTS_DIR) / f"figures_{tag}")
    out.mkdir(parents=True, exist_ok=True)
    maps = maps or sorted(df["map"].unique(), key=lambda m: config.MAP_NAMES.index(m) if m in config.MAP_NAMES else 99)
    kmax = int(df["K"].max())
    kmid = int(sorted(df["K"].unique())[len(df["K"].unique()) // 2])
    saved = []

    def save(fig, name):
        path = out / name
        fig.savefig(path, dpi=130)
        plt.close(fig)
        saved.append(path)

    save(plot_runtime(df), "runtime.png")
    save(plot_weighting_bars(df, K=kmid), f"weighting_bars_all_K{kmid}.png")
    save(plot_score_vs_k(df), "score_vs_k_all.png")
    top = best_algorithm(df, maps[0], kmax) if maps else None
    for algo in ([top] if top else []):
        save(plot_weighting_vs_k(df, algo), f"weighting_vs_k_all_{algo}.png")
    for m in maps:
        save(plot_priority_maps(m), f"priority_{m}.png")
        save(plot_score_vs_k(df, map_name=m), f"score_vs_k_{m}.png")
        save(plot_convergence(details, m, kmax), f"convergence_{m}_K{kmax}.png")
        save(plot_placements(df, details, m, kmax), f"placements_{m}_K{kmax}.png")
    return saved


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="run")
    p.add_argument("--maps", nargs="+")
    p.add_argument("--out")
    a = p.parse_args()
    for path in make_all(a.tag, a.maps, a.out):
        print("saved", path)


if __name__ == "__main__":
    main()
