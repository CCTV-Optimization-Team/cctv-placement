"""
Runs every algorithm on the stub score (tiny map, plain-square visibility) and
writes: results/stub_demo.csv, results/stub_convergence.csv, results/stub_convergence.png
Swap build_stub() for the real masks/<name>.npz to produce the real thing.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from algorithms import ALGORITHMS, run, greedy_cost, results_table, save_history_csv
from stub_score import build_stub, stub_weights

K, SEEDS, MULT = 4, range(20), 5
os.makedirs("results", exist_ok=True)
grid, cands, masks = build_stub(half_width=3, stride=2)
budget = MULT * greedy_cost(len(cands), K)

rows, hist = [], {}
for wname in ("uniform", "hotspot"):
    w = stub_weights(grid, wname)
    for a in ALGORITHMS:
        seeds = SEEDS if a not in ("greedy",) else [0]       # greedy is deterministic
        for s in seeds:
            r = run(a, K=K, weights=w, budget=budget, seed=s, masks=masks,
                    map_name="tiny_stub", weighting=wname)
            rows.append({**r.row(), "raw_coverage": r.raw_coverage})
            if wname == "hotspot":
                hist.setdefault(a, []).append(r.history)

df = pd.DataFrame(rows)
df.to_csv("results/stub_demo.csv", index=False)
print(df.groupby(["weighting", "algorithm"])[["score", "raw_coverage", "runtime", "evaluations"]]
        .mean().round(4).to_string())

# convergence data (mean best score vs evaluations, hotspot weights)
conv = pd.DataFrame({a: np.mean(h, axis=0) for a, h in hist.items()}).rename_axis("evaluation")
conv.to_csv("results/stub_convergence.csv")
fig, ax = plt.subplots(figsize=(7, 4))
for a in conv.columns:
    ax.plot(conv.index, conv[a], label=a)
ax.set_xscale("log"); ax.set_xlabel("score evaluations"); ax.set_ylabel("best weighted coverage")
ax.set_title(f"Convergence on stub score (K={K}, mean of {len(list(SEEDS))} seeds)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("results/stub_convergence.png", dpi=150)
print("\nwrote results/stub_demo.csv, stub_convergence.csv, stub_convergence.png")
