#!/usr/bin/env python3
"""Plot QPS vs Concurrency and Latency (p50) vs Concurrency for all configs."""

import matplotlib.pyplot as plt
import numpy as np

# Data from eval results (c=1,2,4,8)
concurrency = [1, 2, 4, 8]

configs = {
    "FP16":   {"qps": [1.102, 1.153, 1.330, 2.594], "p50": [0.908, 0.952, 2.949, 3.035]},
    "INT8":   {"qps": [0.960, 0.716, 0.189, 0.325], "p50": [1.021, 2.513, 9.167, 24.605]},
    "AWQ4":   {"qps": [1.847, 1.469, 1.533, 3.057], "p50": [0.534, 0.557, 2.568, 2.572]},
    "GPT-4o": {"qps": [1.208, 1.896, 4.100, 7.026], "p50": [0.848, 0.862, 0.827, 0.947]},
}

colors = {"FP16": "#2196F3", "INT8": "#FF9800", "AWQ4": "#4CAF50", "GPT-4o": "#9C27B0"}
markers = {"FP16": "s", "INT8": "D", "AWQ4": "o", "GPT-4o": "^"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ── Plot 1: QPS vs Concurrency ──
for name, data in configs.items():
    ax1.plot(concurrency, data["qps"], marker=markers[name], color=colors[name],
             linewidth=2.2, markersize=8, label=name)

ax1.set_xlabel("Concurrency", fontsize=12)
ax1.set_ylabel("Throughput (QPS)", fontsize=12)
ax1.set_title("Throughput vs Concurrency", fontsize=14, fontweight="bold")
ax1.set_xticks(concurrency)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

# ── Plot 2: Latency p50 vs Concurrency ──
for name, data in configs.items():
    ax2.plot(concurrency, data["p50"], marker=markers[name], color=colors[name],
             linewidth=2.2, markersize=8, label=name)

ax2.set_xlabel("Concurrency", fontsize=12)
ax2.set_ylabel("Latency p50 (seconds)", fontsize=12)
ax2.set_title("Latency (p50) vs Concurrency", fontsize=14, fontweight="bold")
ax2.set_xticks(concurrency)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig("eval_results/throughput_latency_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: eval_results/throughput_latency_comparison.png")
plt.close()
