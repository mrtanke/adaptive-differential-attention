import json, math
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

variants = ["standard", "differential", "headwise", "tokenwise", "token_headwise"]
seeds = [20260920, 20260921, 20260922]
data = {}
for variant in variants:
    for seed in seeds:
        path = Path(f"results/experiment2_dense_seed{seed}_{variant}_final.json")
        if not path.exists(): raise FileNotFoundError(path)
        data[variant, seed] = json.loads(path.read_text())["scores"]

counts = [30, 40, 50, 60, 70, 80, 90, 100, 120]
summary = {}
for variant in variants:
    levels = {}
    for n in counts:
        values = []
        for seed in seeds:
            bucket = [r["accuracy"] for r in data[variant, seed] if r["distractors"] == n]
            values.append(sum(bucket) / len(bucket))
        mean = sum(values) / len(values)
        std = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))
        levels[str(n)] = {"mean_accuracy": mean, "std_accuracy": std, "seed_accuracies": values}
    summary[variant] = levels
Path("results/experiment2_dense_aggregate.json").write_text(json.dumps({"seeds": seeds, "by_variant": summary}, indent=2) + "\n")

plt.figure(figsize=(8, 5))
for variant in variants:
    means = [summary[variant][str(n)]["mean_accuracy"] for n in counts]
    stds = [summary[variant][str(n)]["std_accuracy"] for n in counts]
    plt.plot(counts, means, marker="o", label=variant.replace("_", "+"))
    plt.fill_between(counts, [m-s for m,s in zip(means,stds)], [m+s for m,s in zip(means,stds)], alpha=.14)
plt.ylim(-.03, 1.03); plt.xlabel("Irrelevant distractors"); plt.ylabel("Exact-match retrieval accuracy")
plt.title("Experiment 2: dense transition region"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
Path("figures").mkdir(exist_ok=True); plt.savefig("figures/experiment2_dense_accuracy_vs_distractors.png", dpi=180); plt.close()

lines = ["# Experiment 2: Dense Retrieval Transition", "", "Exact match macro-averaged across early/middle/late depth; mean ± sample standard deviation across three seeds.", "", "| Variant | " + " | ".join(map(str, counts)) + " |", "|---|" + "---:|" * len(counts)]
for variant in variants:
    lines.append("| " + variant + " | " + " | ".join(f"{summary[variant][str(n)]['mean_accuracy']:.3f} ± {summary[variant][str(n)]['std_accuracy']:.3f}" for n in counts) + " |")
Path("results/experiment2_dense_aggregate.md").write_text("\n".join(lines) + "\n")
