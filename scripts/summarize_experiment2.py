import json, math
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

variants = ["standard", "differential", "headwise", "tokenwise", "token_headwise"]
seeds = [20260920, 20260921, 20260922]
raw = {}
for variant in variants:
    for seed in seeds:
        path = Path(f"results/experiment2_seed{seed}_{variant}_final.json")
        if not path.exists():
            raise FileNotFoundError(path)
        raw[(variant, seed)] = json.loads(path.read_text())

by_variant = {}
for variant in variants:
    levels = defaultdict(list)
    per_seed = {}
    for seed in seeds:
        rows = raw[(variant, seed)]["final_scores"]
        grouped = defaultdict(list)
        for row in rows: grouped[row["distractors"]].append(row["accuracy"])
        per_seed[str(seed)] = {str(n): sum(v) / len(v) for n, v in grouped.items()}
        for n, values in grouped.items(): levels[n].append(sum(values) / len(values))
    summary = {}
    for n, values in sorted(levels.items()):
        mean = sum(values) / len(values)
        std = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))
        summary[str(n)] = {"mean_accuracy": mean, "std_accuracy": std, "seed_accuracies": values}
    by_variant[variant] = {"per_seed": per_seed, "by_distractor": summary}

result = {"protocol": "synthetic_niah_answer_only", "seeds": seeds, "variants": by_variant}
Path("results/experiment2_aggregate.json").write_text(json.dumps(result, indent=2) + "\n")

counts = sorted(int(x) for x in by_variant["standard"]["by_distractor"])
plt.figure(figsize=(8, 5))
for variant in variants:
    stats = by_variant[variant]["by_distractor"]
    means = [stats[str(n)]["mean_accuracy"] for n in counts]
    stds = [stats[str(n)]["std_accuracy"] for n in counts]
    plt.plot(counts, means, marker="o", label=variant.replace("_", "+"))
    plt.fill_between(counts, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], alpha=.14)
plt.xscale("symlog", linthresh=4); plt.ylim(-.03, 1.03)
plt.xlabel("Irrelevant distractors"); plt.ylabel("Exact-match retrieval accuracy")
plt.title("Experiment 2: retrieval robustness"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
Path("figures").mkdir(exist_ok=True); plt.savefig("figures/experiment2_accuracy_vs_distractors.png", dpi=180); plt.close()

lines = ["# Experiment 2: Retrieval Robustness", "", "Accuracy is macro-averaged over needle-depth buckets; values are mean ± sample standard deviation across three seeds.", "", "| Variant | " + " | ".join(str(n) for n in counts) + " |", "|---|" + "---:|" * len(counts)]
for variant in variants:
    stats = by_variant[variant]["by_distractor"]
    values = [f"{stats[str(n)]['mean_accuracy']:.3f} ± {stats[str(n)]['std_accuracy']:.3f}" for n in counts]
    lines.append("| " + variant + " | " + " | ".join(values) + " |")
Path("results/experiment2_aggregate.md").write_text("\n".join(lines) + "\n")
