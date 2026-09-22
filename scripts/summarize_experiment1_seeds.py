import json
import statistics
from pathlib import Path
import matplotlib.pyplot as plt

names = {
    "standard": "Standard",
    "differential": "Diff V1",
    "headwise": "Head-wise Diff",
    "tokenwise": "Token-wise Diff",
    "token_headwise": "Token + Head-wise Diff",
}
rows = []
for label, title in names.items():
    files = ["results/experiment1_" + label + "_final.json"]
    files += ["results/experiment1_seed" + str(seed) + "_" + label + "_final.json" for seed in (20260921, 20260922)]
    data = [json.load(open(path)) for path in files]
    metrics = {}
    for key in ("best_validation_loss", "test_loss", "test_ppl", "training_seconds"):
        values = [item[key] for item in data]
        metrics[key] = {"mean": statistics.mean(values), "std": statistics.stdev(values), "values": values}
    rows.append({"variant": title, "files": files, "metrics": metrics})

Path("results").mkdir(exist_ok=True)
Path("results/experiment1_three_seed_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
lines = [
    "| Variant | Best validation loss | Test loss | Test PPL | Training time |",
    "|---|---:|---:|---:|---:|",
]
for row in rows:
    metrics = row["metrics"]
    def fmt(key):
        return "{:.4f} +/- {:.4f}".format(metrics[key]["mean"], metrics[key]["std"])
    lines.append(
        "| {} | {} | {} | {} | {:.2f} +/- {:.2f} min |".format(
            row["variant"], fmt("best_validation_loss"), fmt("test_loss"),
            fmt("test_ppl"), metrics["training_seconds"]["mean"] / 60,
            metrics["training_seconds"]["std"] / 60,
        )
    )
Path("results/experiment1_three_seed_summary.md").write_text("\n".join(lines) + "\n")

plt.figure(figsize=(8, 4.5))
x = range(len(rows))
means = [row["metrics"]["test_ppl"]["mean"] for row in rows]
stds = [row["metrics"]["test_ppl"]["std"] for row in rows]
plt.errorbar(x, means, yerr=stds, fmt="o", capsize=5)
plt.xticks(x, [row["variant"] for row in rows], rotation=20, ha="right")
plt.ylabel("Held-out test perplexity")
plt.title("Experiment 1: three-seed mean +/- std")
plt.tight_layout()
Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/experiment1_three_seed_test_ppl.png", dpi=180)
print("\n".join(lines))
