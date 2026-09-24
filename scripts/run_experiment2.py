import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, json, math, time
import torch
from src.data.synthetic_niah import make_batch
from src.models import DecoderOnlyTransformer, ModelConfig

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--pilot", action="store_true")
a = p.parse_args()
run = json.load(open(a.config))
base = json.load(open(Path(a.config).parent / run["base_config"]))
cfg = {**base, **run}
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for Experiment 2.")
torch.manual_seed(cfg["seed"])
device = torch.device("cuda")
training, evaluation = base["training"], base["evaluation"]
steps = base["pilot"]["max_steps"] if a.pilot else training["max_steps"]
cases = base["pilot"]["eval_cases_per_condition"] if a.pilot else evaluation["training_eval_cases_per_condition"]
model = DecoderOnlyTransformer(ModelConfig(**base["model"], attention_type=cfg["attention_type"])).to(device)
initial_checkpoint = cfg.get("initial_checkpoint")
if not initial_checkpoint:
    raise ValueError("Experiment 2 runs must set initial_checkpoint to the matching Experiment 1 checkpoint.")
state = torch.load(initial_checkpoint, map_location=device, weights_only=False)
model.load_state_dict(state["model"], strict=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])

def evaluate(cases_per_condition, offset):
    model.eval(); rows = []
    with torch.inference_mode():
        for n in evaluation["distractor_counts"]:
            depths = ["early"] if n == 0 else evaluation["depths"]
            for depth_i, depth in enumerate(depths):
                total_correct = total_loss = done = 0
                context_tokens = 4 * (n + 1) + 3
                chunk_size = min(cases_per_condition, max(1, evaluation["max_attention_elements_per_batch"] // context_tokens ** 2))
                while done < cases_per_condition:
                    count = min(chunk_size, cases_per_condition - done)
                    x, y = make_batch(count, n, depth, cfg["seed"] + offset + n * 100000 + depth_i * 1000 + done, device)
                    logits = model(x)[:, -1]
                    total_correct += int((logits.argmax(-1) == y).sum())
                    total_loss += float(torch.nn.functional.cross_entropy(logits, y, reduction="sum"))
                    done += count
                rows.append({"distractors": n, "context_tokens": context_tokens, "needle_depth": depth,
                             "accuracy": total_correct / cases_per_condition, "loss": total_loss / cases_per_condition,
                             "cases": cases_per_condition})
    model.train()
    return rows

Path("checkpoints").mkdir(exist_ok=True); Path("results").mkdir(exist_ok=True)
torch.cuda.reset_peak_memory_stats(); start = time.perf_counter(); curve = []
warmup = min(training["warmup_steps"], max(1, steps // 4))
for step in range(steps):
    progress = step + 1
    if step < warmup:
        lr = training["learning_rate"] * progress / warmup
    else:
        lr = training["min_learning_rate"] + .5 * (training["learning_rate"] - training["min_learning_rate"]) * (1 + math.cos(math.pi * step / steps))
    optimizer.param_groups[0]["lr"] = lr
    rng = torch.Generator().manual_seed(cfg["seed"] + step)
    n = int(torch.randint(4, 31, (1,), generator=rng))
    depths = ("early", "middle", "late")
    depth = depths[step % len(depths)]
    x, y = make_batch(training["batch_size"], n, depth, cfg["seed"] * 100000 + step, device)
    logits = model(x)[:, -1]
    loss = torch.nn.functional.cross_entropy(logits, y)
    if not torch.isfinite(loss): raise RuntimeError("non-finite loss")
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if progress % training["eval_every"] == 0 or progress == steps:
        scores = evaluate(cases, progress * 10000)
        curve.append({"step": progress, "train_loss": loss.detach().item(), "mean_accuracy": sum(r["accuracy"] for r in scores) / len(scores), "scores": scores})
        print(json.dumps(curve[-1]), flush=True)

final_scores = evaluate(base["pilot"]["final_cases_per_condition"], 9000000)
elapsed = time.perf_counter() - start
tag = "pilot" if a.pilot else "final"; name = cfg["run_name"]
checkpoint = f"checkpoints/{name}_{tag}.pt"
torch.save({"model": model.state_dict(), "config": cfg, "steps": steps}, checkpoint)
result = {"run_name": name, "pilot": a.pilot, "protocol": "synthetic_niah_answer_only", "initial_checkpoint": initial_checkpoint, "steps": steps,
          "parameter_count": sum(p.numel() for p in model.parameters()), "training_seconds": elapsed,
          "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(), "checkpoint_path": checkpoint,
          "training_curve": curve, "final_scores": final_scores}
Path(f"results/{name}_{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
