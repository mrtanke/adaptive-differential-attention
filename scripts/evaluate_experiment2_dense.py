"""Evaluation-only dense-grid probe for existing Experiment 2 final checkpoints."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, json
import torch
from src.data.synthetic_niah import make_batch
from src.models import DecoderOnlyTransformer, ModelConfig

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
run = json.load(open(args.config))
base = json.load(open(Path(args.config).parent / run["base_config"]))
cfg = {**base, **run}
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for dense Experiment 2 evaluation.")
device = torch.device("cuda")
checkpoint_path = Path(f"checkpoints/{cfg['run_name']}_final.pt")
if not checkpoint_path.exists():
    raise FileNotFoundError(checkpoint_path)
model = DecoderOnlyTransformer(ModelConfig(**base["model"], attention_type=cfg["attention_type"])).to(device).eval()
model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model"], strict=True)

counts = [30, 40, 50, 60, 70, 80, 90, 100, 120]
cases = base["pilot"]["final_cases_per_condition"]
budget = base["evaluation"]["max_attention_elements_per_batch"]
forward_microbatch = 64
rows = []
with torch.inference_mode():
    for n in counts:
        context_tokens = 4 * (n + 1) + 3
        chunk_size = min(cases, max(1, budget // context_tokens ** 2))
        for depth_i, depth in enumerate(base["evaluation"]["depths"]):
            correct = total_loss = done = 0
            # Matches the full-run final evaluation seed construction exactly.
            while done < cases:
                count = min(chunk_size, cases - done)
                seed = cfg["seed"] + 9000000 + n * 100000 + depth_i * 1000 + done
                # Generate the exact logical batch used by the full-run protocol,
                # then forward it in safe slices. This changes neither prompts nor
                # targets, and avoids holding a 505 x 487 x 8192 logits tensor.
                x_cpu, y_cpu = make_batch(count, n, depth, seed, torch.device("cpu"))
                for start in range(0, count, forward_microbatch):
                    x = x_cpu[start:start + forward_microbatch].to(device)
                    y = y_cpu[start:start + forward_microbatch].to(device)
                    logits = model(x)[:, -1]
                    correct += int((logits.argmax(-1) == y).sum())
                    total_loss += float(torch.nn.functional.cross_entropy(logits, y, reduction="sum"))
                done += count
            rows.append({"distractors": n, "context_tokens": context_tokens, "needle_depth": depth,
                         "accuracy": correct / cases, "loss": total_loss / cases, "cases": cases})

result = {"run_name": cfg["run_name"], "source_checkpoint": str(checkpoint_path),
          "protocol": "experiment2_full_final_evaluation_generation", "distractor_counts": counts,
          "scores": rows}
Path("results").mkdir(exist_ok=True)
output = Path(f"results/experiment2_dense_{cfg['run_name'].removeprefix('experiment2_')}_final.json")
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
