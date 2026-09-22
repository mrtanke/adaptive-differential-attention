import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse,json,math,time
import torch
from src.models import DecoderOnlyTransformer,ModelConfig
from src.data.tinystories_lm import cached_splits,encode_batch,split_metadata

p=argparse.ArgumentParser()
p.add_argument("--config",required=True)
p.add_argument("--pilot",action="store_true")
a=p.parse_args()
run=json.load(open(a.config))
base=json.load(open(Path(a.config).parent/run["base_config"]))
cfg={**base,**run}
torch.manual_seed(cfg["seed"])
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for Experiment 1 pilot/full runs, but no GPU is visible to PyTorch.")
device=torch.device("cuda")
data=cfg["data"]; train,val,test=cached_splits(data["cache_dir"],data["validation_fraction"],data["split_seed"])
model=DecoderOnlyTransformer(ModelConfig(**base["model"],attention_type=cfg["attention_type"])).to(device)
opt=torch.optim.AdamW(model.parameters(),lr=base["training"]["learning_rate"],weight_decay=base["training"]["weight_decay"])
steps=base["pilot"]["max_steps"] if a.pilot else base["training"]["max_steps"]
warmup=min(base["training"]["warmup_steps"],max(1,steps//4)) if a.pilot else base["training"]["warmup_steps"]
bs=base["training"]["batch_size"]; ctx=data["context_length"]; eval_batches=base["pilot"]["eval_batches"] if a.pilot else base["training"]["eval_batches"]
def evaluate(dataset,batches):
    model.eval(); losses=[]
    with torch.no_grad():
        for j in range(batches):
            x,y=encode_batch(dataset,data["tokenizer_path"],ctx,j*bs,bs)
            _,loss=model(x.to(device),y.to(device)); losses.append(loss.item())
    model.train(); return sum(losses)/len(losses)
torch.cuda.reset_peak_memory_stats(); Path("checkpoints").mkdir(exist_ok=True); start=time.perf_counter(); losses=[]; curve=[]; best_validation=float("inf"); best_checkpoint=None
for step in range(steps):
    if step<warmup: lr=base["training"]["learning_rate"]*(step+1)/warmup
    else: lr=base["training"]["min_learning_rate"]+.5*(base["training"]["learning_rate"]-base["training"]["min_learning_rate"])*(1+math.cos(math.pi*step/base["training"]["max_steps"]))
    opt.param_groups[0]["lr"]=lr
    x,y=encode_batch(train,data["tokenizer_path"],ctx,step*bs,bs)
    _,loss=model(x.to(device),y.to(device))
    if not torch.isfinite(loss): raise RuntimeError("non-finite loss")
    opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    if (step+1)%base["training"]["eval_every"]==0 or step+1==steps:
        current_validation=evaluate(val,eval_batches); curve.append({"step":step+1,"train_loss":loss.item(),"validation_loss":current_validation})
        if current_validation<best_validation:
            best_validation=current_validation; best_checkpoint="checkpoints/"+cfg["run_name"]+"_best.pt"; torch.save({"model":model.state_dict(),"config":cfg,"step":step+1},best_checkpoint)
validation_loss=evaluate(val,eval_batches); test_loss=evaluate(test,eval_batches)
elapsed=time.perf_counter()-start; name=cfg["run_name"]; tag="pilot" if a.pilot else "final"
Path("checkpoints").mkdir(exist_ok=True); checkpoint="checkpoints/"+name+"_"+tag+".pt"; torch.save({"model":model.state_dict(),"config":cfg},checkpoint)
lambda_stats=[]
for i,block in enumerate(model.blocks):
    attn=block.attn
    if hasattr(attn,"lambda_full"): lambda_stats.append({"layer":i,"lambda_layer":float(attn.lambda_full().detach().cpu())})
result={"run_name":name,"pilot":a.pilot,"train_loss_first":losses[0],"train_loss_last":losses[-1],"validation_loss":validation_loss,"best_validation_loss":best_validation,"best_validation_ppl":math.exp(best_validation),"training_curve":curve,"best_checkpoint_path":best_checkpoint,"test_loss":test_loss,"validation_ppl":math.exp(validation_loss),"test_ppl":math.exp(test_loss),"parameter_count":sum(p.numel() for p in model.parameters()),"training_seconds":elapsed,"tokens":steps*bs*ctx,"tokens_per_second":steps*bs*ctx/elapsed,"peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(),"checkpoint_path":checkpoint,"lambda_statistics":lambda_stats,"splits":split_metadata(data["cache_dir"],data["validation_fraction"],data["split_seed"])}
Path("results").mkdir(exist_ok=True); Path("results/"+name+"_"+tag+".json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
