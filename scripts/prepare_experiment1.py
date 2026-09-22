import argparse,json
from pathlib import Path
from tokenizers import Tokenizer,models,pre_tokenizers,trainers
from src.data.tinystories_lm import cached_splits,split_metadata
p=argparse.ArgumentParser();p.add_argument("--config",default="configs/experiment1_base.json");p.add_argument("--tokenizer-stories",type=int,default=100000);a=p.parse_args();c=json.load(open(a.config));d=c["data"]
train,_,_=cached_splits(d["cache_dir"],d["validation_fraction"],d["split_seed"]);Path("artifacts").mkdir(exist_ok=True)
t=Tokenizer(models.BPE(unk_token="<unk>"));t.pre_tokenizer=pre_tokenizers.ByteLevel();t.train_from_iterator((train[i]["text"] for i in range(min(a.tokenizer_stories,len(train)))),trainers.BpeTrainer(vocab_size=d["vocab_size"],special_tokens=["<unk>","<pad>"]));t.save(d["tokenizer_path"])
Path("artifacts/experiment1_splits.json").write_text(json.dumps(split_metadata(d["cache_dir"],d["validation_fraction"],d["split_seed"]),indent=2)+"\n")
print(Path("artifacts/experiment1_splits.json").read_text())
