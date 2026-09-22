from pathlib import Path
import json, numpy as np
from datasets import Dataset, concatenate_datasets
from tokenizers import Tokenizer

def cached_splits(cache_dir, validation_fraction=.5, seed=20260920):
 root=Path(cache_dir)/"roneneldan___tiny_stories/default/0.0.0"
 version=next(p for p in root.iterdir() if p.is_dir())
 train=concatenate_datasets([Dataset.from_file(str(p)) for p in sorted(version.glob("tiny_stories-train-*.arrow"))])
 official_validation=Dataset.from_file(str(version/"tiny_stories-validation.arrow"))
 order=np.random.default_rng(seed).permutation(len(official_validation)); cut=int(len(order)*validation_fraction)
 return train, official_validation.select(order[:cut].tolist()), official_validation.select(order[cut:].tolist())

def split_metadata(cache_dir, validation_fraction=.5, seed=20260920):
 train,val,test=cached_splits(cache_dir,validation_fraction,seed)
 return {"official_train":len(train),"official_validation":len(val)+len(test),"validation":len(val),"test":len(test),"seed":seed}

def encode_batch(dataset, tokenizer_path, context_length, start, batch_size):
 tok=Tokenizer.from_file(tokenizer_path); rows=[]
 for i in range(batch_size):
  ids=tok.encode(dataset[(start+i)%len(dataset)]["text"]).ids
  if len(ids)<2: ids=[0,0]
  ids=(ids+[0]*context_length)[:context_length+1]; rows.append(ids)
 import torch
 x=torch.tensor(rows,dtype=torch.long); return x[:,:-1],x[:,1:]
