import torch
from src.models import *
for kind in ("standard","diff_v1","diff_headwise","diff_tokenwise","diff_token_headwise"):
 torch.manual_seed(7);m=DecoderOnlyTransformer(ModelConfig(32,32,1,4,64,kind));o=torch.optim.AdamW(m.parameters(),lr=.003);x=torch.arange(9).repeat(8,1)%32;q=[]
 for _ in range(20):
  _,l=m(x[:,:-1],x[:,1:]);o.zero_grad();l.backward();o.step();q.append(l.item())
 assert q[-1]<q[0],(kind,q);print(kind,round(q[0],3),round(q[-1],3),sum(p.numel() for p in m.parameters()))
