import torch
from src.models import *
torch.manual_seed(7);m=DecoderOnlyTransformer(ModelConfig(32,32,1,4,64,"diff_v1"));o=torch.optim.AdamW(m.parameters(),lr=.003);x=torch.arange(9).repeat(8,1)%32;q=[]
for _ in range(30):
 _,l=m(x[:,:-1],x[:,1:]);o.zero_grad();l.backward();o.step();q.append(l.item())
assert q[-1]<q[0];print("tiny overfit:",q[0],q[-1])
