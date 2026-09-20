import math,torch
from torch import nn
import torch.nn.functional as F
class RMSNorm(nn.Module):
 def __init__(s,d,eps=1e-5): super().__init__();s.eps=eps;s.weight=nn.Parameter(torch.ones(d))
 def forward(s,x): return x*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+s.eps).to(x.dtype)*s.weight
class RoPE(nn.Module):
 def __init__(s,d): super().__init__();s.register_buffer("f",1/(10000**(torch.arange(0,d,2).float()/d)),False)
 def forward(s,x):
  p=torch.outer(torch.arange(x.size(1),device=x.device,dtype=s.f.dtype),s.f).to(x.dtype);c,z=p.cos()[None,:,None,:],p.sin()[None,:,None,:];a,b=x[...,::2],x[...,1::2];return torch.stack((a*c-b*z,a*z+b*c),-1).flatten(-2)
def causal_mask(t,dev): return torch.ones(t,t,device=dev,dtype=torch.bool).tril()
def lambda_init_fn(depth): return .8-.6*math.exp(-.3*depth)
class CausalSelfAttention(nn.Module):
 def __init__(s,d,h):
  super().__init__();assert d%h==0;s.h=h;s.k=d//h;s.q=nn.Linear(d,d,False);s.kp=nn.Linear(d,d,False);s.v=nn.Linear(d,d,False);s.o=nn.Linear(d,d,False);s.rope=RoPE(s.k)
 def forward(s,x,return_attention=False):
  b,t,_=x.shape;q,k,v=(p(x).view(b,t,s.h,s.k) for p in(s.q,s.kp,s.v));q,k=s.rope(q),s.rope(k);w=F.softmax(((q.transpose(1,2)@k.transpose(1,2).transpose(-2,-1))*s.k**-.5).masked_fill(~causal_mask(t,x.device),float("-inf")),dim=-1,dtype=torch.float32).to(x.dtype);y=s.o((w@v.transpose(1,2)).transpose(1,2).reshape(b,t,-1));return(y,w)if return_attention else y
class DifferentialAttentionV1(nn.Module):
 def __init__(s,d,h,depth):
  super().__init__();assert h%2==0 and d%h==0;s.h=h//2;s.k=d//h;s.lambda_init=lambda_init_fn(depth);s.q=nn.Linear(d,d,False);s.kp=nn.Linear(d,d,False);s.v=nn.Linear(d,d,False);s.o=nn.Linear(d,d,False);s.rope=RoPE(s.k);s.lambda_q1=nn.Parameter(torch.randn(s.k)*.1);s.lambda_k1=nn.Parameter(torch.randn(s.k)*.1);s.lambda_q2=nn.Parameter(torch.randn(s.k)*.1);s.lambda_k2=nn.Parameter(torch.randn(s.k)*.1);s.subln=RMSNorm(2*s.k)
 def lambda_full(s): return torch.exp((s.lambda_q1*s.lambda_k1).sum().float())-torch.exp((s.lambda_q2*s.lambda_k2).sum().float())+s.lambda_init
 def forward(s,x,return_attention=False):
  b,t,d=x.shape;q=s.rope(s.q(x).view(b,t,2*s.h,s.k));k=s.rope(s.kp(x).view(b,t,2*s.h,s.k));v=s.v(x).view(b,t,s.h,2*s.k);scores=(q.transpose(1,2)@k.transpose(1,2).transpose(-2,-1))*s.k**-.5;scores=torch.nan_to_num(scores);maps=F.softmax(scores.masked_fill(~causal_mask(t,x.device),float("-inf")),dim=-1,dtype=torch.float32).to(x.dtype).view(b,s.h,2,t,t);w=maps[:,:,0]-s.lambda_full().to(x.dtype)*maps[:,:,1];y=s.o((s.subln(w@v.transpose(1,2))*(1-s.lambda_init)).transpose(1,2).reshape(b,t,d));return(y,w)if return_attention else y
