import math
import torch
from torch import nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5): super().__init__(); self.eps=eps; self.weight=nn.Parameter(torch.ones(dim))
    def forward(self,x): return x*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.eps).to(x.dtype)*self.weight
class RoPE(nn.Module):
    def __init__(self, dim): super().__init__(); self.register_buffer("f",1/(10000**(torch.arange(0,dim,2).float()/dim)),False)
    def forward(self,x):
        p=torch.outer(torch.arange(x.size(1),device=x.device,dtype=self.f.dtype),self.f).to(x.dtype); c,s=p.cos()[None,:,None,:],p.sin()[None,:,None,:]; a,b=x[...,::2],x[...,1::2]
        return torch.stack((a*c-b*s,a*s+b*c),-1).flatten(-2)
def causal_mask(t,device): return torch.ones(t,t,device=device,dtype=torch.bool).tril()
def lambda_init_fn(depth): return .8-.6*math.exp(-.3*depth)

class CausalSelfAttention(nn.Module):
    def __init__(self,d,h):
        super().__init__(); assert d%h==0; self.h=h; self.k=d//h; self.q_proj=nn.Linear(d,d,False); self.k_proj=nn.Linear(d,d,False); self.v_proj=nn.Linear(d,d,False); self.out_proj=nn.Linear(d,d,False); self.rope=RoPE(self.k)
    def forward(self,x,return_attention=False):
        b,t,_=x.shape; q,k,v=(p(x).view(b,t,self.h,self.k) for p in (self.q_proj,self.k_proj,self.v_proj)); q,k=self.rope(q),self.rope(k); score=torch.nan_to_num((q.transpose(1,2)@k.transpose(1,2).transpose(-2,-1))*self.k**-.5); w=F.softmax(score.masked_fill(~causal_mask(t,x.device),float("-inf")),dim=-1,dtype=torch.float32).to(x.dtype); y=self.out_proj((w@v.transpose(1,2)).transpose(1,2).reshape(b,t,-1)); return (y,w) if return_attention else y

class DifferentialAttentionBase(nn.Module):
    def __init__(self,d,h,depth):
        super().__init__(); assert h%2==0 and d%h==0; self.num_heads=h//2; self.head_dim=d//h; self.d_model=d; self.lambda_init=lambda_init_fn(depth); self.q_proj=nn.Linear(d,d,False); self.k_proj=nn.Linear(d,d,False); self.v_proj=nn.Linear(d,d,False); self.out_proj=nn.Linear(d,d,False); self.rope=RoPE(self.head_dim); self.subln=RMSNorm(2*self.head_dim)
    def lambda_values(self,x): raise NotImplementedError
    def forward(self,x,return_attention=False):
        b,t,d=x.shape; q=self.rope(self.q_proj(x).view(b,t,2*self.num_heads,self.head_dim)); k=self.rope(self.k_proj(x).view(b,t,2*self.num_heads,self.head_dim)); v=self.v_proj(x).view(b,t,self.num_heads,2*self.head_dim); score=torch.nan_to_num((q.transpose(1,2)@k.transpose(1,2).transpose(-2,-1))*self.head_dim**-.5); maps=F.softmax(score.masked_fill(~causal_mask(t,x.device),float("-inf")),dim=-1,dtype=torch.float32).to(x.dtype).view(b,self.num_heads,2,t,t); lam=self.lambda_values(x).to(x.dtype); w=maps[:,:,0]-lam*maps[:,:,1]; y=self.out_proj((self.subln(w@v.transpose(1,2))*(1-self.lambda_init)).transpose(1,2).reshape(b,t,d)); return (y,w) if return_attention else y

class DifferentialAttentionV1(DifferentialAttentionBase):
    def __init__(self,d,h,depth):
        super().__init__(d,h,depth); self.lambda_q1=nn.Parameter(torch.randn(self.head_dim)*.1); self.lambda_k1=nn.Parameter(torch.randn(self.head_dim)*.1); self.lambda_q2=nn.Parameter(torch.randn(self.head_dim)*.1); self.lambda_k2=nn.Parameter(torch.randn(self.head_dim)*.1)
    def lambda_full(self): return torch.exp((self.lambda_q1*self.lambda_k1).sum().float())-torch.exp((self.lambda_q2*self.lambda_k2).sum().float())+self.lambda_init
    def lambda_values(self,x): return self.lambda_full()

class HeadwiseDifferentialAttention(DifferentialAttentionBase):
    def __init__(self,d,h,depth): super().__init__(d,h,depth); self.lambda_head=nn.Parameter(torch.full((self.num_heads,),self.lambda_init))
    def lambda_values(self,x): return self.lambda_head[None,:,None,None]

class TokenwiseDifferentialAttention(DifferentialAttentionBase):
    def __init__(self,d,h,depth):
        super().__init__(d,h,depth); self.lambda_proj=nn.Linear(d,1); nn.init.zeros_(self.lambda_proj.weight); nn.init.constant_(self.lambda_proj.bias,self.lambda_init)
    def lambda_values(self,x): return self.lambda_proj(x).transpose(1,2).unsqueeze(-1)

class TokenHeadwiseDifferentialAttention(DifferentialAttentionBase):
    def __init__(self,d,h,depth):
        super().__init__(d,h,depth); self.lambda_proj=nn.Linear(d,self.num_heads); nn.init.zeros_(self.lambda_proj.weight); nn.init.constant_(self.lambda_proj.bias,math.log(self.lambda_init/(1-self.lambda_init)))
    def lambda_values(self,x): return torch.sigmoid(self.lambda_proj(x)).transpose(1,2).unsqueeze(-1)
