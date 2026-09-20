import torch
from torch import nn
import torch.nn.functional as F
from .attention import RMSNorm,CausalSelfAttention,DifferentialAttentionV1
class SwiGLU(nn.Module):
 def __init__(s,d,f): super().__init__();s.g=nn.Linear(d,f,False);s.u=nn.Linear(d,f,False);s.o=nn.Linear(f,d,False)
 def forward(s,x): return s.o(F.silu(s.g(x))*s.u(x))
class Block(nn.Module):
 def __init__(s,c,i):
  super().__init__();s.an=RMSNorm(c.d_model);s.fn=RMSNorm(c.d_model);s.attn=CausalSelfAttention(c.d_model,c.n_heads) if c.attention_type=="standard" else DifferentialAttentionV1(c.d_model,c.n_heads,i);s.ff=SwiGLU(c.d_model,c.ffn_dim)
 def forward(s,x): x=x+s.attn(s.an(x));return x+s.ff(s.fn(x))
class DecoderOnlyTransformer(nn.Module):
 def __init__(s,c):
  super().__init__();s.config=c;s.token_embedding=nn.Embedding(c.vocab_size,c.d_model);s.blocks=nn.ModuleList(Block(c,i)for i in range(c.n_layers));s.final_norm=RMSNorm(c.d_model);s.lm_head=nn.Linear(c.d_model,c.vocab_size,False)
 def forward(s,ids,targets=None):
  x=s.token_embedding(ids)
  for b in s.blocks:x=b(x)
  z=s.lm_head(s.final_norm(x));return z if targets is None else(z,F.cross_entropy(z.flatten(0,1),targets.flatten()))
