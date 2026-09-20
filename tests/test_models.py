import torch
from src.models import *
def cfg(t): return ModelConfig(31,32,2,4,64,t)
for_type=("standard","diff_v1")
def test_forward():
 for t in for_type: assert DecoderOnlyTransformer(cfg(t))(torch.randint(0,31,(2,7))).shape==(2,7,31)
def test_backward_finite():
 for t in for_type:
  m=DecoderOnlyTransformer(cfg(t));x=torch.randint(0,31,(2,7));_,l=m(x[:,:-1],x[:,1:]);l.backward();assert torch.isfinite(l) and all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
def test_causal():
 _,w=CausalSelfAttention(32,4)(torch.randn(1,6,32),True);f=torch.triu(torch.ones(6,6),1).bool();assert (w[...,f]==0).all()
def test_lambda_layerwise():
 m=DecoderOnlyTransformer(cfg("diff_v1"));a,b=(x.attn for x in m.blocks);assert a.lambda_q1.ndim==1 and a.lambda_q1 is not b.lambda_q1 and a.lambda_full().ndim==0
