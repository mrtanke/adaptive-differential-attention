import torch
from src.models import *
TYPES=("standard","diff_v1","diff_headwise","diff_tokenwise","diff_token_headwise")
def cfg(t): return ModelConfig(31,32,2,4,64,t)
def test_forward_shape():
 for t in TYPES: assert DecoderOnlyTransformer(cfg(t))(torch.randint(0,31,(2,7))).shape==(2,7,31)
def test_backward_and_finite():
 for t in TYPES:
  m=DecoderOnlyTransformer(cfg(t));x=torch.randint(0,31,(2,7));_,l=m(x[:,:-1],x[:,1:]);l.backward();assert torch.isfinite(l) and all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
def test_causal_masking():
 for t in TYPES:
  a=DecoderOnlyTransformer(cfg(t)).blocks[0].attn;_,w=a(torch.randn(1,6,32),True);f=torch.triu(torch.ones(6,6),1).bool();assert (w[...,f]==0).all()
def test_lambda_granularity():
 m=DecoderOnlyTransformer(cfg("diff_v1"));assert m.blocks[0].attn.lambda_full().ndim==0
 m=DecoderOnlyTransformer(cfg("diff_headwise"));assert m.blocks[0].attn.lambda_head.shape==(2,)
 x=torch.randn(3,5,32);m=DecoderOnlyTransformer(cfg("diff_tokenwise"));assert m.blocks[0].attn.lambda_values(x).shape==(3,1,5,1)
 m=DecoderOnlyTransformer(cfg("diff_token_headwise"));assert m.blocks[0].attn.lambda_values(x).shape==(3,2,5,1)
def test_no_nan_inf():
 for t in TYPES: assert torch.isfinite(DecoderOnlyTransformer(cfg(t))(torch.randint(0,31,(2,33)))).all()
