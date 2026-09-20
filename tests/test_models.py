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
  _,w=DecoderOnlyTransformer(cfg(t)).blocks[0].attn(torch.randn(1,6,32),True);f=torch.triu(torch.ones(6,6),1).bool();assert (w[...,f]==0).all()
def test_lambda_granularity_and_zero_init():
 x=torch.randn(3,5,32);v=DifferentialAttentionV1(32,4,1); assert v.lambda_values(x).ndim==0
 h=HeadwiseDifferentialAttention(32,4,1); assert h.delta_lambda_head.shape==(2,) and torch.equal(h.delta_lambda_head,torch.zeros_like(h.delta_lambda_head))
 t=TokenwiseDifferentialAttention(32,4,1); assert t.lambda_values(x).shape==(3,1,5,1) and not t.delta_lambda_token.weight.any() and not t.delta_lambda_token.bias.any()
 th=TokenHeadwiseDifferentialAttention(32,4,1); assert th.lambda_values(x).shape==(3,2,5,1) and not th.delta_lambda_token_head.weight.any() and not th.delta_lambda_token_head.bias.any()
def test_adaptive_variants_match_v1_at_init():
 torch.manual_seed(2);x=torch.randn(2,6,32);v=DifferentialAttentionV1(32,4,1)
 for cls in (HeadwiseDifferentialAttention,TokenwiseDifferentialAttention,TokenHeadwiseDifferentialAttention):
  a=cls(32,4,1);a.load_state_dict(v.state_dict(),strict=False);assert torch.allclose(v(x),a(x),atol=1e-6,rtol=1e-6)
def test_no_nan_inf():
 for t in TYPES: assert torch.isfinite(DecoderOnlyTransformer(cfg(t))(torch.randint(0,31,(2,33)))).all()
