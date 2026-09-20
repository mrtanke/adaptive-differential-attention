
## Lambda variants

- diff_v1: original layer-shared scalar computed from four learned head-dimension vectors.
- diff_headwise: one learned static scalar per differential head.
- diff_tokenwise: one learned query-token scalar shared by all differential heads.
- diff_token_headwise: learned query-token/head logits, passed through sigmoid, following V2 lambda projection semantics while retaining V1 paired-map/value/RMSNorm/output structure.

V2 differences: no FlashAttention/GQA, no V2 projection/value layout changes, and a bias is initialized so lambda starts at the V1 depth initialization instead of V2s uninitialized linear projection behavior.
