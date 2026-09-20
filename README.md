
## Controlled adaptive lambda variants

All four Differential Attention models use the exact Diff V1 paired-map formulation: A_diff = A1 - lambda * A2. They retain the same Q/K/V layout, RoPE, V1 layer lambda, headwise RMSNorm, V1 output scale, and output projection. The only difference is lambda granularity.

- diff_v1: lambda = lambda_layer, the original learned V1 scalar.
- diff_headwise: lambda = lambda_layer + delta_lambda_head; delta has one zero-initialized learned scalar per differential head.
- diff_tokenwise: lambda = lambda_layer + delta_lambda_token(x_t); the zero-initialized projection maps d_model to one value shared by heads.
- diff_token_headwise: lambda = lambda_layer + delta_lambda_token_head(x_t); the zero-initialized projection maps d_model to one value per differential head.

These are controlled extensions of Diff V1, not V2 reproductions: no sigmoid constraints, FlashAttention, GQA, or V2 projection/value-layout changes are used. All adaptive variants exactly match a V1 module with identical weights at initialization.
