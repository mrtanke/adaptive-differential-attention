## Adaptive Differential Attention

Standard self-attention computes an attention map from the similarity between queries and keys:

$$
A = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right).
$$

[Differential Transformer](https://proceedings.iclr.cc/paper_files/paper/2025/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html) (Ye et al., ICLR 2025) extends this idea by computing two attention maps, $A_1$ and $A_2$, and subtracting the second from the first:

$$
A_{\mathrm{diff}} = A_1 - \lambda A_2.
$$

The coefficient $\lambda$ controls the strength of the subtraction. 

This project studies four Differential Attention variants that differ only in the granularity of $\lambda$.

### Differential Attention

The original formulation uses one coefficient per layer:

$$
A_{\mathrm{diff}}^{(l)}
=
A_1^{(l)}
-
\lambda_l A_2^{(l)}.
$$

### Head-wise Differential Attention

Each differential attention head has its own coefficient:

$$
A_{\mathrm{diff}}^{(l,h)}
=
A_1^{(l,h)}
-
\lambda_{l,h} A_2^{(l,h)}.
$$

### Token-wise Differential Attention

Each query token has its own coefficient, shared across heads:

$$
A_{\mathrm{diff}}^{(l,t)}
=
A_1^{(l,t)}
-
\lambda_{l,t} A_2^{(l,t)}.
$$

### Token + Head-wise Differential Attention

Each query token and attention head has its own coefficient:

$$
A_{\mathrm{diff}}^{(l,h,t)}
=
A_1^{(l,h,t)}
-
\lambda_{l,h,t} A_2^{(l,h,t)}.
$$

All adaptive variants are initialized to match the original Differential Attention behavior. The remaining Transformer architecture is kept unchanged to enable a controlled comparison.