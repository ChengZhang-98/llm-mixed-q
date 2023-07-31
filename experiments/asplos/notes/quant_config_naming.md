# Naming of quantisation config toml

## BERT

including bert-base, and bert-large

The `i` -th Transformer block has the following layer config entries:

| Transformer block | layer entry | Notes |
| ----------------- | ------------ | ----- |
| `Q <= X_n W_Q + b` | `model_layer_i.attention.query` | |
| `K <= X_n W_K + b` | `model_layer_i.attention.key` | |
| `V <= X_n W_V + b` | `model_layer_i.attention.value` | |
| `A <= QK^T` | `model_layer_i.attention.matmul_0` | |
| `B <= A_hat V` | `model_layer_i.attention.matmul_1` | |
| `B0 <= B_c W_0 + b` | `model_layer_i.attention.output.dense` | The output linear of attention |
| `B1 <= B0 W_1 + b` | `model_layer_i.intermediate.dense` | The 1st linear of FFN |
| `B2 <= B1 W_2 + b` | `model_layer_i.output.dense` | The 2nd linear of FFN |

## OPT

including opt-125m, opt-350m

| Transformer block | layer entry | Notes |
| ----------------- | ------------ | ----- |
| `Q <= X_n W_Q + b` | `model_layer_i.self_attn.q_proj` | |
| `K <= X_n W_K + b` | `model_layer_i.self_attn.k_proj` | |
| `V <= X_n W_V + b` | `model_layer_i.self_attn.v_proj` | |
| `A <= QK^T` | `model_layer_i.self_attn.bmm_0` | |
| `B <= A_hat V` | `model_layer_i.self_attn.bmm_1` | |
| `B0 <= B_c W_0 + b` | `model_layer_i.self_attn.out_proj` | The output linear of attention |
| `B1 <= B0 W_1 + b` | `model_layer_i.fc1` | The 1st linear of FFN |
| `B2 <= B1 W_2 + b` | `model_layer_i.fc2` | The 2nd linear of FFN |

For opt-1.3b, opt-2.7b, opt-6.7b, the only block config with entry `model_layer` is shared across all blocks.

## Llama

including llama-160m

All Transformer blocks use the same integer quantisation config. Rotary positional encoding is 4-bit integer with fraction point between 3 and 4 (`0bx.xxx`).

| Transformer block | layer entry | Notes |
| ----------------- | ------------ | ----- |
| `Q <= X_n W_Q` | `model_layer_i.self_attn.q_proj` | |
| `Q <= RoPE(Q)` | `model_layer_i.self_attn.rotary_positional_encoding` | 4-bit **integer** quantisation |
| `K <= X_n W_K` | `model_layer_i.self_attn.k_proj` | |
| `V <= X_n W_V` | `model_layer_i.self_attn.v_proj` | |
| `A <= QK^T` | `model_layer_i.self_attn.matmul_0` | |
| `B <= A_hat V` | `model_layer_i.self_attn.matmul_1` | |
| `B0 <= B_c W_0` | `model_layer_i.self_attn.o_proj` | The output linear of attention |
| `G <= B_n W_G` | `model_layer_i.mlp.gate_proj` | gate linear |
| `U <= B_n W_U` | `model_layer_i.mlp.up_proj` | up linear |
| `D <= [SiLU(G ⊗ U)] W_D` | `model_layer_i.mlp.down` | down linear |

For llama-7b, alpaca-b, and vicuna-7b, the only block config with entry `model_layer` is shared across all blocks.

```text
# Llama Transformer block
# Norm
X_n = RMSNorm(X)

# Multi-head attention
for 0 in 0,1,2,...,num_heads-1
  Q <= X_n W_Q
  K <= X_n W_K
  V <= X_n W_V
  Q <= RoPE(Q)
  K <= RoPE(Q)
  A <= QK^T/sqrt(d_k)
  A_hat <= softmax(A, axis=-1)
  B <= A_hat V
end for
B_c <= concat(B_0, ..., B_H-1)
B_0 <= B_c W_0

# Residual + Norm
B_n <= RMSNorm(B_0 + X)

# MLP/FFN
G <= B_n W_G
U <= B_n W_U
D <= [SiLU(G ⊗ U)] W_D

# Reisdual
O <= D + B_0 + X
```
