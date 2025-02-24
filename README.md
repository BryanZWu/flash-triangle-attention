# Triton Implementation of Triangular Attention

This repository contains an efficient implementation of 
triangular attention using Triton, specifically designed for 
protein structure prediction models like AlphaFold2. 
The implementation is based on the Flash Attention 2.0
algorithm but modified to handle the unique requirements 
of triangular attention: 5d tensors operating over pairwise
representations (B, H, L, L, D), and a bias term over
the attention weights, through which we wish to
propagate gradients.

## Technical Details

### Input Format
- Query (Q): `(Batch, Heads, L1, L2, Dims)`
- Key (K): `(Batch, Heads, L1, L2, Dims)`
- Value (V): `(Batch, Heads, L1, L2, Dims)`
- Bias (B): `(Batch, Heads, L1, L2)`

### Key Differences from Standard Attention

1. **Additional Bias Term**: Unlike standard attention
    which only uses QK^T similarity, this implementation
    includes an additional bias term that modulates the
    attention scores. 
2. **5D Tensors**: Works with 5D tensors (B, H, L1, L2, D)
    instead of the standard 4D tensors used in 
    transformer attention

## Performance Results

Benchmarked on an NVIDIA L4 GPU with the following parameters:
- Batch Size: 4
- Heads: 8
- Head Dimension: 32

### Memory Usage
![Memory Usage](test_results/fused-attention-batch4-head8-d32-fwd-memory.png)

### Latency
![Latency](test_results/fused-attention-batch4-head8-d32-fwd-latency.png)

Key findings:
- Triton implementation shows significant memory savings compared to vanilla PyTorch
- Performance advantages become more pronounced at longer sequence lengths
- Maintains numerical stability while achieving better memory efficiency

## Usage

```python
import torch
from flash_tt_attn import attention
# Initialize inputs
q = torch.randn(batch, heads, seq_len, seq_len, head_dim, dtype=torch.float16, device="cuda")
k = torch.randn_like(q)
v = torch.randn_like(q)
bias = torch.randn(batch, heads, seq_len, seq_len, dtype=torch.float16, device="cuda")
scale = 1.0 / (head_dim ** 0.5)

# Compute attention
output = attention(q, k, v, bias, scale)
```



