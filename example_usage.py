import torch
from flash_tt_attn import attention
# Initialize inputs
batch = 4
heads = 8
seq_len = 128
head_dim = 32
q = torch.randn(batch, heads, seq_len, seq_len, head_dim, dtype=torch.float16, device="cuda", requires_grad=True)
k = torch.randn_like(q)
v = torch.randn_like(q)
bias = torch.randn(batch, heads, seq_len, seq_len, dtype=torch.float16, device="cuda", requires_grad=True)
scale = 1.0 / (head_dim ** 0.5)

# Compute attention
output = attention(q, k, v, bias, scale)
loss = torch.nn.functional.mse_loss(output, torch.randn_like(output))
loss.backward()

print(q.grad)
print(k.grad)
print(v.grad)
print(bias.grad)
